"""Processing of data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import pickle
import pandas as pd
from os.path import join

import torch

from src import utils
from collections import defaultdict as ddict


class Data(object):
    '''The abustrct class that defines interfaces for holding all data.
    '''

    def __init__(self, args):
        # concept vocab
        self.cons = []
        # rel vocab
        self.rels = []
        # transitive rels vocab
        self.index_cons = {}  # {string: index}
        self.index_rels = {}  # {string: index}
        # save triples as array of indices
        self.triples = np.array([0])  # training dataset
        self.val_triples = np.array([0])  # validation dataset
        self.test_triples = np.array([0])  # test dataset

        self.hr_map = {} # ndcg test
        self.tr_map = {}

        self.hr2t_all = ddict(set)
        self.rt2h_all = ddict(set)

        # use for rgcn
        self.adj_list = None
        self.rels_list = []
        # (h,r,t) tuples(int), no w
        # set containing train, val, test (for negative sampling).
        self.triples_record = set([])
        self.weights = np.array([0])

        self.neg_triples = np.array([0])
        # map for sigma
        # head per tail and tail per head (for each relation). used for bernoulli negative sampling
        self.hpt = np.array([0])
        self.tph = np.array([0])

        # test for rank
        self.head_candidate = set([])
        self.tail_candidate = set([])

        # recorded for tf_parts
        self.threshold = args.threshold
        self.dim = args.dim
        self.batch_size = args.batch_size
        self.L1 = False
        self.last_c = -1
        self.last_r = -1

    def load_triples(self, filename, splitter='\t', line_end='\n'):
        '''Load the dataset'''
        triples = []

        hr_map = {}
        tr_map = {}

        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            if self.index_cons.get(line[0]) == None:
                self.cons.append(line[0])
                self.last_c += 1
                self.index_cons[line[0]] = self.last_c
            if self.index_cons.get(line[2]) == None:
                self.cons.append(line[2])
                self.last_c += 1
                self.index_cons[line[2]] = self.last_c
            if self.index_rels.get(line[1]) == None:
                self.rels.append(line[1])
                self.last_r += 1
                self.index_rels[line[1]] = self.last_r

            h = self.index_cons[line[0]]
            r = self.index_rels[line[1]]
            t = self.index_cons[line[2]]
            w = float(line[3])

            self.hr2t_all[(h, r)].add(t)
            self.rt2h_all[(r, t)].add(h)


            triples.append([h, r, t, w])
            self.triples_record.add((h, r, t))
        return np.array(triples)
    
    def get_head_candidates(self):
        """获取数据集中出现过的头实体候选集合"""
        return list(self.head_candidate)
    
    def get_tail_candidates(self):
        """获取数据集中出现过的尾实体候选集合"""
        return list(self.tail_candidate)

    def load_data(self, file_train, file_val, file_test, splitter='\t', line_end='\n'):

        self.triples = self.load_triples(file_train, splitter, line_end)
        self.val_triples = self.load_triples(file_val, splitter, line_end)
        self.test_triples = self.load_triples(file_test, splitter, line_end)


        # init adj for every rel
        for rel in self.rels:
            self.rels_list.append(self.index_rels[rel])

        # 优化的图构建 - 使用向量化操作，同时计算tph和hpt统计
        self._build_optimized_adj_list()

    def _build_optimized_adj_list(self):
        """优化的邻接列表构建，使用向量化操作，同时计算tph和hpt统计"""
        
        # 计算tph和hpt统计信息
        tph_array = np.zeros((len(self.rels), len(self.cons)))
        hpt_array = np.zeros((len(self.rels), len(self.cons)))
        
        for h_, r_, t_, w in self.triples:  # 使用所有训练数据计算统计信息
            h, r, t = int(h_), int(r_), int(t_)
            tph_array[r][h] += 1.
            hpt_array[r][t] += 1.
        
        self.tph = np.mean(tph_array, axis=1)
        self.hpt = np.mean(hpt_array, axis=1)
        
        # 筛选高质量三元组构建图
        train_data = self.triples
        mask = train_data[:, 3] >= self.threshold
        filtered_triples = train_data[mask]
        
        print(f"Filtered {len(filtered_triples)} high-quality triples from {len(train_data)} total triples")
        
        self.adj_list = {}
        
        # 按关系分组处理
        for rel_idx in self.rels_list:
            # 使用向量化操作筛选该关系的三元组
            rel_mask = filtered_triples[:, 1] == rel_idx
            rel_triples = filtered_triples[rel_mask]
            
            if len(rel_triples) > 0:
                # 直接构建PyTorch张量，避免Python列表操作
                src_nodes = torch.tensor(rel_triples[:, 0], dtype=torch.long)
                dst_nodes = torch.tensor(rel_triples[:, 2], dtype=torch.long)
                weights = torch.tensor(rel_triples[:, 3], dtype=torch.float)
                
                # 构建边索引张量
                edge_index = torch.stack([src_nodes, dst_nodes], dim=0)
                
                self.adj_list[rel_idx] = (edge_index, weights)
                # print(f"Relation {rel_idx}: {len(rel_triples)} edges")
            else:
                # 为没有边的关系创建空张量
                empty_edge_index = torch.zeros((2, 0), dtype=torch.long)
                empty_weights = torch.zeros(0, dtype=torch.float)
                self.adj_list[rel_idx] = (empty_edge_index, empty_weights)


    # add more triples to self.triples_record to 'filt' negative sampling
    def record_more_data(self, filename, splitter='\t', line_end='\n'):
        for line in open(filename):
            line = line.rstrip(line_end).split(splitter)
            if len(line) < 3:
                continue
            h = self.con_str2index(line[0])
            r = self.rel_str2index(line[1])
            t = self.con_str2index(line[2])
            w = line[3]
            if h != None and r != None and t != None:
                self.triples_record.add((h, r, t))
        # print("Loaded %s to triples_record." % (filename))
        # print("Update: total number of triples in set:", len(self.triples_record))

    def load_hr_map(self, filename, splitter='\t', line_end='\n'):
        """
        Initialize self.hr_map.
        Load self.hr_map={h:{r:t:w}}}, not restricted to test data
        :return:
        """

        with open(join(filename, 'test.tsv')) as f:
            for line in f:
                line = line.rstrip(line_end).split(splitter)
                h = self.con_str2index(line[0])
                r = self.rel_str2index(line[1])
                t = self.con_str2index(line[2])
                w = float(line[3])

                # construct hr_map
                if self.hr_map.get(h) == None:
                    self.hr_map[h] = {}
                if self.hr_map[h].get(r) == None:
                    self.hr_map[h][r] = {t: w}
                else:
                    self.hr_map[h][r][t] = w

                if self.tr_map.get(t) == None:
                    self.tr_map[t] = {}
                if self.tr_map[t].get(r) == None:
                    self.tr_map[t][r] = {h: w}
                else:
                    self.tr_map[t][r][h] = w

                self.head_candidate.add(h)
                self.tail_candidate.add(t)

        count = 0
        for h in self.hr_map:
            count += len(self.hr_map[h])
        print('Loaded ranking test queries. Number of (h,r,?t) queries: %d' % count)

        supplement_t_files = ['train.tsv', 'val.tsv','test.tsv']
        for file in supplement_t_files:
            with open(join(filename, file)) as f:
                for line in f:
                    line = line.rstrip(line_end).split(splitter)
                    h = self.con_str2index(line[0])
                    r = self.rel_str2index(line[1])
                    t = self.con_str2index(line[2])
                    w = float(line[3])

                    # update hr_map
                    if h in self.hr_map and r in self.hr_map[h]:
                        self.hr_map[h][r][t] = w

                    if t in self.tr_map and r in self.tr_map[t]:
                        self.tr_map[t][r][h] = w
    def num_cons(self):
        '''Returns number of ontologies.

        This means all ontologies have index that 0 <= index < num_onto().
        '''
        return len(self.cons)

    def num_rels(self):
        '''Returns number of relations.

        This means all relations have index that 0 <= index < num_rels().
        Note that we consider *ALL* relations, e.g. $R_O$, $R_h$ and $R_{tr}$.
        '''
        return len(self.rels)

    def rel_str2index(self, rel_str):
        '''For relation `rel_str` in string, returns its index.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.index_rels.get(rel_str)

    def rel_index2str(self, rel_index):
        '''For relation `rel_index` in int, returns its string.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.rels[rel_index]

    def con_str2index(self, con_str):
        '''For ontology `con_str` in string, returns its index.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.index_cons.get(con_str)

    def con_index2str(self, con_index):
        '''For ontology `con_index` in int, returns its string.

        This is not used in training, but can be helpful for visualizing/debugging etc.'''
        return self.cons[con_index]

    def rel(self):
        return np.array(range(self.num_rels()))

    def corrupt_pos(self, triple, pos):
        """
        :param triple: [h, r, t]
        :param pos: index position to replace (0 for h, 2 fot t)
        :return: [h', r, t] or [h, r, t']
        """
        hit = True
        res = None
        while hit:
            res = np.copy(triple)
            samp = np.random.randint(self.num_cons())
            while samp == triple[pos]:
                samp = np.random.randint(self.num_cons())
            res[pos] = samp
            # # debug
            # if tuple(res) in self.triples_record:
            #     print('negative sampling: rechoose')
            #     print(res)
            if tuple(res) not in self.triples_record:
                hit = False
        return res

    # bernoulli negative sampling
    def corrupt(self, triple, neg_per_positive, tar=None):
        """
        :param triple: [h r t]
        :param tar: 't' or 'h'
        :return: np.array [[h,r,t1],[h,r,t2],...]
        """
        # print("array.shape:", res.shape)
        if tar == 't':
            position = 2
        elif tar == 'h':
            position = 0
        res = [self.corrupt_pos(triple, position) for i in range(neg_per_positive)]
        return np.array(res)

    class index_dist:
        def __init__(self, index, dist):
            self.dist = dist
            self.index = index
            return

        def __lt__(self, other):
            return self.dist > other.dist

    # bernoulli negative sampling on a batch
    def corrupt_batch(self, t_batch, neg_per_positive, tar=None):
        res = np.array([self.corrupt(triple, neg_per_positive, tar) for triple in t_batch])
        return res

    def save(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.__dict__, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        # print("Save data object as", filename)

    def load(self, filename):
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)
        print("Loaded data object from", filename)

    def save_meta_table(self, save_dir):
        """
        save index-con, index-rel table to file.
        File: idx_concept.csv, idx_relation.csv
        :return:
        """
        idx_con_path = join(save_dir, 'idx_concept.csv')
        df_con = pd.DataFrame({'index': list(self.index_cons.values()), 'concepts': list(self.index_cons.keys())})
        df_con.sort_values(by='index').to_csv(idx_con_path, index=None)

        idx_rel_path = join(save_dir, 'idx_relation.csv')
        df_rel = pd.DataFrame({'index': list(self.index_rels.values()), 'relations': list(self.index_rels.keys())})
        df_rel.sort_values(by='index').to_csv(idx_rel_path, index=None)


class BatchLoader():
    def __init__(self, data_obj, batch_size, neg_per_positive):
        self.this_data = data_obj  # Data() object
        self.shuffle = True
        self.batch_size = batch_size
        self.neg_per_positive = neg_per_positive

    def gen_batch(self, forever=False, shuffle=True):
        """
        """
        l = self.this_data.triples.shape[0]
        while True:
            triples = self.this_data.triples  # np.float64 [[h,r,t,w]]
            if shuffle:
                np.random.shuffle(triples)
            for i in range(0, l, self.batch_size):
                batch = triples[i: i + self.batch_size, :]
                if batch.shape[0] < self.batch_size:
                    batch = np.concatenate((batch, self.this_data.triples[:self.batch_size - batch.shape[0]]),
                                           axis=0)
                    assert batch.shape[0] == self.batch_size

                h_batch, r_batch, t_batch, w_batch = batch[:, 0].astype(int), batch[:, 1].astype(int), batch[:,
                                                                                                       2].astype(
                    int), batch[:, 3]
                hrt_batch = batch[:, 0:3].astype(int)

                # all_neg_hn_batch = self.corrupt_batch(hrt_batch, self.neg_per_positive, "h")
                # all_neg_tn_batch = self.corrupt_batch(hrt_batch, self.neg_per_positive, "t")

                neg_hn_batch, neg_rel_hn_batch, \
                neg_t_batch, neg_h_batch, \
                neg_rel_tn_batch, neg_tn_batch \
                    = self.corrupt_batch(h_batch, r_batch, t_batch)

                yield h_batch.astype(np.int64), r_batch.astype(np.int64), t_batch.astype(
                    np.int64), w_batch.astype(
                    np.float32), \
                    neg_hn_batch.astype(np.int64), neg_rel_hn_batch.astype(np.int64), \
                    neg_t_batch.astype(np.int64), neg_h_batch.astype(np.int64), \
                    neg_rel_tn_batch.astype(np.int64), neg_tn_batch.astype(np.int64)

            if not forever:
                break

    def corrupt_batch(self, h_batch, r_batch, t_batch):
        N = self.this_data.num_cons()  # number of entities

        neg_hn_batch = np.random.randint(0, N, size=(
        self.batch_size, self.neg_per_positive))  # random index without filtering
        neg_rel_hn_batch = np.tile(r_batch, (self.neg_per_positive, 1)).transpose()  # copy
        neg_t_batch = np.tile(t_batch, (self.neg_per_positive, 1)).transpose()

        neg_h_batch = np.tile(h_batch, (self.neg_per_positive, 1)).transpose()
        neg_rel_tn_batch = neg_rel_hn_batch
        neg_tn_batch = np.random.randint(0, N, size=(self.batch_size, self.neg_per_positive))

        return neg_hn_batch, neg_rel_hn_batch, neg_t_batch, neg_h_batch, neg_rel_tn_batch, neg_tn_batch
