from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import copy
import codecs
import numpy as np
import os
from src import utils

class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNLayer, self).__init__()
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        row, col = edge_index
        aggr = torch.mean(x[col], dim=0, keepdim=True)  # 聚合邻居特征
        return F.relu(self.fc(aggr))


class GNN(nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GNN, self).__init__()
        self.layer1 = GNNLayer(num_node_features, hidden_channels)
        self.layer2 = GNNLayer(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        return x


class GCNLayer(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, batch_norm=False):
        super(GCNLayer, self).__init__()
        self.weight = torch.Tensor(in_features, out_features)
        self.weight = nn.Parameter(nn.init.xavier_uniform_(self.weight))
        if bias:
            self.bias = torch.Tensor(out_features)
            self.bias = nn.Parameter(nn.init.xavier_uniform_(self.bias))
        else:
            self.register_parameter('bias', None)

        self.bn = nn.BatchNorm1d(out_features) if batch_norm else None

    def forward(self, input, adj, batch_norm=True):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)

        if self.bias is not None:
            output = output + self.bias

        if self.bn is not None and batch_norm:
            output = self.compute_bn(output)

        return output

    def compute_bn(self, x):
        if len(x.shape) == 2:
            return self.bn(x)
        else:
            return self.bn(x.view(-1, x.size(-1))).view(x.size())


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, graph_hops, dropout, batch_norm=False):
        super(GCN, self).__init__()
        self.dropout = dropout

        self.graph_encoders = nn.ModuleList()
        self.graph_encoders.append(GCNLayer(nfeat, nhid, batch_norm=batch_norm))

        for _ in range(graph_hops - 2):
            self.graph_encoders.append(GCNLayer(nhid, nhid, batch_norm=batch_norm))

        self.graph_encoders.append(GCNLayer(nhid, nclass, batch_norm=False))

    def forward(self, x, node_anchor_adj):
        for i, encoder in enumerate(self.graph_encoders[:-1]):
            x = F.relu(encoder(x, node_anchor_adj))
            x = F.dropout(x, self.dropout, training=self.training)

        x = self.graph_encoders[-1](x, node_anchor_adj)
        return x

# class RGCNLayer(nn.Module):
#     def __init__(self, in_features, out_features, rel_list, device, bias=True):
#         super(RGCNLayer, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.num_relations = len(rel_list)
#         self.rel_list = rel_list
#         self.bias = bias
#
#         self.device = device
#
#         # 初始化权重和偏置
#         self.weight = nn.Parameter(torch.FloatTensor(self.num_relations, in_features, out_features))
#
#         if bias:
#             self.bias = nn.Parameter(torch.FloatTensor(self.num_relations, out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.weight)
#         if self.bias is not None:
#             nn.init.xavier_uniform_(self.bias)
#
#     def forward(self, x, adj_list):
#         # x: 节点特征矩阵 (N, in_features)
#         # adj_list: 加权邻接矩阵列表
#
#         outputs = []
#         for r in self.rel_list:
#             adj = adj_list[r].to(self.device)
#             weigth_r = self.weight[r]
#             output_r = torch.matmul(adj, torch.matmul(x, weigth_r))
#             if self.bias is not None:
#                 output_r += self.bias[r]
#
#             outputs.append(output_r)
#
#         return torch.stack(outputs, dim=0).sum(dim=0)
#
#
# class RGCN(nn.Module):
#     def __init__(self, in_features, hidden_features, out_features, graph_hops, rel_list, dropout, device):
#         super(RGCN, self).__init__()
#         self.dropout = dropout
#
#         self.graph_encoders = nn.ModuleList()
#         self.graph_encoders.append(RGCNLayer(in_features, hidden_features, rel_list, device))
#
#         for _ in range(graph_hops - 2):
#             self.graph_encoders.append(RGCNLayer(hidden_features, hidden_features, rel_list, device))
#
#         self.graph_encoders.append(RGCNLayer(hidden_features, out_features, rel_list, device))
#
#     def forward(self, x, adj_list):
#         for i, encoder in enumerate(self.graph_encoders[:-1]):
#             x = F.relu(encoder(x, adj_list))
#             x = F.dropout(x, self.dropout, training=self.training)
#
#         x = self.graph_encoders[-1](x, adj_list)
#         return x

# 内存优化的W-RGCN实现
class OptimizedWRGCNLayer(nn.Module):
    """优化的单层W-RGCN（保持原始逻辑但提升效率）"""
    
    def __init__(self, input_dim, output_dim, num_relations, device, dropout=0.3, batch_size=50000, bias=True):
        super(OptimizedWRGCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relations = num_relations
        self.dropout = dropout
        self.batch_size = batch_size  # 增大批处理大小，减少循环次数
        self.device = device
        self.bias = bias
        
        # 为每个关系创建权重矩阵
        self.relation_weights = nn.Parameter(torch.FloatTensor(num_relations, input_dim, output_dim))
        # 自环权重
        self.self_weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        
        # 偏置参数
        if bias:
            self.bias_param = nn.Parameter(torch.FloatTensor(num_relations, output_dim))
        else:
            self.register_parameter('bias_param', None)
        
        # 初始化权重
        self.reset_parameters()
        
        self.dropout_layer = nn.Dropout(dropout)
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.relation_weights)
        nn.init.xavier_uniform_(self.self_weight)
        if self.bias_param is not None:
            nn.init.zeros_(self.bias_param)
        
    def forward(self, entity_embeddings, adj_list, rel_list):
        """
        优化的前向传播：减少循环次数，提升内存访问效率
        """
        num_entities = entity_embeddings.size(0)
        
        # 初始化输出嵌入（使用与输入相同的设备和数据类型）
        out_embeddings = torch.zeros(num_entities, self.output_dim, 
                                    device=entity_embeddings.device, 
                                    dtype=entity_embeddings.dtype)
        
        # 预先将所有数据移到正确的设备（批量操作，减少单独检查）
        device = entity_embeddings.device
        
        # 处理每种关系类型
        for r_idx, r in enumerate(rel_list):
            if r not in adj_list:
                continue
                
            edge_index, edge_weights = adj_list[r]
            
            # 一次性设备转移（如果需要的话）
            if edge_index.device != device:
                edge_index = edge_index.to(device, non_blocking=True)
            if edge_weights.device != device:
                edge_weights = edge_weights.to(device, non_blocking=True)
            
            if edge_index.size(1) == 0:  # 跳过空关系
                continue
            
            num_edges = edge_index.size(1)
            src_nodes = edge_index[0]  # 源节点索引
            tgt_nodes = edge_index[1]  # 目标节点索引
            
            # 智能批处理：如果边数较少，直接处理；否则分批
            if num_edges <= self.batch_size:
                # 小规模：一次性处理所有边（Mine的策略）
                src_embeddings = entity_embeddings[src_nodes]
                transformed_embeddings = torch.matmul(src_embeddings, self.relation_weights[r_idx])
                weighted_embeddings = transformed_embeddings * edge_weights.unsqueeze(1)
                out_embeddings.index_add_(0, tgt_nodes, weighted_embeddings)
                
                # 处理偏置（优化：只对唯一的目标节点处理一次）
                if self.bias_param is not None:
                    unique_tgt_nodes = torch.unique(tgt_nodes)
                    if len(unique_tgt_nodes) > 0:
                        out_embeddings[unique_tgt_nodes] += self.bias_param[r_idx]
            else:
                # 大规模：分批处理（保持原始逻辑）
                for start_idx in range(0, num_edges, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, num_edges)
                    
                    # 当前批次的边信息
                    batch_src_nodes = src_nodes[start_idx:end_idx]
                    batch_tgt_nodes = tgt_nodes[start_idx:end_idx]
                    batch_edge_weights = edge_weights[start_idx:end_idx]
                    
                    # 获取源节点的嵌入
                    src_embeddings = entity_embeddings[batch_src_nodes]
                    
                    # 应用关系特定的权重变换
                    transformed_embeddings = torch.matmul(src_embeddings, self.relation_weights[r_idx])
                    
                    # 应用置信度权重
                    weighted_embeddings = transformed_embeddings * batch_edge_weights.unsqueeze(1)
                    
                    # 聚合到目标节点
                    out_embeddings.index_add_(0, batch_tgt_nodes, weighted_embeddings)
                
                # 处理偏置（分批情况下）
                if self.bias_param is not None:
                    unique_tgt_nodes = torch.unique(tgt_nodes)
                    if len(unique_tgt_nodes) > 0:
                        out_embeddings[unique_tgt_nodes] += self.bias_param[r_idx]
        
        # 添加自环
        self_embeddings = torch.matmul(entity_embeddings, self.self_weight)
        out_embeddings = out_embeddings + self_embeddings
        
        return out_embeddings


class WRGCN(nn.Module):
    """多层W-RGCN网络（优化版本）"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_relations, rel_list, device, num_layers=2, dropout=0.3, batch_size=50000):
        super(WRGCN, self).__init__()
        self.num_layers = num_layers
        self.rel_list = rel_list
        self.layers = nn.ModuleList()
        
        # 构建多层网络
        if num_layers == 1:
            self.layers.append(OptimizedWRGCNLayer(input_dim, output_dim, num_relations, device, dropout, batch_size))
        else:
            # 第一层：input_dim -> hidden_dim
            self.layers.append(OptimizedWRGCNLayer(input_dim, hidden_dim, num_relations, device, dropout, batch_size))
            
            # 中间层：hidden_dim -> hidden_dim
            for _ in range(num_layers - 2):
                self.layers.append(OptimizedWRGCNLayer(hidden_dim, hidden_dim, num_relations, device, dropout, batch_size))
            
            # 最后一层：hidden_dim -> output_dim（最后一层不使用dropout和激活函数）
            self.layers.append(OptimizedWRGCNLayer(hidden_dim, output_dim, num_relations, device, 0.0, batch_size))
    
    def forward(self, entity_embeddings, adj_list):
        """
        优化的前向传播
        Args:
            entity_embeddings: [num_entities, input_dim] 实体嵌入
            adj_list: dict {rel_id: (edge_index, edge_weights)}
        Returns:
            output_embeddings: [num_entities, output_dim] 输出嵌入
        """
        x = entity_embeddings
        
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_list, self.rel_list)
            
            # 除了最后一层，都应用ReLU激活函数和dropout
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, layer.dropout, training=self.training)
        
        return x

# 创建一个简单的GRU模型
class TripletGRU(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_layers, dropout):
        super(TripletGRU, self).__init__()
        self.gru = nn.GRU(emb_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)  # 输出一个置信度分数

        # 使用xavier_uniform_初始化GRU层的权重
        self.init_weights()

    def init_weights(self):
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.xavier_uniform_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.01)

    def forward(self, x):
        # x的形状是[batch_size, seq_length, emb_dim]
        _, h_n = self.gru(x)
        # 取最后一个时间步的输出
        out = h_n[-1]
        # 通过全连接层得到置信度分数，使用sigmoid确保输出在[0,1]
        confidence = torch.sigmoid(self.fc(out))
        return confidence
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, node_features):
        # node_features: List of tensors of shape [num_nodes, input_dim] for each view
        # 假设所有视图都有相同数量的节点

        num_nodes, input_dim = node_features[0].size()
        num_views = len(node_features)

        # 初始化聚合特征
        aggregated_features = torch.zeros((num_nodes, input_dim)).cuda()

        for i in range(num_views):
            # 计算当前视图的Q, K, V
            Q = self.query(node_features[i]).cuda()  # [num_nodes, input_dim]
            K = self.key(node_features[i]).cuda()  # [num_nodes, input_dim]
            V = self.value(node_features[i]).cuda()  # [num_nodes, input_dim]

            # 计算注意力权重
            attention_scores = torch.matmul(Q, K.t()) / (input_dim ** 0.5)  # [num_nodes, num_nodes]
            attention_weights = F.softmax(attention_scores, dim=-1)  # [num_nodes, num_nodes]

            # 聚合特征
            aggregated_features += torch.matmul(attention_weights, V)

        # 归一化聚合特征
        aggregated_features /= num_views

        return aggregated_features


class unKG_GSL(nn.Module):

    @property
    def num_cons(self):
        return self._num_cons

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def dim(self):
        return self._dim

    @property
    def batch_size(self):

        return self._batch_size

    @property
    def neg_batch_size(self):
        return self._neg_per_positive * self._batch_size

    def __init__(self, num_rels, num_cons, dim, batch_size, neg_per_positive, reg_scale, p_neg, function, data,
                 args, device):
        super(unKG_GSL, self).__init__()
        self._num_rels = num_rels
        self._num_cons = num_cons
        self._dim = dim  # dimension of both relation and ontology.
        self._batch_size = batch_size
        self._neg_per_positive = neg_per_positive
        self._epoch_loss = 0
        self._p_neg = p_neg
        self._soft_size = 1
        self._prior_psl = 0
        self.reg_scale = reg_scale
        self.function = function
        self.data = data
        self.device = device
        self.args = args
        
        # 链接预测损失权重参数
        self.link_loss_weight = getattr(args, 'link_loss_weight', 0)

        # init embedding
        if self.args.is_gcn:
            self.ent_embedding = nn.Embedding(num_embeddings=self.num_cons, embedding_dim=self.args.init_size)
        else:
            self.ent_embedding = nn.Embedding(num_embeddings=self.num_cons, embedding_dim=self.dim)
        self.rel_embedding = nn.Embedding(num_embeddings=self.num_rels, embedding_dim=self.dim)
        self.ent_embedding_update = None # entity embedding after gcn

        self.liner = torch.nn.Linear(1, 1).cuda()

        self.__data_init()

        self.gcn = WRGCN(self.args.init_size, self.args.hid_size_gcn, self.dim, len(self.data.rels_list), self.data.rels_list, self.device, self.args.graph_hop, self.args.dropout, batch_size=50000)

        self.rnn = TripletGRU(self.dim, self.args.hid_size_rnn, self.args.graph_hop, self.args.dropout)
        # self.attention = AttentionLayer(self.dim).to(device)
        
        # 预计算entity索引，避免每次重复创建
        self.entity_indices = torch.LongTensor([self.data.index_cons[cons] for cons in self.data.cons]).cuda()

    def __data_init(self):

        nn.init.normal_(self.liner.weight, mean=0, std=0.3)
        nn.init.normal_(self.liner.bias, mean=0, std=0.3)
        nn.init.xavier_uniform_(self.ent_embedding.weight)
        nn.init.xavier_uniform_(self.rel_embedding.weight)

    def forward(self, h, r, t, w, n_hn, n_rel_hn, n_t, n_h, n_rel_tn, n_tn):

        # 使用预计算的entity索引，避免重复创建张量
        if self.args.is_gcn:
            input_gcn = self.ent_embedding(self.entity_indices)
            self.ent_embedding_update = self.gcn(input_gcn, self.data.adj_list)
        else:
            self.ent_embedding_update = self.ent_embedding(self.entity_indices)

        h = torch.tensor(h, dtype=torch.int64).cuda()
        r = torch.tensor(r, dtype=torch.int64).cuda()
        t = torch.tensor(t, dtype=torch.int64).cuda()
        w = torch.tensor(w, dtype=torch.float32).cuda()
        n_hn = torch.tensor(n_hn, dtype=torch.int64).cuda()
        n_rel_hn = torch.tensor(n_rel_hn, dtype=torch.int64).cuda()
        n_t = torch.tensor(n_t, dtype=torch.int64).cuda()
        n_h = torch.tensor(n_h, dtype=torch.int64).cuda()
        n_rel_tn = torch.tensor(n_rel_tn, dtype=torch.int64).cuda()
        n_tn = torch.tensor(n_tn, dtype=torch.int64).cuda()

        # get score
        hrt = self.cal_score(h, r, t)
        h_n_rt = self.cal_score(n_hn, n_rel_hn, n_t, mode="negtive")
        hrt_n = self.cal_score(n_h, n_rel_tn, n_tn, mode="negtive")

        # task loss
        main_loss = self.main_loss(hrt, w, h_n_rt, hrt_n, self.link_loss_weight)

        # regularizer loss
        regularizer_loss = self.regularizer_loss(h, r, t)

        loss = main_loss + self.reg_scale * regularizer_loss

        return loss

    def cal_score(self, h, r, t, mode="positive"):

        head = self.ent_embedding_update[h]
        rel = self.rel_embedding(r)
        tail = self.ent_embedding_update[t]

        if hasattr(self.args, 'is_gru') and self.args.is_gru:
            # 双分支：GRU序列分数 + UKGE语义分数
            score_sequence = self.cal_score_sem(head, rel, tail, mode)   # [0,1]
            score_semantic = self.cal_confidence(head, rel, tail, mode)  # [0,1]
            
            # 加权融合两个分数
            confidence = score_semantic * self.args.contact_a + score_sequence * (1 - self.args.contact_a)
        else:
            # 单分支：只使用UKGE语义分数
            confidence = self.cal_confidence(head, rel, tail, mode)

        return confidence

    def cal_score_sem(self, head, rel, tail, mode):

        # 将实体和关系嵌入堆叠成一个序列
        if mode == "positive":
            # 正样本：[batch_size, dim] -> [batch_size, 3, dim]
            stacked_tensors = torch.stack((head, rel, tail), dim=1)
            score_semantic = self.rnn(stacked_tensors)  # [batch_size, 1]
        else:
            # 负样本优化：展开计算后重塑
            # head, rel, tail: [batch_size, neg_per_positive, dim]
            batch_size, neg_num, dim = head.shape
            
            # 使用 reshape 避免连续性问题
            head_flat = head.reshape(batch_size * neg_num, dim)
            rel_flat = rel.reshape(batch_size * neg_num, dim)
            tail_flat = tail.reshape(batch_size * neg_num, dim)
            
            # 堆叠为 [batch_size * neg_per_positive, 3, dim]
            stacked_flat = torch.stack((head_flat, rel_flat, tail_flat), dim=1)
            
            # GRU处理：[batch_size * neg_per_positive, 3, dim] -> [batch_size * neg_per_positive, 1]
            scores_flat = self.rnn(stacked_flat)
            
            # 重塑回原始形状：[batch_size, neg_per_positive, 1]
            score_semantic = scores_flat.reshape(batch_size, neg_num, 1)

        return score_semantic

    # def cal_confidence(self, head, rel, tail):
    #
    #     htr = torch.sum(rel * (head * tail), dim=-1)  # [batch]
    #     htr = torch.unsqueeze(htr, dim=-1)            # [batch, 1]
    #     return htr

    def cal_confidence(self, head, rel, tail, mode):

        if mode == "positive":
            htr = torch.sum(rel * (head * tail), dim=1)
        else:
            htr = torch.sum(rel * (head * tail), dim=2)

        # 先通过liner层进行线性变换
        htr_expanded = htr.unsqueeze(-1)  # [batch, 1] 或 [batch, neg_num, 1]
        linear_output = self.liner(htr_expanded)  # 通过liner层
        linear_output = linear_output.squeeze(-1)  # 移除最后一维
        
        # 然后根据function参数选择激活函数
        if self.function == 'logi':
            confidence = torch.sigmoid(linear_output)
        else:  # 'rect' - 使用有限整流器
            confidence = torch.clamp(linear_output, min=0, max=1)
        
        return confidence.unsqueeze(-1)


    def link_prediction_loss(self, hrt, w, h_n_rt, hrt_n, margin=1.0):
        """
        链接预测排序损失：确保正例分数高于负例分数
        :param hrt: 正例预测分数 (batch_size, 1)
        :param w: 正例实际分数 (batch_size, 1) 
        :param h_n_rt: 负例头部替换的预测分数 (batch_size, neg_num, 1)
        :param hrt_n: 负例尾部替换的预测分数 (batch_size, neg_num, 1)
        :param margin: 边界参数
        :return: 链接预测排序损失
        """
        # 扩展正例分数以匹配负例维度
        hrt_expanded = hrt.unsqueeze(1)  # (batch_size, 1, 1)
        
        # 计算头部替换的排序损失 - 正例应该比负例分数高
        margin_loss_h = torch.clamp(margin - hrt_expanded + h_n_rt, min=0)
        ranking_loss_h = torch.mean(margin_loss_h, dim=1)  # (batch_size, 1)
        
        # 计算尾部替换的排序损失 - 正例应该比负例分数高
        margin_loss_t = torch.clamp(margin - hrt_expanded + hrt_n, min=0)
        ranking_loss_t = torch.mean(margin_loss_t, dim=1)  # (batch_size, 1)
        
        # 总排序损失
        total_ranking_loss = torch.mean(ranking_loss_h + ranking_loss_t)
        
        return total_ranking_loss

    def confidence_prediction_loss(self, hrt, w, h_n_rt, hrt_n):
        """
        置信度预测损失：原始的MSE损失，专注于准确预测置信度值
        :param hrt: 正例预测分数 (batch_size, 1)
        :param w: 正例实际分数 (batch_size, 1) 
        :param h_n_rt: 负例头部替换的预测分数 (batch_size, neg_num, 1)
        :param hrt_n: 负例尾部替换的预测分数 (batch_size, neg_num, 1)
        :return: 置信度预测损失
        """
        w = torch.unsqueeze(w, dim=-1)
        
        # 正例的置信度预测损失
        f_loss_h = torch.square(hrt - w)
        
        # 负例的置信度预测损失（期望负例分数接近0）
        f_loss_hn = torch.mean(torch.square(h_n_rt), dim=1)
        f_loss_tn = torch.mean(torch.square(hrt_n), dim=1)
        
        # 总置信度损失
        confidence_loss = (torch.sum(((f_loss_tn + f_loss_hn) / 2.0) * self._p_neg + f_loss_h)) / self.batch_size
        
        return confidence_loss

    def main_loss(self, hrt, w, h_n_rt, hrt_n, link_weight=0.3):
        """
        融合损失函数：结合置信度预测和链接预测损失
        :param hrt: 正例预测分数
        :param w: 正例实际分数
        :param h_n_rt: 负例头部替换分数
        :param hrt_n: 负例尾部替换分数
        :param link_weight: 链接预测损失的权重 (0-1)
        :return: 融合损失
        """
        # 计算置信度预测损失
        confidence_loss = self.confidence_prediction_loss(hrt, w, h_n_rt, hrt_n)
        
        # 计算链接预测损失
        ranking_loss = self.link_prediction_loss(hrt, w, h_n_rt, hrt_n, margin=0.1)
        
        # 加权融合两种损失
        main_loss = (1 - link_weight) * confidence_loss + link_weight * ranking_loss
        
        return main_loss

    def regularizer_loss(self, h, r, t):
        head = self.ent_embedding_update[h]
        rel = self.rel_embedding(r)
        tail = self.ent_embedding_update[t]
        
        # 原有的embedding正则化
        regularizer_loss = ((torch.sum(torch.square(head)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(tail)) / 2.0) / self.batch_size) + \
                      ((torch.sum(torch.square(rel)) / 2.0) / self.batch_size)

        # 添加GRU参数正则化（如果使用GRU）
        if hasattr(self.args, 'is_gru') and self.args.is_gru and hasattr(self.args, 'gru_reg_scale'):
            gru_reg = 0.0
            for param in self.rnn.parameters():
                gru_reg += torch.sum(torch.square(param))
            regularizer_loss += self.args.gru_reg_scale * gru_reg / self.batch_size

        return regularizer_loss

