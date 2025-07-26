from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

if './src' not in sys.path:
    sys.path.append('./src')

if './' not in sys.path:
    sys.path.append('./')

import os
from os.path import join
from src.data import Data
import numpy as np
import random
import datetime
import argparse

import torch

from src.trainer import Trainer
from src.utils import *



def get_model_identifier(model, function):
    prefix = model
    now = datetime.datetime.now()
    date = '%02d%02d%02d%02d%02d' % (now.year, now.month, now.day, now.hour, now.minute)
    identifier = prefix + '_' + function +'_'+ date
    return identifier

def seed_everything(seed_value):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

    torch.manual_seed(seed_value)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子（只用一块GPU）
    torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子（多块GPU）

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def set_parser():

    parser = argparse.ArgumentParser()
    # only test
    parser.add_argument('--only_test', action='store_true', help="is or not only_test.")

    parser.add_argument("--seed", default=3407, type=int, help='Random seed.')
    # required
    parser.add_argument('--data', type=str, default='ppi5k', help="the dir path where you store data (train.tsv, val.tsv, test.tsv). Default: ppi5k")
    parser.add_argument('-m', '--model', type=str, default='un_gcn_gru', help="choose model . default: 'ukg-gsl'")
    # optional
    parser.add_argument("--verbose", help="print detailed info for debugging",action="store_true")
    parser.add_argument('--function', type=str, default='rect', help="choose maping function rect or logi. default: rect")
    parser.add_argument('-d', '--dim', type=int, default=512, help="set dimension. default: 128")
    parser.add_argument('--epoch', type=int, default=500, help="set number of epochs. default: 500")
    parser.add_argument('--lr', type=float, default=0.0002, help="set learning rate. default: 0.001")
    parser.add_argument('--batch_size', type=int, default=4096, help="set batch size. default: 1024")
    parser.add_argument('--n_neg', type=int, default=10, help="Number of negative samples per (h,r,t). default: 10")
    parser.add_argument('--models_dir', type=str, default='./trained_models', help="the dir path where you store trained models. A new directory will be created inside it.")

    parser.add_argument('--is_gcn', type=bool, default=True, help="is or not use gcn.")
    parser.add_argument('--is_gru', type=bool, default=True, help="is or not use gru for sequence scoring.")
    parser.add_argument("--threshold", type=float, default=0.7,help="0 or 1 for construct adj")

    # regularizer coefficient (lambda)
    parser.add_argument('--p_neg', type=float, default=1.0,help="The scale for neg sample. Default 1.0")
    parser.add_argument('--contact_a', type=float, default=0.5, help="The scale for two score. Default 0.5")

    # loss
    parser.add_argument('--reg_scale', type=float, default=0.0001,help="The scale for regularizer (lambda) of calculate confidence. Default 0.005")
    parser.add_argument('--gru_reg_scale', type=float, default=0.0001,help="The scale for GRU regularizer. Default 0.001")
    parser.add_argument('--link_loss_weight', type=float, default=0, help="不使用这个损失")
    # parser.add_argument('--reg_scale_graph', type=float, default=0.001,help="The scale for regularizer (lambda) of graph. Default 0.001")

    # early_stop
    parser.add_argument('--early_stop', type=bool, default=True, help="early stop.")
    parser.add_argument('--early_stop_patience', type=int, default=30, help="early stop patience. default: 10")

    #gnn param
    parser.add_argument('--init_size', type=int, default=1024, help="set embedding size of init")
    parser.add_argument('--hid_size_gcn', type=int, default=512, help="set hidden size of gcn")
    parser.add_argument('--hid_size_rnn', type=int, default=256, help="set hidden size of rnn")
    parser.add_argument('--graph_hop', type=int, default=2, help="grqph hop of gnn")
    parser.add_argument('--dropout', type=float, default=0.2, help="set dropout of all model")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = set_parser()  # 设置参数

    seed_everything(args.seed) # 设置随机数
    print('seed is: ', args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # path to save
    identifier = get_model_identifier(args.model, args.function)
    save_dir = join(args.models_dir, args.data, identifier)  # the directory where we store this model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print('Trained models will be stored in: ', save_dir)

    # input files
    data_dir = join('./data', args.data)
    file_train = join(data_dir, 'train.tsv')  # training data
    file_val = join(data_dir, 'val.tsv')  # validation datan
    file_test = join(data_dir, 'test.tsv')

    more_filt = [file_val, join(data_dir, 'test.tsv')]
    print('Read train.tsv from', data_dir)

    # load data
    this_data = Data(args)
    this_data.load_data(file_train=file_train, file_val=file_val, file_test=file_test)
    for f in more_filt:
        this_data.record_more_data(f)
    this_data.save_meta_table(save_dir)  # output: idx_concept.csv, idx_relation.csv

    m_train = Trainer(args, device)
    m_train.build(this_data, save_dir, lr=args.lr, modelname=args.model)

    # Model will be trained, validated, and saved in './trained_models'
    if args.only_test:
        m_train.test("test")
    else:
        m_train.train()
    # m_train.test(filename=data_dir)