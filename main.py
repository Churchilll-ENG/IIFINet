import torch
import argparse

from src.train import evaluate_model
from src.utils import *
from torch.utils.data import DataLoader
from src import train
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulT1',
                    help='name of the model to use (Transformer, etc.)')        # 指定使用模型

# Tasks
parser.add_argument('--vonly', action='store_true',
                    help='use the crossmodal fusion into v (default: False)')   # 仅使用Visual数据进行模态融合
parser.add_argument('--aonly', action='store_true',
                    help='use the crossmodal fusion into a (default: False)')   # 仅使用Audio数据进行模态融合
parser.add_argument('--lonly', action='store_true',
                    help='use the crossmodal fusion into l (default: False)')   # 仅使用text数据进行模态融合
parser.add_argument('--aligned', default=True,
                    help='consider aligned experiment or not (default: False)') # 是否对齐实验
parser.add_argument('--dataset', type=str, default='iemo',
                    help='dataset to use (default: mosei_senti)')               # 指定数据集名称
parser.add_argument('--data_path', type=str, default='./data',
                    help='path for storing the dataset')                        # 指定数据集路径

# Dropouts                                                                      # 控制不同层的dropout
parser.add_argument('--attn_dropout', type=float, default=0.1,
                    help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0,
                    help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0,
                    help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.3,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.1,
                    help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=4,
                    help='number of layers in the network (default: 5)')              # 网络层数
parser.add_argument('--num_heads', type=int, default=10,
                    help='number of heads for the transformer network (default: 5)')  # 注意力头的数量
parser.add_argument('--attn_mask', default='True',
                    help='use attention mask for Transformer (default: true)')        # 是否使用注意力掩码

# Tuning
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.9,
                    help='gradient clip value (default: 0.8)')                         # 梯度裁剪值
parser.add_argument('--lr', type=float, default=1e-4,
                    help='initial learning rate (default: 1e-3)')                      # 初始学习率
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')                           # 优化器设置
parser.add_argument('--num_epochs', type=int, default=70,
                    help='number of epochs (default: 40)')                             # 训练轮数
parser.add_argument('--when', type=int, default=25,
                    help='when to decay learning rate (default: 20)')                  # 学习率衰减轮次
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')                    # 批次块？

# Logistics
parser.add_argument('--log_interval', type=int, default=40,
                    help='frequency of result logging (default: 30)')                  # 每隔30轮记录一次结果
parser.add_argument('--seed', type=int, default=3407,
                    help='random seed')                                                # 随机种子
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='do not use cuda')                                            # 是否使用GPU,默认使用
parser.add_argument('--name', type=str, default='mult',
                    help='name of the trial (default: "mult")')                        # 实验名称
parser.add_argument('--eval',default=True,
                    help='evaluate model')                                             # 是否只进行评估

args = parser.parse_args()

torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
valid_partial_mode = args.lonly + args.vonly + args.aonly

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

use_cuda = False

output_dim_dict = {
    'mosi': 1,
    'mosei_senti': 1,
    'iemocap': 8,
    'iemo': 8,
    'sur': 8,
    'sad': 8,
    'dis': 8,
    'ang': 8,
    'fea': 8,
    'hap': 8
}

criterion_dict = {
    'iemocap': 'CrossEntropyLoss',
    'sur': 'CrossEntropyLoss',
    'iemo': 'CrossEntropyLoss',
    'sad': 'CrossEntropyLoss',
    'dis': 'CrossEntropyLoss',
    'ang': 'CrossEntropyLoss',
    'fea': 'CrossEntropyLoss',
    'hap': 'CrossEntropyLoss',
}

torch.set_default_tensor_type('torch.FloatTensor')
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        use_cuda = True

####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################

print("Start loading the data....")

train_data = get_data(args, dataset, 'train')
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')
   
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

print('Finish loading the data....')
if not args.aligned:
    print("### Note: You are running in unaligned mode.")

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')


if __name__ == '__main__':
    if not args.eval:
        test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)
    else:
        train.initiate_eval(hyp_params, train_loader, valid_loader, test_loader)

