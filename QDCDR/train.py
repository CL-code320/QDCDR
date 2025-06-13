import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from model.trainer import QdTrainer
from utils.loader import DataLoader
from utils.GraphMaker import GraphMaker, SharedGraphMaker
from utils import torch_utils, helper
from pathlib import Path
import matplotlib.pyplot as plt
import json
import codecs
import copy


parser = argparse.ArgumentParser()
# dataset part
parser.add_argument('--dataset', type=str, default='Sport_Cloth', help='')

# model part
parser.add_argument('--model', type=str, default="QDCDR", help="The model name.")
parser.add_argument('--feature_dim', type=int, default=128, help='Initialize network embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=128, help='GNN network hidden embedding dimension.')
parser.add_argument('--GNN', type=int, default=2, help='GNN layer.')

parser.add_argument('--dropout', type=float, default=0.3, help='GNN layer dropout rate.')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam',
                    help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--lr', type=float, default=0.001, help='Applies to sgd and adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--decay_epoch', type=int, default=10, help='Decay learning rate after this epoch.')
parser.add_argument('--leakey', type=float, default=0.1)
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--beta', type=float, default=0.9)

parser.add_argument('--lambda_kl', type=float, default=0.1)
parser.add_argument('--ratio', type=float, default=0.5)
# train part
parser.add_argument('--num_epoch', type=int, default=300, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=1024, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=200, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=100, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saves', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--seed', type=int, default=2040)
parser.add_argument('--load', dest='load', action='store_true', default=False,  help='Load pretrained model.')
parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')

def seed_everything(seed=1111):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


args = parser.parse_args()
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)
init_time = time.time()
# make opt
opt = vars(args)
seed_everything(opt["seed"])


current_path = Path(__file__).resolve()
dataset_path = current_path.parent.parent / "dataset"


if "QDCDR" in opt["model"]:
    filename  = opt["dataset"]
    source_graph = dataset_path / filename / "train.txt"
    source_G = GraphMaker(opt, source_graph)
    source_UV = source_G.UV
    source_VU = source_G.VU
    source_adj = source_G.adj

    filename = filename.split("_")
    filename = filename[1] + "_" + filename[0]
    target_train_data = dataset_path / filename / "train.txt"
    target_G = GraphMaker(opt, target_train_data)
    target_UV = target_G.UV
    target_VU = target_G.VU
    target_adj = target_G.adj


    share_G = SharedGraphMaker(opt, source_graph, target_train_data)
    share_UV = share_G.UV
    share_VU = share_G.VU
    share_adj = share_G.adj
    print("graph loaded!")


model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)
# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'],
                                header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

# print model info
helper.print_config(opt)


print("Loading data from {} with batch size {}...".format(opt['dataset'], opt['batch_size']))
train_batch = DataLoader(opt['dataset'], opt['batch_size'], opt, evaluation = -1)
source_dev_batch = DataLoader(opt['dataset'], opt["batch_size"], opt, evaluation = 1)
target_dev_batch = DataLoader(opt['dataset'], opt["batch_size"], opt, evaluation = 2)


print("user_num", opt["source_user_num"])
print("source_item_num", opt["source_item_num"])
print("target_item_num", opt["target_item_num"])
print("share_item_num", opt["share_item_num"])
print("source train data : {}, target train data {}, share train data : {}, source test data : {}, target test data{}, share test data : {}".format(len(train_batch.source_train_data),len(train_batch.target_train_data),len(train_batch.shared_train_data),len(train_batch.source_test_data),len(train_batch.target_test_data),len(train_batch.shared_test_data)))

if opt["cuda"]:
    source_UV = source_UV.cuda()
    source_VU = source_VU.cuda()
    source_adj = source_adj.cuda()

    target_UV = target_UV.cuda()
    target_VU = target_VU.cuda()
    target_adj = target_adj.cuda()

    share_UV = share_UV.cuda()
    share_VU = share_VU.cuda()
    share_adj = share_adj.cuda()

# model
if not opt['load']:
    trainer = QdTrainer(opt)
else:
    # load pretrained model
    model_file = opt['model_file']
    print("Loading model from {}".format(model_file))
    model_opt = torch_utils.load_config(model_file)
    model_opt['optim'] = opt['optim']
    trainer = QdTrainer(opt)
    trainer.load(model_file)

dev_score_history = [0]
current_lr = opt['lr']
global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(train_batch) * opt['num_epoch']
best_hr_s_10, best_ndcg_s_10, best_epoch_s = 0., 0., 0
best_hr_t_10, best_ndcg_t_10, best_epoch_t = 0., 0., 0
# 记录数据
epochs = []
avg_source_trues = []
avg_target_trues = []
# 设置保存目录
save_dir = "./Plots_final" 
file_name = opt['dataset'] + "_avg_true_plot.png"
# start training
for epoch in range(1, opt['num_epoch'] + 1):
    train_loss = 0
    mask_dict = {}
    start_time = time.time()
    for i, batch in enumerate(train_batch):
        global_step += 1
        loss = trainer.reconstruct_graph(batch, source_UV, source_VU, target_UV, target_VU, share_UV, share_VU, source_adj, target_adj, epoch)
        train_loss += loss
        mask_dict = trainer.mask_dict

    source_num_true_per_user = mask_dict['source'].sum(dim=1)  # (B,)
    target_num_true_per_user = mask_dict['target'].sum(dim=1) 
    # 计算平均值
    avg_source_true = source_num_true_per_user.float().mean().item()
    avg_target_true = target_num_true_per_user.float().mean().item()

    # 记录数据
    if epoch % 10 == 0:
        epochs.append(epoch)
        avg_source_trues.append(avg_source_true)
        avg_target_trues.append(avg_target_true)

    print(f"Epoch {epoch}: Source True Avg = {avg_source_true:.2f}, Target True Avg = {avg_target_true:.2f}")


    duration = time.time() - start_time
    train_loss = train_loss/len(train_batch)
    print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                                    opt['num_epoch'], train_loss, duration, current_lr))

    if epoch % 10:
        # pass
        continue

    # 评估模型
    print("Evaluating on dev set...")
    trainer.model.eval()
    trainer.evaluate_embedding(source_UV, source_VU, target_UV, target_VU, share_UV, share_VU, source_adj, target_adj)

    # 计算 Source HR@10, NDCG@10
    NDCG_10, HT_10 = 0.0, 0.0
    valid_entity = 0.0

    for i, batch in enumerate(source_dev_batch):
        predictions = trainer.source_predict(batch)
        for pred in predictions:
            rank = (-pred).argsort().argsort()[0].item()  # 获取排名

            valid_entity += 1
            if rank < 10:
                NDCG_10 += 1 / np.log2(rank + 2)
                HT_10 += 1

            if valid_entity % 100 == 0:
                print('.', end='')

    s_ndcg_10 = NDCG_10 / valid_entity
    s_hit_10 = HT_10 / valid_entity


    # 计算 Target HR@10, NDCG@10, HR@20, NDCG@20
    NDCG_10, HT_10 = 0.0, 0.0
    valid_entity = 0.0

    for i, batch in enumerate(target_dev_batch):
        predictions = trainer.target_predict(batch)
        for pred in predictions:
            rank = (-pred).argsort().argsort()[0].item()

            valid_entity += 1
            if rank < 10:
                NDCG_10 += 1 / np.log2(rank + 2)
                HT_10 += 1

            if valid_entity % 100 == 0:
                print('.', end='')

    t_ndcg_10 = NDCG_10 / valid_entity
    t_hit_10 = HT_10 / valid_entity

    # 更新最佳指标
    if s_hit_10 >= best_hr_s_10:
        best_hr_s_10 = s_hit_10
        best_ndcg_s_10 = s_ndcg_10
        best_epoch_s = epoch

    if t_hit_10 >= best_hr_t_10:
        best_hr_t_10 = t_hit_10
        best_ndcg_t_10 = t_ndcg_10
        best_epoch_t = epoch

    # 输出结果
    print("train_loss:{:.6f}".format(train_loss))
    print("source_hit@10:{:.4f} source_ndcg@10:{:.4f}".format(s_hit_10, s_ndcg_10))
    print("target_hit@10:{:.4f} target_ndcg@10:{:.4f}".format(t_hit_10, t_ndcg_10))


    dev_score = t_ndcg_10  # 仍然使用 NDCG@10 作为开发集评估指标
    file_logger.log(
        "{}\t{:.6f}\t{:.4f}\t{:.4f}".format(epoch, train_loss, dev_score, max([dev_score] + dev_score_history))
    )

    # save
    if epoch == 1 or dev_score > max(dev_score_history):
        print("new best model saved.")
    # if epoch % opt['save_epoch'] != 0:
    #     pass

    # lr schedule
    if len(dev_score_history) > opt['decay_epoch'] and dev_score <= dev_score_history[-1] and opt['optim'] in ['sgd', 'adagrad', 'adadelta', 'adam']:
        current_lr *= opt['lr_decay']
        trainer.update_lr(current_lr)

    dev_score_history += [dev_score]

    if np.isnan(train_loss):
        print('ERROR: loss is nan.')
        print('All done!')
        sys.exit()
    print("")

print('All done!')
print(f"Source Domain[{best_epoch_s}]: HR@10 = {best_hr_s_10:.4f}, NDCG@10 = {best_ndcg_s_10:.4f}")
print(f"Target Domain[{best_epoch_t}]: HR@10 = {best_hr_t_10:.4f}, NDCG@10 = {best_ndcg_t_10:.4f}")

