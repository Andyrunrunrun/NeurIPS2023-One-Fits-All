from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
from models.DLinear import DLinear
# from models.MultiModelTS import MultiModelTS

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import pandas as pd
from datetime import datetime

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

def save_results_to_csv(args, results_list, final_stats, filename=None):
    """
    将实验参数和结果保存到CSV文件中，支持追加模式
    
    Args:
        args: 实验参数对象
        results_list (list): 包含每次迭代结果的列表
        final_stats (dict): 最终统计结果
        filename (str): CSV文件名，如果为None则使用默认名称
    """
    if filename is None:
        filename = "experiment_results.csv"  # 使用固定文件名以支持追加
    
    # (1) 准备实验参数数据 - 提取所有重要的超参数
    params_dict = {
        'model_id': args.model_id,
        'base_model': args.base_model,
        'model': args.model,
        'seq_len': args.seq_len,
        'pred_len': args.pred_len,
        'label_len': args.label_len,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'train_epochs': args.train_epochs,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'e_layers': args.e_layers,
        'gpt_layers': args.gpt_layers,
        'd_ff': args.d_ff,
        'dropout': args.dropout,
        'patch_size': args.patch_size,
        'kernel_size': args.kernel_size,
        'loss_func': args.loss_func,
        'pretrain': args.pretrain,
        'freeze': args.freeze,
        'stride': args.stride,
        'hid_dim': args.hid_dim,
        'data_path': args.data_path,
        'features': args.features,
        'target': args.target,
        'embed': args.embed,
        'percent': args.percent,
        'freq': args.freq,
        'enc_in': args.enc_in,
        'c_out': args.c_out,
        'decay_fac': args.decay_fac,
        'lradj': args.lradj,
        'patience': args.patience,
        'max_len': args.max_len,
        'tmax': args.tmax,
        'itr': args.itr,
        'cos': args.cos
    }
    
    # (2) 创建当前实验的详细结果数据 - 包含每次迭代的完整信息
    current_experiment_data = []
    for i, result in enumerate(results_list):
        row = params_dict.copy()  # 复制参数字典
        row.update({
            'iteration': i + 1,
            'mse': result['mse'],
            'mae': result['mae'],
            'train_time': result.get('train_time', 0),
            'final_train_loss': result.get('final_train_loss', 0),
            'final_vali_loss': result.get('final_vali_loss', 0),
            'epochs_trained': result.get('epochs_trained', 0),
            'experiment_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        current_experiment_data.append(row)
    
    # (3) 处理文件追加逻辑
    if os.path.exists(filename):
        # 如果文件已存在，读取现有数据并追加新实验
        print(f"检测到现有文件 {filename}，将追加新的实验结果...")
        try:
            existing_df = pd.read_csv(filename, encoding='utf-8-sig')
            # 删除之前的SUMMARY行（如果存在）
            existing_df = existing_df[existing_df['iteration'] != 'SUMMARY'].copy()
            
            # 将新实验数据追加到现有数据
            all_data = existing_df.to_dict('records') + current_experiment_data
            
            print(f"成功追加 {len(current_experiment_data)} 条新记录到现有的 {len(existing_df)} 条记录中")
        except Exception as e:
            print(f"读取现有文件时出错: {e}，将创建新文件")
            all_data = current_experiment_data
    else:
        # 如果文件不存在，创建新文件
        print(f"创建新的实验结果文件: {filename}")
        all_data = current_experiment_data
    
    # (4) 计算整体统计信息 - 基于所有历史数据
    all_mses = []
    all_maes = []
    experiment_groups = {}  # 按实验时间戳分组
    
    for record in all_data:
        if isinstance(record['iteration'], int):  # 排除SUMMARY行
            all_mses.append(record['mse'])
            all_maes.append(record['mae'])
            
            # 按实验分组统计
            exp_timestamp = record['experiment_timestamp']
            if exp_timestamp not in experiment_groups:
                experiment_groups[exp_timestamp] = []
            experiment_groups[exp_timestamp].append(record)
    
    # (5) 添加整体统计汇总行
    if all_mses and all_maes:
        summary_row = params_dict.copy()
        summary_row.update({
            'iteration': 'SUMMARY',
            'mse': np.mean(all_mses),
            'mae': np.mean(all_maes),
            'mse_std': np.std(all_mses),
            'mae_std': np.std(all_maes),
            'total_iterations': len(all_data),
            'total_experiments': len(experiment_groups),
            'experiment_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'train_time': '',
            'final_train_loss': '',
            'final_vali_loss': '',
            'epochs_trained': ''
        })
        all_data.append(summary_row)
    
    # (6) 保存到CSV文件
    df = pd.DataFrame(all_data)
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    print(f"实验结果已保存到: {filename}")
    print(f"当前文件包含 {len(experiment_groups)} 个实验组，共 {len([d for d in all_data if isinstance(d['iteration'], int)])} 次迭代")
    
    return filename

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='GPT4TS')

parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--base_model', type=str, required=True, default='gpt2')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
parser.add_argument('--data_path', type=str, default='traffic.csv')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=1)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=10)


parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--patience', type=int, default=3)

parser.add_argument('--gpt_layers', type=int, default=3)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=862)
parser.add_argument('--c_out', type=int, default=862)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=10)

parser.add_argument('--itr', type=int, default=3)
parser.add_argument('--cos', type=int, default=0)



args = parser.parse_args()

SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

mses = []
maes = []
results_list = []  # 存储每次迭代的详细结果

for ii in range(args.itr):

    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len, args.pred_len,
                                                                    args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                    args.d_ff, args.embed, ii)
    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)

    if args.freq == 0:
        args.freq = 'h'

    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')

    if args.freq != 'h':
        args.freq = SEASONALITY_MAP[test_data.freq]
        print("freq = {}".format(args.freq))

    device = torch.device('cuda:0')

    time_now = time.time()
    train_start_time = time.time()  # 记录训练开始时间
    train_steps = len(train_loader)

    if args.model == 'PatchTST':
        model = PatchTST(args, device)
        model.to(device)
    elif args.model == 'DLinear':
        model = DLinear(args, device)
        model.to(device)
    else:
        model = GPT4TS(args, device)
    # mse, mae = test(model, test_data, test_loader, args, device, ii)

    params = model.parameters()
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    if args.loss_func == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_func == 'smape':
        class SMAPE(nn.Module):
            def __init__(self):
                super(SMAPE, self).__init__()
            def forward(self, pred, true):
                return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
        criterion = SMAPE()
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

    final_train_loss = 0
    final_vali_loss = 0
    epochs_trained = 0

    for epoch in range(args.train_epochs):
        epochs_trained = epoch + 1
        
        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader)):

            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)

            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            outputs = model(batch_x, ii)

            outputs = outputs[:, -args.pred_len:, :]
            batch_y = batch_y[:, -args.pred_len:, :].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())

            if (i + 1) % 1000 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            loss.backward()
            model_optim.step()

        
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))

        final_train_loss = np.average(train_loss)
        final_vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
        # test_loss = vali(model, test_data, test_loader, criterion, args, device, ii)
        # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}, Test Loss: {4:.7f}".format(
        #     epoch + 1, train_steps, train_loss, vali_loss, test_loss))
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, final_train_loss, final_vali_loss))

        if args.cos:
            scheduler.step()
            print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
        else:
            adjust_learning_rate(model_optim, epoch + 1, args)
        early_stopping(final_vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    train_end_time = time.time()
    train_time = train_end_time - train_start_time  # 计算训练总时间

    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))
    print("------------------------------------")
    mse, mae = test(model, test_data, test_loader, args, device, ii)
    mses.append(mse)
    maes.append(mae)
    
    # 收集当前迭代的结果数据（暂不保存到文件）
    iteration_result = {
        'mse': mse,
        'mae': mae,
        'train_time': train_time,
        'final_train_loss': final_train_loss,
        'final_vali_loss': final_vali_loss,
        'epochs_trained': epochs_trained
    }
    results_list.append(iteration_result)

# ============ 所有迭代完成，开始保存最终结果 ============
mses = np.array(mses)
maes = np.array(maes)

# 计算整个实验的统计信息
final_stats = {
    'mse_mean': np.mean(mses),
    'mse_std': np.std(mses),
    'mae_mean': np.mean(maes),
    'mae_std': np.std(maes)
}

print("mse_mean = {:.4f}, mse_std = {:.4f}".format(final_stats['mse_mean'], final_stats['mse_std']))
print("mae_mean = {:.4f}, mae_std = {:.4f}".format(final_stats['mae_mean'], final_stats['mae_std']))

# 一次性保存整个实验的所有结果到CSV文件
csv_filename = save_results_to_csv(args, results_list, final_stats)
print(f"\n完整的实验结果已保存到CSV文件: {csv_filename}")