"""
app_uq_ensemble_test.py - Dedicated Testing Script with Quality-Filtered Ensemble

This is the TESTING script for post-training evaluation with quality-filtered ensemble.
For training, use `app_uq_ensemble.py`.

Pipeline Overview:
    1. Training Phase (app_uq_ensemble.py): Train model, save checkpoints every 10 epochs
    2. Model Selection: Manually select main model and auxiliary models from saved checkpoints
    3. Testing Phase (this script): Load selected models, apply quality-filtered ensemble

Key Features:
    - Supports both naive ensemble and quality-filtered ensemble
    - Uses multiple auxiliary models for noise quality evaluation
    - Entropy-based filtering and inverse-entropy weighting
    - Runs 10 times for statistical analysis (mean ± std)

Usage:
    # Naive ensemble (default)
    python app_uq_ensemble_test.py --dataset Earthquake --n_ensemble 50

    # Quality-filtered ensemble
    python app_uq_ensemble_test.py --dataset Earthquake --n_ensemble 50 \\
        --enable_filtered_ensemble \\
        --aux_model_dir ./ModelSave/dataset_Earthquake_timesteps_1000_xxx/ \\
        --aux_model_epochs "100,150,200,250"

        # PS: aux-models 思考可以选一次选连的不同epoch对应的models，也可以尝试选择不同seed训练的models
"""

import torch
import torch.nn as nn
import numpy as np
import math
from DSTPP import GaussianDiffusion_ST, Transformer, Transformer_ST, Model_all, ST_Diffusion
from DSTPP import RectifiedFlow, RF_Diffusion
from DSTPP.RF_Model_all import RF_Model_all
from DSTPP.Metric import get_calibration_score
from torch.optim import AdamW, Adam
import argparse
from scipy.stats import kstest
from DSTPP.Dataset import get_dataloader
import time
import setproctitle
from torch.utils.tensorboard import SummaryWriter
import datetime
import pickle
import os
from tqdm import tqdm
import random
import json
import datetime

# Import model utilities and enhanced ensemble functions
from DSTPP.model_utils import create_model, load_model, load_multiple_models, find_model_checkpoints, parse_epochs_string
from gu_ensemble import quality_filtered_ensemble


def ensemble_sample(model, batch_size, cond, n_samples=100, dim=2):
    """
    Perform naive ensemble sampling with uniform weights.
    Returns weights for consistency with enhanced ensemble.
    """
    sampled_temporal_all = []
    sampled_spatial_all = []

    for _ in range(n_samples):
        sampled_seq = model.diffusion.sample(batch_size=batch_size, cond=cond)
        sampled_temporal_all.append(sampled_seq[:, 0, :1])  # temporal component
        sampled_spatial_all.append(sampled_seq[:, 0, -dim:])  # spatial component

    # Uniform weights for naive ensemble
    weights = torch.ones(n_samples) / n_samples
    return sampled_temporal_all, sampled_spatial_all, weights


def setup_init(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_name():
    TIME = int(time.time())
    TIME = time.localtime(TIME)
    return time.strftime("%Y-%m-%d %H:%M:%S", TIME)


def normalization(x, MAX, MIN):
    # normalize the data to [0, 1]
    return (x - MIN) / (MAX - MIN)


def denormalization(x, MAX, MIN, log_normalization=False):
    """
    Denormalize the data from [0, 1] back to original scale.
    If log_normalization is True, also apply exp() to reverse the log transform.
    
    Args:
        x: normalized data (tensor)
        MAX: maximum value used in normalization (scalar or tensor)
        MIN: minimum value used in normalization (scalar or tensor)
        log_normalization: whether log transform was applied before normalization
    
    Returns:
        Denormalized data in original scale
    """
    x_cpu = x.detach().cpu()
    # Convert MAX/MIN to tensor if they are lists
    if isinstance(MAX, list):
        MAX = torch.tensor(MAX)
    if isinstance(MIN, list):
        MIN = torch.tensor(MIN)

    if log_normalization:
        return torch.exp(x_cpu * (MAX - MIN) + MIN)
    else:
        return x_cpu * (MAX - MIN) + MIN


def get_args():
    parser = argparse.ArgumentParser(description='DSTPP with Uncertainty Quantification')
    parser.add_argument('--seed', type=int, default=1234, help='')
    parser.add_argument('--mode', type=str, default='train', help='')
    parser.add_argument('--total_epochs', type=int, default=200, help='')
    parser.add_argument('--machine', type=str, default='none', help='')
    parser.add_argument('--loss_type', type=str, default='l2', choices=['l1', 'l2', 'Euclid'], help='')
    parser.add_argument('--beta_schedule', type=str, default='cosine', choices=['linear', 'cosine'], help='')
    parser.add_argument('--dim', type=int, default=2, help='', choices=[1, 2, 3])
    parser.add_argument(
        '--dataset',
        type=str,
        default='Earthquake',
        choices=['Citibike', 'Earthquake', 'HawkesGMM', 'Pinwheel', 'COVID19', 'Mobility', 'HawkesGMM_2d', 'Crime', 'Football', 'Independent'],
        help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--timesteps', type=int, default=50, help='')
    parser.add_argument('--samplingsteps', type=int, default=50, help='')
    parser.add_argument('--objective', type=str, default='pred_noise', help='')
    parser.add_argument('--cuda_id', type=str, default='0', help='')
    # NEW: 添加模型类型参数
    parser.add_argument('--model_type', type=str, default='rf', choices=['ddpm', 'rf'], help='使用的模型类型')
    # NEW: UQ评估参数
    parser.add_argument('--enable_uq', action='store_true', help='启用uncertainty quantification评估')
    parser.add_argument('--n_ensemble', type=int, default=5, help='ensemble采样数量')
    # cpu核数
    parser.add_argument('--cpu_num', type=int, default=6, help='CPU核数')

    # Enhanced ensemble arguments (quality-filtered ensemble)
    parser.add_argument('--enable_filtered_ensemble', action='store_true', help='启用quality-filtered ensemble（需要多个辅助模型）')
    parser.add_argument('--aux_model_dir', type=str, default=None, help='Directory containing auxiliary model checkpoints')
    parser.add_argument('--aux_model_epochs', type=str, default=None, help='Comma-separated epochs to load, e.g., "100,120,140"')
    parser.add_argument('--filter_ratio', type=float, default=0.2, help='Ratio of low-quality noises to filter out')
    parser.add_argument('--use_weighting', action='store_true', default=True, help='Use inverse-entropy weighting for ensemble')
    # Log normalization for temporal data
    parser.add_argument('--log_normalization', type=int, default=1, help='是否对时间间隔进行log变换 (1=是, 0=否)')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args


opt = get_args()
device = torch.device("cuda:{}".format(opt.cuda_id) if opt.cuda else "cpu")

if opt.dataset == 'HawkesGMM':
    opt.dim = 1


def data_loader():
    f = open('dataset/{}/data_train.pkl'.format(opt.dataset), 'rb')
    train_data = pickle.load(f)
    train_data = [[list(i) for i in u] for u in train_data]

    f = open('dataset/{}/data_val.pkl'.format(opt.dataset), 'rb')
    val_data = pickle.load(f)
    val_data = [[list(i) for i in u] for u in val_data]

    f = open('dataset/{}/data_test.pkl'.format(opt.dataset), 'rb')
    test_data = pickle.load(f)
    test_data = [[list(i) for i in u] for u in test_data]

    # Compute time intervals (d_t) with optional log transform
    if not opt.log_normalization:
        # Standard: d_t = t_i - t_{i-1}
        train_data = [[[i[0], i[0] - u[index - 1][0] if index > 0 else i[0]] + i[1:] for index, i in enumerate(u)] for u in train_data]
        val_data = [[[i[0], i[0] - u[index - 1][0] if index > 0 else i[0]] + i[1:] for index, i in enumerate(u)] for u in val_data]
        test_data = [[[i[0], i[0] - u[index - 1][0] if index > 0 else i[0]] + i[1:] for index, i in enumerate(u)] for u in test_data]
    else:
        # Log transform: log(max(d_t, 1e-4)) to handle long-tail distribution
        train_data = [[[i[0], math.log(max(i[0] - u[index - 1][0], 1e-4)) if index > 0 else math.log(max(i[0], 1e-4))] + i[1:]
                       for index, i in enumerate(u)] for u in train_data]
        val_data = [[[i[0], math.log(max(i[0] - u[index - 1][0], 1e-4)) if index > 0 else math.log(max(i[0], 1e-4))] + i[1:]
                     for index, i in enumerate(u)] for u in val_data]
        test_data = [[[i[0], math.log(max(i[0] - u[index - 1][0], 1e-4)) if index > 0 else math.log(max(i[0], 1e-4))] + i[1:]
                      for index, i in enumerate(u)] for u in test_data]

    data_all = train_data + test_data + val_data

    Max, Min = [], []
    for m in range(opt.dim + 2):
        if m > 0:
            Max.append(max([i[m] for u in data_all for i in u]))
            Min.append(min([i[m] for u in data_all for i in u]))
        else:
            Max.append(1)
            Min.append(0)

    # Only check Min[1] >= 0 when not using log normalization
    # (log values can be negative, e.g., log(1e-4) ≈ -9.2)
    if not opt.log_normalization:
        assert Min[1] >= 0, "Time interval should be non-negative when not using log normalization"

    # normalize d_time and location (vector)
    train_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in train_data]
    test_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in test_data]
    val_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in val_data]

    trainloader = get_dataloader(train_data, opt.batch_size, D=opt.dim, shuffle=True, num_workers=opt.cpu_num)
    testloader = get_dataloader(test_data, len(test_data) if len(test_data) <= 1000 else 1000, D=opt.dim, shuffle=False, num_workers=opt.cpu_num)
    valloader = get_dataloader(val_data, len(val_data) if len(val_data) <= 1000 else 1000, D=opt.dim, shuffle=False, num_workers=opt.cpu_num)

    return trainloader, testloader, valloader, (Max, Min)


def Batch2toModel(batch, transformer):
    if opt.dim == 1:
        event_time_origin, event_time, lng = map(lambda x: x.to(device), batch)
        event_loc = lng.unsqueeze(dim=2)

    if opt.dim == 2:
        event_time_origin, event_time, lng, lat = map(lambda x: x.to(device), batch)
        event_loc = torch.cat((lng.unsqueeze(dim=2), lat.unsqueeze(dim=2)), dim=-1)

    if opt.dim == 3:
        event_time_origin, event_time, lng, lat, height = map(lambda x: x.to(device), batch)
        event_loc = torch.cat((lng.unsqueeze(dim=2), lat.unsqueeze(dim=2), height.unsqueeze(dim=2)), dim=-1)

    event_time = event_time.to(device)
    event_time_origin = event_time_origin.to(device)
    event_loc = event_loc.to(device)

    enc_out, mask = transformer(event_loc, event_time_origin)

    enc_out_non_mask = []
    event_time_non_mask = []
    event_loc_non_mask = []
    for index in range(mask.shape[0]):
        length = int(sum(mask[index]).item())
        if length > 1:
            enc_out_non_mask += [i.unsqueeze(dim=0) for i in enc_out[index][:length - 1]]
            event_time_non_mask += [i.unsqueeze(dim=0) for i in event_time[index][1:length]]
            event_loc_non_mask += [i.unsqueeze(dim=0) for i in event_loc[index][1:length]]

    enc_out_non_mask = torch.cat(enc_out_non_mask, dim=0)
    event_time_non_mask = torch.cat(event_time_non_mask, dim=0)
    event_loc_non_mask = torch.cat(event_loc_non_mask, dim=0)

    event_time_non_mask = event_time_non_mask.reshape(-1, 1, 1)
    event_loc_non_mask = event_loc_non_mask.reshape(-1, 1, opt.dim)
    enc_out_non_mask = enc_out_non_mask.reshape(event_time_non_mask.shape[0], 1, -1)

    return event_time_non_mask, event_loc_non_mask, enc_out_non_mask


def LR_warmup(lr, epoch_num, epoch_current):
    return lr * (epoch_current + 1) / epoch_num


if __name__ == "__main__":
    setup_init(opt)
    setproctitle.setproctitle("RF-STPP-UQ-Test")

    print('dataset:{}'.format(opt.dataset))
    print('mode:', opt.mode)
    print('enable_filtered_ensemble:', opt.enable_filtered_ensemble)

    # Model path configuration
    MODEL_PATH = './ModelSave/dataset_Earthquake_timesteps_1000_2025-06-09-10h/model_280.pkl'
    # MODEL_PATH = './ModelSave/dataset_Earthquake_timesteps_50_2025-05-27-09h/model_140.pkl'
    # MODEL_PATH = './ModelSave/dataset_Crime_timesteps_50_2025-05-27-09h/model_190.pkl'
    # MODEL_PATH = './ModelSave/dataset_Football_timesteps_500_2025-06-09-10h/model_1220.pkl'

    # ============ Model Creation (using utility function) ============
    Model = create_model(opt, device)
    print("Model created successfully!")

    # ============ Load Main Model ============
    if opt.mode == 'test':
        model_path = MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model path does not exist: {}".format(model_path))
        print("Loading model from:", model_path)
        Model.load_state_dict(torch.load(model_path, map_location=device))
        Model.to(device)
        Model.eval()
        print("Model loaded successfully!")
    else:
        # 报错提示仅支持测试模式并退出
        raise RuntimeError("Only test mode is supported. Exiting.")
        # Model.to(device)

    # ============ Load Auxiliary Models for Enhanced Ensemble ============
    auxiliary_models = None
    if opt.enable_filtered_ensemble and opt.aux_model_dir is not None:
        print("Loading auxiliary models for quality-filtered ensemble...")
        epochs = parse_epochs_string(opt.aux_model_epochs)
        try:
            checkpoint_paths = find_model_checkpoints(opt.aux_model_dir, epochs)
            if len(checkpoint_paths) >= 2:
                # load_multiple_models 内部已经处理 device 和 eval 模式
                auxiliary_models = load_multiple_models(opt, device, checkpoint_paths)
                print(f"Enhanced ensemble enabled with {len(auxiliary_models)} auxiliary models.")
            else:
                print("Warning: Less than 2 auxiliary models. Falling back to naive ensemble.")
        except FileNotFoundError as e:
            print(f"Warning: {e}. Falling back to naive ensemble.")

    trainloader, testloader, valloader, (MAX, MIN) = data_loader()
    print("Data loaded successfully!")

    # warmup_steps = 5
    # optimizer = AdamW(Model.parameters(), lr=opt.lr, betas=(0.9, 0.99))
    # step, early_stop = 0, 0
    min_loss_test = 1e20

    print('TEST Set evaluation with UQ - Running 10 times for statistical analysis!')

    # 存储10次运行的结果
    all_mae_temporal = []
    all_rmse_temporal = []
    all_mae_spatial = []
    all_cs_time_mean = []
    all_cs_loc_mean = []
    all_cs2_time = []
    all_cs2_loc = []

    # 运行10次测试
    for run_idx in range(10):
        print(f'\nRun {run_idx + 1}/10:')

        with torch.no_grad():
            Model.eval()

            # Test set evaluation with UQ
            loss_test_all, vb_test_all, vb_test_temporal_all, vb_test_spatial_all = 0.0, 0.0, 0.0, 0.0
            mae_temporal, rmse_temporal, mae_spatial, total_num = 0.0, 0.0, 0.0, 0.0

            # UQ metrics accumulators
            target_levels = np.linspace(0.5, 0.9, 5)  # 0.5 0.6 0.7 0.8 0.9
            cs_time_all = torch.zeros(len(target_levels))
            cs_loc_all = torch.zeros(len(target_levels))
            cs2_time_all = torch.zeros(len(target_levels))
            cs2_loc_all = torch.zeros(len(target_levels))

            for batch in testloader:
                event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)

                # Ensemble sampling for UQ
                # 根据是否启用质量过滤ensemble选择不同的采样策略
                if opt.enable_filtered_ensemble and auxiliary_models:
                    # 使用质量过滤的ensemble (需要多个模型)
                    sampled_temporal_all, sampled_spatial_all, ensemble_weights = quality_filtered_ensemble(model=Model,
                                                                                                            auxiliary_models=auxiliary_models,
                                                                                                            batch_size=event_time_non_mask.shape[0],
                                                                                                            cond=enc_out_non_mask,
                                                                                                            n_noises=opt.n_ensemble,
                                                                                                            dim=opt.dim,
                                                                                                            filter_ratio=opt.filter_ratio,
                                                                                                            use_weighting=opt.use_weighting,
                                                                                                            device=device)
                    # 注意: 过滤后 ensemble_weights 长度可能小于 n_ensemble
                    n_kept = len(sampled_temporal_all)
                    if run_idx == 0:  # 只在第一次运行时打印
                        print(f"  [Filtered Ensemble] Kept {n_kept} / {opt.n_ensemble} samples after quality filtering")
                else:
                    # 使用朴素ensemble (单模型多次采样)
                    sampled_temporal_all, sampled_spatial_all, ensemble_weights = ensemble_sample(Model, event_time_non_mask.shape[0],
                                                                                                  enc_out_non_mask, opt.n_ensemble, opt.dim)

                # Single sample for basic metrics
                # sampled_seq = Model.diffusion.sample(batch_size=event_time_non_mask.shape[0], cond=enc_out_non_mask)

                # loss = Model.diffusion(torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)
                """
                # NLL calculation for DDPM or RF models
                if opt.model_type == 'ddpm':
                    vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1),
                                                                            enc_out_non_mask)
                else:
                    vb, vb_temporal, vb_spatial = Model.diffusion.calculate_neg_log_likelihood(
                        torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)

                vb_test_all += vb
                vb_test_temporal_all += vb_temporal
                vb_test_spatial_all += vb_spatial
                # loss_test_all += loss.item() * event_time_non_mask.shape[0]
                """

                # # Basic metrics - 使用单词预测结果
                # real = (event_time_non_mask[:, 0, :].detach().cpu()) * (MAX[1] - MIN[1]) + MIN[1]
                # gen = (sampled_seq[:, 0, :1].detach().cpu()) * (MAX[1] - MIN[1]) + MIN[1]
                # mae_temporal += torch.abs(real - gen).sum().item()
                # rmse_temporal += ((real - gen)**2).sum().item()

                # real = event_loc_non_mask[:, 0, :].detach().cpu()
                # real = real * (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]])) + torch.tensor([MIN[2:]])
                # gen = sampled_seq[:, 0, -opt.dim:].detach().cpu()
                # gen = gen * (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]])) + torch.tensor([MIN[2:]])
                # mae_spatial += torch.sqrt(torch.sum((real - gen)**2, dim=-1)).sum().item()

                # total_num += gen.shape[0]

                # Basic metrics - 使用ensemble集成结果
                # 计算ensemble预测的加权均值作为最终预测，预期：ensemble的加权均值预测通常比单次采样更稳定和准确
                temporal_stack = torch.stack(sampled_temporal_all, dim=0)  # [n_filtered, bsz, 1]
                spatial_stack = torch.stack(sampled_spatial_all, dim=0)  # [n_filtered, bsz, dim]

                # 使用ensemble权重进行加权平均
                # 注意: ensemble_weights 可能是 torch.Tensor 或 list，需要统一处理
                if isinstance(ensemble_weights, torch.Tensor):
                    weights_tensor = ensemble_weights.to(device=temporal_stack.device, dtype=temporal_stack.dtype)
                else:
                    weights_tensor = torch.tensor(ensemble_weights, device=temporal_stack.device, dtype=temporal_stack.dtype)
                weights_tensor = weights_tensor.view(-1, 1, 1)  # [n_filtered, 1, 1]

                ensemble_temporal_mean = (temporal_stack * weights_tensor).sum(dim=0)  # [bsz, 1]
                ensemble_spatial_mean = (spatial_stack * weights_tensor).sum(dim=0)  # [bsz, dim]

                # Temporal metrics - use denormalization function with log_normalization support
                real_time_gt = denormalization(event_time_non_mask[:, 0, :], MAX[1], MIN[1], opt.log_normalization)
                gen_temporal = denormalization(ensemble_temporal_mean, MAX[1], MIN[1], opt.log_normalization)
                mae_temporal += torch.abs(real_time_gt - gen_temporal).sum().item()
                rmse_temporal += ((real_time_gt - gen_temporal)**2).sum().item()

                # Spatial metrics - use denormalization function (no log transform for spatial data)
                real_loc_gt = denormalization(event_loc_non_mask[:, 0, :], MAX[2:], MIN[2:], log_normalization=False)
                gen_spatial = denormalization(ensemble_spatial_mean, MAX[2:], MIN[2:], log_normalization=False)
                mae_spatial += torch.sqrt(torch.sum((real_loc_gt - gen_spatial)**2, dim=-1)).sum().item()

                total_num += gen_temporal.shape[0]

                # UQ evaluation: calculate calibration scores
                # 使用实际保留的样本数量进行判断（过滤后可能减少）
                actual_n_samples = len(sampled_temporal_all)
                if actual_n_samples >= 10:  # Only perform UQ evaluation if ensemble size is sufficient
                    sampled_temporal_denorm = []
                    sampled_spatial_denorm = []

                    for temp_sample in sampled_temporal_all:
                        temp_denorm = denormalization(temp_sample, MAX[1], MIN[1], opt.log_normalization)
                        sampled_temporal_denorm.append(temp_denorm.unsqueeze(1))

                    for spat_sample in sampled_spatial_all:
                        spat_denorm = denormalization(spat_sample, MAX[2:], MIN[2:], log_normalization=False)
                        sampled_spatial_denorm.append(spat_denorm.unsqueeze(1))

                    # Calculate calibration scores
                    calibration_score = get_calibration_score(
                        sampled_temporal_denorm,
                        sampled_spatial_denorm,
                        None,  # No marks in DSTPP Task
                        real_time_gt,
                        real_loc_gt,
                        target_levels=target_levels,
                        model='DSTPP')

                    cs_time_all += calibration_score[0]
                    cs_loc_all += calibration_score[1]
                    cs2_time_all += calibration_score[2]
                    cs2_loc_all += calibration_score[3]

            # Normalize UQ metrics
            cs_time_all /= total_num
            cs_loc_all /= total_num
            cs2_time_all /= total_num
            cs2_loc_all /= total_num

            # 记录本次运行的结果
            current_mae_temporal = mae_temporal / total_num
            current_rmse_temporal = np.sqrt(rmse_temporal / total_num)
            current_mae_spatial = mae_spatial / total_num
            current_cs_time_mean = cs_time_all.mean().item()
            current_cs_loc_mean = cs_loc_all.mean().item()

            all_mae_temporal.append(current_mae_temporal)
            all_rmse_temporal.append(current_rmse_temporal)
            all_mae_spatial.append(current_mae_spatial)
            all_cs_time_mean.append(current_cs_time_mean)
            all_cs_loc_mean.append(current_cs_loc_mean)
            all_cs2_time.append(cs2_time_all.numpy())
            all_cs2_loc.append(cs2_loc_all.numpy())

            # Print current run results
            print(f'  MAE Temporal: {current_mae_temporal:.4f}')
            print(f'  RMSE Temporal: {current_rmse_temporal:.4f}')
            print(f'  MAE Spatial: {current_mae_spatial:.4f}')
            print(f'  Calibration Score (Mean) - Time: {current_cs_time_mean:.4f}')
            print(f'  Calibration Score (Mean) - Location: {current_cs_loc_mean:.4f}')

    # 计算所有运行的统计信息
    print('\n' + '=' * 40)
    print('FINAL STATISTICAL SUMMARY (10 runs):')
    print('=' * 40)

    # 基本指标统计
    mae_temporal_mean = np.mean(all_mae_temporal)
    mae_temporal_std = np.std(all_mae_temporal)
    rmse_temporal_mean = np.mean(all_rmse_temporal)
    rmse_temporal_std = np.std(all_rmse_temporal)
    mae_spatial_mean = np.mean(all_mae_spatial)
    mae_spatial_std = np.std(all_mae_spatial)

    print(f'MAE Temporal: {mae_temporal_mean:.4f} ± {mae_temporal_std:.4f}')
    print(f'RMSE Temporal: {rmse_temporal_mean:.4f} ± {rmse_temporal_std:.4f}')
    print(f'MAE Spatial: {mae_spatial_mean:.4f} ± {mae_spatial_std:.4f}')

    # UQ指标统计
    cs_time_mean_avg = np.mean(all_cs_time_mean)
    cs_time_mean_std = np.std(all_cs_time_mean)
    cs_loc_mean_avg = np.mean(all_cs_loc_mean)
    cs_loc_mean_std = np.std(all_cs_loc_mean)

    print(f'Calibration Score (Mean) - Time: {cs_time_mean_avg:.4f} ± {cs_time_mean_std:.4f}')
    print(f'Calibration Score (Mean) - Location: {cs_loc_mean_avg:.4f} ± {cs_loc_mean_std:.4f}')

    # 各置信水平的校准分数统计
    all_cs2_time_array = np.array(all_cs2_time)  # shape: (10, 5)
    all_cs2_loc_array = np.array(all_cs2_loc)  # shape: (10, 5)

    print('\nCalibration Scores by Confidence Level:')
    print('Time Calibration:')
    for i, level in enumerate(target_levels):
        mean_val = np.mean(all_cs2_time_array[:, i])
        std_val = np.std(all_cs2_time_array[:, i])
        print(f'  Level {level:.1f}: {mean_val:.4f} ± {std_val:.4f}')

    print('Location Calibration:')
    for i, level in enumerate(target_levels):
        mean_val = np.mean(all_cs2_loc_array[:, i])
        std_val = np.std(all_cs2_loc_array[:, i])
        print(f'  Level {level:.1f}: {mean_val:.4f} ± {std_val:.4f}')

    print('=' * 80)

    # 保存结果到文件
    results_summary = {
        'mae_temporal': {
            'mean': mae_temporal_mean,
            'std': mae_temporal_std,
            'all_values': all_mae_temporal
        },
        'rmse_temporal': {
            'mean': rmse_temporal_mean,
            'std': rmse_temporal_std,
            'all_values': all_rmse_temporal
        },
        'mae_spatial': {
            'mean': mae_spatial_mean,
            'std': mae_spatial_std,
            'all_values': all_mae_spatial
        },
        'cs_time_mean': {
            'mean': cs_time_mean_avg,
            'std': cs_time_mean_std,
            'all_values': all_cs_time_mean
        },
        'cs_loc_mean': {
            'mean': cs_loc_mean_avg,
            'std': cs_loc_mean_std,
            'all_values': all_cs_loc_mean
        },
        'cs2_time_by_level': {
            'means': np.mean(all_cs2_time_array, axis=0).tolist(),
            'stds': np.std(all_cs2_time_array, axis=0).tolist(),
            'all_values': all_cs2_time_array.tolist()
        },
        'cs2_loc_by_level': {
            'means': np.mean(all_cs2_loc_array, axis=0).tolist(),
            'stds': np.std(all_cs2_loc_array, axis=0).tolist(),
            'all_values': all_cs2_loc_array.tolist()
        },
        'target_levels': target_levels.tolist()
    }

    # 保存到json文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_without_ext = MODEL_PATH.split('/')[-1].split('.')[0]
    results_file = f'./jsons/{filename_without_ext}_uq_test_results_{opt.dataset}_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f'Results saved to: {results_file}')
