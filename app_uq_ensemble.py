"""
app_uq_ensemble.py - Training Script for DSTPP with UQ

This is the TRAINING script for Spatio-temporal Diffusion Point Processes (DSTPP).
For dedicated testing with quality-filtered ensemble, use `app_uq_ensemble_test.py`.

Pipeline Overview:
    1. Training Phase (this script): Train model, periodic validation with naive ensemble
    2. Testing Phase (app_uq_ensemble_test.py): Load trained models, apply quality-filtered ensemble

Key Features:
    - Supports DDPM and Rectified Flow (RF) models
    - Naive ensemble sampling for UQ during training validation
    - Saves model checkpoints every 10 epochs for later selection
    - Unique experiment naming to prevent collisions between runs

Experiment Naming Convention:
    {dataset}_{model_type}_T{timesteps}_S{samplingsteps}_seed{seed}_{datetime}_{uuid}
    
    Example: Earthquake_rf_T1000_S50_seed1234_20251201_1430_a1b2c3
    
    This ensures:
    - Different seeds → different directories
    - Different model_type → different directories
    - Different timesteps/samplingsteps → different directories
    - Same params at same time → UUID prevents collision

Output Structure:
    ./ModelSave/{exp_name}/
        ├── config.json          # All hyperparameters for reproducibility
        ├── model_0.pkl          # Checkpoint at epoch 0
        ├── model_10.pkl         # Checkpoint at epoch 10
        └── ...
    
    ./logs/{exp_name}/           # TensorBoard logs

Usage:
    Run this script via command line with arguments to specify the dataset, model type, and hyperparameters.
    Example:
        python -u app_uq_ensemble.py --dataset Earthquake --model_type rf --enable_uq --n_ensemble 50


Arguments:
    --dataset: Choice of dataset (e.g., 'Earthquake', 'Crime').
    --model_type: 'ddpm' or 'rf' (Rectified Flow).
    --enable_uq: Flag to enable uncertainty quantification evaluation.
    --n_ensemble: Number of ensemble samples for UQ.
    --mode: 'train' or 'test'.
    ... (see get_args() for full list)
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

# Import model utilities
from DSTPP.model_utils import create_model


def ensemble_sample(model, batch_size, cond, n_samples=100, dim=2):
    """
    Perform naive ensemble sampling for uncertainty quantification during training.
    
    Args:
        model: The trained model (RF_Model_all or Model_all)
        batch_size: Number of samples in a batch
        cond: Conditioning information from the transformer encoder
        n_samples: Number of ensemble samples
        dim: Spatial dimension (1, 2, or 3)
    
    Returns:
        sampled_temporal_all: List of temporal predictions
        sampled_spatial_all: List of spatial predictions
        weights: Uniform weights [n_samples]
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
    parser.add_argument('--dataset',
                        type=str,
                        default='Earthquake',
                        choices=[
                            'Citibike', 'Earthquake', 'HawkesGMM', 'Pinwheel', 'COVID19', 'Mobility', 'HawkesGMM_2d', 'Crime', 'Football',
                            'Independent', 'earth-stpp-smash'
                        ],
                        help='')
    parser.add_argument('--batch_size', type=int, default=128, help='')
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
    # NEW: 添加集成聚合策略参数
    parser.add_argument('--ensemble_agg', type=str, default='mean', choices=['mean', 'median'], help='集成聚合策略: mean (加权平均) 或 median (中位数)')
    # cpu核数
    parser.add_argument('--cpu_num', type=int, default=12, help='CPU核数')
    # Log normalization for temporal data
    parser.add_argument('--log_normalization', type=int, default=1, help='是否对时间间隔进行log变换 (1=是, 0=否)')
    # PriorNet: History-Adaptive Prior (HAP)
    parser.add_argument('--use_prior_net', action='store_true', help='使用历史自适应先验网络 (HAP/PriorNet)')
    parser.add_argument('--prior_loss_weight', type=float, default=0.1, help='Prior NLL损失权重 (仅当use_prior_net时生效)')
    parser.add_argument('--prior_hidden_dim', type=int, default=128, help='PriorNet隐藏层维度')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args


opt = get_args()
device = torch.device("cuda:{}".format(opt.cuda_id) if opt.cuda else "cpu")

if opt.dataset == 'HawkesGMM':
    opt.dim = 1


def data_loader(writer):
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


def generate_experiment_name(opt):
    """
    Generate a unique and descriptive experiment name based on key hyperparameters.
    
    Naming convention:
        {dataset}_{model_type}[_HAP{prior_loss_weight}]_T{timesteps}_S{samplingsteps}_seed{seed}_{date}_{short_uuid}
    
    This ensures:
        1. Human-readable: key params visible in name
        2. Unique: short UUID prevents any collision
        3. Sortable: date format allows chronological sorting
        4. HAP identifiable: HAP experiments clearly marked with loss weight
    """
    import uuid

    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M")
    short_uuid = str(uuid.uuid4())[:6]  # 6 chars is enough for uniqueness

    # Build experiment name with key hyperparameters
    # Include HAP marker if using prior_net
    hap_marker = ""
    if getattr(opt, 'use_prior_net', False):
        prior_weight = getattr(opt, 'prior_loss_weight', 0.1)
        hap_marker = f"_HAP{prior_weight}"

    # --log_normalization marker
    lognorm_marker = "LogNorm" if opt.log_normalization else "NoLogNorm"

    exp_name = (f"{opt.dataset}_"
                f"{opt.model_type}"
                f"{hap_marker}_"
                f"{lognorm_marker}_"
                f"T{opt.timesteps}_"
                f"S{opt.samplingsteps}_"
                f"seed{opt.seed}_"
                f"{date_str}_"
                f"{short_uuid}")

    return exp_name


def save_experiment_config(opt, save_dir):
    """
    Save all experiment configurations to a JSON file for reproducibility.
    """
    config = vars(opt).copy()
    config_path = os.path.join(save_dir, 'config.json')

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print(f"Experiment config saved to: {config_path}")


if __name__ == "__main__":
    setup_init(opt)
    setproctitle.setproctitle("RF-STPP-UQ-Training")

    print('=' * 60)
    print('Experiment Configuration:')
    print(f'  Dataset: {opt.dataset}')
    print(f'  Model Type: {opt.model_type}')
    print(f'  Timesteps: {opt.timesteps}, Sampling Steps: {opt.samplingsteps}')
    print(f'  Seed: {opt.seed}')
    print(f'  Log Normalization: {bool(opt.log_normalization)}')
    print(f'  UQ Enabled: {opt.enable_uq}')
    if opt.enable_uq:
        print(f'  Ensemble Samples: {opt.n_ensemble}')
    # PriorNet configuration (HAP)
    if opt.use_prior_net:
        print(f'  HAP/PriorNet: Enabled (prior_loss_weight={opt.prior_loss_weight}, hidden_dim={opt.prior_hidden_dim})')
    else:
        print(f'  HAP/PriorNet: Disabled (standard Gaussian prior)')
    print('=' * 60)

    # ============ Generate Unique Experiment Name ============
    # Format: {dataset}_{model_type}_T{timesteps}_S{samplingsteps}_seed{seed}_{datetime}_{uuid}
    # This prevents any collision between experiments with different hyperparameters
    exp_name = generate_experiment_name(opt)
    print(f'Experiment Name: {exp_name}')

    # Create directories with unique experiment name
    logdir = f"./logs/{exp_name}"
    model_path = f"./ModelSave/{exp_name}/"

    if not os.path.exists('./ModelSave'):
        os.mkdir('./ModelSave')
    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    if 'train' in opt.mode and not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
        # Save experiment configuration for reproducibility
        save_experiment_config(opt, model_path)

    writer = SummaryWriter(log_dir=logdir, flush_secs=5)
    print(f'TensorBoard logs: {logdir}')
    print(f'Model checkpoints: {model_path}')

    Model = create_model(opt, device)  # 根据opt.model_type创建模型rf或ddpm
    print("Model created successfully!")

    trainloader, testloader, valloader, (MAX, MIN) = data_loader(writer)
    print("Data loaded successfully!")

    warmup_steps = 5
    optimizer = AdamW(Model.parameters(), lr=opt.lr, betas=(0.9, 0.99))
    step, early_stop = 0, 0
    min_loss_test = 1e20

    for itr in range(opt.total_epochs):
        print('epoch:{}'.format(itr))

        if (itr % 10 == 0) or (itr == opt.total_epochs - 1):
            print('Evaluate!')
            with torch.no_grad():
                Model.eval()

                # Validation set evaluation =================================================

                print('Validation evaluation!')
                loss_test_all = 0.0
                mae_temporal, rmse_temporal, mae_spatial, total_num = 0.0, 0.0, 0.0, 0.0
                vb_test_all, vb_test_temporal_all, vb_test_spatial_all = 0.0, 0.0, 0.0

                for batch in valloader:
                    event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)

                    sampled_seq = Model.diffusion.sample(batch_size=event_time_non_mask.shape[0], cond=enc_out_non_mask)

                    loss = Model.diffusion(torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)
                    loss_test_all += loss.item() * event_time_non_mask.shape[0]
                    # Temporal - use denormalization function with log_normalization support
                    real = denormalization(event_time_non_mask[:, 0, :], MAX[1], MIN[1], opt.log_normalization)
                    gen = denormalization(sampled_seq[:, 0, :1], MAX[1], MIN[1], opt.log_normalization)
                    mae_temporal += torch.abs(real - gen).sum().item()
                    rmse_temporal += ((real - gen)**2).sum().item()
                    # Spatial - use denormalization function (no log transform for spatial data)
                    real = denormalization(event_loc_non_mask[:, 0, :], MAX[2:], MIN[2:], log_normalization=False)
                    gen = denormalization(sampled_seq[:, 0, -opt.dim:], MAX[2:], MIN[2:], log_normalization=False)
                    mae_spatial += torch.sqrt(torch.sum((real - gen)**2, dim=-1)).sum().item()

                    total_num += gen.shape[0]
                if loss_test_all > min_loss_test:
                    early_stop += 1
                    if early_stop >= 200:  # TODO: Check if the logic and patience is suitable for early stopping?
                        break
                else:
                    early_stop = 0

                torch.save(Model.state_dict(), model_path + 'model_{}.pkl'.format(itr))
                min_loss_test = min(min_loss_test, loss_test_all)

                # Log validation metrics
                writer.add_scalar(tag='Evaluation/loss_val', scalar_value=loss_test_all / total_num, global_step=itr)
                # writer.add_scalar(tag='Evaluation/NLL_val', scalar_value=vb_test_all / total_num, global_step=itr)
                writer.add_scalar(tag='Evaluation/mae_temporal_val', scalar_value=mae_temporal / total_num, global_step=itr)
                writer.add_scalar(tag='Evaluation/rmse_temporal_val', scalar_value=np.sqrt(rmse_temporal / total_num), global_step=itr)
                writer.add_scalar(tag='Evaluation/distance_spatial_val', scalar_value=mae_spatial / total_num, global_step=itr)

                # Test set evaluation =================================================

                print('TEST set evaluation with UQ!')
                # Test set evaluation with UQ
                if opt.enable_uq:
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

                        # Naive ensemble sampling for UQ (training-time validation)
                        sampled_temporal_all, sampled_spatial_all, ensemble_weights = ensemble_sample(Model, event_time_non_mask.shape[0],
                                                                                                      enc_out_non_mask, opt.n_ensemble, opt.dim)

                        loss = Model.diffusion(torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)

                        # Calculate NLL for test set
                        if opt.model_type == 'ddpm':
                            vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1),
                                                                                  enc_out_non_mask)
                        else:  # flow matching
                            vb, vb_temporal, vb_spatial = Model.diffusion.calculate_neg_log_likelihood(
                                torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)

                        vb_test_all += vb
                        vb_test_temporal_all += vb_temporal
                        vb_test_spatial_all += vb_spatial
                        loss_test_all += loss.item() * event_time_non_mask.shape[0]

                        # Single sample for basic metrics [NOT USED in UQ evaluation]
                        # sampled_seq = Model.diffusion.sample(batch_size=event_time_non_mask.shape[0], cond=enc_out_non_mask)
                        # # Basic metrics - 使用单次预测结果
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

                        # Basic metrics - 使用ensemble集成结果（支持加权平均）
                        # 计算ensemble预测的加权均值作为最终预测；ensemble_weights: [n_samples], 权重和为1
                        stacked_temporal = torch.stack(sampled_temporal_all, dim=0)  # [n_samples, bsz, 1]
                        stacked_spatial = torch.stack(sampled_spatial_all, dim=0)  # [n_samples, bsz, dim]

                        if opt.ensemble_agg == 'median':
                            ensemble_temporal_mean = torch.median(stacked_temporal, dim=0).values
                            ensemble_spatial_mean = torch.median(stacked_spatial, dim=0).values
                        else:
                            weights_view = ensemble_weights.view(-1, 1, 1).to(stacked_temporal.device)
                            ensemble_temporal_mean = (stacked_temporal * weights_view).sum(dim=0)  # [bsz, 1]
                            ensemble_spatial_mean = (stacked_spatial * weights_view).sum(dim=0)  # [bsz, dim]

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
                            None,  # TODO: No marks in DSTPP Task now
                            real_time_gt,
                            real_loc_gt,
                            target_levels=target_levels,
                            model='DSTPP')  # SMASH中, `model`控制ece计算方式 (mark probs or samples众数)

                        cs_time_all += calibration_score[0]
                        cs_loc_all += calibration_score[1]
                        cs2_time_all += calibration_score[2]
                        cs2_loc_all += calibration_score[3]

                    # Log test metrics
                    writer.add_scalar(tag='Evaluation/loss_test', scalar_value=loss_test_all / total_num, global_step=itr)
                    writer.add_scalar(tag='Evaluation/NLL_test', scalar_value=vb_test_all / total_num, global_step=itr)
                    writer.add_scalar(tag='Evaluation/NLL_temporal_test', scalar_value=vb_test_temporal_all / total_num, global_step=itr)
                    writer.add_scalar(tag='Evaluation/NLL_spatial_test', scalar_value=vb_test_spatial_all / total_num, global_step=itr)
                    writer.add_scalar(tag='Evaluation/mae_temporal_test', scalar_value=mae_temporal / total_num, global_step=itr)
                    writer.add_scalar(tag='Evaluation/rmse_temporal_test', scalar_value=np.sqrt(rmse_temporal / total_num), global_step=itr)
                    writer.add_scalar(tag='Evaluation/distance_spatial_test', scalar_value=mae_spatial / total_num, global_step=itr)

                    # Normalize and log UQ metrics
                    cs_time_all /= total_num  # cs score is to measure the calibration of the model (diff between predicted and true levels)
                    cs_loc_all /= total_num
                    cs2_time_all /= total_num  # predicted quantile scores
                    cs2_loc_all /= total_num

                    # Print UQ results
                    print(f'Epoch {itr} - UQ Evaluation Results:')
                    print(f'Calibration Score (Quantile) - Time: {cs2_time_all}')
                    print(f'Calibration Score (Quantile) - Location: {cs2_loc_all}')
                    print(f'Calibration Score (Mean) - Time: {cs_time_all.mean().item():.4f}')
                    print(f'Calibration Score (Mean) - Location: {cs_loc_all.mean().item():.4f}')
                    print(f'NLL Temporal: {vb_test_temporal_all / total_num:.4f}')  # NLL
                    print(f'NLL Spatial: {vb_test_spatial_all / total_num:.4f}')
                    print(f'NLL Total: {vb_test_all / total_num:.4f}')
                    print(f'MAE Temporal: {mae_temporal / total_num:.4f}')
                    print(f'RMSE Temporal: {np.sqrt(rmse_temporal / total_num):.4f}')
                    print(f'MAE Spatial: {mae_spatial / total_num:.4f}')

                    # Log UQ metrics to tensorboard
                    writer.add_scalar(tag='UQ/calibration_score_time_mean', scalar_value=cs_time_all.mean().item(), global_step=itr)
                    writer.add_scalar(tag='UQ/calibration_score_loc_mean', scalar_value=cs_loc_all.mean().item(), global_step=itr)
                    for i, level in enumerate(target_levels):
                        writer.add_scalar(tag=f'UQ/calibration_time_{level:.1f}', scalar_value=cs2_time_all[i].item(), global_step=itr)
                        writer.add_scalar(tag=f'UQ/calibration_loc_{level:.1f}', scalar_value=cs2_loc_all[i].item(), global_step=itr)

        # Training ================================================================

        # TODO: Learning rate scheduling
        lr_init = opt.lr
        if itr < warmup_steps:
            for param_group in optimizer.param_groups:
                lr = LR_warmup(lr_init, warmup_steps, itr)
                param_group["lr"] = lr
        else:
            for param_group in optimizer.param_groups:
                lr = lr_init - (lr_init - 5e-5) * (itr - warmup_steps) / opt.total_epochs
                param_group["lr"] = lr
        # elif itr == 100:
        #     for param_group in optimizer.param_groups:
        #         lr = lr_init * 0.1
        #         param_group["lr"] = lr
        # elif itr == 200:
        #     for param_group in optimizer.param_groups:
        #         lr = lr_init * 0.01
        #         param_group["lr"] = lr

        writer.add_scalar(tag='Statistics/lr', scalar_value=lr, global_step=itr)

        Model.train()
        loss_all, total_num = 0.0, 0.0

        for batch in trainloader:
            event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)
            loss = Model.diffusion(torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)

            optimizer.zero_grad()
            loss.backward()
            loss_all += loss.item() * event_time_non_mask.shape[0]

            writer.add_scalar(tag='Training/loss_step', scalar_value=loss.item(), global_step=step)

            torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.)
            optimizer.step()

            step += 1
            total_num += event_time_non_mask.shape[0]

        with torch.cuda.device("cuda:{}".format(opt.cuda_id)):
            torch.cuda.empty_cache()

        writer.add_scalar(tag='Training/loss_epoch', scalar_value=loss_all / total_num, global_step=itr)
