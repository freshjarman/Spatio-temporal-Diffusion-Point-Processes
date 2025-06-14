import torch
import torch.nn as nn
import numpy as np
from DSTPP import GaussianDiffusion_ST, Transformer, Transformer_ST, Model_all, ST_Diffusion
from DSTPP import RectifiedFlow, RF_Diffusion
from DSTPP.RF_Model_all import RF_Model_all
from DSTPP.Metric import get_calibration_score
from torch.optim import AdamW, Adam
import argparse
from scipy.stats import kstest, pearsonr
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
import matplotlib.pyplot as plt


def ensemble_sample(model, batch_size, cond, n_samples=100, dim=2):
    """
    Perform ensemble sampling with multiple samples for uncertainty quantification
    """
    sampled_temporal_all = []
    sampled_spatial_all = []

    for _ in range(n_samples):
        sampled_seq = model.diffusion.sample(batch_size=batch_size, cond=cond)
        sampled_temporal_all.append(sampled_seq[:, 0, :1])  # temporal component
        sampled_spatial_all.append(sampled_seq[:, 0, -dim:])  # spatial component

    return sampled_temporal_all, sampled_spatial_all


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
    train_data = [[[i[0], i[0] - u[index - 1][0] if index > 0 else i[0]] + i[1:] for index, i in enumerate(u)] for u in train_data]

    f = open('dataset/{}/data_val.pkl'.format(opt.dataset), 'rb')
    val_data = pickle.load(f)
    val_data = [[list(i) for i in u] for u in val_data]
    val_data = [[[i[0], i[0] - u[index - 1][0] if index > 0 else i[0]] + i[1:] for index, i in enumerate(u)] for u in val_data]

    f = open('dataset/{}/data_test.pkl'.format(opt.dataset), 'rb')
    test_data = pickle.load(f)
    test_data = [[list(i) for i in u] for u in test_data]
    test_data = [[[i[0], i[0] - u[index - 1][0] if index > 0 else i[0]] + i[1:] for index, i in enumerate(u)] for u in test_data]

    data_all = train_data + test_data + val_data

    Max, Min = [], []
    for m in range(opt.dim + 2):
        if m > 0:
            Max.append(max([i[m] for u in data_all for i in u]))
            Min.append(min([i[m] for u in data_all for i in u]))
        else:
            Max.append(1)
            Min.append(0)

    assert Min[1] >= 0
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
    setproctitle.setproctitle("RF-STPP-UQ")

    print('dataset:{}'.format(opt.dataset))
    MODEL_PATH = './ModelSave/dataset_Earthquake_timesteps_1000_2025-06-09-10h/model_280.pkl'
    # MODEL_PATH = './ModelSave/dataset_Earthquake_timesteps_50_2025-05-27-09h/model_140.pkl'
    # MODEL_PATH = './ModelSave/dataset_Crime_timesteps_50_2025-05-27-09h/model_190.pkl'
    # MODEL_PATH = './ModelSave/dataset_Football_timesteps_500_2025-06-09-10h/model_1220.pkl'  # 1230

    # Spatio-temporal Encoder
    transformer = Transformer_ST(d_model=64,
                                 d_rnn=256,
                                 d_inner=128,
                                 n_layers=4,
                                 n_head=4,
                                 d_k=16,
                                 d_v=16,
                                 dropout=0.1,
                                 device=device,
                                 loc_dim=opt.dim,
                                 CosSin=True).to(device)

    if opt.model_type == 'ddpm':
        # 原有DDPM模型创建代码
        model = ST_Diffusion(n_steps=opt.timesteps, dim=1 + opt.dim, condition=True, cond_dim=64).to(device)
        diffusion = GaussianDiffusion_ST(model,
                                         loss_type=opt.loss_type,
                                         seq_length=1 + opt.dim,
                                         timesteps=opt.timesteps,
                                         sampling_timesteps=opt.samplingsteps,
                                         objective=opt.objective,
                                         beta_schedule=opt.beta_schedule).to(device)
        Model = Model_all(transformer, diffusion)
    elif opt.model_type == 'rf':
        # 新的Rectified Flow模型创建代码
        model = RF_Diffusion(n_steps=opt.timesteps, dim=1 + opt.dim, condition=True, cond_dim=64).to(device)
        rf = RectifiedFlow(model, loss_type=opt.loss_type, seq_length=1 + opt.dim, timesteps=opt.timesteps,
                           sampling_timesteps=opt.samplingsteps).to(device)
        Model = RF_Model_all(transformer, rf)
    else:
        raise ValueError("Unsupported model type: {}".format(opt.model_type))

    print("Model created successfully!")

    if opt.mode == 'test':
        model_path = MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError("Model path does not exist: {}".format(model_path))
        print("Loading model from:", model_path)
        Model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    else:
        print("Training mode, no model loading.")
    Model.to(device)

    trainloader, testloader, valloader, (MAX, MIN) = data_loader()
    print("Data loaded successfully!")

    min_loss_test = 1e20

    print('TEST Set evaluation with Variance-Error Correlation Analysis!')

    # 存储所有样本的预测方差和预测误差
    all_temporal_var = []
    all_temporal_mae = []
    all_spatial_var = []
    all_spatial_mae = []

    with torch.no_grad():
        Model.eval()

        # Test set evaluation
        total_num = 0.0

        for batch in tqdm(testloader, desc="Processing test batch"):
            event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)

            # Ensemble sampling
            sampled_temporal_all, sampled_spatial_all = ensemble_sample(Model, event_time_non_mask.shape[0], enc_out_non_mask, opt.n_ensemble,
                                                                        opt.dim)

            # 收集所有样本的预测结果以计算方差和误差
            # 转换为tensor进行计算
            sampled_temporal_stack = torch.stack(sampled_temporal_all, dim=0)  # [n_ensemble, bsz, 1]
            sampled_spatial_stack = torch.stack(sampled_spatial_all, dim=0)  # [n_ensemble, bsz, dim]

            # 计算每个样本的ensemble预测均值
            ensemble_temporal_mean = sampled_temporal_stack.mean(dim=0)  # [bsz, 1]
            ensemble_spatial_mean = sampled_spatial_stack.mean(dim=0)  # [bsz, dim]

            # 计算每个样本的ensemble预测方差
            temporal_var = ((sampled_temporal_stack - ensemble_temporal_mean)**2).mean(dim=0).squeeze()  # [bsz]
            spatial_var = ((sampled_spatial_stack - ensemble_spatial_mean)**2).sum(dim=2).mean(dim=0)  # [bsz]

            # 将方差反归一化到原始尺度
            temporal_var = temporal_var.detach().cpu() * ((MAX[1] - MIN[1])**2)  # 方差需要平方缩放因子
            spatial_var = spatial_var.detach().cpu() * (torch.mean((torch.tensor(MAX[2:]) - torch.tensor(MIN[2:]))**2))

            # 计算真实值和预测均值之间的误差
            real_time_gt = (event_time_non_mask[:, 0, :].detach().cpu()) * (MAX[1] - MIN[1]) + MIN[1]
            gen_temporal = (ensemble_temporal_mean.detach().cpu()) * (MAX[1] - MIN[1]) + MIN[1]
            temporal_mae = torch.abs(real_time_gt - gen_temporal).squeeze()  # [bsz]

            real_loc_gt = event_loc_non_mask[:, 0, :].detach().cpu()
            real_loc_gt = real_loc_gt * (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]])) + torch.tensor([MIN[2:]])
            gen_spatial = ensemble_spatial_mean.detach().cpu()
            gen_spatial = gen_spatial * (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]])) + torch.tensor([MIN[2:]])
            spatial_mae = torch.sqrt(torch.sum((real_loc_gt - gen_spatial)**2, dim=-1))  # [bsz]

            # 保存每个样本的方差和MAE
            all_temporal_var.extend(temporal_var.tolist())
            all_temporal_mae.extend(temporal_mae.tolist())
            all_spatial_var.extend(spatial_var.tolist())
            all_spatial_mae.extend(spatial_mae.tolist())

            total_num += gen_temporal.shape[0]

    print(f"Processed {total_num} test samples")

    # 计算相关系数
    temp_corr, temp_p = pearsonr(all_temporal_var, all_temporal_mae)
    spat_corr, spat_p = pearsonr(all_spatial_var, all_spatial_mae)

    print('=' * 40)
    print('VARIANCE-ERROR CORRELATION ANALYSIS:')
    print('=' * 40)
    print(f'Temporal Variance-MAE Correlation: {temp_corr:.4f} (p-value: {temp_p:.4e})')
    print(f'Spatial Variance-MAE Correlation: {spat_corr:.4f} (p-value: {spat_p:.4e})')

    # 添加归一化函数
    def min_max_normalize(arr):
        min_val = min(arr)
        max_val = max(arr)
        # 避免除零错误
        if max_val == min_val:
            return [0.5] * len(arr)  # 如果所有值都相等，返回0.5
        return [(x - min_val) / (max_val - min_val) for x in arr]

    # 归一化数据以便更好比较
    temp_var_norm = min_max_normalize(all_temporal_var)
    temp_mae_norm = min_max_normalize(all_temporal_mae)
    spat_var_norm = min_max_normalize(all_spatial_var)
    spat_mae_norm = min_max_normalize(all_spatial_mae)

    # 计算归一化后的相关系数 (结果应与原始相关系数相同)
    temp_corr_norm, temp_p_norm = pearsonr(temp_var_norm, temp_mae_norm)
    spat_corr_norm, spat_p_norm = pearsonr(spat_var_norm, spat_mae_norm)

    print('归一化后的相关系数:')
    print(f'Temporal Variance-MAE Correlation (Normalized): {temp_corr_norm:.4f}')
    print(f'Spatial Variance-MAE Correlation (Normalized): {spat_corr_norm:.4f}')

    # 创建更好的可视化效果
    plt.figure(figsize=(18, 12))

    # 1. 时间预测的折线图 (按方差排序)
    plt.subplot(2, 2, 1)
    temp_indices = np.argsort(temp_var_norm)
    temp_var_sorted = np.array(temp_var_norm)[temp_indices]
    temp_mae_sorted = np.array(temp_mae_norm)[temp_indices]
    sample_rate = max(1, len(temp_var_sorted) // 1000)

    plt.plot(temp_var_sorted[::sample_rate], label='Temporal Variance (Norm)', color='blue', alpha=0.7)
    plt.plot(temp_mae_sorted[::sample_rate], label='Temporal MAE (Norm)', color='red', alpha=0.7)
    plt.title(f'Temporal Prediction: Normalized Variance vs MAE (Correlation: {temp_corr_norm:.4f})')
    plt.xlabel('Sorted Sample Index')
    plt.ylabel('Normalized Value [0,1]')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 2. 空间预测的折线图 (按方差排序)
    plt.subplot(2, 2, 2)
    spat_indices = np.argsort(spat_var_norm)
    spat_var_sorted = np.array(spat_var_norm)[spat_indices]
    spat_mae_sorted = np.array(spat_mae_norm)[spat_indices]

    plt.plot(spat_var_sorted[::sample_rate], label='Spatial Variance (Norm)', color='blue', alpha=0.7)
    plt.plot(spat_mae_sorted[::sample_rate], label='Spatial MAE (Norm)', color='red', alpha=0.7)
    plt.title(f'Spatial Prediction: Normalized Variance vs MAE (Correlation: {spat_corr_norm:.4f})')
    plt.xlabel('Sorted Sample Index')
    plt.ylabel('Normalized Value [0,1]')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    # 3. 时间预测的散点图与回归线
    plt.subplot(2, 2, 3)
    # 为了使散点图清晰，可能需要采样
    sample_size = min(5000, len(temp_var_norm))
    sample_idx = np.random.choice(len(temp_var_norm), sample_size, replace=False)

    plt.scatter(np.array(temp_var_norm)[sample_idx], np.array(temp_mae_norm)[sample_idx], alpha=0.4, s=10, color='purple')

    # 添加线性回归线
    from scipy.stats import linregress
    slope, intercept, _, _, _ = linregress(temp_var_norm, temp_mae_norm)
    x_vals = np.array([min(temp_var_norm), max(temp_var_norm)])
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, 'r--', linewidth=2)

    plt.title(f'Temporal: Variance vs MAE Scatter (Corr: {temp_corr_norm:.4f})')
    plt.xlabel('Normalized Variance')
    plt.ylabel('Normalized MAE')
    plt.grid(True, linestyle='--', alpha=0.5)

    # 4. 空间预测的散点图与回归线
    plt.subplot(2, 2, 4)
    plt.scatter(np.array(spat_var_norm)[sample_idx], np.array(spat_mae_norm)[sample_idx], alpha=0.4, s=10, color='purple')

    # 添加线性回归线
    slope, intercept, _, _, _ = linregress(spat_var_norm, spat_mae_norm)
    x_vals = np.array([min(spat_var_norm), max(spat_var_norm)])
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, 'r--', linewidth=2)

    plt.title(f'Spatial: Variance vs MAE Scatter (Corr: {spat_corr_norm:.4f})')
    plt.xlabel('Normalized Variance')
    plt.ylabel('Normalized MAE')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # 保存更新后的图像
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_without_ext = os.path.basename(MODEL_PATH).split('.')[0]
    fig_path = f'var_error_correlation_normalized_{opt.dataset}_{filename_without_ext}_{timestamp}.png'
    plt.savefig(fig_path)
    print(f'Normalized figure saved to: {fig_path}')

    # 保存归一化的数据
    data_to_save = {
        # 原始数据
        'temporal_variance': all_temporal_var,
        'temporal_mae': all_temporal_mae,
        'spatial_variance': all_spatial_var,
        'spatial_mae': all_spatial_mae,
        'temporal_correlation': temp_corr,
        'temporal_pvalue': temp_p,
        'spatial_correlation': spat_corr,
        'spatial_pvalue': spat_p,

        # 归一化数据
        'temporal_variance_normalized': temp_var_norm,
        'temporal_mae_normalized': temp_mae_norm,
        'spatial_variance_normalized': spat_var_norm,
        'spatial_mae_normalized': spat_mae_norm,
        'temporal_correlation_normalized': temp_corr_norm,
        'spatial_correlation_normalized': spat_corr_norm,
        'dataset': opt.dataset,
        'model_path': MODEL_PATH,
        'n_ensemble': opt.n_ensemble
    }

    data_path = f'var_error_data_{opt.dataset}_{filename_without_ext}_{timestamp}.json'
    with open(data_path, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    print(f'Data saved to: {data_path}')

    # 展示图像（如果在有GUI的环境中）
    plt.show()
