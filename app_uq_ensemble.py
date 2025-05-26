import torch
import torch.nn as nn
import numpy as np
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
    parser.add_argument('--dataset',
                        type=str,
                        default='Earthquake',
                        choices=['Citibike', 'Earthquake', 'HawkesGMM', 'Pinwheel', 'COVID19', 'Mobility', 'HawkesGMM_2d', 'Independent'],
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
    # cpu核数
    parser.add_argument('--cpu_num', type=int, default=6, help='CPU核数')
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

    assert Min[1] > 0
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
    setproctitle.setproctitle("RF-STPP-UQ-Training")

    print('dataset:{}'.format(opt.dataset))
    print('UQ enabled:', opt.enable_uq)
    if opt.enable_uq:
        print('Ensemble samples need num = :', opt.n_ensemble)

    # Specify a directory for logging data
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d-%Hh")
    logdir = "./logs/{}_timesteps_{}_{}".format(opt.dataset, opt.timesteps, date)
    model_path = './ModelSave/dataset_{}_timesteps_{}_{}/'.format(opt.dataset, opt.timesteps, date)

    if not os.path.exists('./ModelSave'):
        os.mkdir('./ModelSave')

    if 'train' in opt.mode and not os.path.exists(model_path):
        os.mkdir(model_path)

    writer = SummaryWriter(log_dir=logdir, flush_secs=5)

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

                # Validation set evaluation
                print('Validation evaluation!')
                loss_test_all = 0.0
                mae_temporal, rmse_temporal, mae_spatial, total_num = 0.0, 0.0, 0.0, 0.0
                vb_test_all, vb_test_temporal_all, vb_test_spatial_all = 0.0, 0.0, 0.0

                for batch in valloader:
                    event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)

                    sampled_seq = Model.diffusion.sample(batch_size=event_time_non_mask.shape[0], cond=enc_out_non_mask)

                    loss = Model.diffusion(torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)

                    # Unnecessary in dev-set: Calculate the negative log likelihood (NLL) for validation

                    # if opt.model_type == 'ddpm':
                    #     vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(
                    #         torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)
                    # else:
                    #     vb, vb_temporal, vb_spatial = Model.diffusion.calculate_neg_log_likelihood(
                    #         torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)

                    # vb_test_all += vb
                    # vb_test_temporal_all += vb_temporal
                    # vb_test_spatial_all += vb_spatial
                    loss_test_all += loss.item() * event_time_non_mask.shape[0]
                    # Temporal
                    real = (event_time_non_mask[:, 0, :].detach().cpu()) * (MAX[1] - MIN[1]) + MIN[1]
                    gen = (sampled_seq[:, 0, :1].detach().cpu()) * (MAX[1] - MIN[1]) + MIN[1]
                    mae_temporal += torch.abs(real - gen).sum().item()
                    rmse_temporal += ((real - gen)**2).sum().item()
                    # Spatial
                    real = event_loc_non_mask[:, 0, :].detach().cpu()
                    real = real * (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]])) + torch.tensor([MIN[2:]])
                    gen = sampled_seq[:, 0, -opt.dim:].detach().cpu()
                    gen = gen * (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]])) + torch.tensor([MIN[2:]])
                    mae_spatial += torch.sqrt(torch.sum((real - gen)**2, dim=-1)).sum().item()

                    total_num += gen.shape[0]

                # TODO: Check if the logic and patience is suitable for early stopping?
                if loss_test_all > min_loss_test:
                    early_stop += 1
                    if early_stop >= 5:
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

                print('TEST set evaluation with UQ!')
                # Test set evaluation with UQ
                if opt.enable_uq:
                    loss_test_all, vb_test_all, vb_test_temporal_all, vb_test_spatial_all = 0.0, 0.0, 0.0, 0.0
                    mae_temporal, rmse_temporal, mae_spatial, total_num = 0.0, 0.0, 0.0, 0.0

                    # UQ metrics accumulators
                    target_levels = np.linspace(0.5, 0.9, 5)
                    cs_time_all = torch.zeros(len(target_levels))
                    cs_loc_all = torch.zeros(len(target_levels))
                    cs2_time_all = torch.zeros(len(target_levels))
                    cs2_loc_all = torch.zeros(len(target_levels))

                    for batch in testloader:
                        event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)

                        # Ensemble sampling for UQ
                        sampled_temporal_all, sampled_spatial_all = ensemble_sample(Model, event_time_non_mask.shape[0], enc_out_non_mask,
                                                                                    opt.n_ensemble, opt.dim)

                        # Single sample for basic metrics
                        sampled_seq = Model.diffusion.sample(batch_size=event_time_non_mask.shape[0], cond=enc_out_non_mask)

                        loss = Model.diffusion(torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)

                        if opt.model_type == 'ddpm':
                            vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1),
                                                                                  enc_out_non_mask)
                        else:
                            vb, vb_temporal, vb_spatial = Model.diffusion.calculate_neg_log_likelihood(
                                torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)

                        vb_test_all += vb
                        vb_test_temporal_all += vb_temporal
                        vb_test_spatial_all += vb_spatial
                        loss_test_all += loss.item() * event_time_non_mask.shape[0]

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
                        # 计算ensemble预测的均值作为最终预测，预期：ensemble的均值预测通常比单次采样更稳定和准确
                        ensemble_temporal_mean = torch.stack(sampled_temporal_all, dim=0).mean(dim=0)  # [bsz, 1]
                        ensemble_spatial_mean = torch.stack(sampled_spatial_all, dim=0).mean(dim=0)  # [bsz, dim]

                        # Temporal metrics
                        real_time_gt = (event_time_non_mask[:, 0, :].detach().cpu()) * (MAX[1] - MIN[1]) + MIN[1]
                        gen_temporal = (ensemble_temporal_mean.detach().cpu()) * (MAX[1] - MIN[1]) + MIN[1]
                        mae_temporal += torch.abs(real_time_gt - gen_temporal).sum().item()
                        rmse_temporal += ((real_time_gt - gen_temporal)**2).sum().item()

                        # Spatial metrics
                        real_loc_gt = event_loc_non_mask[:, 0, :].detach().cpu()
                        real_loc_gt = real_loc_gt * (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]])) + torch.tensor([MIN[2:]])
                        gen_spatial = ensemble_spatial_mean.detach().cpu()
                        gen_spatial = gen_spatial * (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]])) + torch.tensor([MIN[2:]])
                        mae_spatial += torch.sqrt(torch.sum((real_loc_gt - gen_spatial)**2, dim=-1)).sum().item()

                        total_num += gen_temporal.shape[0]

                        # UQ evaluation: calculate calibration scores
                        sampled_temporal_denorm = []
                        sampled_spatial_denorm = []

                        for temp_sample in sampled_temporal_all:
                            temp_denorm = temp_sample.detach().cpu() * (MAX[1] - MIN[1]) + MIN[1]
                            sampled_temporal_denorm.append(temp_denorm.unsqueeze(1))

                        for spat_sample in sampled_spatial_all:
                            spat_denorm = spat_sample.detach().cpu() * (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]])) + torch.tensor([MIN[2:]])
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

                    # Log test metrics
                    writer.add_scalar(tag='Evaluation/loss_test', scalar_value=loss_test_all / total_num, global_step=itr)
                    writer.add_scalar(tag='Evaluation/NLL_test', scalar_value=vb_test_all / total_num, global_step=itr)
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
                    print(f'MAE Temporal: {mae_temporal / total_num:.4f}')
                    print(f'RMSE Temporal: {np.sqrt(rmse_temporal / total_num):.4f}')
                    print(f'MAE Spatial: {mae_spatial / total_num:.4f}')

                    # Log UQ metrics to tensorboard
                    writer.add_scalar(tag='UQ/calibration_score_time_mean', scalar_value=cs_time_all.mean().item(), global_step=itr)
                    writer.add_scalar(tag='UQ/calibration_score_loc_mean', scalar_value=cs_loc_all.mean().item(), global_step=itr)
                    for i, level in enumerate(target_levels):
                        writer.add_scalar(tag=f'UQ/calibration_time_{level:.1f}', scalar_value=cs2_time_all[i].item(), global_step=itr)
                        writer.add_scalar(tag=f'UQ/calibration_loc_{level:.1f}', scalar_value=cs2_loc_all[i].item(), global_step=itr)

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

        # Training
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
