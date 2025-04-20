import torch
import torch.nn as nn
import numpy as np
from DSTPP import GaussianDiffusion_ST, Transformer, Transformer_ST, Model_all, ST_Diffusion
from DSTPP import RectifiedFlow, RF_Diffusion
from DSTPP.RF_Model_all import RF_Model_all
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
    parser = argparse.ArgumentParser(description='test')
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
                            'Citibike', 'Earthquake', 'HawkesGMM', 'Pinwheel', 'COVID19', 'Mobility', 'HawkesGMM_2d',
                            'Independent'
                        ],
                        help='')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--timesteps', type=int, default=500, help='')
    parser.add_argument('--samplingsteps', type=int, default=500, help='')
    parser.add_argument('--objective', type=str, default='pred_noise', help='')
    parser.add_argument('--cuda_id', type=str, default='0', help='')
    # NEW: 添加模型类型参数
    parser.add_argument('--model_type', type=str, default='ddpm', choices=['ddpm', 'rf'], help='使用的模型类型')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args


opt = get_args()
device = torch.device("cuda:{}".format(opt.cuda_id) if opt.cuda else "cpu")

if opt.dataset == 'HawkesGMM':
    opt.dim = 1

# os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_id)


def data_loader(writer):

    f = open('dataset/{}/data_train.pkl'.format(opt.dataset), 'rb')
    train_data = pickle.load(f)
    train_data = [[list(i) for i in u] for u in train_data]
    train_data = [[[i[0], i[0] - u[index - 1][0] if index > 0 else i[0]] + i[1:] for index, i in enumerate(u)]
                  for u in train_data]

    f = open('dataset/{}/data_val.pkl'.format(opt.dataset), 'rb')
    val_data = pickle.load(f)
    val_data = [[list(i) for i in u] for u in val_data]
    val_data = [[[i[0], i[0] - u[index - 1][0] if index > 0 else i[0]] + i[1:] for index, i in enumerate(u)]
                for u in val_data]

    f = open('dataset/{}/data_test.pkl'.format(opt.dataset), 'rb')
    test_data = pickle.load(f)
    test_data = [[list(i) for i in u] for u in test_data]
    test_data = [[[i[0], i[0] - u[index - 1][0] if index > 0 else i[0]] + i[1:] for index, i in enumerate(u)]
                 for u in test_data]

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
    # normalzie d_time and location (vector)
    train_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in train_data]
    test_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in test_data]
    val_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in val_data]

    trainloader = get_dataloader(train_data, opt.batch_size, D=opt.dim, shuffle=True)
    testloader = get_dataloader(test_data, len(test_data) if len(test_data) <= 1000 else 1000, D=opt.dim, shuffle=False)
    valloader = get_dataloader(test_data, len(val_data) if len(val_data) <= 1000 else 1000, D=opt.dim, shuffle=False)

    return trainloader, testloader, valloader, (Max, Min)


def Batch2toModel(batch, transformer):

    if opt.dim == 1:
        event_time_origin, event_time, lng = map(lambda x: x.to(device), batch)
        event_loc = lng.unsqueeze(dim=2)

    if opt.dim == 2:
        event_time_origin, event_time, lng, lat = map(lambda x: x.to(device), batch)

        event_loc = torch.cat((lng.unsqueeze(dim=2), lat.unsqueeze(dim=2)), dim=-1)  # (batch_size, seq_len, opt.dim)

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
    for index in range(mask.shape[0]):  # combine all batches in a long sequence, match condition and event pairwisely
        length = int(sum(mask[index]).item())
        if length > 1:
            enc_out_non_mask += [i.unsqueeze(dim=0) for i in enc_out[index][:length - 1]]  # condition, drop last one
            event_time_non_mask += [i.unsqueeze(dim=0) for i in event_time[index][1:length]]  # event time
            event_loc_non_mask += [i.unsqueeze(dim=0) for i in event_loc[index][1:length]]  # event loc

    enc_out_non_mask = torch.cat(enc_out_non_mask, dim=0)  # (batch_all_condition_events, 3 * d_model)
    event_time_non_mask = torch.cat(event_time_non_mask, dim=0)
    event_loc_non_mask = torch.cat(event_loc_non_mask, dim=0)

    event_time_non_mask = event_time_non_mask.reshape(-1, 1, 1)  # (batch_all_condition_events, 1, 1)
    event_loc_non_mask = event_loc_non_mask.reshape(-1, 1, opt.dim)  # (batch_all_condition_events, 1, opt.dim)
    # (batch_all_condition_events, 1, 3 * d_model)
    enc_out_non_mask = enc_out_non_mask.reshape(event_time_non_mask.shape[0], 1, -1)

    return event_time_non_mask, event_loc_non_mask, enc_out_non_mask


def LR_warmup(lr, epoch_num, epoch_current):
    return lr * (epoch_current + 1) / epoch_num


if __name__ == "__main__":

    setup_init(opt)
    setproctitle.setproctitle("Model-Training")

    print('dataset:{}'.format(opt.dataset))

    # Specify a directory for logging data
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d-%Hh")
    logdir = "./logs/{}_timesteps_{}_{}".format(opt.dataset, opt.timesteps, date)
    # logdir = "./logs/{}_timesteps_{}".format(opt.dataset, opt.timesteps)
    model_path = './ModelSave/dataset_{}_timesteps_{}_{}/'.format(opt.dataset, opt.timesteps, date)

    if not os.path.exists('./ModelSave'):
        os.mkdir('./ModelSave')

    if 'train' in opt.mode and not os.path.exists(model_path):
        os.mkdir(model_path)

    writer = SummaryWriter(log_dir=logdir, flush_secs=5)

    model = ST_Diffusion(n_steps=opt.timesteps, dim=1 + opt.dim, condition=True, cond_dim=64).to(device)

    diffusion = GaussianDiffusion_ST(model,
                                     loss_type=opt.loss_type,
                                     seq_length=1 + opt.dim,
                                     timesteps=opt.timesteps,
                                     sampling_timesteps=opt.samplingsteps,
                                     objective=opt.objective,
                                     beta_schedule=opt.beta_schedule).to(device)

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

    Model = Model_all(transformer, diffusion)

    trainloader, testloader, valloader, (MAX, MIN) = data_loader(writer)

    warmup_steps = 5

    # training
    optimizer = AdamW(Model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    step, early_stop = 0, 0
    min_loss_test = 1e20
    for itr in range(opt.total_epochs):

        print('epoch:{}'.format(itr))

        if (itr % 10 == 0) or (iter == opt.total_epochs - 1):
            print('Evaluate!')
            with torch.no_grad():

                Model.eval()

                # validation set
                loss_test_all, vb_test_all, vb_test_temporal_all, vb_test_spatial_all = 0.0, 0.0, 0.0, 0.0
                mae_temporal, rmse_temporal, mae_spatial, mae_lng, mae_lat, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for batch in valloader:
                    event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)

                    sampled_seq = Model.diffusion.sample(batch_size=event_time_non_mask.shape[0], cond=enc_out_non_mask)

                    # sampled_seq_temporal_all, sampled_seq_spatial_all = [], []
                    # for _ in range(100):  # 准备基于多次采样计算统计指标（如平均RMSE、置信区间等）
                    #     sampled_seq = Model.diffusion.sample(batch_size=event_time_non_mask.shape[0],
                    #                                          cond=enc_out_non_mask)
                    #     sampled_seq_temporal_all.append(  # FIX: origin denormalize is wrong
                    #         (sampled_seq[:, 0, :1].detach().cpu()) * (MAX[1] - MIN[1]) + MIN[1])
                    #     sampled_seq_spatial_all.append((sampled_seq[:, 0, -opt.dim:].detach().cpu()) *
                    #                                    (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]])) +
                    #                                    torch.tensor([MIN[2:]]).unsqueeze(dim=1))

                    loss = Model.diffusion(torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1),
                                           enc_out_non_mask)
                    # Variational lower bound to approximate the NLL of the data
                    vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(
                        torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)

                    vb_test_all += vb
                    vb_test_temporal_all += vb_temporal
                    vb_test_spatial_all += vb_spatial

                    loss_test_all += loss.item() * event_time_non_mask.shape[0]

                    real = (event_time_non_mask[:, 0, :].detach().cpu()) * (MAX[1] - MIN[1]) + MIN[1]  # (N, 1)
                    gen = (sampled_seq[:, 0, :1].detach().cpu()) * (MAX[1] - MIN[1]) + MIN[1]
                    assert real.shape == gen.shape
                    # assert real.shape == sampled_seq_temporal_all.shape
                    mae_temporal += torch.abs(real - gen).sum().item()
                    rmse_temporal += ((real - gen)**2).sum().item()
                    # rmse_temporal_mean += ((real-sampled_seq_temporal_all)**2).sum().item()

                    real = event_loc_non_mask[:, 0, :].detach().cpu()  # (N, 2)
                    assert real.shape[1:] == torch.tensor(MIN[2:]).shape
                    real = real * (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]])) + torch.tensor([MIN[2:]])
                    gen = sampled_seq[:, 0, -opt.dim:].detach().cpu()
                    gen = gen * (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]])) + torch.tensor([MIN[2:]])
                    assert real.shape == gen.shape
                    # assert real.shape == sampled_seq_spatial_all.shape  # TODO: sampled_seq_spatial_all is a list, check the logic?
                    mae_spatial += torch.sqrt(torch.sum((real - gen)**2, dim=-1)).sum().item()
                    # mae_spatial_mean += torch.sqrt(torch.sum((real-sampled_seq_spatial_all)**2,dim=-1)).sum().item()

                    total_num += gen.shape[0]

                    assert gen.shape[0] == event_time_non_mask.shape[0]

                if loss_test_all > min_loss_test:
                    early_stop += 1
                    if early_stop >= 100:
                        break

                else:
                    early_stop = 0

                torch.save(Model.state_dict(), model_path + 'model_{}.pkl'.format(itr))

                min_loss_test = min(min_loss_test, loss_test_all)

                writer.add_scalar(tag='Evaluation/loss_val', scalar_value=loss_test_all / total_num, global_step=itr)

                writer.add_scalar(tag='Evaluation/NLL_val', scalar_value=vb_test_all / total_num, global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_temporal_val',
                                  scalar_value=vb_test_temporal_all / total_num,
                                  global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_spatial_val',
                                  scalar_value=vb_test_spatial_all / total_num,
                                  global_step=itr)

                writer.add_scalar(tag='Evaluation/mae_temporal_val',
                                  scalar_value=mae_temporal / total_num,
                                  global_step=itr)
                writer.add_scalar(tag='Evaluation/rmse_temporal_val',
                                  scalar_value=np.sqrt(rmse_temporal / total_num),
                                  global_step=itr)
                # writer.add_scalar(tag='Evaluation/rmse_temporal_mean_val',scalar_value=np.sqrt(rmse_temporal_mean/total_num),global_step=itr)

                writer.add_scalar(tag='Evaluation/distance_spatial_val',
                                  scalar_value=mae_spatial / total_num,
                                  global_step=itr)
                # writer.add_scalar(tag='Evaluation/distance_spatial_mean_val',scalar_value=mae_spatial_mean/total_num,global_step=itr)

                ### TEST
                loss_test_all, vb_test_all, vb_test_temporal_all, vb_test_spatial_all = 0.0, 0.0, 0.0, 0.0
                mae_temporal, rmse_temporal, mae_spatial, mae_lng, mae_lat, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                print('TEST!')
                for batch in testloader:
                    event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)

                    sampled_seq = Model.diffusion.sample(batch_size=event_time_non_mask.shape[0], cond=enc_out_non_mask)

                    loss = Model.diffusion(torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1),
                                           enc_out_non_mask)

                    vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(
                        torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)

                    vb_test_all += vb
                    vb_test_temporal_all += vb_temporal
                    vb_test_spatial_all += vb_spatial

                    loss_test_all += loss.item() * event_time_non_mask.shape[0]

                    # add codes for real和gen来计算MAE for spatial & RMSE for temporal
                    real = (event_time_non_mask[:, 0, :].detach().cpu()) * (MAX[1] - MIN[1]) + MIN[1]  # (N, 1)
                    gen = (sampled_seq[:, 0, :1].detach().cpu()) * (MAX[1] - MIN[1]) + MIN[1]
                    assert real.shape == gen.shape
                    # assert real.shape == sampled_seq_temporal_all.shape
                    mae_temporal += torch.abs(real - gen).sum().item()
                    rmse_temporal += ((real - gen)**2).sum().item()
                    # rmse_temporal_mean += ((real-sampled_seq_temporal_all)**2).sum().item()

                    real = event_loc_non_mask[:, 0, :].detach().cpu()  # (N, 2)
                    assert real.shape[1:] == torch.tensor(MIN[2:]).shape
                    real = real * (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]])) + torch.tensor([MIN[2:]])
                    gen = sampled_seq[:, 0, -opt.dim:].detach().cpu()
                    gen = gen * (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]])) + torch.tensor([MIN[2:]])
                    assert real.shape == gen.shape
                    # assert real.shape == sampled_seq_spatial_all.shape  # TODO: sampled_seq_spatial_all is a list, check the logic?
                    mae_spatial += torch.sqrt(torch.sum((real - gen)**2, dim=-1)).sum().item()
                    # mae_spatial_mean += torch.sqrt(torch.sum((real-sampled_seq_spatial_all)**2,dim=-1)).sum().item()

                    total_num += gen.shape[0]

                    assert gen.shape[0] == event_time_non_mask.shape[0]

                writer.add_scalar(tag='Evaluation/loss_test', scalar_value=loss_test_all / total_num, global_step=itr)

                writer.add_scalar(tag='Evaluation/NLL_test', scalar_value=vb_test_all / total_num, global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_temporal_test',
                                  scalar_value=vb_test_temporal_all / total_num,
                                  global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_spatial_test',
                                  scalar_value=vb_test_spatial_all / total_num,
                                  global_step=itr)

                writer.add_scalar(tag='Evaluation/mae_temporal_test',
                                  scalar_value=mae_temporal / total_num,
                                  global_step=itr)
                writer.add_scalar(tag='Evaluation/rmse_temporal_test',
                                  scalar_value=np.sqrt(rmse_temporal / total_num),
                                  global_step=itr)
                # writer.add_scalar(tag='Evaluation/rmse_temporal_mean_test',scalar_value=np.sqrt(rmse_temporal_mean/total_num),global_step=itr)

                writer.add_scalar(tag='Evaluation/distance_spatial_test',
                                  scalar_value=mae_spatial / total_num,
                                  global_step=itr)
                # writer.add_scalar(tag='Evaluation/distance_spatial_mean_test',scalar_value=mae_spatial_mean/total_num,global_step=itr)

        if itr < warmup_steps:
            for param_group in optimizer.param_groups:
                lr = LR_warmup(1e-3, warmup_steps, itr)
                param_group["lr"] = lr

        else:
            for param_group in optimizer.param_groups:
                lr = 1e-3 - (1e-3 - 5e-5) * (itr - warmup_steps) / opt.total_epochs
                param_group["lr"] = lr

        writer.add_scalar(tag='Statistics/lr', scalar_value=lr, global_step=itr)
        ### TRAIN
        Model.train()

        loss_all, vb_all, vb_temporal_all, vb_spatial_all, total_num = 0.0, 0.0, 0.0, 0.0, 0.0
        for batch in trainloader:

            event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)
            loss = Model.diffusion(torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)

            optimizer.zero_grad()
            loss.backward()
            loss_all += loss.item() * event_time_non_mask.shape[0]
            # Variational lower bound to approximate the NLL of the data
            vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(
                torch.cat((event_time_non_mask, event_loc_non_mask), dim=-1), enc_out_non_mask)

            vb_all += vb
            vb_temporal_all += vb_temporal
            vb_spatial_all += vb_spatial

            writer.add_scalar(tag='Training/loss_step', scalar_value=loss.item(), global_step=step)

            torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.)  # 如果总范数超过设定阈值（1），则按比例缩小所有梯度，使总范数等于阈值
            optimizer.step()

            step += 1

            total_num += event_time_non_mask.shape[0]  # num of events (to predict) per batch

        with torch.cuda.device("cuda:{}".format(opt.cuda_id)):
            torch.cuda.empty_cache()

        writer.add_scalar(tag='Training/loss_epoch', scalar_value=loss_all / total_num, global_step=itr)

        writer.add_scalar(tag='Training/NLL_epoch', scalar_value=vb_all / total_num, global_step=itr)
        writer.add_scalar(tag='Training/NLL_temporal_epoch', scalar_value=vb_temporal_all / total_num, global_step=itr)
        writer.add_scalar(tag='Training/NLL_spatial_epoch', scalar_value=vb_spatial_all / total_num, global_step=itr)
