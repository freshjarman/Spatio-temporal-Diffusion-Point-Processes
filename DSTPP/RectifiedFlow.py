import math
from functools import partial
from sympy import rf
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
import numpy as np
from torchdiffeq import odeint
from zmq import device


# 还是说归一化到[-1, 1]是为了保证数据和噪声的均值和尺度更加匹配？从而在线性插值过程中均值始终为0，从而降低v的学习难度？
# 这点选择了和DSTPP的实现（class GaussianDiffusion_ST）保持一致
def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d


# --- 1. Hutchinson Trace Estimator for Divergence ---
def divergence_approx(f, x, t, e=None, self_cond=None, cond=None):
    # 计算 v(x, t) 关于 x 的散度的随机近似 $\nabla \cdot v = \text{Tr}(\nabla v) = \mathbb{E}_{\epsilon}[\epsilon^T (\nabla v) \epsilon] $
    """

    参数:
        f: 可调用函数，表示速度场 v(x, t, self_cond, cond)
        x: 评估点
        t: 时间点
        e: 随机向量，如果为None则随机生成
        self_cond: 自条件项，默认None
        cond: 条件信息，默认None
    返回:
        div(f) 在点 x 处的随机近似
    """
    if e is None:
        e = torch.randn_like(x)

    x_requires_grad = x.requires_grad
    with torch.enable_grad():
        x.requires_grad_(True)
        fx = f(x, t, self_cond, cond)
        # 计算 e^T * [ v(x, t) 关于 x 的Jacobian矩阵]
        e_J = torch.autograd.grad(fx, x, e, create_graph=True)[0]
        e_J_e = (e_J * e).sum(dim=tuple(range(1, x.dim())))  # Sum over all non-batch dims

    x.requires_grad_(x_requires_grad)
    return e_J_e  # 返回每个 batch 元素(v(x,t))的散度 的近似值


# --- 2. 定义耦合 ODE 的动力学 ---
class ODEFunc(nn.Module):
    """
    Defines the dynamics for the coupled ODE system:
    dx/dt = -v_theta(x, t)      (Forward ODE velocity field f = -v_theta)
    da/dt = +div_x v_theta(x, t) (Because d(log p)/dt = -div(f) = -div(-v_theta) = +div(v_theta))
    """

    def __init__(self, model_v, cond=None):
        super().__init__()
        self.model_v = model_v  # model_v learns v_theta approx x0 - x1
        self.cond = cond  # 存储条件信息

    def forward(self, t, state):
        # state 是一个元组 (x, logp_integral_accumulator)
        x, _ = state
        batch_size = x.shape[0]

        if t.numel() == 1:
            t_batch = t.expand(batch_size)
        else:
            t_batch = t

        # Calculate v_theta (learned velocity, approx x0 - x1), None 是 self_cond 参数
        v_theta = self.model_v(x, t_batch, None, self.cond)

        # Calculate divergence of v_theta using Hutchinson estimator
        e = torch.randn_like(x)
        # divergence_approx returns approximation of div(v_theta)
        div_v_theta = divergence_approx(self.model_v, x, t_batch, e, None, self.cond)

        # Dynamics:
        dxdt = -v_theta  # Forward ODE velocity field f = -v_theta
        dadt = div_v_theta  # Corrected: da/dt = +div(v_theta)

        return (dxdt, dadt)  # 返回耦合 ODE 系统的右式（即如上的两个ODE的右式）


class RectifiedFlow(nn.Module):
    """
    基于Rectified Flow的时空点过程模型
    学习直接的ODE轨迹而不是逐步去噪过程
    """

    def __init__(
            self,
            model,
            *,
            seq_length,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l2',
            use_dynamic_loss_scaling=True,  # 是否使用动态损失缩放
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.seq_length = seq_length  # loc's dim + 1
        self.num_timesteps = timesteps
        self.loss_type = loss_type
        self.use_dynamic_loss_scaling = use_dynamic_loss_scaling

        # 采样相关参数
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps

        # 注册时间步长缓冲区 - 从0到1均匀分布
        self.register_buffer('timesteps', torch.linspace(0, 1, timesteps))

        # 计算损失权重 - 基于Rectified Flow论文的建议
        if use_dynamic_loss_scaling:
            # 动态权重随时间变化
            weight = torch.ones(timesteps)
            for i in range(timesteps):
                t = i / (timesteps - 1)
                # 在t接近0和1时增加权重
                weight[i] = 1.0 / (0.5 + (t - 0.5)**2)
            self.register_buffer('loss_weight', weight / weight.mean())
        else:
            self.register_buffer('loss_weight', torch.ones(timesteps))

    def straight_path_interpolation(self, x_start, t, noise=None):
        """
        计算直线路径插值
        x_t = (1-t) * x_start + t * 噪声
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        x_t = (1 - t.view(-1, 1, 1)) * x_start + t.view(-1, 1, 1) * noise
        return x_t, noise

    def velocity_vector(self, x_start, t):
        """
        计算速度向量: v(x_t, t) = x_0 - x_1
        对于直线路径：x_1为噪声，x_0为原始数据
        """
        noise = torch.randn_like(x_start)
        x_t, _ = self.straight_path_interpolation(x_start, t, noise)  # noise要保证和x_start唯一对应

        # 速度向量指向x_start (从噪声指向数据)
        velocity = x_start - noise

        return x_t, velocity

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t_indices, cond=None):
        """
        计算损失：预测的速度向量与真实速度向量之间的差异
        """
        batch_size = x_start.shape[0]

        # 获取实际时间步长
        t = self.timesteps[t_indices]

        # 计算当前点和真实速度向量
        x_t, true_velocity = self.velocity_vector(x_start, t)  # [bsz, 1, dim] (1 + opt.dim)

        # 模型预测速度向量
        pred_velocity = self.model(x_t, t, None, cond)

        # 计算损失
        loss = self.loss_fn(pred_velocity, true_velocity, reduction='none')  # [bsz, 1, dim] (1 + opt.dim)

        # 区分时间和空间维度的损失
        loss_temporal = loss[:, :, :1].mean()
        loss_spatial = loss[:, :, 1:].mean()

        # 应用损失权重
        if self.use_dynamic_loss_scaling:
            loss_weight = self.loss_weight[t_indices].view(-1, 1, 1)
            loss = loss * loss_weight

        loss = loss.mean()

        return loss, loss_temporal, loss_spatial

    @torch.no_grad()
    def sample(self, batch_size=16, cond=None, steps=None, euler_only=False):
        """
        从噪声采样生成数据
        使用预测的速度场进行指导
        
        参数:
            batch_size: 生成样本数量
            cond: 条件信息
            steps: 采样步数，默认使用初始化时指定的步数
            euler_only: 是否只使用欧拉法 (一阶)，默认False使用Heun法 (二阶)
        """
        device = next(self.parameters()).device
        steps = default(steps, self.sampling_timesteps)

        # 从标准高斯分布开始
        shape = (batch_size, self.channels, self.seq_length)
        x = torch.randn(shape, device=device)

        # 时间步长
        step_size = 1.0 / steps

        # 积分步长 - 欧拉法或Heun法，比简单的欧拉法更精确（2-order），带有预测-校正步骤
        solver_name = "Euler" if euler_only else "Heun"
        for i in tqdm(range(steps), desc=f'RF sampling with ({solver_name}) solver'):
            # 当前时间：从1到0
            t_now = 1.0 - i * step_size
            t_next = max(1.0 - (i + 1) * step_size, 0.0)

            t_tensor = torch.full((batch_size, ), t_now, device=device)

            # 预测当前速度
            v_now = self.model(x, t_tensor, None, cond)

            # 预测步骤(欧拉法)
            x_pred = x + step_size * v_now

            if not euler_only and t_next > 0:  # 只有在非欧拉模式且非最后一步时执行校正步骤
                # 在预测位置上评估速度
                t_tensor_next = torch.full((batch_size, ), t_next, device=device)
                v_next = self.model(x_pred, t_tensor_next, None, cond)

                # 校正步骤(Heun方法)
                x = x + 0.5 * step_size * (v_now + v_next)
            else:
                x = x_pred  # 欧拉法或最后一步

        # 归一化到[0,1]
        x = unnormalize_to_zero_to_one(x)
        return x

    # NOT USED!
    def NLL_cal(self, x_start, cond):
        """
        计算整个轨迹的负对数似然的近似指标，他不是NLL，但它也是指标越小代表模型越好
        RF不是显式概率模型，这是启发式方法，即他不是严格的VLB，数值上不能近似NLL
        """
        # 相较于DiffusionModel.py中的NLL_cal(self, img, cond, noise=None)，这里只是v上的l2损失，数值上应该不完全等于VLB？
        x_start = normalize_to_neg_one_to_one(x_start)
        batch_size = x_start.shape[0]
        device = x_start.device

        # 选择评估点
        sample_count = min(100, self.num_timesteps)  # 用100个点来近似
        t_indices = torch.linspace(0, self.num_timesteps - 1, sample_count, dtype=torch.long, device=device)

        total_loss, temporal_loss, spatial_loss = 0.0, 0.0, 0.0

        for idx in t_indices:
            # 为每个batch样本获取相同的时间点
            t_batch = torch.full((batch_size, ), idx, device=device, dtype=torch.long)

            with torch.no_grad():
                loss, loss_temporal, loss_spatial = self.p_losses(x_start, t_batch, cond)

            total_loss += loss.item()
            temporal_loss += loss_temporal.item()
            spatial_loss += loss_spatial.item()

        # 计算平均值
        total_loss /= len(t_indices)
        temporal_loss /= len(t_indices)
        spatial_loss /= len(t_indices)

        return total_loss, temporal_loss, spatial_loss

    @torch.no_grad()
    def calculate_neg_log_likelihood(self, x_start, cond=None, rtol=1e-5, atol=1e-5, method='dopri5'):
        """
        Calculates the exact log-likelihood log p0(x0) using the change of variables formula.
        Integrates the forward ODE dx/dt = -v_theta(x, t) from t=0 to t=1.
        并分别估计时间和空间维度的对数似然贡献。
        返回负对数似然值，与 DDPM 接口兼容。
        """
        self.model.eval()

        # ODEFunc now correctly defines the forward dynamics dx/dt = -v_theta
        # and the log-density accumulator da/dt = +div(v_theta)
        # logp1(x1) = logp0(x0) + integral[0,1] da/dt dt
        ode_func = ODEFunc(self.model, cond)

        x0_norm = normalize_to_neg_one_to_one(x_start)  # Start at real data x0 (t=0)
        a0 = torch.zeros(x0_norm.shape[0], device=x0_norm.device)  # Initial logp accumulator
        initial_state = (x0_norm, a0)

        t_span = torch.tensor([0.0, 1.0], device=x0_norm.device)  # Integrate forward t=0 to t=1

        final_state_tuple = odeint(ode_func, initial_state, t_span, rtol=rtol, atol=atol, method=method)

        x1 = final_state_tuple[0][-1]  # [bsz, 1, dim], State at t=1 (should approx noise)
        a1 = final_state_tuple[1][-1]  # [bsz], Accumulated logp change a1 = integral[0,1] da/dt dt

        # Calculate log prior probability of the state at t=1
        D = x1.shape[1:].numel()
        log_prior_p1 = -0.5 * (D * math.log(2 * math.pi) + torch.sum(x1**2, dim=tuple(range(1, x1.dim()))))
        # Change of variables: log p0(x0) = log p1(x1) - integral[0,1] div(dx/dt) dt
        # From ODEFunc: da/dt = +div(v_theta)
        # Also, d(log p)/dt = -div(dx/dt) = -div(-v_theta) = +div(v_theta) = da/dt
        # So, a1 = integral[0,1] da/dt dt = integral[0,1] d(log p)/dt dt = log p1(x1) - log p0(x0)
        # Therefore: log p0(x0) = log p1(x1) - a1
        log_likelihood = log_prior_p1 - a1  # [bsz]

        # 计算单独的时间和空间对数似然
        # 1. 单独评估时间和空间维度的散度贡献
        # 创建一个仅在时间维度上有梯度的掩码函数
        def temporal_model(x, t, self_cond, cond):
            # 获取完整预测
            full_pred = self.model(x, t, self_cond, cond)
            # 只保留时间维度的预测 (第一个维度)
            time_pred = torch.zeros_like(full_pred)
            time_pred[:, :, 0:1] = full_pred[:, :, 0:1]
            return time_pred

        # 创建一个仅在空间维度上有梯度的掩码函数
        def spatial_model(x, t, self_cond, cond):
            # 获取完整预测
            full_pred = self.model(x, t, self_cond, cond)
            # 只保留空间维度的预测 (除第一维外)
            space_pred = torch.zeros_like(full_pred)
            space_pred[:, :, 1:] = full_pred[:, :, 1:]
            return space_pred

        # 计算最终状态x1处的散度
        t_final = torch.ones(x1.shape[0], device=x1.device)

        # 估计时间和空间维度的散度比例
        div_temporal = divergence_approx(temporal_model, x1, t_final, None, None, cond)
        div_spatial = divergence_approx(spatial_model, x1, t_final, None, None, cond)

        # 计算散度的相对比例
        div_total = div_temporal.abs() + div_spatial.abs() + 1e-8  # 防止除零
        temporal_ratio = div_temporal.abs() / div_total
        spatial_ratio = div_spatial.abs() / div_total

        # 按散度比例分配 a1
        a1_temporal = a1 * temporal_ratio
        a1_spatial = a1 * spatial_ratio

        # 计算先验概率的时间和空间分量
        x1_temporal = x1[:, :, 0:1]
        D_temporal = x1_temporal.shape[1:].numel()
        log_prior_p1_temporal = -0.5 * (D_temporal * math.log(2 * math.pi) +
                                        torch.sum(x1_temporal**2, dim=tuple(range(1, x1_temporal.dim()))))

        x1_spatial = x1[:, :, 1:]
        D_spatial = x1_spatial.shape[1:].numel()
        log_prior_p1_spatial = -0.5 * (D_spatial * math.log(2 * math.pi) +
                                       torch.sum(x1_spatial**2, dim=tuple(range(1, x1_spatial.dim()))))

        # 计算子空间的对数似然
        log_likelihood_temporal = log_prior_p1_temporal - a1_temporal
        log_likelihood_spatial = log_prior_p1_spatial - a1_spatial

        # 转换为负对数似然 (NLL)，并使用 .sum().item() 匹配 DDPM 的接口
        nll = -log_likelihood.sum().item()
        nll_temporal = -log_likelihood_temporal.sum().item()
        nll_spatial = -log_likelihood_spatial.sum().item()

        return nll, nll_temporal, nll_spatial

    def forward(self, img, cond):
        """
        模型前向传播：随机采样时间点并计算损失
        """
        b, c, n, device = *img.shape, img.device
        assert n == self.seq_length, f'输入序列长度必须为 {self.seq_length}'

        # 随机采样时间索引
        t_indices = torch.randint(0, self.num_timesteps, (b, ), device=device)  # [bsz]

        # 归一化输入
        img = normalize_to_neg_one_to_one(img)

        # 计算损失
        loss, _, _ = self.p_losses(img, t_indices, cond)
        return loss


if __name__ == '__main__':
    # TODO: 测试代码
    # 假设你已经定义了一个模型实例 model 和数据 img 和 cond
    from RF_Diffusion import RF_Diffusion
    from RF_Model_all import RF_Model_all
    from Models import Transformer_ST
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create (& load) model
    transformer = Transformer_ST(d_model=64,
                                 d_rnn=256,
                                 d_inner=128,
                                 n_layers=4,
                                 n_head=4,
                                 d_k=16,
                                 d_v=16,
                                 dropout=0.1,
                                 device=device,
                                 loc_dim=2,
                                 CosSin=True).to(device)
    # rf_diffsuion = RF_Diffusion(n_steps=opt.timesteps, dim=1 + 2, condition=True, cond_dim=64).to(device)
    # rf = RectifiedFlow(rf_diffsuion,
    #                    loss_type=opt.loss_type,
    #                    seq_length=1 + 2,
    #                    timesteps=opt.timesteps,
    #                    sampling_timesteps=opt.samplingsteps).to(device)
    # model = RF_Model_all(transformer, rf)

    # model = None

    # mock data 【！！！shape未必对】
    img = torch.randn(16, 3, 10)  # 示例数据
    cond = torch.randn(16, 3, 10)  # 示例条件；应该包括 t_i + s_i + h_i-1 参考 eq(10)

    rf_instance = RectifiedFlow(model, seq_length=10)  # seq_length对应数据集的：loc的维度+1(i.e. time)
    loss = rf_instance(img, cond)
    print(f'Loss: {loss.item()}')

    # ！测试 NLL 计算
    x_batch = torch.randn(16, 3, 10)  # 示例数据
    nll, nll_temporal, nll_spatial = rf_instance.calculate_neg_log_likelihood(x_batch)
    print(f'NLL: {nll}, Temporal NLL: {nll_temporal}, Spatial NLL: {nll_spatial}')

    # 测试采样
    # sampled_data = rf_instance.sample(batch_size=16)
    # print(f'Sampled Data: {sampled_data.shape}')  # 输出采样数据的形状
