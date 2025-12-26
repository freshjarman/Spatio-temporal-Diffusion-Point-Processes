"""
RectifiedFlow.py

注意：在该项目的RF实现中，我基于了DSTPP的代码；所以t=0表示数据分布，t=1表示噪声分布，与标准的RF实现相反，因此差值/loss/ll计算时需要注意时间的方向。


This module implements the Rectified Flow framework for the Spatio-temporal Diffusion Point Processes model.
Rectified Flow is a generative model that learns a transport map between two distributions (e.g., Gaussian noise
and data distribution) by solving an Ordinary Differential Equation (ODE). It serves as an alternative to
standard diffusion models, often providing straighter paths and potentially faster sampling.

Key Components:
- `RectifiedFlow`: The main class managing the training and sampling processes.
    - **Training**: Implements the loss function (`p_losses`) which minimizes the mean squared error between
      the predicted velocity and the target velocity (x_data - x_noise).
    - **Sampling**: Provides methods to generate samples by solving the ODE from noise to data (`sample`, `sample_ode`).
    - **Likelihood Estimation**: Includes methods to estimate the Negative Log-Likelihood (NLL) using the
      instantaneous change of variables formula and Hutchinson's trace estimator (`NLL_cal`).

- `ODEFunc`: A helper `nn.Module` defining the system of ODEs for likelihood estimation.
    - It models the joint evolution of the sample `x` and the log-probability density, allowing for
      continuous normalizing flow-style likelihood computation.

- Helper Functions:
    - `divergence_approx`: Implements the Hutchinson Trace Estimator to approximate the divergence of the
      velocity field, which is required for tracking the change in log-density during ODE solving.
    - Normalization utilities (`normalize_to_neg_one_to_one`, `unnormalize_to_zero_to_one`).

This implementation supports conditional generation (using `cond`) and handles spatio-temporal data structures.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


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
def divergence_approx(f, x, t, mask, e=None, self_cond=None, cond=None):
    # 计算 v(x, t) 关于 x 的散度的随机近似 $\nabla \cdot v = \text{Tr}(\nabla v) = \mathbb{E}_{\epsilon}[\epsilon^T (\nabla v) \epsilon] $
    """

    参数:
        f: 可调用函数，表示速度场 v(x, t, self_cond, cond)
        x: 评估点
        t: 时间点
        mask: 用于选择有效的样本, 形同 x 的 bool (0、1) 张量，只在被激活维度保留随机噪声
        e: 随机向量，如果为None则随机生成
        self_cond: 自条件项，默认None
        cond: 条件信息，默认None
    返回:
        div(f) 在点 x 处的随机近似
    """
    if e is None:
        e = torch.randn_like(x)
    e = e * mask  # 仅在被激活的维度上保留随机噪声
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

    def __init__(self, model_v, seq_length, cond=None):
        super().__init__()
        self.model_v = model_v  # model_v learns v_theta approx x0 - x1
        self.cond = cond  # 存储条件信息
        self.seq_length = seq_length  # loc's dim + 1

        # construct 2 mask
        mask_time = torch.zeros(1, 1, seq_length)
        mask_space = torch.ones(1, 1, seq_length)
        mask_time[:, :, 0] = 1.  # t
        mask_space[:, :, 0] = 0.  # loc
        self.register_buffer('mask_t', mask_time)
        self.register_buffer('mask_s', mask_space)

    def forward(self, t, state):
        # state 是一个元组 (x, logp_integral_accumulator)
        x, a_all, a_t, a_s = state
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
        div_v_theta = divergence_approx(self.model_v, x, t_batch, torch.ones_like(x).to(e), e, None, self.cond)
        div_t = divergence_approx(self.model_v, x, t_batch, self.mask_t.expand_as(x).to(e), e, None, self.cond)
        div_s = divergence_approx(self.model_v, x, t_batch, self.mask_s.expand_as(x).to(e), e, None, self.cond)

        # Dynamics:
        dxdt = -v_theta  # Forward ODE velocity field f = -v_theta
        dadt = div_v_theta  # Corrected: da/dt = +div(v_theta)
        da_t = div_t
        da_s = div_s

        return (dxdt, dadt, da_t, da_s)  # 返回耦合 ODE 系统的右式（即如上的两个ODE的右式）


class RectifiedFlow(nn.Module):
    """
    基于Rectified Flow的时空点过程模型
    学习直接的ODE轨迹而不是逐步去噪过程
    
    HAP (History Adaptive Prior) Design:
        - PriorNet outputs μ(H) and σ(H) as informative prior
        - μ is supervised by ground truth via NLL loss
        - σ represents calibrated prediction uncertainty
        - No KL regularization (we WANT informative prior, not standard Gaussian)
    
    Args:
        model: 速度场预测网络 (RF_Diffusion)
        seq_length: 序列长度 (1 + loc_dim)
        timesteps: 训练时间步数
        sampling_timesteps: 采样步数
        loss_type: 损失类型 ('l1' or 'l2')
        use_dynamic_loss_scaling: 是否使用时间相关的损失权重
        prior_net: 可选的 PriorNet 实例，用于历史自适应先验 (HAP)
        prior_loss_weight: Prior NLL 损失权重 (仅当 prior_net 不为 None 时生效)
    """

    def __init__(
            self,
            model,
            *,
            seq_length,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l2',
            use_dynamic_loss_scaling=True,
            prior_net=None,
            prior_loss_weight=0.1,  # HAP: Prior NLL loss weight (renamed from kl_weight)
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.seq_length = seq_length  # loc's dim + 1
        self.num_timesteps = timesteps
        self.loss_type = loss_type
        self.use_dynamic_loss_scaling = use_dynamic_loss_scaling

        # PriorNet for History-Adaptive Prior (HAP)
        self.prior_net = prior_net
        self.prior_loss_weight = prior_loss_weight  # HAP: renamed from kl_weight

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

    def _extract_history_encoding(self, cond):
        """
        从条件信息中提取历史编码用于 PriorNet。
        
        cond 格式: [batch, 1, 3*d_model] = [enc_temporal, enc_spatial, enc_joint]
        我们使用 enc_joint (最后 d_model 维) 作为整体历史表示。
        
        Args:
            cond: [batch, 1, 3*d_model] 条件信息
        
        Returns:
            history_enc: [batch, d_model] 用于 PriorNet 的历史编码
        """
        if cond is None:
            return None
        d_model = cond.shape[-1] // 3
        # 取 enc_joint 部分 (最后 d_model 维)
        history_enc = cond[:, 0, 2 * d_model:]  # [batch, d_model]
        return history_enc

    def _sample_prior(self, shape, device, cond=None):
        """
        从先验分布采样噪声。
        
        如果有 PriorNet，从 N(μ(H), σ²(H)) 采样；否则从 N(0, I) 采样。
        
        Args:
            shape: 输出形状 (batch, channels, seq_length)
            device: 设备
            cond: 条件信息
        
        Returns:
            noise: [batch, channels, seq_length] 采样的噪声
        """
        if self.prior_net is not None and cond is not None:
            history_enc = self._extract_history_encoding(cond)
            # PriorNet 输出: [batch, seq_length]
            noise_flat = self.prior_net.sample(history_enc)  # [batch, seq_length]
            # 调整形状: [batch, 1, seq_length]
            noise = noise_flat.unsqueeze(1)
        else:
            noise = torch.randn(shape, device=device)
        return noise

    def _compute_prior_log_prob(self, z, cond=None):
        """
        计算先验分布下的对数概率，分解为时间和空间分量。
        
        Args:
            z: [batch, 1, seq_length] 样本 (在归一化空间 [-1, 1])
            cond: 条件信息
        
        Returns:
            log_prob: [batch] 总对数概率
            log_prob_t: [batch] 时间分量
            log_prob_s: [batch] 空间分量
        """
        D = z.shape[-1]
        z_flat = z.squeeze(1)  # [batch, seq_length]

        if self.prior_net is not None and cond is not None:
            history_enc = self._extract_history_encoding(cond)
            log_prob, log_prob_t, log_prob_s = self.prior_net.log_prob_decomposed(z_flat, history_enc)
        else:
            # 标准高斯先验
            log_prob = -0.5 * (D * math.log(2 * math.pi) + torch.sum(z_flat**2, dim=-1))
            log_prob_t = -0.5 * (math.log(2 * math.pi) + z_flat[:, 0]**2)
            log_prob_s = -0.5 * ((D - 1) * math.log(2 * math.pi) + torch.sum(z_flat[:, 1:]**2, dim=-1))

        return log_prob, log_prob_t, log_prob_s

    def _compute_prior_nll_loss(self, x_real, cond):
        """
        计算 PriorNet 的 NLL 损失 (HAP 核心)。
        
        这是 HAP 的核心：让 PriorNet 的 μ 接近真实值 x_real，
        同时 σ 学习成为校准的不确定性估计。
        
        NLL = 0.5 * [D*log(2π) + sum(log(σ²)) + sum((x_real - μ)² / σ²)]
        
        Args:
            x_real: [batch, 1, seq_length] 真实数据 (已归一化到 [-1, 1])
            cond: 条件信息
        
        Returns:
            prior_loss: 标量，Prior NLL 损失的 batch 均值
        """
        if self.prior_net is None or cond is None:
            return torch.tensor(0.0, device=x_real.device)

        history_enc = self._extract_history_encoding(cond)
        x_flat = x_real.squeeze(1)  # [batch, seq_length]
        nll = self.prior_net.nll_loss(x_flat, history_enc)  # [batch]
        return nll.mean()

    def straight_path_interpolation(self, x_start, t, noise):
        """
        计算直线路径插值: x_t = (1-t) * x_start + t * noise
        
        Args:
            x_start: [batch, 1, seq_length] 真实数据
            t: [batch] 时间点
            noise: [batch, 1, seq_length] 噪声
        
        Returns:
            x_t: 插值结果
        """
        x_t = (1 - t.view(-1, 1, 1)) * x_start + t.view(-1, 1, 1) * noise
        return x_t

    def velocity_vector(self, x_start, t, cond=None):
        """
        计算速度向量: v(x_t, t) = x_0 - x_1
        对于直线路径：x_1为噪声，x_0为原始数据
        
        当使用 PriorNet 时，噪声从 N(μ(H), σ²(H)) 采样。
        
        Args:
            x_start: [batch, 1, seq_length] 真实数据
            t: [batch] 时间点
            cond: 条件信息 (用于 PriorNet)
        
        Returns:
            x_t: 插值状态
            velocity: 目标速度向量
        """
        noise = self._sample_prior(x_start.shape, x_start.device, cond)
        x_t = self.straight_path_interpolation(x_start, t, noise)
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
        
        Returns:
            loss: 总损失 (flow matching loss，不含 KL)
            loss_temporal: 时间维度损失
            loss_spatial: 空间维度损失
        """
        # 获取实际时间步长
        t = self.timesteps[t_indices]

        # 计算当前点和真实速度向量 (使用自适应先验采样噪声)
        x_t, true_velocity = self.velocity_vector(x_start, t, cond)  # [bsz, 1, dim]

        # 模型预测速度向量
        pred_velocity = self.model(x_t, t, None, cond)

        # 计算损失
        loss = self.loss_fn(pred_velocity, true_velocity, reduction='none')  # [bsz, 1, dim]

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
    def sample(self, batch_size=16, cond=None, steps=None, euler_only=False, noise=None):
        """
        从噪声采样生成数据
        使用预测的速度场进行指导
        
        参数:
            batch_size: 生成样本数量
            cond: 条件信息
            steps: 采样步数，默认使用初始化时指定的步数
            euler_only: 是否只使用欧拉法 (一阶)，默认False使用Heun法 (二阶)
            noise: 可选的初始噪声张量 [batch_size, channels, seq_length]，如果为None则随机生成
        """
        device = next(self.parameters()).device
        steps = default(steps, self.sampling_timesteps)
        shape = (batch_size, self.channels, self.seq_length)

        # 从先验分布采样起始噪声 (自适应或标准高斯)
        if noise is None:
            x = self._sample_prior(shape, device, cond)
        else:
            x = noise.to(device)

        # 时间步长
        step_size = 1.0 / steps

        # 积分步长 - 欧拉法或Heun法，比简单的欧拉法更精确（2-order），带有预测-校正步骤
        solver_name = "Euler" if euler_only else "Heun"
        # for i in tqdm(range(steps), desc=f'RF sampling with ({solver_name}) solver'):
        for i in range(steps):
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
        return x  # [bsz, 1, dim] (1 + loc_dim)

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
        ode_func = ODEFunc(self.model, self.seq_length, cond)

        x0_norm = normalize_to_neg_one_to_one(x_start)  # Start at real data x0 (t=0)
        a0 = torch.zeros(x0_norm.shape[0], device=x0_norm.device)  # Initial logp accumulator
        a0_t = torch.zeros(x0_norm.shape[0], device=x0_norm.device)
        a0_s = torch.zeros(x0_norm.shape[0], device=x0_norm.device)
        initial_state = (x0_norm, a0, a0_t, a0_s)  # # (x , a , a_t , a_s)
        t_span = torch.tensor([0.0, 1.0], device=x0_norm.device)  # Integrate forward t=0 to t=1

        final_state_tuple = odeint(ode_func, initial_state, t_span, rtol=rtol, atol=atol, method=method)
        # [bsz, 1, dim], [bsz], [bsz], [bsz]
        x1, a1, a1_t, a1_s = [final_state_tuple[i][-1] for i in range(4)]

        # x1: State at t=1 (should approx noise from prior)
        # a1: Accumulated log-density change = integral[0,1] div(v_theta) dt

        # Calculate log prior probability of x1 (使用自适应或标准先验)
        # Change of variables: log p0(x0) = log p1(x1) - a1
        log_prior_p1, log_p1_t, log_p1_s = self._compute_prior_log_prob(x1, cond)

        log_p0 = log_prior_p1 - a1  # [bsz]
        log_p0_t = log_p1_t - a1_t  # [bsz]
        log_p0_s = log_p1_s - a1_s  # [bsz]

        nll = -log_p0.sum().item()
        nll_temp = -log_p0_t.sum().item()
        nll_spat = -log_p0_s.sum().item()

        return nll, nll_temp, nll_spat

    def forward(self, img, cond):
        """
        模型前向传播：随机采样时间点并计算损失
        
        HAP Loss: L_total = L_flow + λ * L_prior
        - L_flow: Flow Matching MSE loss (velocity prediction)
        - L_prior: PriorNet NLL loss (μ and σ supervision)
        
        Returns:
            loss: 总损失 = flow_matching_loss + prior_loss_weight * prior_nll_loss
        """
        b, c, n, device = *img.shape, img.device
        assert n == self.seq_length, f'输入序列长度必须为 {self.seq_length}'

        # 随机采样时间索引
        t_indices = torch.randint(0, self.num_timesteps, (b, ), device=device)

        # 归一化输入
        img_norm = normalize_to_neg_one_to_one(img)

        # 计算 flow matching 损失
        fm_loss, _, _ = self.p_losses(img_norm, t_indices, cond)

        # 计算 Prior NLL 损失 (HAP 核心)
        prior_loss = self._compute_prior_nll_loss(img_norm, cond)

        # 总损失
        loss = fm_loss + self.prior_loss_weight * prior_loss
        return loss

    def forward_with_details(self, img, cond):
        """
        带详细损失分解的前向传播 (用于日志记录)。
        
        HAP Loss: L_total = L_flow + λ * L_prior
        
        Returns:
            loss: 总损失
            loss_temporal: 时间维度 flow matching 损失
            loss_spatial: 空间维度 flow matching 损失  
            prior_loss: Prior NLL 损失 (HAP)
        """
        b, c, n, device = *img.shape, img.device
        assert n == self.seq_length, f'输入序列长度必须为 {self.seq_length}'

        t_indices = torch.randint(0, self.num_timesteps, (b, ), device=device)
        img_norm = normalize_to_neg_one_to_one(img)

        fm_loss, loss_temporal, loss_spatial = self.p_losses(img_norm, t_indices, cond)
        prior_loss = self._compute_prior_nll_loss(img_norm, cond)

        loss = fm_loss + self.prior_loss_weight * prior_loss
        return loss, loss_temporal, loss_spatial, prior_loss


if __name__ == '__main__':
    """
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
"""
