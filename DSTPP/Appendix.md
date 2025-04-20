# RF-TPP Appendix
# 1. 负对数似然（NLL）近似

确实，Rectified Flow (RF) 的标准训练目标（速度场的 MSE 损失）与 DDPM 的变分下界 (VLB) 在形式和含义上都不同。

虽然 RF 本身通常不直接优化 VLB，但我们可以借鉴连续时间生成模型（如 Continuous Normalizing Flows, CNFs, 或基于 SDE 的模型）的理论来计算 RF 模型下的精确（或近似精确）对数似然，而不是像 DDPM 那样计算一个下界。这通常涉及到求解一个包含散度项的常微分方程 (ODE)。

## 理论：精确对数似然计算 (基于 ODE)

对于一个由 ODE $\frac{dx_t}{dt} = v(x_t, t)$ 定义的、从先验分布 $p_1$ (通常是标准高斯 $N(0, I)$ at $t=1$) 变换到数据分布 $p_0$ (at $t=0$) 的流模型，数据点 $x_0$ 的精确对数似然可以通过瞬时变量变换公式得到：

$$ \log p_0(x_0) = \log p_1(x_1) - \int_0^1 \nabla \cdot v_\theta(x_t, t) dt $$

这里：

$x_t$ 是 ODE 从 $t=0$ 时的 $x_0$ 积分到 $t=1$ 时的 $x_1$ 的轨迹（或者反向积分）。
$v_\theta(x_t, t)$ 是我们训练好的神经网络模型预测的速度场。
$\nabla \cdot v_\theta(x_t, t)$ 是速度场在点 $(x_t, t)$ 处的散度 (divergence)，也等于模型雅可比矩阵 $\nabla_{x_t} v_\theta(x_t, t)$ 的迹 (trace)。
$\log p_1(x_1)$ 是 $x_1$ 在先验分布下的对数似然（对于标准高斯，这是一个简单的二次型）。
积分项 $\int_0^1 \nabla \cdot v_\theta(x_t, t) dt$ 计算了在流变换过程中对数密度的总变化量。

## 挑战与实现：散度积分

计算这个积分是主要挑战：

高维散度：直接计算雅可比矩阵并求迹在高维空间中计算成本极高 ($O(D^2)$ 或更高)。
积分：我们需要沿着 ODE 轨迹对散度进行积分。
解决方案：Hutchinson 迹估计器 + ODE 求解器

Hutchinson 迹估计器：我们可以使用随机迹估计器（如 Hutchinson 方法）来近似散度（迹），而无需计算完整的雅可比矩阵。对于一个向量场 $v$ 和一个随机向量 $\epsilon$（满足 $\mathbb{E}[\epsilon]=0, \mathbb{E}[\epsilon \epsilon^T]=I$，例如来自标准正态分布或 Rademacher 分布），有： $$ \nabla \cdot v = \text{Tr}(\nabla v) = \mathbb{E}_{\epsilon}[\epsilon^T (\nabla v) \epsilon] $$ 在实践中，我们通常只采样一个或少数几个 $\epsilon$ 来获得无偏估计。计算 $\epsilon^T (\nabla v) \epsilon$ 可以通过两次反向传播（或一次前向和一次反向）高效完成，计算成本约为两次模型前向传播。

耦合 ODE 系统：我们将状态 $x_t$ 的 ODE 和累积对数密度变化 $\Delta \log p_t = -\int \nabla \cdot v_\theta dt$ 的 ODE 耦合起来求解： $$ \frac{d}{dt} \begin{pmatrix} x_t \ a_t \end{pmatrix} = \begin{pmatrix} v_\theta(x_t, t) \ -\nabla \cdot v_\theta(x_t, t) \end{pmatrix} $$ 其中 $a_t = \log p_t(x_t)$。我们从 $t=0$ 积分到 $t=1$，初始条件为 $(x_0, a_0=0)$。最终得到的 $a_1$ 就是积分项 $-\int_0^1 \nabla \cdot v_\theta dt$。然后 $\log p_0(x_0) \approx \log p_1(x_1) + a_1$。

## 代码实现思路

我们需要一个能够求解 ODE 的库（如 torchdiffeq）和一个计算散度近似的函数

```python
# 需要安装: pip install torchdiffeq
import torch
import torch.nn as nn
from torchdiffeq import odeint

# --- 1. Hutchinson Trace Estimator for Divergence ---
def divergence_approx(f, x, t, e=None):
    """ 计算 f(x, t) 关于 x 的散度的随机近似 """
    # f: 模型 v_theta(x, t)
    # x: 输入状态 [batch, ...]
    # t: 时间 [batch]
    # e: 随机噪声向量，与 x 形状相同，通常来自 N(0, I)
    if e is None:
        e = torch.randn_like(x)

    # 确保 x 需要梯度
    x_requires_grad = x.requires_grad
    with torch.enable_grad():
        x.requires_grad_(True)
        fx = f(x, t) # 计算 v_theta(x, t)
        # 计算 vjp: e^T * J
        # autograd.grad 计算的是 gradient * vector product (对于标量输出)
        # 或 vector * jacobian product (对于向量输出)
        # 我们需要计算 e^T * J * e
        # 第一步：计算 J * e (Jacobian-vector product)
        # PyTorch 的 vjp 计算的是 grad_outputs^T * J
        # 为了得到 J * e，我们可以计算 grad( (f(x,t) * e).sum(), x )
        f_dot_e = (fx * e).sum()
        # 计算梯度，即 J^T * e，但我们需要 J*e
        # 使用技巧：计算 vjp(e) -> e^T * J
        # 然后再与 e 点积
        # 更直接的方法：
        vjp_e = torch.autograd.grad(outputs=fx, inputs=x, grad_outputs=e, create_graph=True)[0]
        # 现在 vjp_e 包含 J^T * e (如果 fx 是行向量) 或 J*e (如果 fx 是列向量)
        # 假设 fx 和 x 都是类似 [batch, dim] 的形状
        # grad(outputs=fx, inputs=x, grad_outputs=e) 计算的是 sum_i (e_i * d(fx_i)/dx) ? 不对
        # grad(outputs=(fx*e).sum(), inputs=x) 计算的是 d((fx*e).sum())/dx = d(sum_j fx_j * e_j)/dx
        # = sum_j ( (d(fx_j)/dx) * e_j + fx_j * d(e_j)/dx )  <- 如果 e 依赖 x
        # 假设 e 不依赖 x
        # = sum_j (d(fx_j)/dx) * e_j = J^T * e
        # 我们需要 e^T * J * e
        # 试试直接计算两次
        e_vjp_e = (vjp_e * e).sum() # 这应该是 e^T * J^T * e ? 不对

        # 重新思考：我们需要计算 Tr(J) = sum_i (J_ii)
        # Hutchinson: E[eps^T J eps]
        # 计算 J*eps:
        # 需要设置 create_graph=True 以便计算二阶导数（如果需要的话）
        # 但这里我们只需要 J*eps
        # 使用 torch.func.jvp (forward-mode AD) 可能更直接
        # 或者利用 autograd hack:
        # grad(sum(f(x, t) * e), x, create_graph=True) -> J^T * e
        # grad(sum(J^T * e * e), x) -> ???

        # 最常用的方法：
        # 计算 vjp: e^T * J
        e_J = torch.autograd.grad(fx, x, e, create_graph=True)[0]
        # 计算 (e^T * J) * e
        e_J_e = (e_J * e).sum(dim=tuple(range(1, x.dim()))) # Sum over all non-batch dims

    # 恢复 x 的 requires_grad 状态
    x.requires_grad_(x_requires_grad)
    return e_J_e # 返回每个 batch 元素的散度近似值

# --- 2. 定义耦合 ODE 的动力学 ---
class ODEFunc(nn.Module):
    def __init__(self, model_v):
        super().__init__()
        self.model_v = model_v # 这是我们训练好的 RF 模型 v_theta

    def forward(self, t, state):
        # state 是一个元组 (x, logp_integral_accumulator)
        x, _ = state
        batch_size = x.shape[0]

        # 确保 t 是一个标量，或者扩展成 batch 大小
        if t.numel() == 1:
            t_batch = t.expand(batch_size) # RF 模型需要 batch 化的时间输入
        else:
            t_batch = t # 如果 ODE 求解器已经提供了 batch 化的时间

        # 计算速度场 v_theta(x, t)
        v = self.model_v(x, t_batch)

        # 计算散度近似 -nabla_x dot v_theta(x, t)
        # 使用与 x 相同设备上的随机噪声
        e = torch.randn_like(x)
        neg_div_v = -divergence_approx(self.model_v, x, t_batch, e)

        # 返回状态的导数 (dx/dt, d(logp)/dt)
        return (v, neg_div_v)

# --- 3. 在 RectifiedFlow 类中添加计算对数似然的方法 ---
class RectifiedFlow(nn.Module):
    # ... (之前的 __init__, p_losses, sample 等方法) ...

    @torch.no_grad()
    def calculate_log_likelihood(self, x_start, cond=None, rtol=1e-5, atol=1e-5, method='dopri5'):
        """
        计算给定数据点 x_start 的精确对数似然 (相对于先验)
        """
        self.model.eval() # 确保模型在评估模式

        # 准备 ODE 函数
        ode_func = ODEFunc(self.model)

        # 初始状态
        x0 = normalize_to_neg_one_to_one(x_start) # 确保输入在 [-1, 1]
        logp_init = torch.zeros(x0.shape[0], device=x0.device) # 初始 logp 变化为 0
        initial_state = (x0, logp_init)

        # 定义积分时间点 (从 t=0 到 t=1)
        t_span = torch.tensor([0.0, 1.0], device=x0.device)

        # 使用 ODE 求解器积分
        # 注意：torchdiffeq 的 odeint 需要 t 作为第一个参数传递给 ODEFunc
        # 我们需要确保 ODEFunc 能处理标量或向量 t
        # odeint 返回每个时间点的状态列表，我们只需要最后一个时间点 t=1 的状态
        final_state_tuple = odeint(
            ode_func,
            initial_state,
            t_span,
            rtol=rtol,
            atol=atol,
            method=method
        )

        # 提取 t=1 时的状态
        x1 = final_state_tuple[0][-1] # 轨迹在 t=1 时的位置
        logp_integral = final_state_tuple[1][-1] # 累积的 logp 变化 (- integral div v dt)

        # 计算 x1 在先验 N(0, I) 下的对数似然
        # log p(z) = -0.5 * [ D*log(2pi) + sum(z_i^2) ]
        D = x1.shape[1:].numel() # 数据维度
        log_prior_p1 = -0.5 * (D * math.log(2 * math.pi) + torch.sum(x1**2, dim=tuple(range(1, x1.dim()))))

        # 计算最终的 log p0(x0)
        log_likelihood = log_prior_p1 + logp_integral

        return log_likelihood # 返回每个 batch 元素的对数似然值
```

如何使用:
```python
# 假设 x_start 是一个 batch 的数据点，cond 是条件输入
# model 是 训练好的RectifiedFlow 模型
x_start = torch.randn(64, 2) # 64 个样本，2 维数据
cond = None # 如果没有条件输入
model = RectifiedFlow() # 初始化模型
log_likelihood = model.calculate_log_likelihood(x_start, cond)
average_nll = -log_likelihoods.mean() # 计算平均负对数似然
``` 

重要说明:

- 计算成本：这种方法比简单的 MSE 计算成本高得多，因为它需要在每个 ODE 步骤中计算散度近似（涉及反向传播），并且需要运行 ODE 求解器。**它主要用于评估模型，而不是作为主要的训练目标。**

- 近似：Hutchinson 迹估计器本身是随机的，所以计算结果是对数似然的一个无偏估计。多次运行或使用多个 e 样本可以提高精度。ODE 求解器也有数值误差。

- 与 VLB 的关系：这个计算得到的是 $\log p_\theta(x_0)$ 的估计值，而不是像 VLB 那样的 $\mathbb{E}_{q(\dots)}[\dots] \le \log p(x_0)$。因此，它不是一个严格的下界，而是对目标值本身的估计。但它可以像 VLB 一样，用于比较不同模型或同一模型在不同检查点时的似然性能。

- 这个实现提供了一种在 Rectified Flow 框架下评估模型对数似然的方法，其作用类似于 DDPM 中的 VLB，但机制是基于连续时间流模型的精确似然计算原理。

## ==== 以下为废案 ====

```
DSTPP\RectifiedFlow.py
- line 173: NLL_cal(self, x_start, cond)
- 计算整个轨迹的负对数似然的近似
  注意：RF不是显式概率模型，这是启发式方法
```

## ！！！参考DSTPP equation 3（VLB）

**Sonnet Thinking解释如下：**

在深度学习中，Rectified Flow (RF) 模型是一种用于生成建模的框架。它通过将数据分布映射到一个简单的分布（如高斯分布）来实现生成任务。与传统的生成模型不同，RF模型不直接定义概率密度函数，而是通过学习一个变换来实现数据的生成。

Rectified Flow (RF) 模型不是一个显式的概率模型，因此没有直接的负对数似然表达式。但在生成模型评估中，我们需要一种方式来评估模型的生成质量，特别是样本可能性（likelihood）。

从理论上讲，如果我们有一个连续时间随机微分方程(SDE)描述的生成模型，负对数似然可以通过以下方式表示：

$$\text{NLL}(x) = -\log p(x) \approx \mathbb{E}_{t \sim U[0,T]}\left[\frac{1}{2}|v\theta(x_t, t) - v_\text{true}(x_t, t)|^2\right] + C$$

其中：

- $v_\theta$ 是模型预测的速度场
- $v_\text{true}$ 是真实的速度场
- $C$ 是一个常数项

## 代码实现解释
该方法通过以下步骤近似计算NLL：

1. 在时间轴上均匀采样点（最多100个）
2. 对每个时间点评估模型预测速度与真实速度之间的MSE
3. 对所有时间点的损失取平均值

从数学上看，整个轨迹的负对数似然可以表示为($x_{start}$为真实数据)：

$$\text{NLL}(x_\text{start}) \approx \frac{1}{N} \sum_{i=1}^{N} |v_\theta(x_{t_i}, t_i) - v_\text{true}(x_{t_i}, t_i)|^2$$

其中 $t_i$ 是从轨迹上均匀采样的时间点。

## 为何这是有效的近似？
- **路径积分视角：** 在扩散模型和流模型中，负对数似然与沿生成路径的速度场误差积分相关

- **变分下界：** 这个近似可以被视为真实负对数似然的变分上界

- **实用价值：** 尽管是启发式方法，但这提供了一种评估和比较模型性能的实用方式；在RF模型中，我们不求精确计算复杂的似然函数，而是用这种近似方法来评估模型性能和进行模型选择。

总结来说，这个函数计算的确实是MSE损失，但它通过在整个生成轨迹上取平均，为RF模型提供了一个实用的负对数似然近似值。