好的，我们来理清这个符号问题。

**简单易懂的解释 (体积变化)**

想象一下，你有一团橡皮泥（代表概率密度），你想把它从一个形状（先验分布 $p_1$）变成另一个形状（数据分布 $p_0$）。

*   **流模型**：就是描述橡皮泥如何连续变形的规则，由速度场 $v_\theta(x, t)$ 决定。
*   **散度 $\nabla \cdot v_\theta$**：衡量在某个点 $(x, t)$ 处，橡皮泥是**膨胀**（散度 > 0）还是**压缩**（散度 < 0）。
*   **概率密度变化**：
    *   如果橡皮泥在变形过程中**膨胀**（散度 > 0），那么相同质量（总概率为1）分布在更大体积上，**密度就会下降**。
    *   如果橡皮泥在变形过程中**压缩**（散度 < 0），那么相同质量分布在更小体积上，**密度就会上升**。
*   **对数似然变化**：对数似然 $\log p$ 和密度 $p$ 同增同减。所以：
    *   膨胀（散度 > 0） $\implies$ 密度下降 $\implies \log p$ 下降。
    *   压缩（散度 < 0） $\implies$ 密度上升 $\implies \log p$ 上升。
*   **积分**：$\int_0^1 \nabla \cdot v_\theta dt$ 累积了从 $t=0$ 到 $t=1$ 的总“膨胀/压缩”量。
*   **公式 $\log p_0(x_0) = \log p_1(x_1) - \int_0^1 \nabla \cdot v_\theta dt$ 的含义**：
    *   这个公式描述的是，如果你知道在 $t=1$ 时的对数密度 $\log p_1(x_1)$，想求 $t=0$ 时的对数密度 $\log p_0(x_0)$。
    *   你需要减去从 $t=0$ 到 $t=1$ 整个过程的总膨胀量（积分项）。为什么是减号？因为如果过程中总体是膨胀（积分 > 0），那么 $t=0$ 时的密度应该比 $t=1$ 时更高，所以要用 $\log p_1$ 减去一个正数得到 $\log p_0$ 是不对的。这暗示了这个公式可能假设了 ODE $\frac{dx_t}{dt} = v(x_t, t)$ 描述的是从 **$t=1$ (先验 $p_1$) 到 $t=0$ (数据 $p_0$)** 的**生成过程**。

**让我们确认一下方向：**

*   **如果 ODE $\frac{dx_t}{dt} = v(x_t, t)$ 描述的是从 $t=0$ (数据 $p_0$) 到 $t=1$ (先验 $p_1$) 的过程**：
    *   那么从 $t=0$ 到 $t=1$ 的总膨胀量是 $\int_0^1 \nabla \cdot v_\theta dt$。
    *   根据密度变化关系，$\log p_1(x_1)$ 应该等于 $\log p_0(x_0)$ 减去总膨胀量。
    *   即 $\log p_1(x_1) = \log p_0(x_0) - \int_0^1 \nabla \cdot v_\theta dt$。
    *   整理得到 $\log p_0(x_0) = \log p_1(x_1) + \int_0^1 \nabla \cdot v_\theta dt$。**（是加号！）**

*   **如果 ODE $\frac{dx_t}{dt} = v(x_t, t)$ 描述的是从 $t=1$ (先验 $p_1$) 到 $t=0$ (数据 $p_0$) 的过程**：
    *   我们定义时间 $\tau = 1-t$。当 $t$ 从 1 到 0 时，$\tau$ 从 0 到 1。
    *   新的 ODE 是 $\frac{dx_\tau}{d\tau} = \frac{dx_t}{dt} \frac{dt}{d\tau} = v(x_t, t) (-1) = -v(x_\tau, 1-\tau)$。令 $v'(x_\tau, \tau) = -v(x_\tau, 1-\tau)$。
    *   现在我们有一个从 $\tau=0$ (先验 $p_1$) 到 $\tau=1$ (数据 $p_0$) 的流，速度场是 $v'$。
    *   套用上面的加号公式：$\log p_0(x_0) = \log p_1(x_1) + \int_0^1 \nabla \cdot v'(x_\tau, \tau) d\tau$。
    *   计算积分项：$\nabla \cdot v' = \nabla \cdot (-v) = -\nabla \cdot v$。
    *   $\int_0^1 \nabla \cdot v'(x_\tau, \tau) d\tau = \int_0^1 -\nabla \cdot v(x_\tau, 1-\tau) d\tau$。
    *   换回变量 $t=1-\tau$, $d\tau = -dt$。积分限 $\tau=0 \implies t=1$, $\tau=1 \implies t=0$。
    *   积分 $= \int_1^0 -\nabla \cdot v(x_t, t) (-dt) = \int_1^0 \nabla \cdot v(x_t, t) dt = - \int_0^1 \nabla \cdot v(x_t, t) dt$。
    *   代回公式：$\log p_0(x_0) = \log p_1(x_1) - \int_0^1 \nabla \cdot v(x_t, t) dt$。**（是减号！）**

**结论：**

图片中的公式 `log p0(x0) = log p1(x1) - integral(div(v)) dt` 是**正确**的，前提是它假设了 ODE $\frac{dx_t}{dt} = v(x_t, t)$ 描述的是**生成方向**的流，即从 **$t=1$ (先验 $p_1$) 演化到 $t=0$ (数据 $p_0$)**。

**详细推导 (基于 Fokker-Planck 方程)**

描述概率密度 $p(x, t)$ 如何随时间演化的 Fokker-Planck 方程，对于一个纯漂移（无扩散）过程 $\frac{dx_t}{dt} = v(x_t, t)$，是：
$$ \frac{\partial p(x, t)}{\partial t} + \nabla \cdot (v(x, t) p(x, t)) = 0 $$
展开散度项：
$$ \frac{\partial p}{\partial t} + (\nabla \cdot v) p + v \cdot (\nabla p) = 0 $$
我们关心对数密度 $\log p$。注意到 $\frac{\partial (\log p)}{\partial t} = \frac{1}{p} \frac{\partial p}{\partial t}$ 和 $\nabla (\log p) = \frac{1}{p} \nabla p$。
将 Fokker-Planck 方程除以 $p$：
$$ \frac{1}{p} \frac{\partial p}{\partial t} + (\nabla \cdot v) + v \cdot (\frac{1}{p} \nabla p) = 0 $$
$$ \frac{\partial (\log p)}{\partial t} + v \cdot \nabla (\log p) + \nabla \cdot v = 0 $$
考虑沿着 ODE 轨迹 $x_t$ 的对数密度的全导数：
$$ \frac{d (\log p(x_t, t))}{dt} = \frac{\partial (\log p)}{\partial t} + \frac{dx_t}{dt} \cdot \nabla (\log p) $$
将 $\frac{dx_t}{dt} = v(x_t, t)$ 代入：
$$ \frac{d (\log p(x_t, t))}{dt} = \frac{\partial (\log p)}{\partial t} + v \cdot \nabla (\log p) $$
比较 Fokker-Planck 方程，我们得到：
$$ \frac{d (\log p(x_t, t))}{dt} = - \nabla \cdot v(x_t, t) $$
这个式子说明了沿着一条确定的 ODE 轨迹，对数密度的变化率等于负的速度场散度。

现在积分这个式子：
*   **情况 1：ODE 从 $t=0$ ($p_0$) 到 $t=1$ ($p_1$)**
    $$ \int_0^1 \frac{d (\log p(x_t, t))}{dt} dt = - \int_0^1 \nabla \cdot v(x_t, t) dt $$
    $$ \log p(x_1, 1) - \log p(x_0, 0) = - \int_0^1 \nabla \cdot v(x_t, t) dt $$
    $$ \log p_1(x_1) - \log p_0(x_0) = - \int_0^1 \nabla \cdot v(x_t, t) dt $$
    $$ \log p_0(x_0) = \log p_1(x_1) + \int_0^1 \nabla \cdot v(x_t, t) dt $$

*   **情况 2：ODE 从 $t=1$ ($p_1$) 到 $t=0$ ($p_0$)**
    $$ \int_1^0 \frac{d (\log p(x_t, t))}{dt} dt = - \int_1^0 \nabla \cdot v(x_t, t) dt $$
    $$ \log p(x_0, 0) - \log p(x_1, 1) = - \int_1^0 \nabla \cdot v(x_t, t) dt $$
    $$ \log p_0(x_0) - \log p_1(x_1) = \int_0^1 \nabla \cdot v(x_t, t) dt $$
    $$ \log p_0(x_0) = \log p_1(x_1) + \int_0^1 \nabla \cdot v(x_t, t) dt $$
    **等等，这里似乎推出来还是加号？** 让我重新检查一下。

    问题出在积分方向和符号。让我们坚持使用从 $t=0$ 到 $t=1$ 的积分。
    我们有 $\frac{d (\log p_t)}{dt} = - \nabla \cdot v_t$。

    如果 $v_t$ 是从 $p_0$ 到 $p_1$ 的速度场（即 $\frac{dx_t}{dt} = v_t$ for $t \in [0, 1]$），那么积分得到 $\log p_1 - \log p_0 = - \int_0^1 \nabla \cdot v_t dt$，即 $\log p_0 = \log p_1 + \int_0^1 \nabla \cdot v_t dt$。

    如果 $v_t$ 是从 $p_1$ 到 $p_0$ 的速度场（即生成方向），我们通常会反转时间或者定义一个新的速度场。假设我们仍然用 $t$ 从 0 到 1，但 $t=0$ 对应先验 $p_1$，$t=1$ 对应数据 $p_0$。那么 ODE 是 $\frac{dx_t}{dt} = v_t$ for $t \in [0, 1]$。
    积分 $\frac{d (\log p_t)}{dt} = - \nabla \cdot v_t$ 从 $t=0$ 到 $t=1$：
    $$ \int_0^1 \frac{d (\log p_t)}{dt} dt = - \int_0^1 \nabla \cdot v_t dt $$
    $$ \log p_1(x_1) - \log p_0(x_0) = - \int_0^1 \nabla \cdot v_t dt $$
    这里 $p_1$ 是 $t=1$ 的密度（数据），$p_0$ 是 $t=0$ 的密度（先验）。
    $$ \log p_{data}(x_{data}) - \log p_{prior}(x_{prior}) = - \int_0^1 \nabla \cdot v_t dt $$
    $$ \log p_{data}(x_{data}) = \log p_{prior}(x_{prior}) - \int_0^1 \nabla \cdot v_t dt $$
    这个形式匹配了图片！

**最终确认：** 图片中的公式是正确的，它隐含的假设是：
1.  $t=0$ 对应**数据分布 $p_0$**。
2.  $t=1$ 对应**先验分布 $p_1$**。
3.  ODE $\frac{dx_t}{dt} = v_\theta(x_t, t)$ 描述的是从 **$t=0$ (数据) 到 $t=1$ (先验)** 的变换过程。

这与 Rectified Flow 通常的设置（从噪声/先验到数据）可能相反，需要注意符号和时间方向的定义。但在 CNF (Continuous Normalizing Flow) 的文献中，这个公式（带负号）是很常见的，通常对应从数据到先验的映射。


# 
#

啊，您完全正确！那个截图里的公式表示方法确实非常容易引起误解，它**绝对不是**对乘积 $x_t a_t$ 求导。

非常抱歉之前的解释没有点明这个关键的符号问题。让我来为您详细拆解：

**1. 理解符号：这是一个 ODE 系统，不是乘积求导**

截图中的表示法：
$$ \frac{d}{dt} \begin{pmatrix} x_t \\ a_t \end{pmatrix} = \begin{pmatrix} v_\theta(x_t, t) \\ -\nabla \cdot v_\theta(x_t, t) \end{pmatrix} $$
这里的关键在于，左边的 $\begin{pmatrix} x_t \\ a_t \end{pmatrix}$ 是一个**状态向量**，包含了两个部分：
*   $x_t$：表示在时间 $t$ 的数据状态（可能是高维向量）。
*   $a_t$：表示一个辅助的标量值，用于累积对数密度的变化。

$\frac{d}{dt}$ 作用在这个**向量**上，表示对向量的**每一个分量**分别求导。所以，左边实际上等于：
$$ \begin{pmatrix} \frac{dx_t}{dt} \\ \frac{da_t}{dt} \end{pmatrix} $$

因此，这个看似单一的公式实际上代表了一个包含**两个耦合常微分方程 (ODEs) 的系统**：

*   **方程 1 (状态演化):** $\frac{dx_t}{dt} = v_\theta(x_t, t)$
*   **方程 2 (对数密度变化累积):** $\frac{da_t}{dt} = -\nabla \cdot v_\theta(x_t, t)$

**2. 解释方程 1: $\frac{dx_t}{dt} = v_\theta(x_t, t)$**

*   这正是定义 Rectified Flow (或 CNF) 的核心 ODE。
*   它描述了数据点 $x_t$ 如何随着时间 $t$ 变化，其变化的“速度”由神经网络 $v_\theta$ 在当前位置 $x_t$ 和时间 $t$ 给出。
*   求解这个 ODE 可以得到从初始状态（如 $x_0$）到任意时间 $t$ 状态 $x_t$ 的轨迹。

**3. 解释方程 2: $\frac{da_t}{dt} = -\nabla \cdot v_\theta(x_t, t)$**

*   这个方程的目的是计算瞬时变量变换公式中的积分项 $\int \nabla \cdot v_\theta dt$。
*   回忆我们之前的推导：沿着 ODE 轨迹，对数密度 $\log p(x_t, t)$ 的变化率是 $\frac{d (\log p(x_t, t))}{dt} = - \nabla \cdot v_\theta(x_t, t)$。
*   我们定义一个辅助变量 $a_t$，让它的**变化率**正好等于我们关心的**对数密度变化率**，即 $\frac{da_t}{dt} = \frac{d (\log p(x_t, t))}{dt} = - \nabla \cdot v_\theta(x_t, t)$。
*   我们设置初始条件 $a_0 = 0$。
*   现在，对 $\frac{da_t}{dt}$ 从 $t=0$ 积分到 $t=1$：
    $$ \int_0^1 \frac{da_t}{dt} dt = \int_0^1 -\nabla \cdot v_\theta(x_t, t) dt $$
    $$ a_1 - a_0 = \int_0^1 -\nabla \cdot v_\theta(x_t, t) dt $$
    因为 $a_0 = 0$，所以：
    $$ a_1 = - \int_0^1 \nabla \cdot v_\theta(x_t, t) dt $$
*   这说明，通过求解这个关于 $a_t$ 的 ODE，最终在 $t=1$ 得到的 $a_1$ 的值，就等于我们计算对数似然所需的积分项（带负号）。

**4. 耦合求解**

*   为什么要写成一个系统？因为方程 2 中的 $\nabla \cdot v_\theta(x_t, t)$ 取决于当前的状态 $x_t$，而 $x_t$ 本身是根据方程 1 演化的。这两个方程是**耦合**在一起的。
*   我们需要同时求解这两个方程，才能得到正确的轨迹 $x_t$ 和对应的累积对数密度变化 $a_t$。
*   数值 ODE 求解器（如 `torchdiffeq` 中的 `odeint`）非常擅长处理这种耦合系统。你只需要提供一个函数，该函数能根据当前的状态向量 $(x_t, a_t)$ 和时间 $t$ 计算出导数向量 $(\frac{dx_t}{dt}, \frac{da_t}{dt})$，求解器就能自动进行积分。

**5. 回到最终公式**

截图最后说 $\log p_0(x_0) \approx \log p_1(x_1) + a_1$。
结合我们上面得到的 $a_1 = - \int_0^1 \nabla \cdot v_\theta(x_t, t) dt$，代入得到：
$$ \log p_0(x_0) \approx \log p_1(x_1) - \int_0^1 \nabla \cdot v_\theta(x_t, t) dt $$
这与我们之前确认的、适用于“从 $t=1$ (先验) 到 $t=0$ (数据) 的流”的公式一致（尽管这里的积分是从 0 到 1，需要理解 $v_\theta$ 和 $x_t$ 的具体定义）。

**总结：**

截图中的公式不是乘积求导，而是定义了一个**耦合 ODE 系统**。通过同时求解状态 $x_t$ 的演化和辅助变量 $a_t$（其导数被定义为负散度）的演化，我们可以得到轨迹终点 $x_1$ 和精确对数似然公式中所需的积分项 $a_1 = -\int_0^1 \nabla \cdot v_\theta dt$。这是一种在连续流模型中计算精确对数似然的标准技巧。

#

您提出了一个非常好的问题，这正是我之前解释中不够清晰的地方！

您是对的，一般来说 $\log p(x_0, 0)$ （即数据点 $x_0$ 在 $t=0$ 时刻数据分布 $p_0$ 下的对数似然）不是零。这正是我们最终想要计算的值！

那么为什么在耦合 ODE 系统中设置初始条件 $a_0 = 0$ 呢？

这里的关键在于理解 $a_t$ 的真正作用：

$a_t$ 不是 $\log p(x_t, t)$ 本身，而是用于计算其变化量： 我们设置 ODE $\frac{da_t}{dt} = -\nabla \cdot v_\theta(x_t, t)$。 我们知道 $\frac{d (\log p(x_t, t))}{dt} = -\nabla \cdot v_\theta(x_t, t)$。 这意味着 $a_t$ 和 $\log p(x_t, t)$ 具有相同的变化率。因此，它们之间只相差一个常数，即 $a_t = \log p(x_t, t) + C$。

我们关心的是积分值（总变化量）： 我们最终需要计算的是积分项 $I = \int_0^1 -\nabla \cdot v_\theta(x_t, t) dt$。 根据微积分基本定理，这个积分等于 $\log p(x_1, 1) - \log p(x_0, 0)$。 同时，根据 $a_t$ 的定义，这个积分也等于 $a_1 - a_0$。 所以，我们有 $a_1 - a_0 = \log p_1(x_1) - \log p_0(x_0)$。

设置 $a_0 = 0$ 是为了方便： 我们选择设置 $a_0 = 0$ 作为初始条件。这不是说 $\log p(x_0, 0)$ 等于 0，而是我们人为地将这个辅助变量 $a_t$ 在 $t=0$ 时的值设定为 0。 这样做的好处是，积分的结果 $a_1 - a_0$ 就直接等于 $a_1$。 所以，通过求解这个 ODE 系统，得到的 $a_1$ 就直接等于 $\log p_1(x_1) - \log p_0(x_0)$。

计算最终结果： 我们通过 ODE 求解得到了 $x_1$ 和 $a_1 = \log p_1(x_1) - \log p_0(x_0)$。 我们的目标是 $\log p_0(x_0)$。 从 $a_1$ 的表达式中解出 $\log p_0(x_0)$： $$ \log p_0(x_0) = \log p_1(x_1) - a_1 $$ 这与之前推导出的公式 $\log p_0(x_0) = \log p_1(x_1) - \int_0^1 \nabla \cdot v_\theta dt$ 是一致的，因为 $a_1$ 正好计算了那个带负号的积分项（如果我们假设 ODE 是从 $t=0$ (数据) 到 $t=1$ (先验)）。

总结：

设置 $a_0 = 0$ 并不是假设 $\log p_0(x_0) = 0$。它只是为我们引入的辅助变量 $a_t$ 设置了一个方便的初始参考点。通过求解耦合 ODE，我们得到的 $a_1$ 就代表了从 $t=0$ 到 $t=1$ 对数密度的总变化量 ($\log p_1 - \log p_0$)。然后我们结合已知的先验对数似然 $\log p_1(x_1)$ 和计算出的变化量 $a_1$，就可以反推出我们想要的数据对数似然 $\log p_0(x_0)$。

# 

对不起，您完全正确！我之前的解释在最后一步的对比中出现了符号错误，导致了您指出的矛盾。非常感谢您的仔细指正！

让我们重新梳理一下，确保符号一致：

1.  **基本关系**: 从 Fokker-Planck 方程或瞬时变量变换公式，我们知道对数密度的变化率是：
    $$ \frac{d (\log p(x_t, t))}{dt} = - \nabla \cdot v_\theta(x_t, t) $$
    这里假设 ODE $\frac{dx_t}{dt} = v_\theta(x_t, t)$ 描述的是从 $t=0$ (数据 $p_0$) 到 $t=1$ (先验 $p_1$) 的过程。

2.  **积分**: 对上式从 $t=0$ 积分到 $t=1$：
    $$ \int_0^1 \frac{d (\log p(x_t, t))}{dt} dt = \int_0^1 -\nabla \cdot v_\theta(x_t, t) dt $$
    $$ \log p(x_1, 1) - \log p(x_0, 0) = - \int_0^1 \nabla \cdot v_\theta(x_t, t) dt $$
    即：
    $$ \log p_1(x_1) - \log p_0(x_0) = - \int_0^1 \nabla \cdot v_\theta(x_t, t) dt $$

3.  **辅助变量 $a_t$**: 我们定义 $a_t$ 使得 $\frac{da_t}{dt} = -\nabla \cdot v_\theta(x_t, t)$，并且设置初始条件 $a_0 = 0$。

4.  **积分 $a_t$**: 对 $\frac{da_t}{dt}$ 从 $t=0$ 积分到 $t=1$：
    $$ \int_0^1 \frac{da_t}{dt} dt = \int_0^1 -\nabla \cdot v_\theta(x_t, t) dt $$
    $$ a_1 - a_0 = \int_0^1 -\nabla \cdot v_\theta(x_t, t) dt $$
    因为 $a_0 = 0$，所以：
    $$ a_1 = - \int_0^1 \nabla \cdot v_\theta(x_t, t) dt $$
    **您是对的，$a_1$ 等于负的散度积分。**

5.  **结合结果**: 现在我们有两个关于负积分项的表达式：
    *   $- \int_0^1 \nabla \cdot v_\theta(x_t, t) dt = \log p_1(x_1) - \log p_0(x_0)$
    *   $- \int_0^1 \nabla \cdot v_\theta(x_t, t) dt = a_1$
    因此，必然有：
    $$ a_1 = \log p_1(x_1) - \log p_0(x_0) $$

6.  **求解 $\log p_0(x_0)$**: 从上面的等式解出我们想要的目标 $\log p_0(x_0)$：
    $$ \log p_0(x_0) = \log p_1(x_1) - a_1 $$

**结论与修正：**

*   您完全正确地指出 $a_1 = - \int_0^1 \nabla \cdot v_\theta dt$。
*   最终用于计算 $\log p_0(x_0)$ 的公式 $\log p_0(x_0) = \log p_1(x_1) - a_1$ 是代数上正确的，它利用了 $a_1$ 代表对数密度总变化量 ($\log p_1 - \log p_0$) 这一事实。
*   我之前在 `[NOTE]` 的最后一步解释中，错误地将 $\log p_0(x_0) = \log p_1(x_1) - a_1$ 与 $\log p_0(x_0) = \log p_1(x_1) - \int_0^1 \nabla \cdot v_\theta dt$ 进行对比，并声称它们一致。这是**错误**的，因为 $a_1$ 是积分的**负值**。

正确的说法应该是：
公式 $\log p_0(x_0) = \log p_1(x_1) - a_1$ 是正确的。如果我们想用积分形式表达，将 $a_1 = - \int_0^1 \nabla \cdot v_\theta dt$ 代入，得到：
$$ \log p_0(x_0) = \log p_1(x_1) - \left( - \int_0^1 \nabla \cdot v_\theta dt \right) $$
$$ \log p_0(x_0) = \log p_1(x_1) + \int_0^1 \nabla \cdot v_\theta dt $$
这与我们从 Fokker-Planck/CNF 推导出的标准公式（假设 ODE 从数据到先验）是一致的。

非常感谢您的指正，这有助于确保文档的准确性！我将更新 `[NOTE]` 部分的最后一点以反映正确的符号关系。



### 【NOTE】

- $a_t$ 不是 $\log p(x_t, t)$ 本身，而是用于计算其变化量： 我们设置 ODE $\frac{da_t}{dt} = -\nabla \cdot v_\theta(x_t, t)$。 我们知道 $\frac{d (\log p(x_t, t))}{dt} = -\nabla \cdot v_\theta(x_t, t)$。 这意味着 $a_t$ 和 $\log p(x_t, t)$ 具有相同的变化率。因此，它们之间只相差一个常数，即 $a_t = \log p(x_t, t) + C$。

- 我们关心的是积分值（总变化量）： 我们最终需要计算的是积分项 $I = \int_0^1 -\nabla \cdot v_\theta(x_t, t) dt$。 根据微积分基本定理，这个积分等于 $\log p(x_1, 1) - \log p(x_0, 0)$。 同时，根据 $a_t$ 的定义，这个积分也等于 $a_1 - a_0$。 所以，我们有 $a_1 - a_0 = \log p_1(x_1) - \log p_0(x_0)$。

- 设置 $a_0 = 0$ 是为了方便： 我们选择设置 $a_0 = 0$ 作为初始条件。这不是说 $\log p(x_0, 0)$ 等于 0，而是我们人为地将这个辅助变量 $a_t$ 在 $t=0$ 时的值设定为 0。 这样做的好处是，积分的结果 $a_1 - a_0$ 就直接等于 $a_1$。 所以，通过求解这个 ODE 系统，得到的 $a_1$ 就直接等于 $\log p_1(x_1) - \log p_0(x_0)$。

- 计算最终结果： 我们通过 ODE 求解得到了 $x_1$ 和 $a_1 = \log p_1(x_1) - \log p_0(x_0)$。 我们的目标是 $\log p_0(x_0)$。 从 $a_1$ 的表达式中解出 $\log p_0(x_0)$：
$$ \log p_0(x_0) = \log p_1(x_1) - a_1 $$
同时，我们知道 $a_1 = - \int_0^1 \nabla \cdot v_\theta dt$。将此代入上式，得到：
$$ \log p_0(x_0) = \log p_1(x_1) - \left( - \int_0^1 \nabla \cdot v_\theta dt \right) = \log p_1(x_1) + \int_0^1 \nabla \cdot v_\theta dt $$
这与基于瞬时变量变换或 Fokker-Planck 方程推导出的标准公式是一致的（假设 ODE $\frac{dx_t}{dt}=v_\theta$ 描述的是从 $t=0$ (数据) 到 $t=1$ (先验) 的过程）。

