# Flow Matching for Spatio-temporal Point Processes

[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/tsinghua-fib-lab/Spatio-temporal-Diffusion-Point-Processes)


## 潜在风险

- 我是在DSTPP框架上扩展支持了Flow Matching的建模方案；因此，我沿用了DDPM中“t=0对应真实数据分布，t=1对应噪声分布”的设定，这与Rectified Flow中“t=0对应噪声分布，t=1对应真实数据分布”的设定是相反的。这在代码实现中会体现在训练时的插值路径设计，采样函数和NLL计算函数的时间步长处理上（即Rectified Flow的插值，采样和NLL计算（详情参考`DSTPP/Appendix.md`中的数学理论比较与我的论文的`Appendix.A`）需要对时间步长进行反转处理）。这可能会引起一些混淆，尤其是对于熟悉Rectified Flow的用户来说。

> 在使用本代码时，请务必注意此类潜在风险，以免引起混淆。

## Overall Framework
**Next Event Prediction in STPPs**

![OverallFramework](./assets/framework.png "Our proposed framework")


## Experiments Management

- Git (github; gitee)
- Tensorboard (/logs)
- Model Checkpoints (/ModelSave, with `config.json`)
- Argparse (refer to `README.md` & `run.sh` for usage guide)

> PS: recommend to use Hydra (or other config management tools) and WandB in future for better experiment management.

## Data Reprocessing

- **The dataset used in this project are [0,1] normalized (specifically, normalized d_time & normalized location), which may affect `NLL` computation in Table2 in `DSTPP` paper! i.e. `Spatial NLL` for `DeepSTPP` which isn't normalized in location.** You can refer to [issues](https://github.com/tsinghua-fib-lab/Spatio-temporal-Diffusion-Point-Processes/issues/4) on Github and [mathematical basics](https://kimi.moonshot.cn/share/cune319l51jflplgpllg) from my KIMI-Chat.

- **For `Earthquake/Crime/Football` datasets, if you want to use log-normalization for time-intervals (to alleviate long-tail problem of d_t), please make sure to set `--log_normalization 1` both in training and testing stage to keep consistent with `SMASH` framework. (Note that the original `DSTPP` framework didn't implement log-normalization, which I used before AAAI'26.)**

## Installation

### Environment
- Tested OS: Linux
- Python >= 3.7
- PyTorch == 1.7.1
- Tensorboard

### Dependencies
0. WT: Don't pay much attention to the `requirements. txt` file, it's too dirty! Install PyTorch >= 1.7.1 with the correct CUDA version.

## 📚 TEST: app_uq_ensemble_test.py 使用指南

### 🎯 核心概念

| 术语 | 含义 |
|------|------|
| **Naive Ensemble** | 朴素集成：单个模型采样 N 次，均匀加权平均 |
| **Quality-Filtered Ensemble** | 质量过滤集成：多模型评估噪声质量，过滤低质量噪声，按熵值加权 |
| **Main Model** | 主模型：用于生成最终预测的模型（通过 `--main_model_path` 指定） |
| **Auxiliary Models** | 辅助模型：用于评估噪声质量的多个模型（可来自同一 seed 的不同 epoch，或不同 seed） |


> **注意**：所有命令均是在powershell环境下运行，如需在Linux/MacOS终端运行，请将反引号(`)替换为反斜杠(\)。

---

### 📋 场景一：朴素集成（Naive Ensemble）

**适用情况**：只有一个训练好的模型，想要通过多次采样获得不确定性估计。

```powershell
# 基础用法 - Earthquake 数据集，50次采样
python app_uq_ensemble_test.py --dataset Earthquake --mode test --n_ensemble 50 `
    --main_model_path "./ModelSave/dataset_Earthquake_timesteps_1000_2025-06-09-10h/model_280.pkl"

# 指定模型类型为 Rectified Flow，100次采样
python app_uq_ensemble_test.py --dataset Earthquake --mode test --model_type rf --n_ensemble 100 `
    --main_model_path "./ModelSave/dataset_Earthquake_timesteps_1000_2025-06-09-10h/model_280.pkl"

# Crime 数据集
python app_uq_ensemble_test.py --dataset Crime --mode test --n_ensemble 50 `
    --main_model_path "./ModelSave/dataset_Crime_timesteps_500_xxx/model_190.pkl"
```

**特点**：
- ✅ 简单易用，只需一个模型
- ✅ 计算量相对较小
- ❌ 所有噪声均匀加权，可能包含低质量预测

---

### 📋 场景二：质量过滤集成（Quality-Filtered Ensemble）

**适用情况**：有多个模型 checkpoint，想要通过噪声质量评估来提升集成效果。

#### 2.1 同一 Seed 的不同 Epoch 模型（--aux_model_dir）

适用于从**同一次训练**中选择不同 epoch 的模型作为辅助模型。

```powershell
# 使用 epoch 100, 150, 200, 250 的模型作为辅助模型
python app_uq_ensemble_test.py --dataset Earthquake --mode test --n_ensemble 100 `
    --main_model_path "./ModelSave/dataset_Earthquake_timesteps_1000_2025-06-09-10h/model_280.pkl" `
    --enable_filtered_ensemble `
    --aux_model_dir "./ModelSave/dataset_Earthquake_timesteps_1000_2025-06-09-10h/" `
    --aux_model_epochs "100,150,200,250"
```

#### 2.2 不同 Seed 的模型（--aux_model_paths）【新功能】

适用于从**不同 seed 训练**的模型中选择辅助模型，提供更多样化的预测以更好地评估噪声质量。

```powershell
# 使用不同 seed 训练的模型作为辅助模型，路径用逗号分隔（逗号后无空格）
python app_uq_ensemble_test.py --dataset Crime --mode test --n_ensemble 100 `
    --main_model_path "./ModelSave/dataset_Crime_timesteps_500_2025-12-01-10h_seed_1023/model_200.pkl" `
    --enable_filtered_ensemble `
    --aux_model_paths "./ModelSave/dataset_Crime_timesteps_500_2025-12-01-10h_seed_1023/model_200.pkl,./ModelSave/dataset_Crime_timesteps_500_2025-12-01-10h_seed_5555/model_200.pkl,./ModelSave/dataset_Crime_timesteps_500_2025-12-01-10h_seed_218/model_200.pkl,./ModelSave/dataset_Crime_timesteps_500_2025-12-01-10h_seed_617/model_200.pkl"
```

#### 2.3 混合使用（不同 seed + 不同 epoch）

你也可以灵活混合不同 seed 和不同 epoch 的模型：

```powershell
# 混合使用：不同 seed 的不同 epoch 模型
python app_uq_ensemble_test.py --dataset Crime --mode test --n_ensemble 100 `
    --main_model_path "./ModelSave/dataset_Crime_timesteps_500_seed_1023/model_200.pkl" `
    --enable_filtered_ensemble `
    --aux_model_paths "./ModelSave/dataset_Crime_timesteps_500_seed_1023/model_150.pkl,./ModelSave/dataset_Crime_timesteps_500_seed_1023/model_200.pkl,./ModelSave/dataset_Crime_timesteps_500_seed_5555/model_150.pkl,./ModelSave/dataset_Crime_timesteps_500_seed_5555/model_200.pkl"
```

#### 2.4 调整过滤比例

```powershell
# 过滤掉 30% 的低质量噪声（默认 20%）
python app_uq_ensemble_test.py --dataset Earthquake --mode test --n_ensemble 100 `
    --main_model_path "./ModelSave/xxx/model_280.pkl" `
    --enable_filtered_ensemble `
    --aux_model_dir "./ModelSave/xxx/" `
    --aux_model_epochs "100,150,200,250" `
    --filter_ratio 0.3

# 更激进的过滤：过滤掉 50% 的噪声
python app_uq_ensemble_test.py --dataset Earthquake --mode test --n_ensemble 100 `
    --main_model_path "./ModelSave/xxx/model_280.pkl" `
    --enable_filtered_ensemble `
    --aux_model_dir "./ModelSave/xxx/" `
    --aux_model_epochs "100,150,200,250" `
    --filter_ratio 0.5
```

#### 2.5 不使用熵值加权（仅过滤，均匀加权）

```powershell
# 禁用熵值加权，过滤后的噪声均匀加权
python app_uq_ensemble_test.py --dataset Earthquake --mode test --n_ensemble 100 `
    --main_model_path "./ModelSave/xxx/model_280.pkl" `
    --enable_filtered_ensemble `
    --aux_model_dir "./ModelSave/xxx/" `
    --aux_model_epochs "100,150,200,250" `
    --no-use_weighting
```

> **注意**：`--no-use_weighting` 需要 Python 3.9+ 的 `BooleanOptionalAction` 支持。

**特点**：
- ✅ 过滤低质量噪声，提升预测精度
- ✅ 熵值加权让高质量预测贡献更大
- ✅ 支持同 seed 不同 epoch，或不同 seed 的模型
- ❌ 需要多个模型 checkpoint
- ❌ 计算量较大（需要每个噪声经过多个模型）

---

### 📋 场景三：不同数据集的测试

```powershell
# Earthquake 数据集 (2D 空间)
python app_uq_ensemble_test.py --dataset Earthquake --mode test --dim 2 --n_ensemble 50 `
    --main_model_path "./ModelSave/dataset_Earthquake_xxx/model_280.pkl"

# Crime 数据集 (2D 空间)
python app_uq_ensemble_test.py --dataset Crime --mode test --dim 2 --n_ensemble 50 `
    --main_model_path "./ModelSave/dataset_Crime_xxx/model_190.pkl"

# Football 数据集 (2D 空间)
python app_uq_ensemble_test.py --dataset Football --mode test --dim 2 --n_ensemble 50 `
    --main_model_path "./ModelSave/dataset_Football_xxx/model_1220.pkl"

# HawkesGMM 数据集 (1D 空间 - 自动设置)
python app_uq_ensemble_test.py --dataset HawkesGMM --mode test --n_ensemble 50 `
    --main_model_path "./ModelSave/dataset_HawkesGMM_xxx/model_100.pkl"
```

---

### 📋 场景四：调整模型参数

```powershell
# 使用不同的 timesteps（需与训练时一致）
python app_uq_ensemble_test.py --dataset Earthquake --mode test `
    --timesteps 1000 --samplingsteps 50 --n_ensemble 50 `
    --main_model_path "./ModelSave/xxx/model_280.pkl"

# 使用更少的采样步数（加速推理）
python app_uq_ensemble_test.py --dataset Earthquake --mode test `
    --timesteps 1000 --samplingsteps 20 --n_ensemble 50 `
    --main_model_path "./ModelSave/xxx/model_280.pkl"

```

#### ！log-normalization 时间间隔处理

```powershell
# 关闭时间间隔的 log 变换（默认打开）
python app_uq_ensemble_test.py --dataset Earthquake --mode test `
    --log_normalization 0 --n_ensemble 50 `
    --main_model_path "./ModelSave/xxx/model_280.pkl"
---
```

### 📋 场景五：GPU 和 CPU 性能相关

```powershell
# 使用特定 GPU
python app_uq_ensemble_test.py --dataset Earthquake --mode test --cuda_id 0 --n_ensemble 50 `
    --main_model_path "./ModelSave/xxx/model_280.pkl"

# 调整 CPU 核数（用于数据加载）
python app_uq_ensemble_test.py --dataset Earthquake --mode test --cpu_num 8 --n_ensemble 50 `
    --main_model_path "./ModelSave/xxx/model_280.pkl"

# 调整 batch size
python app_uq_ensemble_test.py --dataset Earthquake --mode test --batch_size 128 --n_ensemble 50 `
    --main_model_path "./ModelSave/xxx/model_280.pkl"
```

---

### 📋 场景六：完整的质量过滤集成测试命令

#### 6.1 同 Seed 不同 Epoch（使用 --aux_model_dir）

```powershell
python app_uq_ensemble_test.py `
    --dataset Earthquake `
    --mode test `
    --model_type rf `
    --dim 2 `
    --timesteps 1000 `
    --samplingsteps 50 `
    --n_ensemble 100 `
    --seed 1234 `
    --cuda_id 0 `
    --cpu_num 6 `
    --log_normalization 1 `
    --main_model_path "./ModelSave/dataset_Earthquake_timesteps_1000_2025-06-09-10h/model_280.pkl" `
    --enable_filtered_ensemble `
    --aux_model_dir "./ModelSave/dataset_Earthquake_timesteps_1000_2025-06-09-10h/" `
    --aux_model_epochs "100,150,200,250,280" `
    --filter_ratio 0.2 `
    --use_weighting
```

#### 6.2 不同 Seed（使用 --aux_model_paths）

```powershell
python app_uq_ensemble_test.py `
    --dataset Crime `
    --mode test `
    --model_type rf `
    --dim 2 `
    --timesteps 500 `
    --samplingsteps 50 `
    --n_ensemble 100 `
    --seed 1234 `
    --cuda_id 0 `
    --main_model_path "./ModelSave/dataset_Crime_timesteps_500_2025-12-01-10h_seed_1023/model_200.pkl" `
    --enable_filtered_ensemble `
    --aux_model_paths "./ModelSave/dataset_Crime_timesteps_500_2025-12-01-10h_seed_1023/model_200.pkl,./ModelSave/dataset_Crime_timesteps_500_2025-12-01-10h_seed_5555/model_200.pkl,./ModelSave/dataset_Crime_timesteps_500_2025-12-01-10h_seed_218/model_200.pkl,./ModelSave/dataset_Crime_timesteps_500_2025-12-01-10h_seed_617/model_200.pkl" `
    --filter_ratio 0.2 `
    --use_weighting
```

---

### 🔧 参数说明表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset` | str | Earthquake | 数据集名称 |
| `--mode` | str | train | 必须设为 `test` |
| `--model_type` | str | rf | 模型类型：`rf` 或 `ddpm` |
| `--dim` | int | 2 | 空间维度 (1/2/3) |
| `--timesteps` | int | 50 | 扩散步数 |
| `--samplingsteps` | int | 50 | 采样步数 |
| `--n_ensemble` | int | 5 | 集成采样数量 |
| `--seed` | int | 1234 | 随机种子 |
| `--cuda_id` | str | 0 | GPU ID |
| `--log_normalization` | int | 1 | 是否对时间间隔 log 变换 |
| `--main_model_path` | str | **必填** | 主模型路径（用于最终预测） |
| `--enable_filtered_ensemble` | flag | False | 启用质量过滤集成 |
| `--aux_model_dir` | str | None | 辅助模型目录（同 seed 不同 epoch） |
| `--aux_model_epochs` | str | None | 辅助模型 epochs，逗号分隔 |
| `--aux_model_paths` | str | None | 辅助模型完整路径，逗号分隔（支持跨 seed）|
| `--filter_ratio` | float | 0.2 | 过滤掉的低质量噪声比例 |
| `--use_weighting` | flag | True | 使用熵值倒数加权 |

---

### ⚠️ 注意事项

0. **数据预处理**：如果使用了时间间隔的 log 变换，确保训练和测试阶段的 `--log_normalization` 参数一致（默认为1以保证和`SMASH`一致，但原`DSTPP`框架默认为0，我AAAI26前使用的是0）
1. **必填参数**：`--main_model_path` 是必填参数，必须指定主模型的完整路径
2. **辅助模型选择**：
   - 使用 `--aux_model_dir` + `--aux_model_epochs`：适用于同一 seed 不同 epoch
   - 使用 `--aux_model_paths`：适用于不同 seed 或任意组合（更灵活）
   - 两者二选一，`--aux_model_paths` 优先级更高
3. **最少辅助模型**：质量过滤集成需要至少 2 个辅助模型
4. **运行次数**：脚本自动运行 10 次测试以计算均值和标准差
5. **结果保存**：结果自动保存到 `./jsons/` 目录
6. **Python 版本**：过滤后依然使用均匀加权可在命令中指定 `--no-use_weighting`；但注意需要 Python 3.9+

---

### 💡 辅助模型选择建议

| 场景 | 推荐方法 | 理由 |
|------|----------|------|
| 快速实验 | 同 seed 不同 epoch | 只需一次训练，方便快捷 |
| 追求更好效果 | 不同 seed | 模型多样性更高，噪声质量评估更可靠 |
| 资源充足 | 混合使用 | 综合多样性和训练成本 |

## Model Training

Use the following command to train DSTPP on `Earthquake` dataset: 

``
python app.py --dataset Earthquake --mode train --timesteps 500 --samplingsteps 500 --batch_size 64 --total_epochs 2000
``

To train DSTPP on other datasets:

``
python app.py --dataset COVID19 --mode train --timesteps 500 --samplingsteps 500 --batch_size 64 --total_epochs 2000
``

``
python app.py --dataset Citibike --mode train --timesteps 500 --samplingsteps 500 --batch_size 128 --total_epochs 2000 
``

``
python app.py --dataset Independent --mode train --timesteps 500 --samplingsteps 500 --batch_size 128 --total_epochs 2000 
``

The trained models are saved in ``ModelSave/``.

The logs are saved in ``logs/``, use `tensorboard` to analyze visually.

``
tensorboard --logdir=./logs
``

## Note

- The base project `DSTPP` was initially described in the full research track paper *[Spatio-temporal Diffusion Point Processes](https://dl.acm.org/doi/10.1145/3580305.3599511)* at KDD 2023 in Long Beach, CA. Contributors to this project are from the *[Future Intelligence laB (FIB)](https://fi.ee.tsinghua.edu.cn/)* at *[Tsinghua University](https://www.tsinghua.edu.cn/en/)*. The implemention of Diffusion is based on *[DDPM](https://github.com/lucidrains/denoising-diffusion-pytorch)*.

- The code is tested under a Linux desktop with torch 1.7 and Python 3.7.10.

