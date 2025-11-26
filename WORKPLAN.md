# Work Plan Overview
- [x] 1. Rectifiedflow framework for DSTPP (model/train/inference)
  - [x] 1.1 Rectifiedflow model
    - Spatio-temporal encoder
      - now use the same `DSTPP/Models/Transformer_ST` as encoder (section 3.1)
    - Spatio-temporal decoder
      - `RF_Diffusion`: Co-attention neural network for modeling $v_\theta$ (section 3.4)
      - `RectifiedFlow`: Rectified flow for training(path interpolation)/sampling/NLL_cal (section 3.2 + 3.3)
  - [x] 1.2 train
  - [x] 1.3 inference
    - [x] Euler sampling (1-order)
    - [x] Heun sampling (2-order)
- [x] 2. NLL_Cal for rectifiedflow
  - [x] 2.1 NLL_cal codes
  - [x] 2.2 NLL_cal math equations etc. (for paper writing)
- [x] 3. Uncertainty (refer to Paper `SMURF-THP` & `SMASH`)
  - 3.1 Uncertainty quantify metrics codes
  - 3.2 Uncertainty calibration codes
- [ ] 4. re-conduct STPP experiments on baselines for fair results, refer to `SMASH` repo
  - 4.1 Data processing part (event marks' effect should be removed)
  - 4.2 Normalization method (log-normalization or **[0, 1] normalization** for all datasets, consistent with all baselines)
  - 4.3 Hyperparams setting (sampling-steps, learning rate schedule, ensemble n_samples etc.)

## 2025.4.18
    -- all by Claude 3.7 Sonnet Thinking
1. create `DSTPP/Appendix.md`, i.e. comments for NLL_cal of RectifiedFlow
2. create `DSTPP/RectifiedFlow.py`
3. create `DSTPP/RF_Diffusion.py`
4. create `DSTPP/RF_Model_all.py`

## 2025.4.20
    -- by Gemini 2.5 Pro
1. add `calculate_log_likelihood` function in `DSTPP/RectifiedFlow.py` with relevant utility functions (but to be verified)
2. create `app_new.py` supporting `opt.model_type` == `DDPM` or `rf`

## 2025.4.22
1. update `Appendix.md` with NLL_cal math equations
2. verify and update `calculate_log_likelihood` function in `DSTPP/RectifiedFlow.py`

## 2025.4.24
1. summarize the model details of `DSTPP`
2. support euler sampling and heun sampling in `DSTPP/RectifiedFlow.py` - function `sample`
3. ! fix the bugs in `RectifiedFlow.py`, including `sample` function and `calculate_log_likelihood` function. update the corresponding math equations (sampling equations & ODEFunc System & ODE Solver) in `Appendix.md`


## 2025.4.25
1. [x] check and update the model structure (including input -> process -> model -> output) of my `RF-STPP` framework

## 2025.4.27
1. Fix bugs in `RectifiedFlow.py`, including:
   1. `calculate_neg_log_likelihood`: to support directly calculating NLL for temporal/spatial/all
   2. Fix bugs in `ODEFunc` and `divergence_approx` about velocity prediction of nn model
   3. Update relevent utilization codes in `app_new.py`
2. Delete NLL Calculation codes during training stage in `app_new.py`, which i think is unnecessary
3. Realize the `SinusodialPosEmb` in `RF_Diffusion/RF_Diffusion` - `self.time_mlp`
4. Debug the whole `RF-STPP` process on earthquake dataset with epoch=200 & sampling-steps=20

## 2025.4.28
1. batched experiments on SEU Platform
2. [x] fix the `divergence_approx` (with mask)/`ODEFunc`/`calculate_neg_log_likelihood` for precise `nll, nll_temp, nll_spat`

## 2025.5.26
1. update ensemble + uncertainty quantify/calibration module
   - [x] update uncertainty quantify/calibration module, confidence levels = [0.5 0.6 0.7 0.8 0.9]
2. modify crime dataset to support `RF-STPP` framework, i.e. the each `data[2]` is event mark, which is not supported by `RF-STPP` framework now (**此处的Crime dataset来自SMASH仓库，并非DSTPP仓库原有的Crime数据集，务必注意！**)
3. why CS metric is so bad in Crime for `RF-STPP`? 

---
---

# After AAAI'26 Phase 2 Weak Rejection (2025.11)

## 2025.11.24
1. refer to `SMASH`, rewrite its docstring

### Previous Results Fault with `SMASH`
2. - [ ] previous STPP results on `SMASH` repo are kinda wrong, because the data processing part is not consistent with `DSTPP` repo, where **event marks' effect should be removed**, need to re-conduct experiments on `SMASH` other than directly using its paper results (**all baselines' real results may be worse than paper results?**)
3. - [ ] re-conduct experiments on `football` dataset with my `RF-STPP`, with better data processing part (e.g. use 'log' normalization for data which is consistent with `SMASH` repo; **Attention:** `SMASH` use log-normalization for all datasets, ours use [0, 1] normalization for all datasets! Is it unfair for baselines? Need to check if the same normalization method is essential for fair comparison!)





---

## Hyperparams setting
1. sampling-steps (affect metrics: temporal-rmse & spatial-distance, further affect uncertainty metrics): less more for FM compared with DDPM, to improve computational efficiency (speed)
2. learning rate schedule: 考虑lr衰减，epoch = 100后考虑lr * 0.1，因为他这个NLL_Spatial就直接转折上去了
3. ensemble: `n_samples` during sampling