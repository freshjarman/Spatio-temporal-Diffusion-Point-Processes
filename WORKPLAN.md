## Work Plan Overview
- [ ] 1. Rectifiedflow framework for DSTPP (model/train/inference)
  - [ ] 1.1 Rectifiedflow model
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
- [ ] 3. Uncertainty (refer to Paper `SMURF-THP` & `SMASH`)
  - [ ] 3.1 Uncertainty quantify metrics codes
  - [ ] 3.2 Uncertainty calibration codes

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

## Other detailed settings
[x] realize the `SinusodialPosEmb` in `RF_Diffusion/RF_Diffusion` - `self.time_mlp`
[ ] complete abstract + introduction + method_summary in one page
[ ] update uncertainty quantify/calibration module

## Hyperparams setting
1. sampling-steps (affect metrics: temporal-rmse & spatial-distance, further affect uncertainty metrics)