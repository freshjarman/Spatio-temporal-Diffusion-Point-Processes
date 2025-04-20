## TODO
- [ ] 1. Rectifiedflow framework for DSTPP (model/train/inference)
  - [ ] 1.1 Rectifiedflow model
  - [ ] 1.2 train
  - [ ] 1.3 inference
    - [ ] Euler sampling (1-order)
    - [ ] Heun sampling (2-order)
- [ ] 2. NLL_Cal for rectifiedflow
  - [ ] 2.1 NLL_cal codes
  - [ ] 2.2 NLL_cal math equations etc. (for paper writing)
- [ ] 3. Uncertainty 
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
