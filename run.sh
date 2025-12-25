python app.py --dataset Earthquake --mode train --timesteps 500 --samplingsteps 500 --batch_size 64 --cuda_id 0 --total_epochs 2000
# 使用Rectified Flow训练，使用很少的samplingsteps
# python app_new.py --dataset Earthquake --mode train --timesteps 500 --samplingsteps 50 --batch_size 64 --model_type rf --lr 5e-4 --total_epochs 2000

python app.py --dataset COVID19 --mode train --timesteps 500 --samplingsteps 500 --batch_size 64 --cuda_id 0 --total_epochs 2000

python app.py --dataset Citybikes --mode train --timesteps 500 --samplingsteps 500 --batch_size 128 --cuda_id 0 --total_epochs 2000 

# Independent就是app.py中的if opt.dataset == 'HawkesGMM': opt.dim = 1吗？
python app.py --dataset Independent --mode train --timesteps 500 --samplingsteps 500 --batch_size 128 --cuda_id 0 --total_epochs 2000 

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # WIN 上解决 "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized." 错误 （ODE计算导致的）
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python app_uq_ensemble.py --dataset Earthquake --mode train --model_type rf --enable_uq --n_ensemble 10 --timesteps 500 --samplingsteps 50 --batch_size 256 --total_epochs 800 --lr 5e-4 --cuda_id 0 
# --timesteps 必须对应正确
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python app_uq_ensemble_test.py --dataset Earthquake --mode test --model_type rf --timesteps 1000 --n_ensemble 100 --samplingsteps 50 --batch_size 64 --cuda_id 0
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python app_uq_ensemble_test.py --dataset Crime --mode test --model_type rf --timesteps 50 --n_ensemble 100 --samplingsteps 5 --batch_size 256
# ./ModelSave/dataset_Crime_timesteps_50_2025-05-27-09h/model_190.pkl
# ./ModelSave/dataset_Earthquake_timesteps_1000_2025-06-09-10h/model_280.pkl

# cost time 
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python app_test_cost_time.py --dataset Earthquake --mode test --model_type rf --timesteps 1000 --n_ensemble 1 --samplingsteps 3 --batch_size 64 --cuda_id 0
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python app_test_cost_time.py --dataset Earthquake --mode test --model_type ddpm --timesteps 500 --n_ensemble 1 --samplingsteps 500 --batch_size 64 --cuda_id 0

# ######################### new args ################################

# common 
# 目前想保持原先的数据预处理设置，必须要加--log_normalization 0
--log_normalization 0 # whether to use log-normalization for d_t data processing, 0: no, 1: yes

# TEST Stage: 完整的质量过滤集成测试命令

#### 1 同 Seed 不同 Epoch（使用 --aux_model_dir）

```linux
python -u app_uq_ensemble_test.py \
    --dataset Earthquake \
    --mode test \
    --model_type rf \
    --dim 2 \
    --timesteps 1000 \
    --samplingsteps 50 \
    --n_ensemble 100 \
    --seed 218 \
    --batch_size 512 \
    # --cuda_id 0 \
    --cpu_num 6 \
    --log_normalization 0 \  # not use log-normalization for d_t
    --main_model_path "./ModelSave/dataset_Earthquake_timesteps_1000_2025-06-09-10h/model_280.pkl" \  # 必须
    --enable_filtered_ensemble \
    --aux_model_dir "./ModelSave/dataset_Earthquake_timesteps_1000_2025-06-09-10h/" \
    --aux_model_epochs "100,150,200,250,280" \
    --filter_ratio 0.2 \
    --use_weighting
```

#### 2 不同 Seed（使用 --aux_model_paths）

```linux
python -u app_uq_ensemble_test.py \
    --dataset Crime \
    --mode test \
    --model_type rf \
    --dim 2 \
    --log_normalization 0 \  # not use log-normalization for d_t
    --timesteps 500 \
    --samplingsteps 10 \
    --n_ensemble 125 \
    --seed 218 \
    --batch_size 512 \
    --main_model_path "./ModelSave/dataset_Crime_timesteps_500_2025-12-01-10h_seed_218/model_200.pkl" \
    --enable_filtered_ensemble \
    --aux_model_paths "./ModelSave/dataset_Crime_timesteps_500_2025-12-01-10h_seed_218/model_200.pkl,./ModelSave/dataset_Crime_timesteps_500_2025-12-01-10h_seed_1023/model_200.pkl,./ModelSave/dataset_Crime_timesteps_500_2025-12-01-10h_seed_5555/model_170.pkl,./ModelSave/dataset_Crime_timesteps_500_2025-12-01-10h_seed_617/model_180.pkl" \
    --filter_ratio 0.2 \
    --use_weighting
```

# Training Stage

python -u app_uq_ensemble.py --dataset Crime --mode train --model_type rf --enable_uq --n_ensemble 100 --samplingsteps 10 --timesteps 500 --batch_size 512 --total_epochs 1000 --lr 5e-4 --seed 218
python -u app_uq_ensemble.py --dataset Crime --mode train --model_type rf --enable_uq --n_ensemble 150 --timesteps 500 --samplingsteps 10 --batch_size 512 --total_epochs 500 --lr 5e-4 --log_normalization 1 --seed 218

### 使用方法

训练（启用 PriorNet）：
python app_uq_ensemble.py --model_type rf --dataset earth-stpp-smash \
    --use_prior_net --kl_weight 0.001 --prior_hidden_dim 128 --seed 218

训练（不启用，标准高斯先验）：
python app_uq_ensemble.py --model_type rf --dataset earth-stpp-smash --seed 218

测试：
python app_uq_ensemble_test.py --model_type rf --dataset earth-stpp-smash \
    --use_prior_net --main_model_path ./ModelSave/xxx/model_xxx.pkl --seed 218