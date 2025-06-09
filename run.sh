python app.py --dataset Earthquake --mode train --timesteps 500 --samplingsteps 500 --batch_size 64 --cuda_id 0 --total_epochs 2000
# 使用Rectified Flow训练，使用很少的samplingsteps
# python app_new.py --dataset Earthquake --mode train --timesteps 500 --samplingsteps 50 --batch_size 64 --model_type rf --lr 5e-4 --total_epochs 2000

python app.py --dataset COVID19 --mode train --timesteps 500 --samplingsteps 500 --batch_size 64 --cuda_id 0 --total_epochs 2000

python app.py --dataset Citybikes --mode train --timesteps 500 --samplingsteps 500 --batch_size 128 --cuda_id 0 --total_epochs 2000 

# Independent就是app.py中的if opt.dataset == 'HawkesGMM': opt.dim = 1吗？
python app.py --dataset Independent --mode train --timesteps 500 --samplingsteps 500 --batch_size 128 --cuda_id 0 --total_epochs 2000 

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
$env:KMP_DUPLICATE_LIB_OK="TRUE" python app_uq_ensemble.py --dataset Earthquake --mode train --model_type rf --enable_uq --n_ensemble 10 --timesteps 500 --samplingsteps 50 --batch_size 256 --total_epochs 800 --lr 5e-4 --cuda_id 0 