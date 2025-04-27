python app.py --dataset Earthquake --mode train --timesteps 500 --samplingsteps 500 --batch_size 64 --cuda_id 0 --total_epochs 2000
# 使用Rectified Flow训练，使用很少的samplingsteps
# python app.py --dataset Earthquake --mode train --timesteps 500 --samplingsteps 20 --batch_size 64 --model_type rf --cuda_id 0 --total_epochs 2000

python app.py --dataset COVID19 --mode train --timesteps 500 --samplingsteps 500 --batch_size 64 --cuda_id 0 --total_epochs 2000

python app.py --dataset Citybikes --mode train --timesteps 500 --samplingsteps 500 --batch_size 128 --cuda_id 0 --total_epochs 2000 

# Independent就是app.py中的if opt.dataset == 'HawkesGMM': opt.dim = 1吗？
python app.py --dataset Independent --mode train --timesteps 500 --samplingsteps 500 --batch_size 128 --cuda_id 0 --total_epochs 2000 
