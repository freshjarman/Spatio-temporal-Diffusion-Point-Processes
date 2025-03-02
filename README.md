# Spatio-temporal Diffusion Point Processes

![OverallFramework](./assets/framework.png "Our proposed framework")

This project was initially described in the full research track paper *[Spatio-temporal Diffusion Point Processes](https://dl.acm.org/doi/10.1145/3580305.3599511)* at KDD 2023 in Long Beach, CA. Contributors to this project are from the *[Future Intelligence laB (FIB)](https://fi.ee.tsinghua.edu.cn/)* at *[Tsinghua University](https://www.tsinghua.edu.cn/en/)*.

The code is tested under a Linux desktop with torch 1.7 and Python 3.7.10.

## Installation

### Environment
- Tested OS: Linux
- Python >= 3.7
- PyTorch == 1.7.1
- Tensorboard

### Dependencies
0. WT: Don't pay much attention to the ``requirements. txt`` file, it's too dirty!
1. Install PyTorch 1.7.1 with the correct CUDA version.
2. Use the ``pip install -r requirements. txt`` command to install all of the Python modules and packages used in this project.

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

The logs are saved in ``logs/``.


## Note

The implemention is based on *[DDPM](https://github.com/lucidrains/denoising-diffusion-pytorch)*.

If you found this library useful in your research, please consider citing:

```
@inproceedings{yuan2023DSTPP,
  author = {Yuan, Yuan and Ding, Jingtao and Shao, Chenyang and Jin, Depeng and Li, Yong},
  title = {Spatio-Temporal Diffusion Point Processes},
  year = {2023},
  booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages = {3173–3184},
}
```
