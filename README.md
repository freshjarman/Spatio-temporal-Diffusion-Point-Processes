# Spatio-temporal Diffusion Point Processes

[![zread](https://img.shields.io/badge/Ask_Zread-_.svg?style=flat&color=00b0aa&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/tsinghua-fib-lab/Spatio-temporal-Diffusion-Point-Processes)

![OverallFramework](./assets/framework.png "Our proposed framework")

This project was initially described in the full research track paper *[Spatio-temporal Diffusion Point Processes](https://dl.acm.org/doi/10.1145/3580305.3599511)* at KDD 2023 in Long Beach, CA. Contributors to this project are from the *[Future Intelligence laB (FIB)](https://fi.ee.tsinghua.edu.cn/)* at *[Tsinghua University](https://www.tsinghua.edu.cn/en/)*.

The code is tested under a Linux desktop with torch 1.7 and Python 3.7.10.

## Attention

- **The dataset used in this project are normalized (specifically, normalized d_time & normalized location), which may affect `NLL` computation in Table2 in paper! i.e. `Spatial NLL` for `DeepSTPP` which isn't normalized in location.** You can refer to [issues](https://github.com/tsinghua-fib-lab/Spatio-temporal-Diffusion-Point-Processes/issues/4) on Github and [mathematical basics](https://kimi.moonshot.cn/share/cune319l51jflplgpllg) from my KIMI-Chat.

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

The logs are saved in ``logs/``, use `tensorboard` to analyze visually.

``
tensorboard --logdir=./logs
``

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
