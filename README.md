# Noise Separation guided Candidate Label Reconstruction for Noisy Partial Label Learning

This is a [PyTorch](http://pytorch.org) implementation for the ICLR25 paper "Noise Separation guided Candidate Label Reconstruction for Noisy Partial Label Learning". 

Note our method is plug-in method. Here we provide implementation base on PaPi [<sup>1</sup>](#refer-anchor-1).


## Running Our method

We provide the following shell codes for PaPi running with our method. 

### Getting started

- Create directory `../data` (if `../data` does not exist)
- We employed BLIP-2 as the feature extractor based on the open-source library LAVIS[<sup>2</sup>](#refer-anchor-1), thereby securing a more reliable $K$-neighbor relationship. Features can be extracted using either of the following two methods:
  - You can independently install LAVIS and download the model weights for BLIP2. Our code will invoke the relevant model to perform feature extraction.
  - Alternatively, you can directly download [the pre-extracted features](https://drive.google.com/drive/folders/1DdtBqI1zjNjbIB1BZqJ8CiqFf1MvHzSb?usp=sharing) and decompress them into the directory `../data`. 


### Start Running

<details>
<summary>
Run CIFAR10 varing deferent noise rate and partial rate
</summary>

```shell
CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --exp-dir ./experiment --dataset cifar10 --data-dir ../data --num-class 10 --seed 1 --lr 0.05 --wd 1e-3 --epochs 500 --batch-size 256 --alpha_weight 1.0 --partial_rate 0.3 --noisy_rate 0.2 --wp 100 --rho_range 0.7,1.0 --rho_epoch 200 --method plrc  --features blip2_feature_extractor  --print2file 
CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --exp-dir ./experiment --dataset cifar10 --data-dir ../data --num-class 10 --seed 1 --lr 0.05 --wd 1e-3 --epochs 500 --batch-size 256 --alpha_weight 1.0 --partial_rate 0.3 --noisy_rate 0.3 --wp 100 --rho_range 0.7,1.0 --rho_epoch 200 --method plrc  --features blip2_feature_extractor  --print2file
CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --exp-dir ./experiment --dataset cifar10 --data-dir ../data --num-class 10 --seed 1 --lr 0.05 --wd 1e-3 --epochs 500 --batch-size 256 --alpha_weight 1.0 --partial_rate 0.3 --noisy_rate 0.4 --wp 150 --rho_range 0.7,1.0 --rho_epoch 250 --method plrc  --features blip2_feature_extractor  --print2file 
CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --exp-dir ./experiment --dataset cifar10 --data-dir ../data --num-class 10 --seed 1 --lr 0.05 --wd 1e-3 --epochs 500 --batch-size 256 --alpha_weight 1.0 --partial_rate 0.4 --noisy_rate 0.2 --wp 100 --rho_range 0.7,1.0 --rho_epoch 200 --method plrc  --features blip2_feature_extractor  --print2file 
CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --exp-dir ./experiment --dataset cifar10 --data-dir ../data --num-class 10 --seed 1 --lr 0.05 --wd 1e-3 --epochs 500 --batch-size 256 --alpha_weight 1.0 --partial_rate 0.4 --noisy_rate 0.3 --wp 100 --rho_range 0.7,1.0 --rho_epoch 200 --method plrc  --features blip2_feature_extractor  --print2file 
CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --exp-dir ./experiment --dataset cifar10 --data-dir ../data --num-class 10 --seed 1 --lr 0.05 --wd 1e-3 --epochs 500 --batch-size 256 --alpha_weight 1.0 --partial_rate 0.4 --noisy_rate 0.4 --wp 150 --rho_range 0.7,1.0 --rho_epoch 250 --method plrc  --features blip2_feature_extractor  --print2file 
CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --exp-dir ./experiment --dataset cifar10 --data-dir ../data --num-class 10 --seed 1 --lr 0.05 --wd 1e-3 --epochs 500 --batch-size 256 --alpha_weight 1.0 --partial_rate 0.5 --noisy_rate 0.2 --wp 100 --rho_range 0.7,1.0 --rho_epoch 200 --method plrc  --features blip2_feature_extractor  --print2file 
CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --exp-dir ./experiment --dataset cifar10 --data-dir ../data --num-class 10 --seed 1 --lr 0.05 --wd 1e-3 --epochs 500 --batch-size 256 --alpha_weight 1.0 --partial_rate 0.5 --noisy_rate 0.3 --wp 100 --rho_range 0.7,1.0 --rho_epoch 200 --method plrc  --features blip2_feature_extractor  --print2file 
CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --exp-dir ./experiment --dataset cifar10 --data-dir ../data --num-class 10 --seed 1 --lr 0.03 --wd 1e-3 --epochs 1200 --batch-size 256 --alpha_weight 1.0 --partial_rate 0.5 --noisy_rate 0.4 --wp 200 --rho_range 0.2,1.0 --rho_epoch 300 --method plrc  --features blip2_feature_extractor  --print2file 
```

</details>



<details>
<summary>
Run CIFAR100 varing deferent noise rate and partial rate
</summary>

```shell
CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --exp-dir ./experiment --dataset cifar100 --data-dir ../data --num-class 100 --seed 1 --lr 0.1 --wd 1e-3 --epochs 500 --batch-size 256 --alpha_weight 1.0 --partial_rate 0.03 --noisy_rate 0.2 --wp 200 --rho_range 0.7,1.0 --rho_epoch 300 --method plrc  --features blip2_feature_extractor  --print2file 
CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --exp-dir ./experiment --dataset cifar100 --data-dir ../data --num-class 100 --seed 1 --lr 0.1 --wd 1e-3 --epochs 500 --batch-size 256 --alpha_weight 1.0 --partial_rate 0.03 --noisy_rate 0.3 --wp 200 --rho_range 0.7,1.0 --rho_epoch 300 --method plrc  --features blip2_feature_extractor  --print2file 
CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --exp-dir ./experiment --dataset cifar100 --data-dir ../data --num-class 100 --seed 1 --lr 0.1 --wd 1e-3 --epochs 500 --batch-size 256 --alpha_weight 1.0 --partial_rate 0.03 --noisy_rate 0.4 --wp 200 --rho_range 0.7,1.0 --rho_epoch 300 --method plrc  --features blip2_feature_extractor  --print2file 
CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --exp-dir ./experiment --dataset cifar100 --data-dir ../data --num-class 100 --seed 1 --lr 0.1 --wd 1e-3 --epochs 500 --batch-size 256 --alpha_weight 1.0 --partial_rate 0.05 --noisy_rate 0.2 --wp 200 --rho_range 0.7,1.0 --rho_epoch 300 --method plrc  --features blip2_feature_extractor  --print2file 
CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --exp-dir ./experiment --dataset cifar100 --data-dir ../data --num-class 100 --seed 1 --lr 0.1 --wd 1e-3 --epochs 500 --batch-size 256 --alpha_weight 1.0 --partial_rate 0.05 --noisy_rate 0.3 --wp 200 --rho_range 0.7,1.0 --rho_epoch 300 --method plrc  --features blip2_feature_extractor  --print2file
CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --exp-dir ./experiment --dataset cifar100 --data-dir ../data --num-class 100 --seed 1 --lr 0.1 --wd 1e-3 --epochs 500 --batch-size 256 --alpha_weight 1.0 --partial_rate 0.05 --noisy_rate 0.4 --wp 200 --rho_range 0.7,1.0 --rho_epoch 300 --method plrc  --features blip2_feature_extractor  --print2file 
CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --exp-dir ./experiment --dataset cifar100 --data-dir ../data --num-class 100 --seed 1 --lr 0.1 --wd 1e-3 --epochs 500 --batch-size 256 --alpha_weight 1.0 --partial_rate 0.1 --noisy_rate 0.2 --wp 200 --rho_range 0.7,1.0 --rho_epoch 300 --method plrc  --features blip2_feature_extractor  --print2file 
CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --exp-dir ./experiment --dataset cifar100 --data-dir ../data --num-class 100 --seed 1 --lr 0.05 --wd 1e-3 --epochs 1200 --batch-size 256 --alpha_weight 1.0 --partial_rate 0.1 --noisy_rate 0.3 --wp 200 --rho_range 0.7,1.0 --rho_epoch 300 --method plrc  --features blip2_feature_extractor  --print2file 
CUDA_VISIBLE_DEVICES=1  nohup python -u train.py --exp-dir ./experiment --dataset cifar100 --data-dir ../data --num-class 100 --seed 1 --lr 0.05 --wd 1e-3 --epochs 1200 --batch-size 256 --alpha_weight 1.0 --partial_rate 0.1 --noisy_rate 0.4 --wp 250 --rho_range 0.7,1.0 --rho_epoch 350 --method plrc  --features blip2_feature_extractor  --print2file 
```

</details>

## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@inproceedings{iclr25peng,
  author       = {Xiaorui Peng and
                  Yuheng jia and
                  Fuchao Yang and
                  Ran Wang and
                  Min-Ling Zhang},
  title        = {Noise Separation guided Candidate Label Reconstruction for Noisy Partial Label Learning},
  booktitle    = {The Thirteenth International Conference on Learning Representations, {ICLR}
                  2025, Singapore, April 24-28, 2025},
  year         = {2025}
}
```
## Reference

<div id="refer-anchor-1"></div>

- [1] [Xia S, Lv J, Xu N, et al. Towards effective visual representations for partial-label learning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023: 15589-15598.](https://openaccess.thecvf.com/content/CVPR2023/papers/Xia_Towards_Effective_Visual_Representations_for_Partial-Label_Learning_CVPR_2023_paper.pdf)

- [2] [Dongxu Li, Junnan Li, Hung Le, Guangsen Wang, Silvio Savarese, and Steven C.H. Hoi. LAVIS: A one-stop library for language-vision intelligence. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pp. 31–41, July 2023a.](https://github.com/salesforce/LAVIS)
