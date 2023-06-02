# Model Debiasing via Gradient-based Explanation on Representation

![framework diagram](https://github.com/DexterJZ/dexterjz.github.io/blob/main/images/dvge.png)

This repo is an implementation of the paper:

*Evaluating Adversarial Attacks on Driving Safety in Vision-Based Autonomous Vehicles* ([arXiv](https://arxiv.org/abs/2305.12178))

By [Jindi Zhang](https://dexterjz.github.io/), Luning Wang, Dan Su, Yongxiang Huang, Caleb Chen Cao, and Lei Chen

## Introduction

Machine learning systems produce biased results towards certain demographic groups, known as the fairness problem. Recent approaches to tackle this problem learn a latent code (i.e., representation) through disentangled representation learning and then discard the latent code dimensions correlated with sensitive attributes (e.g., gender). Nevertheless, these approaches may suffer from incomplete disentanglement and overlook proxy attributes (proxies for sensitive attributes) when processing real-world data, especially for unstructured data, causing performance degradation in fairness and loss of useful information for downstream tasks. To address the issues, we propose a novel fairness framework named DVGE that performs debiasing with regard to both sensitive attributes and proxy attributes, which boosts the prediction performance of downstream task models without complete disentanglement. The main idea is to, first, leverage gradient-based explanation to find two model focuses, 1) one focus for predicting sensitive attributes and 2) the other focus for predicting downstream task labels, and second, use them to perturb the latent code that guides the training of downstream task models towards fairness and utility goals.

## Dependencies

Install the following dependencies.
```
python>=3.6.4
pytorch>=0.4.0
tqdm
torchvision
cv2
numpy
```

NVIDIA GPU is required for model training and testing.

## Data Preparation

The CelebA Dataset can be downloaded from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

To prepare the dataset, first, download img_align_celeba.zip and put it in `DVGE/data/` as below.
```
DVGE
└── data
    └── img_align_celeba.zip
```

Then, extract images to ```DVGE/data/CelebA/``` by running
```
sh scripts/prepare_data.sh CelebA
```

Afterwards, the data folder structure would look like as follows.
```
DVGE
└── data
    ├── CelebA
        └── img_align_celeba
            ├── 000001.jpg
            ├── 000002.jpg
            ├── ...
            └── 202599.jpg
```

## DVGE

### VAE Training

For DVGE-D (DVGE with a disentangled VAE), we use [FactorVAE](http://proceedings.mlr.press/v80/kim18b/kim18b.pdf) as the encoder. Train it by running
```
sh scripts/factor_celeba.sh
```

The hyperparameter `--gamma` in the script is tunable.

For DVGE-N (DVGE with a non-disentangled VAE), we use [VanillaVAE](https://arxiv.org/pdf/1312.6114.pdf) as the encoder. Train it by running
```
sh scripts/vanilla_celebat.sh
```

### Latent Code Generation

For DVGE-D, generate the latent code by runing
```
sh scripts/factor_celebat.sh
```

For DVGE-N, generate the latent code by runing
```
sh scripts/vanilla_celebat.sh
```

### Latent Code Splitting

Under `DVGE/outputs/<vae_name>/z_and_y/`, create two folders `train/` and `val/` by runing
```
mkdir /outputs/<vae_name>/z_and_y/train/ /outputs/<vae_name>/z_and_y/val/
```

And move `000001.npz` ~ `180000.npz` to `train/`, move `180001.npz` ~ `202599.npz` to `val/`. `<vae_name>` can be `factor_celeba` or `vanilla_celeba`.

After splitting the latent code, the output folder structure would look like as below.
```
DVGE
└── outputs
    ├── <vae_name>
        └── z_and_y
            └── train
                ├── 000001.npz
                ├── 000002.npz
                ├── ...
                └── 180000.npz
            └── val
                ├── 180001.npz
                ├── 180002.npz
                ├── ...
                └── 202599.npz
```

### Sensitive Classifier Training

For the scenario of single sensitive attribute, train the sensitive classifier by running
```
sh scripts/sens_cls.sh
```

For the scenario of multiple sensitive attributes, train by running
```
sh scripts/sens_cls1.sh
```

The latent code used for training the sensitive classifier can be specified by setting `--dataset` in the script, i.e., `factor_celeba`or `vanilla_celeba`.

Choose the checkpoint with the best validation accuracy and record the checkpoint index number.

### Downstream Task Model Training

For the scenario of single sensitive attribute, train the downstream task model by running
```
sh scripts/ot_cls.sh
```

For the scenario of multiple sensitive attributes, train by running
```
sh scripts/ot_cls1.sh
```

The latent code used for training the downstream task model can be specified by setting `--dataset` in the script, i.e., `factor_celeba`or `vanilla_celeba`. Please be noted that the latent code for training the sensitive classifier and the downstream task model must be from the same VAE. For example, if `--dataset` for training the sensitive classifier is `factor_celeba`, then `--dataset` for training the downstream task model should also be `factor_celeba`.

The index number of the best sensitive classifier checkpoint can be specified at `--sens_ckpt` in the script.

To generate a Pareto Front to show the fairness-accuracy trade-off, it is needed to sweep a range of values for `--eta1` and `--eta2` in the script which correspond to $\eta_{1}$ and $\eta_{2}$ in the paper.

## Citation

If you find this repo useful, please consider citing our paper:
```
@article{zhang2023model,
  title={Model Debiasing via Gradient-based Explanation on Representation},
  author={Zhang, Jindi and Wang, Luning and Su, Dan and Huang, Yongxiang and Cao, Caleb Chen and Chen, Lei},
  journal={arXiv preprint arXiv:2305.12178},
  year={2023}
}
```

## Acknowledgements

We would like to give credits to the authors of the following repos/codes that we refered to for our implementation.
- [FactorVAE](https://github.com/1Konny/FactorVAE) by [WonKwang Lee](https://github.com/1Konny/).
- [fullgrad-saliency](https://github.com/idiap/fullgrad-saliency) by [Idiap Research Institute](https://github.com/idiap).
- [FFVAE example](https://gist.github.com/ecreager/e152782ba8c459009ddcae680b4c7684) by [Elliot Creager](https://www.cs.toronto.edu/~creager/)
