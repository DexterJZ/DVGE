# Model Debiasing via Gradient-based Explanation on Representation

![framework diagram](https://github.com/DexterJZ/dexterjz.github.io/blob/main/images/dvge.png)

This repo is an implementation of the paper:

*Evaluating Adversarial Attacks on Driving Safety in Vision-Based Autonomous Vehicles* ([arXiv](https://arxiv.org/abs/2305.12178))

By [Jindi Zhang](https://dexterjz.github.io/), Luning Wang, Dan Su, Yongxiang Huang, Caleb Chen Cao, and Lei Chen

## Introduction

Machine learning systems produce biased results towards certain demographic groups, known as the fairness problem. Recent approaches to tackle this problem learn a latent code (i.e., representation) through disentangled representation learning and then discard the latent code dimensions correlated with sensitive attributes (e.g., gender). Nevertheless, these approaches may suffer from incomplete disentanglement and overlook proxy attributes (proxies for sensitive attributes) when processing real-world data, especially for unstructured data, causing performance degradation in fairness and loss of useful information for downstream tasks. In this paper, we propose a novel fairness framework that performs debiasing with regard to both sensitive attributes and proxy attributes, which boosts the prediction performance of downstream task models without complete disentanglement. The main idea is to, first, leverage gradient-based explanation to find two model focuses, 1) one focus for predicting sensitive attributes and 2) the other focus for predicting downstream task labels, and second, use them to perturb the latent code that guides the training of downstream task models towards fairness and utility goals. We show empirically that our framework works with both disentangled and non-disentangled representation learning methods and achieves better fairness-accuracy trade-off on unstructured and structured datasets than previous state-of-the-art approaches.

### Dependencies

Install the following dependencies.
```
python>=3.6.4
pytorch>=0.4.0
tqdm
torchvision
cv2
numpy
```

NVIDIA GPU is required.

### Data Preparation

The CelebA Dataset can be downloaded from [here](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

First, download img_align_celeba.zip and put it in ```DVGE/data/``` as below.
```
DVGE
└── data
    └── img_align_celeba.zip
```

Then, prepare the dataset by running
```
sh scripts/prepare_data.sh CelebA
```

Afterwards, the data directory structure would look like as follows.
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


## Citation

If you find our work useful, please consider citing our paper:

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
