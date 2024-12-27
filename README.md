# Paper title: A deep learning model for predicting tyrosine kinase inhibitor response from histology in gastrointestinal stromal tumor.

## Authers:
### Xue Kong, Jun Shi, Dongdong Sun, Lanqing Cheng, Can Wu, Zhiguo Jiang, Yushan Zheng, Wei Wang, Haibo Wu.


Xue Kong, Jun Shi, Dongdong Sun contribute equally.

Haibo Wu, Wei Wang, and Yushan Zheng are the corresponding author.


### E-mail: 

- Haibo Wu: wuhaibo@ustc.edu.cn

- Wei Wang: weiwang@hmfl.ac.cn

- Yushan Zheng: yszheng@buaa.edu.cn

## Framework:
![framework](images/framework.jpg)

## Environment
![python](https://img.shields.io/badge/python-3.8-blue)
![torch](https://img.shields.io/badge/torch-1.8%2Bcu111-red)
![torchvision](https://img.shields.io/badge/torchvision-0.9.1+cu111-purple)
![numpy](https://img.shields.io/badge/numpy-1.22.3-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.0-orange)
![opencv-python](https://img.shields.io/badge/opencv--python-4.5.5.62-pink)
![einops](https://img.shields.io/badge/einops-0.6.6-brown)
![matplotlib](https://img.shields.io/badge/matplotlib-3.5.1-yellow)

## install 3rd library
```shell
pip install -r requirements.txt
```

## Feature format
```none
- feature_dir
  - slide-1_feature.pth
  - slide-2_feature.pth
  ......
  - slide-n_feature.pth
 xxx_feature.pth -> shape: number_patches, feaure_dim
```

## Training feature extractor (LACL)
### 1. Download [LACL](https://github.com/junl21/lacl)
### 2. Patch sampling
### 3. Training
```shell
python lacl_train.py
```

## Traning wsi classification model

## Citation
....