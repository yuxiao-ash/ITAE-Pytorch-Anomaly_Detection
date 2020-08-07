# ITAE-Pytorch-Anomaly_Detection
An **unofficial reproduced** implementation of 'Inverse-Transform AutoEncoder for Anomaly Detection', paper see https://arxiv.org/abs/1911.10676

## requirements
* python3
* pytorch-1.0 or higher version
* mmcv
* torchvision
* tqdm

## how to use
* run   *python main.py config/config_mnist*   to train and test on mnist dataset
* run   *python main.py config/config_cifar*   to train and test on cifar dataset

## others
This implementation refers to https://github.com/samet-akcay/ganomaly and https://github.com/milesial/Pytorch-UNet.

## note
To get the score in paper, please refer to [these suggestions from chaoqinhuang](https://github.com/FishSmile-syx/ITAE-Pytorch-Anomaly_Detection/issues/1#issuecomment-669802108) .
