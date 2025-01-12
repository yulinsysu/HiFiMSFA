## HiFiMSFA

## Introduction
This repository is the official PyTorch implementation of HiFiMSFA: Robust and High-Fidelity Image Watermarking using Attention Augmented Deep Network. The paper proposes a deep image watermarking framework with multi-scale salient feature attention and adaptive squared Hinge loss.

## Train
If you need to train HiFiMSFA, you should use commond line as following.

      python train.py
Requirements: torch==1.10.1+cu111 torchvision==0.11.2+cu111 kornia==0.6.8

## Test
The pre-trained model of HiFiMSFA is avaliable at the pretrain floder. you can test it by command line as following.

      python test.py

## License
The models are free for non-commercial and scientific research purpose. Please mail us for further licensing terms.
