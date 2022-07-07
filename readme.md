# Benchmarking 3D Face De-identification with Preserving Facial Attributes [1]


This package provides the implementation codes for reproducing the exact experimental results presented in this paper [1].

This package is tested using a computer system: with Ubuntu 18.04.6 LTS and Pytorch 1.11.0.


## Detailed steps:
### 1. Download BUFE3D Dataset and StarGAN pre-trained models.
- Download the BUFE3D dataset from the original weblink [2].
- Download the pre-trained StarGAN_celeba_128 models (200000-D.ckpt and 200000-G.ckpt) from the original weblink [3] and put them in "GAN/".

### 2. 2D/3D Image Pre-processing
- Edit the variable 'datapath' in "matlab/preprocess_images.m" according to the directories of the downloaded dataset.
- Execute "matlab/preprocess_images.m". This step generates the "Data" folder.

### 3. Classification, De-identification and Evaluation
- Execute "./main.sh"
- 3a.Train/Test Classification Models for Biometric, Expression, Gender and Ethnicity Recognition. This step generates the "task" folder.
- 3b.Train the De-identification Models. The AE/GAN backbone is adopted from [4] and [5] respectively. This step generates the "AE/constraint-2D", "AE/constraint-Depth", "GAN/constraint-2D", "GAN/constraint-Depth" folders.
- 3c.Test the De-identification Performance with the Classification Models.



## References
- [1] Kevin H. M. Cheng, Zitong Yu, Haoyu Chen, and Guoying Zhao. Benchmarking 3D Face De-identification with Preserving Facial Attributes. In International Conference on Image Processing (ICIP), 2022.
- [2] https://www.cs.binghamton.edu/~lijun/Research/3DFE/3DFE_Analysis.html
- [3] https://github.com/yunjey/stargan
- [4] V. Mirjalili, S. Raschka, A. Namboodiri and A. Ross. Semi-adversarial networks: Convolutional autoencoders for imparting privacy to face images. International Conference on Biometrics (ICB), 2018.
- [5] Y. Choi, M. Choi, M. Kim, J.W. Ha, S. Kim and J. Choo. Stargan: Unified generative adversarial networks for multi-domain image-to-image translation. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.



