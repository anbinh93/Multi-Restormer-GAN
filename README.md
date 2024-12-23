The implementation of-GAN.

### Dependencies
- python 3.8.10
- torch 1.10.0
- numpy 1.21.4
- pillow 8.4.0
- torchvision 0.11.1
- scikit-image 0.19.3

### Datasets
in server

### Usage
#### Prepare dataset:


#### Train:
`python train.py --data_source /path/to/dataset --experiment your_experiment_name

#### Test:
`python test.py --data_source /path/to/dataset --experiment your_experiment_name --model_name your_model_name --save_image

### PSNR&SSIM
```
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

img1 = cv2.imread('img path1')
img2 = cv2.imread('img path2')
psnr = peak_signal_noise_ratio(img1, img2)
ssim = structural_similarity(img1, img2, multichannel=True, gaussian_weights=True, use_sample_covariance=False)
```
