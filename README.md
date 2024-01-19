# Conditional Diffusion Model
Conditional DDPM/DDIM with Classifier Free Guidance.  
Most codes were obtained from the following GitHub page: [[Link]](https://github.com/tcapelle/Diffusion-Models-pytorch)

### 1. Denoising Diffusion Probabilistic Models [DDPM]
**Paper**: ```NIPS, 2020``` Denoising Diffusion Probabilistic Models [[Link](https://arxiv.org/pdf/2006.11239.pdf)]  
<img width="900" alt="1-1" src="https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/beebfccf-6bb2-401b-abfa-a2598d9ec2ea">
<img width="900" alt="1-2" src="https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/55f08dbb-49df-4c27-a0d4-ad0e94b86b85">

### 2. Denoising Diffusion Implicit Models [DDIM]
**Paper**: ```ICLR, 2021``` DENOISING DIFFUSION IMPLICIT MODELS [[Link](https://arxiv.org/pdf/2010.02502.pdf)]  
![image](https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/27c066ed-1756-4a90-87d1-38a8c5e76516)
<img width="900" alt="2-2" src="https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/b060fce0-f6a6-48df-9d60-89ba2860717b">

### 3. Classifier Free Guidance [CFG]
**Paper**: ```NIPSW, 2021``` CLASSIFIER-FREE DIFFUSION GUIDANCE [[Link](https://arxiv.org/pdf/2207.12598.pdf)]  
  
<img width="900" alt="3-1" src="https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/5a803333-95f1-4e6c-b29d-6e504c96f19e">
<img width="900" alt="3-2" src="https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/093398ab-d90f-4951-802b-81a611638998">

## Training
- Before training, log in to ```wandb``` on your PC.
- Please check ```train.py``` and ```config.py``` files and train the model.
```Shell
# default option for training
python train.py
# unconditional training
python train.py --condition=False
# don't use wandb
python train.py --wandb=False
# change 'epoch_size'.
python train.py --epoch_size=2000
# change 'batch size'.
python train.py --batch_size=128
# Continue training with latest model
python train.py --resume=cifar10_2000
```

## Testing
- Please check ```test.py``` and ```config.py``` files and evaluate the model.
- When generation is completed, the **video** is automatically saved in the ```contents``` directory.
```Shell
# recommended option for testing
python test.py --trained_model=cifar10_2000 --ddim_sampling=False --cfg_w=2.0
# unconditional testing
python test.py --trained_model=cifar10_2000 --condition=False
# save images per timestep
python test.py --trained_model=cifar10_2000 --save_images=True
```

## Dataset & Model
- I resized the ```CIFAR-10-64``` dataset to **32x32** for training.
- Unzip the dataset in the ```data``` directory.
- I trained the model with the settings listed below.
  
|   CIFAR10-64     | Model  |
|:--------------:|:-----------:|
|[kaggle](https://www.kaggle.com/datasets/joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution)|[Google Drive](https://drive.google.com/file/d/1DtRLa_zu5fU2Tj6X9FTDiv0FxEWSxopc/view?usp=drive_link)|

```Shell
# diffusion model config
share_config['time_dim'] = 256
share_config['condition'] = True
share_config['timestep_num'] = 1000
share_config['sampling_timesteps'] = 100
share_config['beta_schedule'] = 'linear'
share_config['ddim_eta'] = 0
share_config['deep_conv'] = True
share_config['cfg_p_uncond'] = 0.1
share_config['cfg_w'] = 0.1

# train config
share_config['batch_size'] = 128
share_config['epoch_size'] = 2000
share_config['lr'] = 1e-4
share_config['ema'] = True
```


## Results (condition: label)
|             DDPM Sampling          |w = 1.0 <br>(low fidelity, high diversity)   |w = 2.0 <br>(middle fidelity, middle diversity)  |w = 4.0 <br>(high fidelity, low diversity)  |
|:--------------:|:-----------:|:-----------:|:-----------:|
| **Img** |![1 0_image](https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/776905b4-4012-4312-94cf-5165eebf5ef3)|![2 0_image](https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/0f70b4ee-4788-4368-80e6-b7f9e6324e81)|![4 0_image](https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/af0ccb20-687d-458e-9dd0-ba8441cd52e7)|



 
