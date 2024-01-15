# Conditional DM with CFG
Conditional DDPM/DDIM with Classifier Free Guidance.  
Most codes were obtained from the following GitHub page: [[Link]](https://github.com/tcapelle/Diffusion-Models-pytorch)

### Classifier Free Guidance [CFG]
![image](https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/5a803333-95f1-4e6c-b29d-6e504c96f19e)
![image](https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/093398ab-d90f-4951-802b-81a611638998)

## Training
- Before training, log in to ```wandb``` on your PC.
- Please check ```train.py``` and ```config.py``` files and train the model.
```Shell
# default option for training
python train.py
# don't use wandb
python train.py --wandb=False
# change 'epoch_size'.
python train.py --epoch_size=1000
# change 'batch size'.
python train.py --batch_size=128
# Continue training with latest model
python train.py --resume=cifar10_1000
```

## Testing
- Please check ```test.py``` and ```config.py``` files and evaluate the model.
- When generation is completed, the **video** is automatically saved in the ```contents``` directory.
```Shell
# recommended option for testing
python --trained_model=cifar10_1000 --ddim_sampling=False --cfg_w=2.0
# save images per timestep
python --trained_model=cifar10_1000 --ddim_sampling=True --save_images=True
```

## Dataset & Model
- I resized the ```CIFAR-10-64``` dataset to **32x32** for training.
- Unzip the dataset in the ```data``` directory.
- I trained the model with the settings listed below.
  
|   CIFAR10-64     | Model  |
|:--------------:|:-----------:|
|[kaggle](https://www.kaggle.com/datasets/joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution)|[Google Drive](https://drive.google.com/file/d/15_JKss-bW9m6ihwEYaIU_DMSFNlDXKZV/view?usp=sharing)|

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
share_config['epoch_size'] = 1000
share_config['lr'] = 1e-4
share_config['ema'] = True
```


## Results (condition: label)
|             DDPM Sampling          |w = 1.0 <br>(low fidelity, high diversity)   |w = 2.0 <br>(middle fidelity, middle diversity)  |w = 4.0 <br>(high fidelity, low diversity)  |
|:--------------:|:-----------:|:-----------:|:-----------:|
| **Img** |![1 0_image](https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/776905b4-4012-4312-94cf-5165eebf5ef3)|![2 0_image](https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/0f70b4ee-4788-4368-80e6-b7f9e6324e81)|![4 0_image](https://github.com/SkiddieAhn/Study-Diffusion-Model/assets/52392658/af0ccb20-687d-458e-9dd0-ba8441cd52e7)|



 
