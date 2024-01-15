import os 
import glob

if not os.path.exists(f"contents"):
    os.makedirs(f"contents")
if not os.path.exists(f"contents/ddpm"):
    os.makedirs(f"contents/ddpm")
if not os.path.exists(f"contents/ddim"):
    os.makedirs(f"contents/ddim")
if not os.path.exists(f"weights"):
    os.makedirs(f"weights")

share_config = {'dataset': 'cifar10',
                'img_size': (3, 32, 32),
                'num_classes' : 10, # 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
                'data_root': 'data/'}  # remember the final '/'

class dict2class:
    def __init__(self, config, mode):
        self.mode = mode
        for k, v in config.items():
            self.__setattr__(k, v)

    def print_cfg(self):
        print('\n' + '-' * 30 + f'{self.mode} cfg' + '-' * 30)
        for k, v in vars(self).items():
            print(f'{k}: {v}')
        print()


def update_config(args=None, mode='train'):
    # diffusion model share config
    share_config['time_dim'] = args.time_dim
    share_config['condition'] = args.condition
    share_config['timestep_num'] = args.timestep_num
    share_config['sampling_timesteps'] = args.sampling_timesteps
    share_config['beta_schedule'] = args.beta_schedule
    share_config['ddim_eta'] = args.ddim_eta
    share_config['deep_conv'] = args.deep_conv
    share_config['cfg_w'] = args.cfg_w

    if mode == 'train':
        share_config['batch_size'] = args.batch_size
        share_config['epoch_size'] = args.epoch_size
        share_config['lr'] = 1e-4
        share_config['resume'] = glob.glob(f'weights/{args.resume}*')[0] if args.resume else None
        share_config['save_interval'] = args.save_interval
        share_config['visual_log_interval'] = args.visual_log_interval
        share_config['ema'] = args.ema
        share_config['wandb'] = args.wandb
        share_config['cfg_p_uncond'] = args.cfg_p_uncond

    elif mode == 'test':
        share_config['trained_model'] = glob.glob(f'weights/{args.trained_model}*')[0] if args.trained_model else None
        share_config['ddim_sampling'] = args.ddim_sampling
        share_config['save_images'] = args.save_images

    return dict2class(share_config, mode)  # change dict keys to class attributes