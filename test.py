import torch 
import argparse
from config import update_config
from model.unet import UNet_conditional
from training.gpu_func import MultiGPU
from utils import str2bool
from testing.ddpm_save import DDPM_SAVE
from testing.ddim_save import DDIM_SAVE

parser = argparse.ArgumentParser(description='Conditional-Diffusion-Model-CIFAR10')
parser.add_argument('--trained_model', default=None, type=str)
parser.add_argument('--time_dim', default=256, type=int)
parser.add_argument('--deep_conv', default=True, type=str2bool, nargs='?', const=True)
parser.add_argument('--condition', default=True, type=str2bool, nargs='?', const=True)
parser.add_argument('--cfg_w', default=0.1, type=float)
parser.add_argument('--timestep_num', default=1000, type=int)
parser.add_argument('--sampling_timesteps', default=100, type=int)
parser.add_argument('--beta_schedule', default='linear', type=str)
parser.add_argument('--ddim_eta', default=0, type=int)
parser.add_argument('--ddim_sampling', default=True, type=str2bool, nargs='?', const=True)
parser.add_argument('--save_images', default=False, type=str2bool, nargs='?', const=True)


def make_models_dict(model, epoch):
    model_dict = {'model': model.state_dict(), 'epoch': int(epoch)}
    return model_dict

def Test(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()

    if cfg.trained_model:
        in_channel = cfg.img_size[0]
        num_classes = cfg.num_classes
        time_dim = cfg.time_dim
        deep_conv = cfg.deep_conv

        # multi-gpu version
        if gpu_count > 1:
            model = MultiGPU(UNet_conditional(c_in=in_channel, c_out=in_channel, time_dim=time_dim, 
                                            num_classes=num_classes, deep_conv=deep_conv, device=device), dim=0).to(device)
        # single-gpu version
        else:
            model = UNet_conditional(c_in=in_channel, c_out=in_channel, time_dim=time_dim, 
                                            num_classes=num_classes, deep_conv=deep_conv, device=device).to(device)
            
        model.load_state_dict(torch.load(cfg.trained_model)['model'])
        model.eval()

        if cfg.ddim_sampling:
            DDIM_SAVE(cfg, model, device)
        else:
            DDPM_SAVE(cfg, model, device)
    else:
        print('no trained model!')


if __name__ == '__main__':
    args = parser.parse_args()
    test_cfg = update_config(args, mode='test')
    test_cfg.print_cfg()
    Test(test_cfg)