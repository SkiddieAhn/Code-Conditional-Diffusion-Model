import torch
import argparse
from dataset import make_loader
from config import update_config
from training.train_pre_func import * 
from training.train_func import Training
from utils import str2bool

def main():
    parser = argparse.ArgumentParser(description='Conditional-Diffusion-Model-CIFAR10')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epoch_size', default=2000, type=int)
    parser.add_argument('--resume', default=None, type=str)
    parser.add_argument('--save_interval', default=50, type=int)
    parser.add_argument('--visual_log_interval', default=10, type=int)
    parser.add_argument('--ema', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--cfg_p_uncond', default=0.1, type=float)
    parser.add_argument('--cfg_w', default=0.1, type=float)
    parser.add_argument('--deep_conv', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--time_dim', default=256, type=int)
    parser.add_argument('--condition', default=True, type=str2bool, nargs='?', const=True)
    parser.add_argument('--timestep_num', default=1000, type=int)
    parser.add_argument('--sampling_timesteps', default=100, type=int)
    parser.add_argument('--beta_schedule', default='linear', type=str)
    parser.add_argument('--ddim_eta', default=0, type=int)
    parser.add_argument('--wandb', default=True, type=str2bool, nargs='?', const=True)

    args = parser.parse_args()
    train_cfg = update_config(args, mode='train')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    print(device)

    '''
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Pre-work for Training
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''

    # init wandb
    if train_cfg.wandb:
        init_wandb(train_cfg)

    # get dataloader
    train_loader = make_loader(train_cfg.data_root, batch_size=train_cfg.batch_size, rsl=train_cfg.img_size[1], mode='train')
    val_loader = make_loader(train_cfg.data_root, batch_size=train_cfg.batch_size, rsl=train_cfg.img_size[1], mode='test')
    print_infor(cfg=train_cfg, dataloader=train_loader)

    # define models, ema
    model, ema_model, ema = def_models_ema(cfg=train_cfg, device=device)

    # define loss
    loss = def_loss()

    # define optimizer and scheduler and scaler
    opt, sch, scaler = def_optim_sch_scaler(train_cfg, train_loader, model)

    # load models
    cur_epoch, min_loss = load_model(train_cfg, model, opt)

    '''
    <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    Training
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    '''

    # train
    Training(train_cfg, train_loader, val_loader, model, ema_model, ema, loss, opt, sch, scaler, cur_epoch, min_loss, device)


if __name__=="__main__":
    main()