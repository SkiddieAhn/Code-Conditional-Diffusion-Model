import torch 
import copy
import torch.nn as nn
from model.unet import UNet_conditional
from model.ema import EMA
import torch.optim as optim
import wandb
from training.gpu_func import MultiGPU


def init_wandb(cfg):
    project_name = cfg.dataset 
    wandb.init(project=project_name)


def print_infor(cfg, dataloader):
    iter_size = len(dataloader) * cfg.epoch_size
    cfg.print_cfg() 

    print('\n===========================================================')
    print('Dataloader Ok!')
    print('-----------------------------------------------------------')
    print('[Data Size]:',len(dataloader.dataset))
    print('[Batch Size]:',cfg.batch_size)
    print('[One epoch]:',len(dataloader),'step   # (Data Size / Batch Size)')
    print('[Epoch & Iteration]:',cfg.epoch_size,'epoch &', iter_size,'step')
    print('-----------------------------------------------------------')
    print('===========================================================')


def def_models_ema(cfg, device):
    in_channel = cfg.img_size[0]
    num_classes = cfg.num_classes
    time_dim = cfg.time_dim
    deep_conv = cfg.deep_conv
    gpu_count = torch.cuda.device_count()

    # multi-gpu version
    if gpu_count > 1:
        model = MultiGPU(UNet_conditional(c_in=in_channel, c_out=in_channel, time_dim=time_dim, 
                                        num_classes=num_classes, deep_conv=deep_conv, device=device), dim=0).to(device)
    # single-gpu version
    else:
        model = UNet_conditional(c_in=in_channel, c_out=in_channel, time_dim=time_dim, 
                                        num_classes=num_classes, deep_conv=deep_conv, device=device).to(device)
    model = model.train()

    if cfg.ema:
        ema = EMA(0.995)
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    else:
        ema = ema_model = None
    return model, ema_model, ema


def def_loss():
    mse_loss = nn.MSELoss()
    return mse_loss


def def_optim_sch_scaler(cfg, dataloader, model):
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, eps=1e-5)
    sch = optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, steps_per_epoch=len(dataloader), epochs=cfg.epoch_size)
    scaler = torch.cuda.amp.GradScaler()
    return opt, sch, scaler


def load_model(cfg, model, opt):
    if cfg.resume:
        cur_epoch = torch.load(cfg.resume)['epoch']
        min_loss = torch.load(cfg.resume)['min_loss']
        model.load_state_dict(torch.load(cfg.resume)['model'])
        opt.load_state_dict(torch.load(cfg.resume)['optimizer'])

        print('\n===========================================================')
        print(f'Load Model Ok!')
        print('===========================================================')
        return cur_epoch, min_loss
    else:
        return 0, 1000