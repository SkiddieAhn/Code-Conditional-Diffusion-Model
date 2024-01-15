import torch 
import wandb
from diffusion import * 
import cv2

def make_models_dict(model, opt, epoch, min_loss):
    model_dict = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
                  'epoch': int(epoch), 'min_loss': float(min_loss)}
    return model_dict


def update_best_model(dataset_name, model, opt, epoch, loss, min_loss):
    if loss < min_loss:
        min_loss = loss
        model_dict = make_models_dict(model, opt, epoch, min_loss)
        save_path = f'weights/{dataset_name}_best.pth'
        torch.save(model_dict, save_path)
        print('<<< Best model save at [%d] epoch! >>>' % (epoch+1))
    return min_loss


def save_image(file_path, model, alpha_bar_list, cfg_w, rsl, in_channels, condition, num_classes, timestep_num, sampling_timesteps, eta, device, is_wandb):
    model.eval()
    # sampling
    g_img = sampling(model=model, alpha_bar_list=alpha_bar_list, condition=condition, cfg_w=cfg_w, rsl=rsl, channels=in_channels, 
                     num=num_classes, total_timesteps=timestep_num, sampling_timesteps=sampling_timesteps, eta=eta, device=device)
    g_img = torch.clip(g_img, -1.0, 1.0)
    save_image = ((g_img + 1.0) / 2.0) * 255.

    # show with wandb
    if is_wandb:
        wandb.log({"sampled_images":     [wandb.Image(img.permute(1,2,0).cpu().detach().numpy()) for img in save_image]})

    # save image
    long_image = torch.cat([i for i in save_image.cpu().detach()], dim=-1).permute(1, 2, 0).cpu() 
    cv_image = cv2.cvtColor(long_image.numpy(), cv2.COLOR_BGR2RGB)
    cv_image = np.uint8(cv_image)
    cv2.imwrite(file_path, cv_image)
    model.train()