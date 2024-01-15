import torch
import cv2
import numpy as np
from diffusion import * 

def DDPM_SAVE(cfg, model, device):
    channels = cfg.img_size[0]
    rsl = cfg.img_size[1]
    out_rsl = rsl * 10
    condition_num = cfg.num_classes
    data_size = condition_num ** 2
    cfg_w = cfg.cfg_w
    save_images = cfg.save_images

    # setting condition
    if cfg.condition:
        c = [(i%condition_num) for i in range(condition_num**2)]
        c = np.array(c)
        c = torch.tensor(c, dtype=torch.long).to(device)
    else:
        c = None
        cfg_w = 0 
    
    # diffusion model parameter
    timestep_num = cfg.timestep_num
    if cfg.beta_schedule == 'linear':
        beta_list = beta_schedule_linear(timestep_num)
    elif cfg.beta_schedule == 'cosine':
        beta_list = beta_schedule_cosine(timestep_num)
    else:
        beta_list = beta_schedule_quadratic(timestep_num)
    alpha_list = get_alpha_list(beta_list)
    alpha_bar_list = get_alpha_bar_list(alpha_list)
    sqrt_one_minus_alpha_bar_list = torch.sqrt(1.0 - alpha_bar_list)
    one_div_sqrt_alpha_list = 1.0 / torch.sqrt(alpha_list)


    with torch.no_grad():
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('./contents/ddpm/ddpm_cifar10.avi', fourcc, 20, (out_rsl, out_rsl))
        out_img = np.zeros((out_rsl,out_rsl,channels)) # [320, 320, 3]

        # get first image (noise)
        g_img = torch.randn((data_size, channels, rsl, rsl), dtype=torch.float32).to(device) # [100, 3, 32, 32]
        
        # Inference 
        for t in range(timestep_num - 1, -1, -1): # 999 ~ 0 => 1000 step to 1 step 
            in_t = torch.full((data_size,), t, dtype=torch.long).to(device)
            in_c = c

            # model forward with CFG
            pred_noise = model(g_img, in_t, in_c)
            if cfg_w > 0:
                uncond_pred_noise = model(g_img, in_t, None)
                pred_noise = (1+cfg_w)*pred_noise - cfg_w*uncond_pred_noise

            # get t-th images 
            if t > 0:
                z = torch.randn((data_size, channels, rsl, rsl), dtype=torch.float32).to(device) # [100, 3, 32, 32]
            else:
                z = torch.zeros_like(g_img)
            mu = one_div_sqrt_alpha_list[t] * (g_img - (beta_list[t] / sqrt_one_minus_alpha_bar_list[t]) * pred_noise)
            g_img = mu + (torch.sqrt(beta_list[t])*z)

            # make numpy images
            g_np = ((torch.clip(g_img, -1.0, 1.0) + 1.0) / 2.0) * 255. # [100, 3, 32, 32]
            g_np = g_np.permute(0,2,3,1).cpu().detach().numpy() # [100, 32, 32, 3]
            
            # reshape numpy images
            rs_img = np.resize(g_np, (condition_num, condition_num, rsl, rsl, channels)) # [10, 10, 32, 32, 3]

            # make numpy images to grid image
            for i in range(condition_num):
                for j in range(condition_num):
                    for cnl in range(channels):
                        out_img[i * rsl:i * rsl +rsl, j * rsl:j * rsl +rsl, cnl] = rs_img[i, j, :, :, cnl]

            # make video 
            out_img = np.uint8(out_img)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
            out.write(out_img)

            # save grid image per 100 time step 
            if ((t+1) < 1000) and ((t+1) % 100 == 0):
                print(f'{t+1} step')
                if save_images:
                    print(f'{t+1} step')
                    file_path = f"./contents/ddpm/{t+1}_step_img.png"
                    cv2.imwrite(file_path, out_img)

        # finish
        file_path = f"./contents/ddpm/last_img.png"
        cv2.imwrite(file_path, out_img)
        out.release()
        print('save video_ddpm ok!')