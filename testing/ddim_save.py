import torch
import cv2
import numpy as np
from diffusion import * 

def DDIM_SAVE(cfg, model, device):
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
    eta = cfg.ddim_eta
    timestep_num = cfg.timestep_num
    sampling_timesteps = cfg.sampling_timesteps
    if cfg.beta_schedule == 'linear':
        beta_list = beta_schedule_linear(timestep_num)
    elif cfg.beta_schedule == 'cosine':
        beta_list = beta_schedule_cosine(timestep_num)
    else:
        beta_list = beta_schedule_quadratic(timestep_num)
    alpha_list = get_alpha_list(beta_list)
    alpha_bar_list = get_alpha_bar_list(alpha_list)


    with torch.no_grad():
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter('./contents/ddim/ddim_cifar10.avi', fourcc, 20, (out_rsl, out_rsl))
        out_img = np.zeros((out_rsl,out_rsl,channels)) # [320, 320, 3]

        # get first image (noise)
        g_img = torch.randn((data_size, channels, rsl, rsl), dtype=torch.float32).to(device) # [100, 3, 32, 32]

        # get time_pairs 
        times = torch.linspace(-1, timestep_num - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        
        # Inference
        for time, time_next in time_pairs:   
            in_t = torch.full((data_size,), time, dtype=torch.long).to(device)
            in_c = c

            # model forward with CFG
            pred_noise = model(g_img, in_t, in_c)
            if cfg_w > 0:
                uncond_pred_noise = model(g_img, in_t, None)
                pred_noise = (1+cfg_w)*pred_noise - cfg_w*uncond_pred_noise

            # set alpha
            alpha = alpha_bar_list[time]
            alpha_next = alpha_bar_list[time_next]
            sqrt_alpha = torch.sqrt(alpha)
            sqrt_alpha_next = torch.sqrt(alpha_next)
        
            # predict x0
            numerator = g_img - (torch.sqrt(1-alpha) * pred_noise)
            pred_x0 = numerator / sqrt_alpha

            if time_next < 0:
                g_img = pred_x0
                
            else:        
                # predict direction_pointing 
                sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha) * (1 - alpha / alpha_next))
                variance = sigma ** 2 
                direction_pointing = torch.sqrt(1-alpha_next-variance) * pred_noise

                # generate next image 
                g_img = (sqrt_alpha_next * pred_x0) + direction_pointing

            # make numpy images
            g_np = ((torch.clip(g_img, -1.0, 1.0) + 1.0) / 2.0) * 255. # [100, 3, 32, 32]
            g_np = g_np.permute(0,2,3,1).cpu().detach().numpy() # [100, 32, 32, 3]
        
            # reshape numpy images
            rs_img = np.resize(g_np, (condition_num, condition_num, rsl, rsl, 3)) # [10, 10, 32, 32, 3]

            # make numpy images to grid image
            for i in range(condition_num):
                for j in range(condition_num):
                    for cnl in range(channels):
                        out_img[i * rsl:i * rsl +rsl, j * rsl:j * rsl +rsl, cnl] = rs_img[i, j, :, :, cnl]

            # make video 
            out_img = np.uint8(out_img)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
            out.write(out_img)

            print(f'{time+1} step')
            # save grid image per sampling time step 
            if save_images:
                file_path = f"./contents/ddim/{time+1}_step_img.png"
                cv2.imwrite(file_path, out_img)

        # finish
        file_path = f"./contents/ddim/last_img.png"
        cv2.imwrite(file_path, out_img)
        out.release()
        print('save video_ddim ok!')