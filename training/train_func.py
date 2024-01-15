import torch 
from training.train_ing_func import make_models_dict, update_best_model, save_image
from fastprogress import progress_bar
import wandb
from training.gpu_func import showGPU
from diffusion import * 


class Training():
    def __init__(self, cfg, train_loader, val_loader, model, ema_model, ema, loss, opt, sch, scaler, cur_epoch, min_loss, device):
        # setting parameter
        self.dataset_name = cfg.dataset
        self.save_interval = cfg.save_interval
        self.visual_log_interval = cfg.visual_log_interval
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.loss = loss
        self.opt = opt
        self.sch = sch
        self.scaler = scaler
        self.start_epoch = cur_epoch
        self.end_epoch = cfg.epoch_size
        self.min_loss = min_loss
        self.device = device
        self.wandb = cfg.wandb
        self.showgpu = True

        # diffusion model parameters
        self.condition = cfg.condition
        self.num_classes = cfg.num_classes
        self.cfg_p_uncond = cfg.cfg_p_uncond
        self.cfg_w = cfg.cfg_w
        self.timestep_num = cfg.timestep_num
        self.sampling_timesteps = cfg.sampling_timesteps
        self.ddim_eta = cfg.ddim_eta
        self.ema_model = ema_model
        self.ema = ema

        if cfg.beta_schedule == 'linear':
            self.beta_list = beta_schedule_linear(self.timestep_num)
        elif cfg.beta_schedule == 'cosine':
            self.beta_list = beta_schedule_cosine(self.timestep_num)
        else:
            self.beta_list = beta_schedule_quadratic(self.timestep_num)
            
        self.alpha_list = get_alpha_list(self.beta_list)
        self.alpha_bar_list = get_alpha_bar_list(self.alpha_list)
        self.sqrt_alpha_bar_list = torch.sqrt(self.alpha_bar_list)
        self.sqrt_one_minus_alpha_bar_list = torch.sqrt(1.0 - self.alpha_bar_list)
        self.one_div_sqrt_alpha_list = 1.0 / torch.sqrt(self.alpha_list)

        # setting image
        self.in_channels = cfg.img_size[0]
        self.rsl = cfg.img_size[1]

        # training start
        self.fit()


    def one_forward(self, inputs, data_size, t, c):
        # forward process
        esp = torch.randn_like(inputs).to(self.device)
        noise_img = torch.zeros((data_size, self.in_channels, self.rsl, self.rsl), dtype=torch.float32).to(self.device)
        for i in range(data_size):
            sqrt_alpha_bar = self.sqrt_alpha_bar_list[t[i]]
            one_minus_sqrt_alpha_bar = self.sqrt_one_minus_alpha_bar_list[t[i]]
            noise_img[i] = sqrt_alpha_bar * inputs[i] + one_minus_sqrt_alpha_bar * esp[i]

        # reverse process 
        output = self.model(noise_img, t, c)
        loss = self.loss(output, esp)
        return loss


    def one_epoch(self, running_loss):
        pbar = progress_bar(self.train_loader, total=len(self.train_loader))
        for _, data in enumerate(pbar):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            data_size = len(inputs)

            # get time, condition
            t = torch.randint(0, self.timestep_num, size=(data_size,), device=inputs.device).long()
            if self.condition:
                c = labels
            else:
                c = None

            # Classifier Free Guidance 
            if np.random.random() < self.cfg_p_uncond:
                c = None

            # get loss
            loss = self.one_forward(inputs, data_size, t, c)
            pbar.comment = f"MSE={loss.item():2.3f}" 
            if self.wandb:
                wandb.log({"train_mse_step": loss.item(),
                            "learning_rate": self.sch.get_last_lr()[0]})

            # optimization
            self.opt.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt)
            self.scaler.update()
            if self.ema != None:
                self.ema.step_ema(self.ema_model, self.model) # EMA for stable training
            self.sch.step()
            running_loss += loss.item()

            # show gpu only once
            if self.showgpu:
                showGPU()
                self.showgpu = False

        avg_loss = np.round(running_loss / len(self.train_loader), 4)
        if self.wandb:
            wandb.log({"train_mse_epoch": avg_loss})
        return avg_loss
    

    def one_val(self, val_running_loss):
        self.model.eval()
        for data in progress_bar(self.val_loader, total=len(self.val_loader)):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            data_size = len(inputs)

            # get time, condition
            t = torch.randint(0, self.timestep_num, size=(data_size,), device=inputs.device).long()
            if self.condition:
                c = labels
            else:
                c = None

            # get loss
            loss = self.one_forward(inputs, data_size, t, c)
            val_running_loss += loss.item()

        avg_loss = np.round(val_running_loss / len(self.val_loader), 4)
        if self.wandb:
            wandb.log({"val_mse_epoch": avg_loss})
        self.model.train()
        return avg_loss


    def fit(self):
        print('\n===========================================================')
        print('Training Start!')
        print('===========================================================')

        for epoch in progress_bar(range(self.start_epoch, self.end_epoch), total=(self.end_epoch-self.start_epoch)):
            running_loss = 0.0
            val_running_loss = 0.0

            # one epoch
            print(f"[Training] {epoch+1}/{self.end_epoch}")
            avg_loss = self.one_epoch(running_loss)
            print(f"Epoch {epoch+1}/{self.end_epoch}, Loss: {avg_loss}")

            # one validation
            print(f"[Validation] {epoch+1}/{self.end_epoch}")
            val_avg_loss = self.one_val(val_running_loss)
            print(f"Epoch {epoch+1}/{self.end_epoch}, Valid Loss: {val_avg_loss}")

            # early stopping
            self.min_loss = update_best_model(self.dataset_name, self.model, self.opt, epoch, val_avg_loss, self.min_loss)

            # save model
            if (epoch+1) % self.save_interval == 0:
                model_dict = make_models_dict(self.model, self.opt, epoch, self.min_loss)
                save_path = f'weights/{self.dataset_name}_{epoch+1}.pth'
                torch.save(model_dict, save_path)
                print('<< model save at [%d] epoch! >>' % (epoch+1))

            # save image
            if (epoch+1) % self.visual_log_interval == 0:
                file_path = f'contents/{self.dataset_name}_{epoch+1}.jpg'
                save_image(file_path, self.model, self.alpha_bar_list, self.cfg_w, self.rsl, self.in_channels, self.condition, 
                           self.num_classes, self.timestep_num, self.sampling_timesteps, self.ddim_eta, self.device, self.wandb)
 