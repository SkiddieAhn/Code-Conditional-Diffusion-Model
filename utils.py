import torch
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def plot_images(images):
    plt.figure(figsize=(32, 32))
    long_image = torch.cat([(i+1.0)/2.0 for i in images.cpu()], dim=-1).permute(1, 2, 0).cpu() 
    plt.imshow(long_image)
    plt.axis('off')
    plt.show()

def save_images(images, path):
    long_image = torch.cat([((i+1.0)/2.0)*255.0 for i in images.cpu()], dim=-1).permute(1, 2, 0).cpu() 
    cv_image = cv2.cvtColor(long_image.numpy(), cv2.COLOR_BGR2RGB)
    cv_image = np.uint8(cv_image)
    cv2.imwrite(path, cv_image)