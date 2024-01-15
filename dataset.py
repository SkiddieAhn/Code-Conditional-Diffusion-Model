import os
import torch
import torchvision
import torchvision.transforms as transforms


def make_loader(path, mode='train', batch_size=128, rsl=32):
    transform = transforms.Compose([transforms.Resize((rsl, rsl)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    dataset = torchvision.datasets.ImageFolder(os.path.join(path, mode), transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader