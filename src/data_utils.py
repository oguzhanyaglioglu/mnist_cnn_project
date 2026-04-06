from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import Config

def build_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))# MNIST için önceden bilinen yaygın istatistikler
        # x' = (x - 0.1307) / (0.3081)
        # amaç normalizsyonu 0'a, standart sapmayı 1'e yaklaştırıp cnn'in daha kolay ve stabil öğrenmesini sağlamak
    ])

def build_dataloaders(cfg: Config):
    t = build_transforms()

    train_ds = datasets.MNIST(root=cfg.data_dir, train=True, download=True, transform=t)
    test_ds = datasets.MNIST(root=cfg.data_dir, train=False, download=True, transform=t)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader



