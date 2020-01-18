"""Utilities."""
import numpy as np
import torchvision
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class Digits(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return len(self.X)


def get_digits(img_size=16, batch_size=1):
    trfs = transforms.Compose([transforms.Resize(size=(img_size, img_size)), transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST(root='~/data/', transform=trfs, download=True)
    X, y = list(), list()
    indices = np.random.permutation(60000)[:9298]
    for idx in indices:
        x_i, y_i = mnist[idx]  
        X.append(x_i[None, ...])
        y.append(y_i)
    X = torch.cat(X)
    y = torch.Tensor(y)
    X_train, y_train = X[:7291], y[:7291] 
    X_val, y_val = X[7291:], y[7291:]
    X_train = (2 * X_train) - 1
    X_val = (2 * X_val) - 1
    train_ds, val_ds = Digits(X_train, y_train), Digits(X_val, y_val)
    train_ld, val_ld = DataLoader(train_ds, batch_size=batch_size), DataLoader(val_ds, batch_size=batch_size)
    return train_ld, val_ld