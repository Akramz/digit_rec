"""
This file defines the core research implementation
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from .utils import get_digits


class MNISTRecognizer(pl.LightningModule):
    
    def __init__(self):
        super(MNISTRecognizer, self).__init__()
        self.h1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=2, padding=2)
        self.h2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=2, padding=2)
        self.flatten = nn.Flatten()
        self.h3 = nn.Linear(in_features=192, out_features=30)
        self.output = nn.Linear(in_features=30, out_features=10)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        y = self.tanh(self.h1(x))
        y = self.tanh(self.h2(y))
        y = self.flatten(y)
        y = self.tanh(self.h3(y))
        y_hat = self.output(y)
        return y_hat
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = F.mse_loss(y_hat, F.one_hot(y.type(torch.LongTensor), 10).view(-1, 10).type(torch.FloatTensor))
        tb_logs = {
            'train_mse': loss,
            'train_log_mse': torch.log(loss), 
            'error_rate': (y_hat.max(1)[1] != y.type(torch.LongTensor)).type(torch.FloatTensor)
        }
        return {'loss': loss, 'log': tb_logs}
    
    def validation_step(self, batch, batch_idx):
        """Not shown until epoch ends"""
        X, y = batch
        y_hat = self.forward(X)
        loss = F.mse_loss(y_hat, F.one_hot(y.type(torch.LongTensor), 10).view(-1, 10).type(torch.FloatTensor))
        logs = {
            'val_loss': loss,
            'val_log_mse': torch.log(loss), 
            'val_error_rate': (y_hat.max(1)[1] != y.type(torch.LongTensor)).type(torch.FloatTensor)
        }
        return logs
    
    def validation_end(self, outputs):
        avg_val_mse = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_log_mse = torch.stack([x['val_log_mse'] for x in outputs]).mean()
        avg_val_error_rate = torch.stack([x['val_error_rate'] for x in outputs]).mean()
        tb_logs = {'val_loss': avg_val_mse, 
                   'val_log_mse': avg_val_log_mse, 
                   'val_error_rate': avg_val_error_rate}
        return {'val_mse': avg_val_mse, 'log': tb_logs}
    
    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.01)
    
    @pl.data_loader
    def train_dataloader(self):
        return get_digits(batch_size=1)[0]
    
    @pl.data_loader
    def val_dataloader(self):
        return get_digits(batch_size=1)[1]

