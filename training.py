"""
====================================================================================================
Package
====================================================================================================
"""
import os
import datetime
import numpy as np
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from unet import Pretrain  # 你自定義的 UNet 模型
from loss import combined_loss  # 包含 MSE, GDL 和 SSIM 損失的函數
from data_handler import MRCTDataset  # 數據集處理

"""
====================================================================================================
Global Constant
====================================================================================================
"""
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
DATA_PATH = "Data_2D"
MODEL_PATH = "saved_model"
RESULTS_PATH = "results"

"""
====================================================================================================
Training
====================================================================================================
"""
class Training():

    """
    ================================================================================================
    Initialize Critical Parameters
    ================================================================================================
    """
    def __init__(self):
        
        # training device
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        print('\n' + 'Training on: ' + str(self.device) + '\n')

        # time and tensorboard writer
        self.time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        print('\n' + 'Start From: ' + self.time + '\n')
        self.train_writer = None
        self.val_writer = None

        # model and optimizer
        self.init_model()
        self.init_optimizer()

    """
    ================================================================================================
    Initialize Model
    ================================================================================================
    """
    def init_model(self):
        print('\n' + 'Initializing Model' + '\n')
        self.model = Pretrain().to(self.device)

    """
    ================================================================================================
    Initialize Optimizer: Adam
    ================================================================================================
    """
    def init_optimizer(self):
        print('\n' + 'Initializing Optimizer' + '\n')
        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)

    """
    ================================================================================================
    Initialize TensorBoard
    ================================================================================================
    """
    def init_tensorboard(self):
        if (self.train_writer is None) or (self.val_writer is None):
            print('\n' + 'Initializing TensorBoard' + '\n')
            log_dir = os.path.join(RESULTS_PATH, 'Metrics', self.time)
            self.train_writer = SummaryWriter(log_dir=(log_dir + '_train'))
            self.val_writer = SummaryWriter(log_dir=(log_dir + '_val'))

    """
    ================================================================================================
    Initialize Data Loaders
    ================================================================================================
    """
    def init_data_loader(self, mode='train'):
        dataset = MRCTDataset(root=DATA_PATH, mode=mode)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=(mode == 'train'))
        return data_loader

    """
    ================================================================================================
    Load Model Parameters if Exists
    ================================================================================================
    """
    def load_model(self):
        if os.path.isfile(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH)
            print('\n' + 'Loading Checkpoint' + '\n')
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            print('\n' + 'Loading Model: ' + checkpoint['model_name'] + '\n')

    """
    ================================================================================================
    Main Training Function
    ================================================================================================
    """
    def main(self):

        # data loaders for training and validation
        train_loader = self.init_data_loader(mode='train')
        val_loader = self.init_data_loader(mode='val')

        # load model parameters if checkpoint exists
        self.load_model()

        # start training
        best_loss = float('inf')
        for epoch in range(1, EPOCHS + 1):
            print(f'\nEpoch {epoch}/{EPOCHS}')
            train_loss = self.train_epoch(epoch, train_loader)
            val_loss = self.validate_epoch(epoch, val_loader)

            # save metrics to tensorboard
            self.train_writer.add_scalar('train_loss', train_loss, epoch)
            self.val_writer.add_scalar('val_loss', val_loss, epoch)
            
            # save the best model
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_model(epoch, val_loss, best=True)

            # save current model
            self.save_model(epoch, val_loss, best=False)

        self.train_writer.close()
        self.val_writer.close()

    """
    ================================================================================================
    Training Loop
    ================================================================================================
    """
    def train_epoch(self, epoch, train_loader):
        self.model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, total=len(train_loader))
        for i, (mr, ct) in enumerate(progress):
            mr = mr.to(self.device)
            ct = ct.to(self.device)

            # forward pass
            preds = self.model(mr)
            loss = combined_loss(preds, ct)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress.set_description(f"Epoch [{epoch}/{EPOCHS}]")
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Average Training Loss: {avg_loss}")
        return avg_loss

    """
    ================================================================================================
    Validation Loop
    ================================================================================================
    """
    def validate_epoch(self, epoch, val_loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            progress = tqdm(val_loader, total=len(val_loader))
            for mr, ct in progress:
                mr = mr.to(self.device)
                ct = ct.to(self.device)

                preds = self.model(mr)
                loss = combined_loss(preds, ct)

                total_loss += loss.item()
                progress.set_postfix(val_loss=loss.item())

        avg_loss = total_loss / len(val_loader)
        print(f"Validation Loss: {avg_loss}")
        return avg_loss

    """
    ================================================================================================
    Save Model
    ================================================================================================
    """ 
    def save_model(self, epoch, loss, best=False):
        state = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epoch': epoch,
            'loss': loss,
        }
        if best:
            model_path = os.path.join(MODEL_PATH, f'best_model_epoch{epoch}.pth')
        else:
            model_path = os.path.join(MODEL_PATH, f'model_epoch{epoch}.pth')
        torch.save(state, model_path)


"""
====================================================================================================
Main Function
====================================================================================================
"""
if __name__ == '__main__':
    trainer = Training()
    trainer.main()
