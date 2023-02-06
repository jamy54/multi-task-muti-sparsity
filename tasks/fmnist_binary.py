import torch
from Models.Models import bNet, Net
from utilites import Utility, DataLoader
from Tester import Tester
from Trainer import  Trainer
from pruner.Pruner import Pruner
import os
from tasks.fmnist_single import FMNIST

class FMNIST_binary:
    def __init__(self, dataloader, utility):
        self.dataloader = dataloader
        self.utility = utility
        self.FMNIST = FMNIST(dataloader, utility)
        self.bcNet = self.load_model()

    def load_model(self):
        return bNet(my_pretrained_model = self.FMNIST.load_trained_model())

    def get_parent_model(self):
        return self.FMNIST.cNet

    def load_trained_model(self):
        if os.path.exists('./saved_models/FMNIST_binary.pth'):
            self.bcNet.load_state_dict(torch.load('./saved_models/FMNIST_binary.pth'))
        else:
            self.utility.freeze_all_parameters(self.bcNet)
            self.bcNet.fc4.weight.requires_grad = True
            print('only last layer')
            Trainer(model=self.bcNet, train_loader=self.dataloader.train_loader, valid_loader=self.dataloader.valid_loader).train_model(10)
            Tester(model=self.bcNet, test_loader=self.dataloader.test_loader, batch_size=self.dataloader.batch_size).test_model()
            self.utility.save_model(self.bcNet,path='./saved_models/FMNIST_binary.pth')
        return self.bcNet


if __name__ == '__main__':
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle', 'boot']