
from Models.Models import  Net
from utilites import Utility, DataLoader
from Trainer import Trainer
from Tester import Tester
import os
import torch


class FMNIST:
    def __init__(self, dataloader, utility):
        self.dataloader = dataloader
        self.utility = utility
        self.cNet = self.load_model()

    def cifar_10(self):
        pass

    def load_model(self):
        return Net()

    def load_trained_model(self):
        if os.path.exists('./saved_models/FMNIST_single.pth'):
            self.cNet.load_state_dict(torch.load('./saved_models/FMNIST_single.pth'))
        else:
            Trainer(model=self.cNet, train_loader=self.dataloader.train_loader, valid_loader=self.dataloader.valid_loader).train_model(30, isBinary=False)
            Tester(model=self.cNet, test_loader=self.dataloader.test_loader, batch_size=self.dataloader.batch_size).test_model(False)
        return self.cNet

# specify the image classes




if __name__ == '__main__':
    data = DataLoader('cifar-10')
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle', 'boot']

    net = Net()
    Trainer(model=Net,train_loader=data.train_loader,valid_loader=data.valid_loader).train_model(30,isBinary=False)
    Tester(model=Net,test_loader=data.test_loader,batch_size=data.batch_size).test_model(False)

    Utility().save_model(Net, './saved_models/cifar_net_single.pth')

