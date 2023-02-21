from Trainer import Trainer
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets.mnist import FashionMNIST

from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

class DataLoader:
    def __init__(self, name):
        if name == 'F-MNIST':
            self.F_MNIST()

    def F_MNIST(self):
        num_workers = 0

        batch_size = 20

        valid_size = 0.2

        transform = transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, ), (0.5, ))
                    ])

        train_data = FashionMNIST('data', train=True,
                                      download=True, transform=transform)
        test_data = FashionMNIST('data', train=False,
                                     download=True, transform=transform)

        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        #print(num_train, len(test_data), batch_size)

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                   sampler=train_sampler, num_workers=num_workers)
        valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                                   sampler=valid_sampler, num_workers=num_workers)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                                  num_workers=num_workers)
        self.batch_size = batch_size
        self.train_loader, self.valid_loader, self.test_loader = train_loader, valid_loader, test_loader

class Utility:
    def freeze_all_parameters(self, model):
        for param in model.parameters():
            param.requires_grad = False

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

    def countZeroWeights(self, model):
        zeros = 0
        for param in model.parameters():
            if param is not None:
                zeros += param.numel() - param.nonzero(as_tuple=False).size(0)
        return zeros

    def store_parmeters(self,model, filename):
        file = open("weight/"+filename, 'w')
        data =""
        for param in model.parameters():
            items = param[1].flatten().tolist()
            if param.dim() == 4:
                for para in param:
                    for par in para:
                        for pa in par:
                            for p in pa:
                                data += str(round(float(p),4)) + ", "
                            data += "\n"
                        data += "\n"
                    data += "\n"

            elif param.dim() == 2:
                for para in param:
                    for par in para:
                        data += str(round(float(par),4)) + ", "
                    data += "\n"
            elif param.dim() == 1:
                for para in param:
                    data += str(round(float(para),4)) + ", "

        file.write(str(data) + "\n")

        file.close()

        #print(data)

