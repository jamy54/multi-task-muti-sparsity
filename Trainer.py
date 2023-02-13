import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch

class Trainer:
    def __init__(self, model, train_loader, valid_loader):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        self.PATH = './saved_models/F-MNIST_binary.pth'
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def train_model(self, ep, isBinary=True, callbacks = None):
        train_losslist = []
        valid_loss_min = np.Inf  # track change in validation loss

        for epoch in range(ep):

            train_loss = 0.0
            valid_loss = 0.0
            self.model.train()
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                if isBinary:
                    for k, label in enumerate(target):
                        target[k] = 1 if (label in [5, 7, 8, 9]) else 0
                loss = self.criterion(output, target)
                loss.backward()
                for p in self.model.parameters():
                    print(p.grad)

                self.optimizer.step()
                train_loss += loss.item() * data.size(0)

                if callbacks is not None:
                    for callback in callbacks:
                        callback()

            self.model.eval()
            for data, target in self.valid_loader:

                output = self.model(data)
                for k, label in enumerate(target):
                    target[k] = 1 if (label in [5, 7, 8, 9]) else 0

                loss = self.criterion(output, target)
                valid_loss += loss.item() * data.size(0)

            train_loss = train_loss / len(self.train_loader.dataset)
            valid_loss = valid_loss / len(self.valid_loader.dataset)
            train_losslist.append(train_loss)

            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))

            if valid_loss <= valid_loss_min:
                valid_loss_min = valid_loss

        print('Finished Training')