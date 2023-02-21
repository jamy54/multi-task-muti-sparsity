import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
from Trainer import Trainer


class Tester():
    def __init__(self, model, test_loader, batch_size):
        self.test_loader = test_loader
        self.model = model
        self.batch_size = batch_size
        self.class_length = 43
        self.classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle', 'boot']

    def test_model(self, isBinary=True):

        class_correct = list(0. for i in range(self.class_length))
        class_total = list(0. for i in range(self.class_length))

        self.model.eval()
        # iterate over test data
        for data, target in self.test_loader:
            output = self.model(data)
            _, pred = torch.max(output, 1)

            if isBinary:
                for k, label in enumerate(target):
                    target[k] = 1 if (label in [5, 7, 8, 9]) else 0

            correct_tensor = pred.eq(target.data.view_as(pred))
            correct = np.squeeze(correct_tensor.cpu().numpy())

            #print(self.batch_size, len(target) , len(correct))
            for i in range(len(target)):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1

        #if not isBinary:
            #for i in range(10):
                #if class_total[i] > 0:
                    #print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    #    self.classes[i], 100 * class_correct[i] / class_total[i],
                    #    np.sum(class_correct[i]), np.sum(class_total[i])))
               # else:
                    #print('Test Accuracy of %5s: N/A (no training examples)' % (self.classes[i]))

        if not isBinary:
            print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))
        else:
            print('\nTest Accuracy (binary): %2d%% (%2d/%2d)' % (
                100. * np.sum(class_correct) / np.sum(class_total),
                np.sum(class_correct), np.sum(class_total)))