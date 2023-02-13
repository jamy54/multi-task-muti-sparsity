from Models.Models import bNet, Net
from utilites import Utility, DataLoader
from Tester import Tester
from Trainer import Trainer
from pruner.Pruner import Pruner
from tasks.fmnist_binary import FMNIST_binary
import sys
import torch
from pytorch_model_summary import summary
import numpy as np



if __name__ == '__main__':
    print(sys.version)
    data = DataLoader('F-MNIST')
    util = Utility()
    net = Net()
    sys.argv.append('F-MNIST binary')

    if sys.argv[1] == 'F-MNIST':
        Trainer(model=net, train_loader=data.train_loader, valid_loader=data.valid_loader).train_model(30, isBinary=False)
        Tester(model=net, test_loader=data.test_loader, batch_size=data.batch_size).test_model(False)
        util.save_model(net,'./saved_models/FMNIST_single.pth')
        n = np.zeros(shape=(1, 1, 32, 32), dtype=np.float32)
        print(summary(net, torch.tensor(n)))

    elif sys.argv[1] == 'F-MNIST binary':
        print('binary')
        bFmnist = FMNIST_binary(data, util)
        bnet = bFmnist.load_trained_model()
        util.store_parmeters(bnet,'Task2_BP.txt')
        util.store_parmeters(bnet.pretrained, 'Task1_BP.txt')

        n = np.zeros(shape=(1, 1, 32, 32), dtype=np.float32)
        print(summary(bnet.pretrained, torch.tensor(n)))
        print(summary(bnet, torch.tensor(n)))

        print('_' * 30)
        Tester(model=bnet, test_loader=data.test_loader, batch_size=data.batch_size).test_model()
        print("Number of Zero Weights: " + str(util.countZeroWeights(bnet)))

        print('Accuracy Before pruning (base model)')
        print('_' * 30)
        Tester(model=bnet.pretrained, test_loader=data.test_loader, batch_size=data.batch_size).test_model(False)
        print("Number of Zero Weights: " + str(util.countZeroWeights(bnet.pretrained)))

        pruner = Pruner(bnet.pretrained)

        print('Accuracy After pruning')
        print('_' * 30)
        Tester(model=bnet, test_loader=data.test_loader, batch_size=data.batch_size).test_model()
        print("Number of Zero Weights: " + str(util.countZeroWeights(bnet)))

        print('Accuracy After pruning (base model)')
        print('_' * 30)
        Tester(model=bnet.pretrained, test_loader=data.test_loader, batch_size=data.batch_size).test_model(False)
        print("Number of Zero Weights: " + str(util.countZeroWeights(bnet.pretrained)))

        print('FineTuning the model')
        print('_' * 30)

        Trainer(model=bnet.pretrained, train_loader=data.train_loader, valid_loader=data.valid_loader).train_model(10, isBinary=False, callbacks = [lambda: pruner.apply(bnet.pretrained)])


        Tester(model=bnet, test_loader=data.test_loader, batch_size=data.batch_size).test_model()
        print("Number of Zero Weights: " + str(util.countZeroWeights(bnet)))
        Tester(model=bnet.pretrained, test_loader=data.test_loader, batch_size=data.batch_size).test_model(False)
        print("Number of Zero Weights: " + str(util.countZeroWeights(bnet.pretrained)))

        util.store_parmeters(bnet, 'Task2_AP.txt')
        util.store_parmeters(bnet.pretrained, 'Task1_AP.txt')
