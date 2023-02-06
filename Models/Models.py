import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3,3), padding=1)
        #self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # fully connected layers
        self.fc1 = nn.Linear(16 * 8 * 8, 10)
        #self.fc2 = nn.Linear(512, 64)
        #self.fc3 = nn.Linear(64, 43)
        # dropout
        self.dropout = nn.Dropout(p=.5)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        # print(x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = self.pool(F.relu(self.conv3(x)))
        # flattening
        #print(x.shape)
        x = x.view(-1, 16 * 8 * 8)
        # fully connected layers
        x = self.fc1(x)
        #x = self.dropout(F.relu(self.fc2(x)))
        #x = self.fc3(x)
        return x


class bNet(nn.Module):
    def __init__(self, my_pretrained_model):
        super(bNet, self).__init__()
        self.pretrained = my_pretrained_model
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pretrained(x)
        x = self.fc4(x)
        return x
