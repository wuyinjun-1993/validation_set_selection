import torch.nn as nn

import torch.nn.functional as F


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from lib.normalize import *

class DNN_two_layers(nn.Module):
    def __init__(self):
        super(DNN_two_layers, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(28*28, 100)
        # self.fc3 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(100, 10)


        # self.fc1 = nn.Linear(28*28, 10)
        self.groupDis = nn.Sequential(
            nn.Linear(100, 20),
            Normalize(2))

    def forward(self, x,two_branch=False):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x.view(x.shape[0],-1)))
        # # x = F.relu(self.fc3(x))
        # # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)


        x_ = F.relu(self.fc1(x.view(x.shape[0],-1)))
        # x = F.relu(self.fc3(x))
        # x = F.dropout(x, training=self.training)
        x = self.fc2(x_)

        if two_branch:
            x2 = self.groupDis(x_)
            return x,x2

        return F.log_softmax(x, dim=1)

    def feature_forward(self, x):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        x = F.relu(self.fc1(x.view(x.shape[0],-1)))
        # x = F.relu(self.fc3(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        return x
        # return F.log_softmax(x, dim=1)


class DNN_three_layers(nn.Module):
    def __init__(self, hidden_dim=100, low_dim = 128):
        super(DNN_three_layers, self).__init__()
        # self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        # self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(28*28, hidden_dim)
        # self.fc3 = nn.Linear(200, 100)
        self.fc2 = nn.Linear(hidden_dim, low_dim)
        self.fc3 = nn.Linear(low_dim, 10)


        # self.fc1 = nn.Linear(28*28, 10)
        self.groupDis = nn.Sequential(
            nn.Linear(hidden_dim, low_dim),
            Normalize(2))

    def forward(self, x,two_branch=False, full_pred=True):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        # x = F.relu(self.fc1(x.view(x.shape[0],-1)))
        # # x = F.relu(self.fc3(x))
        # # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)


        x_ = F.relu(self.fc1(x.view(x.shape[0],-1)))
        # x = F.relu(self.fc3(x))
        # x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x_))

        if two_branch:
            x2 = self.groupDis(x_)
            return x,x2
        else:
            if not full_pred:
                return x
            else:
                x = self.fc3(x)
                return F.log_softmax(x, dim=1)

    def feature_forward(self, x, all_layer=False):
        # x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # x = x.view(-1, 320)
        if all_layer:
            x_ls = []
            x = F.relu(self.fc1(x.view(x.shape[0],-1)))
            x_ls.append(x.clone().view(x.shape[0],-1))

            x = F.relu(self.fc2(x))
            x_ls.append(x.clone().view(x.shape[0], -1))
            return torch.cat(x_ls, dim = 1)
        else:
            x = F.relu(self.fc1(x.view(x.shape[0],-1)))
            x = F.relu(self.fc2(x))
            return x
        # x = F.relu(self.fc3(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        
        # return F.log_softmax(x, dim=1)