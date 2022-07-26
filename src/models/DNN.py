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
        self.fc1 = nn.Linear(28*28, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, low_dim)
        self.fc3 = nn.Linear(low_dim, 10)

    def forward(self, x):
        x_ = F.relu(self.fc1(x.view(x.shape[0],-1)))
        x = F.relu(self.fc2(x_))
        x = self.fc3(x)
        return x

    def feature_forward(self, x, all_layer=False):
        if all_layer:
            x_ls = []
            x1 = F.relu(self.fc1(x.view(x.shape[0],-1)))
            # x_ls.append(x.clone().detach().cpu().view(x.shape[0],-1))
            x_ls.append(x1)

            x = F.relu(self.fc2(x1))
            # x_ls.append(x.clone().detach().cpu().view(x.shape[0], -1))
            x_ls.append(x)
            return x_ls
        else:
            x = F.relu(self.fc1(x.view(x.shape[0],-1)))
            x = F.relu(self.fc2(x))
            return x
        # x = F.relu(self.fc3(x))
        # x = F.dropout(x, training=self.training)
        # x = self.fc2(x)
        
        # return F.log_softmax(x, dim=1)

    def feature_forward2(self, x, all_layer_grad_no_full_loss=False, labels=None):
        out = F.relu(self.fc1(x.view(x.shape[0],-1)))
        out = F.relu(self.fc2(out))
        out2 = self.fc3(out)
        # if all_layer_grad_no_full_loss:
        #     out2 = out2 - F.one_hot(labels, num_classes=out2.shape[1])

        grad_approx = torch.bmm(out.view(out.shape[0], out.shape[1], 1), out2.view(out2.shape[0], 1, out2.shape[1]))
        grad_approx = grad_approx.view(grad_approx.shape[0], -1)

        return grad_approx

    def obtain_gradient_last_full_layer(self, sample_representation_last_layer, target, criterion, is_cuda = False):
        output = self.fc3(sample_representation_last_layer)
        loss = criterion(output, target)
        onehot_target = torch.zeros(output.shape[1])
        onehot_target[target] = 1
        if is_cuda:
            onehot_target = onehot_target.cuda()

        total_loss = loss + torch.sum(onehot_target.view(-1)*output.view(-1))
        sample_representation_last_layer_grad = torch.autograd.grad(total_loss, sample_representation_last_layer)[0]
        return sample_representation_last_layer_grad


    def obtain_gradient_all_layer_by_sample_representation(self, sample, target, criterion):

        x1 = F.relu(self.fc1(sample.view(sample.shape[0],-1)))
        
        # x_ls.append(x.clone().detach().cpu().view(x.shape[0],-1))
        x2 = F.relu(self.fc2(x1))

        output = self.fc3(x2)
        loss = criterion(output, target)
        x1_grad = torch.autograd.grad(loss, x1, retain_graph=True)[0]
        x2_grad = torch.autograd.grad(loss, x2)[0]
        return [x1_grad, x2_grad]
        # output = self.fc3(sample_representation_last_layer)
        # loss = criterion(output, target)
        # sample_representation_last_layer_grad = torch.autograd.grad(loss, sample_representation_last_layer)[0]
        # return sample_representation_last_layer_grad
            # return x
