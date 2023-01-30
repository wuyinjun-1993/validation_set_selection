import torch.nn as nn

import torch

class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        # self.relu5 = nn.ReLU()

    def get_embedding_dim(self):
        return 84

    def features(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        return y
    
    def forward(self, x):
        y = self.features(x)        
        y = self.fc3(y)
        # y = self.relu5(y)
        return y

    def feature_forward(self, x, all_layer=False):
        out = self.features(x)
        return out

    def feature_forward2(self, x, all_layer_grad_no_full_loss=False, labels=None):
        out = self.features(x)
        out2 = self.fc3(out)
        if all_layer_grad_no_full_loss:
            out2 = out2 - torch.nn.functional.one_hot(labels, num_classes=out2.shape[1])

        grad_approx = torch.bmm(out.view(out.shape[0], out.shape[1], 1), out2.view(out2.shape[0], 1, out2.shape[1]))
        grad_approx = grad_approx.view(grad_approx.shape[0], -1)

        return grad_approx

    def forward_with_features(self, x):
        outf = self.features(x)        
        y = self.fc3(outf)
        # y = self.relu5(y)
        return y, outf, [outf]
        # x = self.layer0(x)
        # out1 = self.layer1(x)
        # out2 = self.layer2(out1)
        # out3 = self.layer3(out2)
        # out4 = self.layer4(out3)
        # spatial_size = out4.size(2)
        # x = nn.functional.avg_pool2d(out4, spatial_size, 1)
        # outf = x.view(x.size(0), -1)
        # x = self.fc(outf)
        # return x, outf, [out1, out2, out3, out4]

