from torch import nn

class ToyNN(nn.Module):
    def __init__(self):
        super(ToyNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.feature_forward(x)
        logits = self.fc3(out)
        return logits

    def feature_forward(self, x, all_layer=False):
        out = self.flatten(x)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        return out
