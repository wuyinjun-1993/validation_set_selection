import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftBootstrappingLoss(nn.Module):
    """
    Calculates the soft bootstrapping loss.

    Implemented from equation 6 in https://arxiv.org/pdf/1412.6596.pdf
    """

    def __init__(self, beta=0.95):
        super().__init__()
        self.beta = beta
        self.reduction = 'mean'

    def forward(self, y_pred, y):
        ce = self.beta * F.cross_entropy(y_pred, y, reduction='none')
        y_pred_const = y_pred.detach()
        bootstrapping = -(1.0 - self.beta) * torch.sum(
            F.softmax(y_pred_const, dim=1) * F.log_softmax(y_pred, dim=1),
            dim=1,
        )

        loss = ce + bootstrapping
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        return loss

class HardBootstrappingLoss(nn.Module):
    """
    Calculates the hard bootstrapping loss.

    Implemented from equation 7 in https://arxiv.org/pdf/1412.6596.pdf
    """

    def __init__(self, beta=0.95):
        super().__init__()
        self.beta = beta
        self.reduction = 'mean'

    def forward(self, y_pred, y):
        ce = self.beta * F.cross_entropy(y_pred, y, reduction='none')
        z = F.softmax(y_pred.detach(), dim=1).argmax(dim=1)
        z = F.one_hot(z)
        bootstrapping = -(1.0 - self.beta) * torch.sum(
            z * F.log_softmax(y_pred, dim=1),
            dim=1,
        )

        loss = ce + bootstrapping
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        return loss
