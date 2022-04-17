import numpy
import torch
from torch.utils.data import Dataset

class ImbalanceDataset(Dataset):
    """
    Wrapper for an existing dataset which removes samples to create exponential
    class bias.
    """

    def __init__(self, unbiased_dataset):
        # 1. Get number of classes from the labels.
        assert hasattr(unbiased_dataset, 'targets'), \
            "Unbiased dataset must have a `targets` attribute"

        self.dataset = unbiased_dataset
        orig_numpy = False
        if type(self.dataset.targets) is numpy.ndarray:
            self.dataset.targets = torch.from_numpy(self.dataset.targets)
            orig_numpy = True

        classes = torch.unique(self.dataset.targets)
        num_classes = classes.shape[0]

        # 2. Get average number of samples per class
        samples_per_class = self.dataset.targets.shape[0] / num_classes

        # 3. Delete samples from each class based on some bias
        num_select = samples_per_class
        self.mask = torch.zeros(self.dataset.targets.shape[0], dtype=torch.bool)
        for cls in classes:
            idx = torch.nonzero(self.dataset.targets == cls)[:int(num_select)]
            self.mask[idx] = True
            num_select *= 0.5

        self.dataset.targets = self.dataset.targets[self.mask]
        if orig_numpy:
            self.dataset.targets = self.dataset.targets.numpy()
        self.dataset.data = self.dataset.data[self.mask]

        self.data = self.dataset.data
        self.targets = self.dataset.targets

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
