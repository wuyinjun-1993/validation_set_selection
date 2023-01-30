import numpy
import torch
from torch.utils.data import Dataset

class ImbalanceDataset(Dataset):
    """
    Wrapper for an existing dataset which removes samples to create exponential
    class bias.
    """

    def __init__(self, unbiased_dataset, imb_factor):
        # 1. Get number of classes from the labels.
        assert hasattr(unbiased_dataset, 'targets'), \
            "Unbiased dataset must have a `targets` attribute"

        if type(unbiased_dataset.targets) is numpy.ndarray:
            unbiased_dataset.targets = torch.from_numpy(unbiased_dataset.targets)

        classes = torch.unique(unbiased_dataset.targets)
        num_classes = classes.shape[0]

        # 2. Get average number of samples per class
        samples_per_class = -1
        for cls in range(num_classes):
            curr_samples_per_class = torch.sum(unbiased_dataset.targets == cls)
            if samples_per_class < 0:
                samples_per_class = curr_samples_per_class
            else:
                samples_per_class = torch.min(samples_per_class, curr_samples_per_class)
        print("sample_class_count::", samples_per_class)
        # 3. Delete samples from each class based on some bias
        self.mask = torch.zeros(unbiased_dataset.targets.shape[0], dtype=torch.bool)
        for cls in range(num_classes):
            num_select = samples_per_class * (imb_factor**(-cls / (num_classes - 1.0)))
            all_cls_idx = torch.nonzero(unbiased_dataset.targets == cls)
            shuffle = torch.randperm(len(all_cls_idx))
            idx = all_cls_idx[shuffle][:int(num_select)]
            self.mask[idx] = True

        # self.dataset.targets = self.dataset.targets[self.mask]
        # if orig_numpy:
        #     self.dataset.targets = self.dataset.targets.numpy()
        # self.dataset.data = self.dataset.data[self.mask]

        # self.data = self.dataset.data
        # self.targets = self.dataset.targets

    # def __getitem__(self, index):
    #     return self.dataset[index]

    # def __len__(self):
    #     return len(self.dataset)
