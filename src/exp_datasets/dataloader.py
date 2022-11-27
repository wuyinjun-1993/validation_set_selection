import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import exp_datasets
from torch.utils.data import Subset, Dataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from PIL import ImageFilter
import random
from main.find_valid_set import *

from PIL import Image
import numpy
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from exp_datasets.sst import *
from exp_datasets.imdb import *
from exp_datasets.trec import *
from exp_datasets.craige import *
from sklearn.model_selection import train_test_split
import pandas as pd
# To ensure each process will produce the same dataset separately. Random flips
# of labels become deterministic so we can perform them independently per
# process.
torch.manual_seed(0)
random.seed(0)
numpy.random.seed(0)

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class RetinaDataset(Dataset):
    def __init__(self, df):
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor(),
        ])
        self.df = df
        self.transform = transform
        self.targets = torch.tensor(df['level'].values)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img = Image.open(self.df.iloc[index].image)

        if(self.transform):
            img = self.transform(img)

        return (index, img, self.targets[index])

    @staticmethod
    def get_subset_dataset(dataset, sample_ids, labels=None):
        subset_df = dataset.df.iloc[sample_ids]
        subset_dataset = RetinaDataset(subset_df)
        subset_dataset.targets = dataset.targets[sample_ids]
        return subset_dataset

    @staticmethod
    def to_cuda(data, targets):
        return data.cuda(), targets.cuda()

class ImageNetDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.targets = dataset.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    @staticmethod
    def get_subset_dataset(dataset, sample_ids, labels=None):
        subset_samples = [dataset.samples[i] for i in sample_ids]
        subset_data = copy.deepcopy(dataset)
        subset_data.samples = subset_samples
        subset_data.targets = [s[1] for s in subset_samples]
        subset_dataset = ImageNetDataset(subset_data)
        return subset_dataset

    @staticmethod
    def to_cuda(data, targets):
        return data.cuda(), targets.cuda()

class dataset_wrapper_X(Dataset):
    def __init__(self, data_tensor, transform, three_imgs = False, two_imgs = False):

        # super(new_mnist_dataset, self).__init__(*args, **kwargs)
        self.data = data_tensor
        self.transform = transform
        self.three_imgs = three_imgs
        self.two_imgs = two_imgs

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            if not type(img) is numpy.ndarray:
                img = Image.fromarray(img.numpy(), mode="L")
            else:
                img = Image.fromarray(img)
            if self.transform is not None:
                img1 = self.transform(img)

                if self.two_imgs:
                    img2 = self.transform(img)
                    return (img1, img2), index


                if self.three_imgs:
                    img2 = self.transform(img)
                    img3 = self.transform(img)
                    return (img1, img2, img3), index
        else:
            img1 = img
        return (index, img1)
        # image, target = super(new_mnist_dataset, self).__getitem__(index)

        # return (index, image,target)

    def __len__(self):
        return len(self.data)

class dataset_wrapper(Dataset):
    def __init__(self, data_tensor, label_tensor, transform, three_imgs = False, two_imgs = False):

        # super(new_mnist_dataset, self).__init__(*args, **kwargs)
        self.data = data_tensor
        self.targets = label_tensor
        self.transform = transform
        self.three_imgs = three_imgs
        self.two_imgs = two_imgs

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            if not type(img) is numpy.ndarray:
                img = Image.fromarray(img.numpy(), mode="L")
            else:
                img = Image.fromarray(img)
        
            img1 = self.transform(img)

            if self.two_imgs:
                img2 = self.transform(img)
                return (img1, img2), target, index


            if self.three_imgs:
                img2 = self.transform(img)
                img3 = self.transform(img)
                return (img1, img2, img3), target, index
        else:
            img1 = img
        return (index, img1, target)
        # image, target = super(new_mnist_dataset, self).__getitem__(index)

        # return (index, image,target)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_subset_dataset(dataset, sample_ids, labels = None):
        subset_data = dataset.data[sample_ids]
        if labels is None:
            subset_labels = dataset.targets[sample_ids]
        else:
            subset_labels = labels[sample_ids]
        transform = dataset.transform


        three_imgs = dataset.three_imgs
        two_imgs = dataset.two_imgs

        if not type(dataset.data) is numpy.ndarray:
            subset_data = subset_data.clone()
            subset_labels = subset_labels.clone()
            if len(sample_ids) <= 1:
                subset_data = subset_data.unsqueeze(0)
                subset_labels = subset_labels.unsqueeze(0)
            
        else:
            subset_data = numpy.copy(subset_data)
            subset_labels = numpy.copy(subset_labels)
            if len(sample_ids) <= 1:
                subset_data = np.expand_dims(subset_data, 0)
                subset_labels = np.expand_dims(subset_labels, 0)

        return dataset_wrapper(subset_data, subset_labels, transform, three_imgs, two_imgs)

    @staticmethod
    def subsampling_dataset_by_class(dataset, num_per_class=45):
        if type(dataset.data) is numpy.ndarray:
            label_set = np.unique(dataset.targets)
        else:
            label_set = torch.unique(dataset.targets)

        full_sel_sample_ids = []
        for label in label_set:
            if type(dataset.data) is numpy.ndarray:
                sample_ids_with_curr_labels = np.nonzero((dataset.targets == label))[0].reshape(-1)
                sample_ids_with_curr_labels = torch.from_numpy(sample_ids_with_curr_labels)
            else:
                sample_ids_with_curr_labels = torch.nonzero((dataset.targets == label)).reshape(-1)

            random_sample_ids_with_curr_labels = torch.randperm(len(sample_ids_with_curr_labels))

            selected_sample_ids_with_curr_labels = random_sample_ids_with_curr_labels[0:num_per_class]

            full_sel_sample_ids.append(selected_sample_ids_with_curr_labels)

        full_sel_sample_ids_tensor = torch.cat(full_sel_sample_ids)
        
        if type(dataset.data) is numpy.ndarray:
            return dataset.get_subset_dataset(dataset, full_sel_sample_ids_tensor.numpy())
        else:
            return dataset.get_subset_dataset(dataset, full_sel_sample_ids_tensor)




    @staticmethod
    def concat_validset(dataset1, dataset2):
        valid_data_mat = dataset1.data
        valid_labels = dataset1.targets
        if type(valid_data_mat) is numpy.ndarray:
            if len(dataset2.data.shape) < len(valid_data_mat.shape):
                dataset2.data = np.expand_dims(dataset2.data, 0)
                dataset2.targets = np.expand_dims(dataset2.targets,0)
            valid_data_mat = numpy.concatenate((valid_data_mat, dataset2.data), axis = 0)
            valid_labels = numpy.concatenate((valid_labels, dataset2.targets), axis = 0)
            
        else:

            print("origin_valid data shape::", valid_data_mat.shape)
            print("new valid data shape::", dataset2.data.shape)
            if len(dataset2.data.shape) < len(valid_data_mat.shape):
                dataset2.data = dataset2.data.unsqueeze(0)
                dataset2.targets = dataset2.targets.unsqueeze(0)
            if len(dataset2.data.shape) > len(valid_data_mat.shape):
                dataset2.data = dataset2.data.squeeze(0)
            print("origin_valid data shape::", valid_data_mat.shape)
            print("new valid data shape::", dataset2.data.shape)
            
            
            valid_data_mat = torch.cat([valid_data_mat, dataset2.data], dim = 0)
            valid_labels = torch.cat([valid_labels.view(-1), dataset2.targets.view(-1)], dim = 0)
        valid_set = dataset_wrapper(valid_data_mat, valid_labels, dataset1.transform)
        return valid_set

    @staticmethod
    def to_cuda(data, targets):
        return data.cuda(), targets.cuda()

def get_dataloader(args, add_erasing=False, aug_plus=False):
    if 'cifar' in args.dataset or 'kitchen' in args.dataset:
        if aug_plus:
            # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            transform_train_list = [
                transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        else:
            transform_train_list = [
                transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        if add_erasing:
            transform_train_list.append(transforms.RandomErasing(p=1.0))
        transform_train = transforms.Compose(transform_train_list)

        if 'kitchen' in args.dataset:
            transform_test = transforms.Compose([
                transforms.Resize((32,32), interpolation=2),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

    elif 'stl' in args.dataset:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(size=96, scale=(0.2,1.)),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    if args.dataset == 'cifar10':
        
        if args.continue_label:
            trainset, validset, metaset, origin_labels = load_train_valid_set(args)
            trainset.transform = transform_train
            trainset.two_imgs=args.two_imgs
            trainset.three_imgs=args.three_imgs
        else:
            trainset = exp_datasets.CIFAR10Instance(root=os.path.join(args.data_dir, 'CIFAR-10'), train=True, download=True, transform=transform_train, two_imgs=args.two_imgs, three_imgs=args.three_imgs)
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)



        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, sampler=train_sampler)

        testset = exp_datasets.CIFAR10Instance(root=os.path.join(args.data_dir, 'CIFAR-10'), train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)
        args.pool_len = 4
        ndata = trainset.__len__()

    elif args.dataset == 'cifar100':
        if args.continue_label:
            trainset, validset, metaset, origin_labels = load_train_valid_set(args)
            trainset.transform = transform_train
            trainset.two_imgs=args.two_imgs
            trainset.three_imgs=args.three_imgs
        else:
            trainset = exp_datasets.CIFAR100Instance(root=os.path.join(args.data_dir, 'CIFAR-100'), train=True, download=True, transform=transform_train, two_imgs=args.two_imgs, three_imgs=args.three_imgs)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, sampler=train_sampler)

        testset = exp_datasets.CIFAR100Instance(root=os.path.join(args.data_dir, 'CIFAR-100'), train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)
        args.pool_len = 4
        ndata = trainset.__len__()

    elif args.dataset == 'stl10':
        trainset = exp_datasets.STL10(root=os.path.join(args.data_dir, 'STL10'), split='train', download=True, transform=transform_train, two_imgs=args.two_imgs, three_imgs=args.three_imgs)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, sampler=train_sampler)

        testset = exp_datasets.STL10(root=os.path.join(args.data_dir, 'STL10'), split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)
        args.pool_len = 7
        ndata = trainset.__len__()

    elif args.dataset == 'stl10-full':
        trainset = exp_datasets.STL10(root=os.path.join(args.data_dir, 'STL10'), split='train+unlabeled', download=True, transform=transform_train, two_imgs=args.two_imgs, three_imgs=args.three_imgs)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, 
                            pin_memory=False, sampler=train_sampler)

        labeledTrainset = exp_datasets.STL10(root=os.path.join(args.data_dir, 'STL10'), split='train', download=True, transform=transform_train, two_imgs=args.two_imgs)
        labeledTrain_sampler = torch.utils.data.distributed.DistributedSampler(labeledTrainset)
        labeledTrainloader = torch.utils.data.DataLoader(labeledTrainset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=2, pin_memory=False, sampler=labeledTrain_sampler)
        testset = exp_datasets.STL10(root=os.path.join(args.data_dir, 'STL10'), split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)
        args.pool_len = 7
        ndata = labeledTrainset.__len__()

    elif args.dataset == 'kitchen':
        trainset = exp_datasets.CIFARImageFolder(root=os.path.join(args.data_dir, 'Kitchen-HC/train'), train=True, transform=transform_train, two_imgs=args.two_imgs, three_imgs=args.three_imgs)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)
        testset = exp_datasets.CIFARImageFolder(root=os.path.join(args.data_dir, 'Kitchen-HC/test'), train=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)
        args.pool_len = 4
        ndata = trainset.__len__()

    elif args.dataset == 'MNIST':
        transform_train = torchvision.transforms.Compose([
                                    # transforms.RandomResizedCrop(size=28, scale=(0.2,1.)),
                                    # transforms.RandomApply([
                                    #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                                    # ], p=0.8),
                                    # transforms.RandomGrayscale(p=0.2),
                                    # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                                    # transforms.RandomHorizontalFlip(),
                                    transforms.RandomResizedCrop(28),
                                    transforms.RandomHorizontalFlip(),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])

        if args.continue_label:
            trainset, validset, metaset, origin_labels = load_train_valid_set(args)
            trainset.transform = transform_train
            trainset.two_imgs=args.two_imgs
            trainset.three_imgs=args.three_imgs
        else:
            trainset = torchvision.datasets.MNIST(args.data_dir, train=True, download=True,
                                    transform=transform_train)

        trainset = dataset_wrapper(numpy.copy(trainset.data), numpy.copy(trainset.targets), transform_train, three_imgs=args.three_imgs)

        transform_test = torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                        ])

        testset = torchvision.datasets.MNIST(args.data_dir, train=False, download=True,
                                        transform=transform_test)

        testset = dataset_wrapper(numpy.copy(testset.data), numpy.copy(testset.targets), transform_test)

        train_sampler = DistributedSampler(trainset)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, sampler=train_sampler)
        testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)
        args.pool_len = 4
        ndata = trainset.__len__()

    return trainloader, testloader, ndata

# def concat_valid_set(valid_set, new_valid_set):
#     valid_data_mat = valid_set.data
#     valid_labels = valid_set.targets
#     if type(valid_data_mat) is numpy.ndarray:
#         valid_data_mat = numpy.concatenate((valid_data_mat, new_valid_set.data), axis = 0)
#         valid_labels = numpy.concatenate((valid_labels, new_valid_set.targets), axis = 0)
        
#     else:
#         valid_data_mat = torch.cat([valid_data_mat, new_valid_set.data], dim = 0)
#         valid_labels = torch.cat([valid_labels, new_valid_set.targets], dim = 0)
#     valid_set = dataset_wrapper(valid_data_mat, valid_labels, valid_set.transform)

#     return valid_set

def split_train_valid_set_by_ids(args, train_dataset, origin_labels, valid_ids, update_train_ids):

    if args.noisy_valid:
        clean_labels = train_dataset.targets.copy()
    else:
        clean_labels = origin_labels
    meta_set = train_dataset.get_subset_dataset(train_dataset, valid_ids, clean_labels)
    train_set = train_dataset.get_subset_dataset(train_dataset, update_train_ids)

    # if not type(train_dataset.data) is numpy.ndarray:
    #     meta_data = train_dataset.data[valid_ids].clone()
    #     meta_labels = origin_labels[valid_ids].clone()

    #     train_data = train_dataset.data[update_train_ids].clone()
    #     train_labels = train_dataset.targets[update_train_ids].clone()
    # else:
    #     meta_data = numpy.copy(train_dataset.data[valid_ids])
    #     meta_labels = numpy.copy(origin_labels[valid_ids])

    #     train_data = numpy.copy(train_dataset.data[update_train_ids])
    #     train_labels = numpy.copy(train_dataset.targets[update_train_ids])

    # train_set = dataset_wrapper(train_data, train_labels, transform)
    # meta_set = dataset_wrapper(meta_data, meta_labels, transform)

    return train_set, meta_set

def random_partition_train_valid_dataset0(criterion, optimizer, net, train_dataset, validset, args, origin_labels, cached_sample_weights=None):
    # if validset is not None and args.total_valid_sample_count > 0 and args.total_valid_sample_count <= len(validset):
    #     args.logger.info("already collect enough samples, exit!!!!")
    #     sys.exit(1)

    # if len(validset) + args.valid_count > args.total_valid_sample_count:
    #     args.valid_count = args.total_valid_sample_count - len(validset)
    train_ids = torch.arange(len(train_dataset))
    rand_train_ids = torch.randperm(len(train_ids))

    valid_size = args.valid_count
    valid_ids = rand_train_ids[:valid_size]

    update_train_ids = rand_train_ids[valid_size:]

    torch.save(valid_ids, os.path.join(args.save_path, "valid_dataset_ids"))
    train_set, meta_set = split_train_valid_set_by_ids(args, train_dataset, origin_labels, valid_ids, update_train_ids)
    remaining_origin_labels = origin_labels[update_train_ids]
    return train_set, meta_set, remaining_origin_labels


def random_partition_train_valid_dataset0_by_class(criterion, optimizer, net, train_dataset, validset, args, origin_labels, cached_sample_weights=None, sample_count_per_class = None):

    # if validset is not None and args.total_valid_sample_count > 0 and args.total_valid_sample_count <= len(validset):
    #     args.logger.info("already collect enough samples, exit!!!!")
    #     sys.exit(1)

    # if len(validset) + args.valid_count > args.total_valid_sample_count:
    #     args.valid_count = args.total_valid_sample_count - len(validset)

    unique_label_set = set(origin_labels.tolist())
    valid_size = args.valid_count
    if not type(origin_labels) is torch.Tensor:
        origin_labels = torch.from_numpy(origin_labels)
    label_id = 0
    curr_total_valid_size = 0
    all_selected_train_ids = []
    all_selected_valid_ids = []
    for label in unique_label_set:
        if label_id < len(unique_label_set) - 1:
            if sample_count_per_class is None:
                curr_valid_size = int(valid_size/len(unique_label_set))
            else:
                curr_valid_size = int(valid_size/sum(sample_count_per_class)*sample_count_per_class[label_id])
        else:
            curr_valid_size = valid_size - curr_total_valid_size

        curr_total_valid_size += curr_valid_size
        curr_train_ids = torch.nonzero(origin_labels == label).view(-1)
        rand_train_id_ids = torch.randperm(len(curr_train_ids))
        curr_train_sample_ids = curr_train_ids[rand_train_id_ids]
        assert torch.all(origin_labels[curr_train_sample_ids] == label) == True
        all_selected_train_ids.append(curr_train_sample_ids[curr_valid_size:])
        all_selected_valid_ids.append(curr_train_sample_ids[0:curr_valid_size])
        label_id += 1

    # train_ids = torch.arange(len(train_dataset))
    # rand_train_ids = torch.randperm(len(train_ids))
    valid_ids = torch.cat(all_selected_valid_ids)
    update_train_ids = torch.cat(all_selected_train_ids)
    assert curr_total_valid_size == valid_size
    assert len(valid_ids) + len(update_train_ids) == len(origin_labels)
    # valid_ids = rand_train_ids[:valid_size]

    # update_train_ids = rand_train_ids[valid_size:]

    torch.save(valid_ids, os.path.join(args.save_path, "valid_dataset_ids"))
    train_set, meta_set = split_train_valid_set_by_ids(args, train_dataset, origin_labels, valid_ids, update_train_ids)
    remaining_origin_labels = origin_labels[update_train_ids]
    return train_set, meta_set, remaining_origin_labels


def obtain_representations_for_valid_set(args, valid_set, net, criterion, optimizer):
    validloader = DataLoader(valid_set, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)

    sample_representation_ls = []

    if not args.all_layer_grad:
        full_sample_representation_tensor, all_sample_ids = get_representations_last_layer2(valid_set, args, validloader, criterion, optimizer, net)
        # full_sample_representation_tensor, all_sample_ids = get_representations_last_layer(args, validloader, criterion, optimizer, net)
        return full_sample_representation_tensor
        # with torch.no_grad():

        #     # all_sample_representations = [None]*len(train_loader.dataset)

        #     for batch_id, (sample_ids, data, labels) in enumerate(validloader):

        #         if args.cuda:
        #             data, labels = validloader.dataset.to_cuda(data, labels)
        #             # data = data.cuda()
        #             # labels = labels.cuda()
                
        #         sample_representation = net.feature_forward(data)
        #         sample_representation_ls.append(sample_representation)
        # return torch.cat(sample_representation_ls)
    else:
        full_sample_representation_tensor, all_sample_ids = get_grad_by_example(args, validloader, net, criterion, optimizer)
        return full_sample_representation_tensor



def init_sampling_valid_samples(net, train_dataset, train_transform, args, origin_labels):
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    pred_prob_ls = []

    sample_id_ls = []

    with torch.no_grad():

        # all_sample_representations = [None]*len(train_loader.dataset)

        for batch_id, (sample_ids, data, labels) in enumerate(trainloader):

            if args.cuda:
                data = data.cuda()
                # labels = labels.cuda()
            
            output_probs = F.softmax(net(data), dim=1).view(-1)

            pred_prob_ls.append(output_probs)

            sample_id_ls.append(sample_ids)

    sample_id_ls_tensor = torch.cat(sample_id_ls)

    pred_prob_ls_tensor = torch.cat(pred_prob_ls)

    sorted_pred_prob_ls, sorted_train_ids = torch.sort(torch.max(pred_prob_ls_tensor,dim=1)[0], descending=True)

    valid_ids = sample_id_ls_tensor[sorted_train_ids][0:args.valid_count]

    update_train_ids = torch.ones(len(train_dataset))
    if not args.include_valid_set_in_training:
        update_train_ids[valid_ids] = 0
    update_train_ids = update_train_ids.nonzero().view(-1)


    train_set, meta_set = split_train_valid_set_by_ids(args, train_dataset, origin_labels, valid_ids, update_train_ids)

    remaining_origin_labels = origin_labels[update_train_ids]

    return train_set, valid_set, meta_set, remaining_origin_labels

def obtain_valid_count_ls_per_class(unique_label_set, total_valid_count, sample_count_per_class = None):
    label_id = 0
    curr_total_valid_count = 0
    valid_count_ls = []
    for label in unique_label_set:
        if label_id < len(unique_label_set) - 1:
            if sample_count_per_class is None:
                curr_valid_size = int(total_valid_count/len(unique_label_set))
            else:
                curr_valid_size = int(total_valid_count/sum(sample_count_per_class)*sample_count_per_class[label_id])
        else:
            curr_valid_size = total_valid_count - curr_total_valid_count

        curr_total_valid_count += curr_valid_size
        valid_count_ls.append(curr_valid_size)
        label_id += 1
    
    return valid_count_ls

def find_representative_samples0_by_class(criterion, optimizer, net, train_dataset,validset,  args, origin_labels, cached_sample_weights = None, sample_count_per_class = None):
    unique_label_set = set(origin_labels.tolist())
    valid_labels = None
    if validset is not None:
        valid_labels = validset.targets
    if not type(origin_labels) is torch.Tensor:
        origin_labels = torch.from_numpy(origin_labels)
        if valid_labels is not None:
            valid_labels = torch.from_numpy(valid_labels)
    label_id = 0
    curr_total_valid_size = 0
    all_selected_train_ids = []
    all_selected_valid_ids = []
    res_train_set, res_meta_set, res_remaining_origin_labels = None, None, None
    

    valid_count_ls = obtain_valid_count_ls_per_class(unique_label_set, args.valid_count, sample_count_per_class = sample_count_per_class)
    origin_total_valid_sample_count = args.total_valid_sample_count
    total_valid_sample_count_ls = obtain_valid_count_ls_per_class(unique_label_set, args.total_valid_sample_count, sample_count_per_class = sample_count_per_class)

    for label in unique_label_set:
        args.logger.info("collecting meta set for label %d"%(label))
        curr_sample_ids = torch.nonzero(origin_labels == label).view(-1)
        curr_origin_labels = origin_labels[curr_sample_ids]
        assert torch.all(curr_origin_labels == label) == True
        curr_cached_sample_weights = None
        if not cached_sample_weights is None:
            curr_cached_sample_weights = cached_sample_weights[curr_sample_ids]
        if not type(train_dataset.targets) is torch.Tensor:
            curr_sample_ids = curr_sample_ids.numpy()
        sub_trainset = train_dataset.get_subset_dataset(train_dataset, curr_sample_ids)
        sub_validset = None
        if validset is not None:
            curr_valid_sample_ids = torch.nonzero(valid_labels == label).view(-1)
            if not type(train_dataset.targets) is torch.Tensor:
                curr_valid_sample_ids = curr_valid_sample_ids.numpy()
            sub_validset = validset.get_subset_dataset(validset, curr_valid_sample_ids)
        args.total_valid_sample_count = total_valid_sample_count_ls
        res_sub_train_set, res_sub_meta_set, res_sub_remaining_origin_labels = find_representative_samples0(criterion, optimizer, net, sub_trainset,sub_validset,  args, curr_origin_labels, cached_sample_weights = curr_cached_sample_weights, valid_count=valid_count_ls[label_id])
        if res_train_set is None:
            res_train_set = res_sub_train_set
            res_meta_set = res_sub_meta_set
            res_remaining_origin_labels = res_sub_remaining_origin_labels
        else:
            res_train_set = res_train_set.concat_validset(res_train_set, res_sub_train_set)
            res_meta_set = res_meta_set.concat_validset(res_meta_set, res_sub_meta_set)
            res_remaining_origin_labels = torch.cat([res_remaining_origin_labels, res_sub_remaining_origin_labels])
        label_id += 1
    if not type(train_dataset.targets) is torch.Tensor:
        res_remaining_origin_labels = res_remaining_origin_labels.numpy()

    return res_train_set, res_meta_set, res_remaining_origin_labels



def obtain_sample_pair_distance_bound(train_dataset, metaset, criterion, optimizer, trainloader, args, net, cached_sample_weights, valid_count, validset):
    if not args.cluster_method_two:
        if args.cluster_method_three:
            # valid_ids, new_valid_representations = get_representative_valid_ids3(trainloader, args, net, valid_count, cached_sample_weights = cached_sample_weights, existing_valid_representation = existing_valid_representation)
            full_sample_representation_tensor = get_representative_valid_ids2_4(train_dataset, criterion, optimizer, trainloader, args, net, valid_count, cached_sample_weights = cached_sample_weights, validset = validset, only_sample_representation=True)
            
        else:
            # train_loader, args, net, valid_count, cached_sample_weights = None, existing_valid_representation = None, existing_valid_set = None
            full_sample_representation_tensor = get_representative_valid_ids(trainloader, args, net, valid_count, cached_sample_weights = cached_sample_weights, only_sample_representation=True)
            
    else:

        full_sample_representation_tensor, valid_representations = get_representative_valid_ids2(criterion, optimizer, trainloader, args, net, valid_count, cached_sample_weights = cached_sample_weights, validset = validset, only_sample_representation=True, qualitiative=True)
    
    full_sample_representation_tensor, full_sample_representation_tensor2 = full_sample_representation_tensor

    valid_representations, valid_representations2 = valid_representations

    full_distance = compute_distance(args, args.cosin_dist, True, full_sample_representation_tensor, valid_representations, args.cuda)
    ratio = torch.sort(torch.abs(torch.sum(full_distance,dim=1))/torch.sum(torch.abs(full_distance),dim=1))[0]
    D_value = ((1+ratio)/(1-ratio)).min()
    args.logger.info("D value is::%f"%(D_value))
    full_sim = -(full_distance.cpu() - 1)

    full_train_output_probs_tensor_ls, full_train_output_origin_probs_tensor_ls = [], []

    full_meta_output_probs_tensor_ls, full_meta_output_origin_probs_tensor_ls = [], []

    train_output_probs_grad_tensor, train_output_origin_probs_tensor = obtain_grad_last_layer(trainloader, args, net)

    full_train_output_probs_tensor_ls.append(train_output_probs_grad_tensor)

    full_train_output_origin_probs_tensor_ls.append(train_output_origin_probs_tensor)

    metaloader = DataLoader(metaset, batch_size=args.batch_size, shuffle=False)

    meta_output_probs_grad_tensor, meta_output_origin_probs_tensor = obtain_grad_last_layer(metaloader, args, net)

    full_meta_output_probs_tensor_ls.append(meta_output_probs_grad_tensor)

    full_meta_output_origin_probs_tensor_ls.append(meta_output_origin_probs_tensor)

    if args.use_model_prov:
        args.all_layer = True

        get_extra_output_prob_ls(args, trainloader, net, full_train_output_probs_tensor_ls, full_train_output_origin_probs_tensor_ls)

        get_extra_output_prob_ls(args, metaloader, net, full_meta_output_probs_tensor_ls, full_meta_output_origin_probs_tensor_ls)




    full_sim_mat = torch.mm(train_output_probs_grad_tensor, torch.t(meta_output_probs_grad_tensor))*full_sim

    print(torch.abs(full_sim_mat).max())

    print(full_sim_mat.max())

    


def find_representative_samples0(criterion, optimizer, net, train_dataset,validset,  args, origin_labels, cached_sample_weights = None, valid_count = None):
    # valid_ratio = args.valid_ratio
    prob_gap_ls = torch.zeros(len(train_dataset))

    if validset is not None and args.total_valid_sample_count > 0 and args.total_valid_sample_count <= len(validset):
        args.logger.info("already collect enough samples, exit!!!!")
        sys.exit(1)

    if valid_count is None:
        if validset is not None:
            valid_count = len(validset) + args.valid_count#int(len(train_dataset)*valid_ratio)
        else:
            valid_count = args.valid_count

    pred_labels = torch.zeros(len(train_dataset), dtype =torch.long)

    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    # if validset is not None:
    #     existing_valid_representation = obtain_representations_for_valid_set(args, validset, net, criterion, optimizer)
    # else:
    existing_valid_representation = None

    

    if not args.cluster_method_two:
        if args.cluster_method_three:
            # valid_ids, new_valid_representations = get_representative_valid_ids3(trainloader, args, net, valid_count, cached_sample_weights = cached_sample_weights, existing_valid_representation = existing_valid_representation)
            valid_ids, new_valid_representations = get_representative_valid_ids2_4(train_dataset, criterion, optimizer, trainloader, args, net, valid_count, cached_sample_weights = cached_sample_weights, validset = validset)
            
        else:
            # train_loader, args, net, valid_count, cached_sample_weights = None, existing_valid_representation = None, existing_valid_set = None
            valid_ids, new_valid_representations = get_representative_valid_ids(trainloader, args, net, valid_count, cached_sample_weights = cached_sample_weights)
            if existing_valid_representation is not None:
                valid_ids = determine_new_valid_ids(args, valid_ids, new_valid_representations, existing_valid_representation, valid_count, cosine_dist = args.cosin_dist, is_cuda=args.cuda, all_layer=args.cluster_method_three)
    else:
        if args.qualitiative:
            valid_ids, new_valid_representations = get_representative_valid_ids2(criterion, optimizer, trainloader, args, net, valid_count, cached_sample_weights = cached_sample_weights, validset = validset, qualitiative = args.qualitiative, origin_label = origin_labels)
        else:
            valid_ids, new_valid_representations = get_representative_valid_ids2(criterion, optimizer, trainloader, args, net, valid_count, cached_sample_weights = cached_sample_weights, validset = validset, qualitiative = args.qualitiative)
        # if existing_valid_representation is not None:
        #     valid_ids = determine_new_valid_ids(args, valid_ids, new_valid_representations, existing_valid_representation, valid_count, cosine_dist = args.cosin_dist, is_cuda=args.cuda, all_layer = args.all_layer)
        # valid_ids, new_valid_representations = get_representative_valid_ids(trainloader, args, net, valid_count - len(validset), cached_sample_weights = cached_sample_weights, existing_valid_representation = existing_valid_representation, existing_valid_set=validset)

    torch.save(valid_ids, os.path.join(args.save_path, "valid_dataset_ids"))
    update_train_ids = torch.ones(len(train_dataset))
    if not args.include_valid_set_in_training:
        update_train_ids[valid_ids] = 0
    update_train_ids = update_train_ids.nonzero().view(-1)
    
    train_set, meta_set = split_train_valid_set_by_ids(args, train_dataset, origin_labels, valid_ids, update_train_ids)
    
    if args.qualitiative:
        obtain_sample_pair_distance_bound(train_dataset, meta_set, criterion, optimizer, trainloader, args, net, cached_sample_weights, valid_count, validset)

    remaining_origin_labels = origin_labels[update_train_ids]
    torch.save(origin_labels, os.path.join(args.save_path, "train_and_meta_labels"))

    return train_set, meta_set, remaining_origin_labels


def uncertainty_sample(criterion, optimizer, net, train_dataset, validset, args, origin_labels, cached_sample_weights=None):
    # if validset is not None and args.total_valid_sample_count > 0 and args.total_valid_sample_count <= len(validset):
    #     args.logger.info("already collect enough samples, exit!!!!")
    #     sys.exit(1)

    # if len(validset) + args.valid_count > args.total_valid_sample_count:
    #     args.valid_count = args.total_valid_sample_count - len(validset)
    
    vals = torch.zeros((train_dataset.targets.shape[0],))
    labels = torch.zeros((train_dataset.targets.shape[0],)).long()
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    for _, (indices, data, target) in enumerate(trainloader):
        if args.cuda:
            data = data.cuda()
        with torch.no_grad():
            output = net(data)
        vals[indices] = criterion(output, output).cpu()
        labels[indices] = target.type(torch.long).cpu()

    if args.clustering_by_class:
        valid_ids = []
        update_train_ids = []
        unique_label_set = set(labels.tolist())
        valid_size = int(args.valid_count / len(unique_label_set))
        all_indices = torch.arange(labels.shape[0])
        for l in unique_label_set:
            _, indices = torch.sort(vals[labels == l], descending=True)
            valid_ids.append(all_indices[labels == l][indices[:valid_size]])
            update_train_ids.append(all_indices[labels == l][indices[valid_size:]])
        valid_ids = torch.cat(valid_ids)
        update_train_ids = torch.cat(update_train_ids)
    else:
        valid_size = args.valid_count
        _, indices = torch.sort(vals, descending=True)
        valid_ids = indices[:valid_size]
        update_train_ids = indices[valid_size:]

    # torch.save(valid_ids, os.path.join(args.data_dir, "valid_dataset_ids"))
    train_set, meta_set = split_train_valid_set_by_ids(args, train_dataset, origin_labels, valid_ids, update_train_ids)
    remaining_origin_labels = origin_labels[update_train_ids]
    return train_set, meta_set, remaining_origin_labels

def certainty_sample(criterion, optimizer, net, train_dataset, validset, args, origin_labels, cached_sample_weights=None):
    # if validset is not None and args.total_valid_sample_count > 0 and args.total_valid_sample_count <= len(validset):
    #     args.logger.info("already collect enough samples, exit!!!!")
    #     sys.exit(1)

    # if len(validset) + args.valid_count > args.total_valid_sample_count:
    #     args.valid_count = args.total_valid_sample_count - len(validset)

    vals = torch.zeros((train_dataset.targets.shape[0],))
    labels = torch.zeros((train_dataset.targets.shape[0],)).long()
    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    for _, (indices, data, target) in enumerate(trainloader):
        if args.cuda:
            data = data.cuda()
        with torch.no_grad():
            output = net(data)
        vals[indices] = criterion(output, output).cpu()
        labels[indices] = target.type(torch.long).cpu()

    if args.clustering_by_class:
        valid_ids = []
        update_train_ids = []
        unique_label_set = set(labels.tolist())
        valid_size = int(args.valid_count / len(unique_label_set))
        all_indices = torch.arange(labels.shape[0])
        for l in unique_label_set:
            _, indices = torch.sort(vals[labels == l], descending=False)
            valid_ids.append(all_indices[labels == l][indices[:valid_size]])
            update_train_ids.append(all_indices[labels == l][indices[valid_size:]])
        valid_ids = torch.cat(valid_ids)
        update_train_ids = torch.cat(update_train_ids)
    else:
        valid_size = args.valid_count
        _, indices = torch.sort(vals, descending=False)
        valid_ids = indices[:valid_size]
        update_train_ids = indices[valid_size:]

    # torch.save(valid_ids, os.path.join(args.data_dir, "valid_dataset_ids"))
    train_set, meta_set = split_train_valid_set_by_ids(args, train_dataset, origin_labels, valid_ids, update_train_ids)
    remaining_origin_labels = origin_labels[update_train_ids]
    return train_set, meta_set, remaining_origin_labels


def find_representative_samples1(net, train_dataset,validset, train_transform, args, origin_labels, cached_sample_weights = None):
    # valid_ratio = args.valid_ratio
    prob_gap_ls = torch.zeros(len(train_dataset))

    prev_w_array_delta_ls_tensor = torch.load(os.path.join(args.prev_save_path, "cached_w_array_delta_ls"), map_location=torch.device('cpu'))
    
    # prev_w_array_delta_ls_tensor = prev_w_array_delta_ls_tensor

    prev_w_array_total_delta_tensor,_ = torch.max(torch.abs(prev_w_array_delta_ls_tensor), dim = 0)# 

    prev_w_array_total_delta_tensor = torch.sum(prev_w_array_delta_ls_tensor, dim = 0)# torch.load(os.path.join(args.prev_save_path, "cached_w_array_total_delta"), map_location=torch.device('cpu'))

    if args.cuda:
        prev_w_array_delta_ls_tensor = prev_w_array_delta_ls_tensor.cuda()
        prev_w_array_total_delta_tensor = prev_w_array_total_delta_tensor.cuda()

    sorted_prev_w_array_total_delta_tensor, sorted_prev_w_array_total_delta_tensor_idx = torch.sort(torch.abs(prev_w_array_total_delta_tensor), descending=False)

    all_sample_ids = torch.tensor(list(range(len(train_dataset))))

    valid_count = len(validset) + args.valid_count

    cluster_ids_x, cluster_centers = kmeans( 
        X=torch.transpose(prev_w_array_delta_ls_tensor,0,1), num_clusters=valid_count, distance='euclidean', device = prev_w_array_delta_ls_tensor.device, rand_init=args.rand_init)

    sorted_prev_w_array_idx_cluster_idx = cluster_ids_x[sorted_prev_w_array_total_delta_tensor_idx]

    selected_count = 0

    covered_cluster_id_set = set()

    idx = 0

    valid_idx_ls = []

    while selected_count < args.valid_count:
        curr_cluster_idx = sorted_prev_w_array_idx_cluster_idx[idx].item()
        curr_sample_idx = sorted_prev_w_array_total_delta_tensor_idx[idx].item()
        idx += 1
        # if curr_cluster_idx in covered_cluster_id_set:
        #     continue

        covered_cluster_id_set.add(curr_cluster_idx)

        valid_idx_ls.append(curr_sample_idx)

        selected_count += 1

    valid_ids = torch.tensor(valid_idx_ls)





    # valid_count = len(validset) + args.valid_count#int(len(train_dataset)*valid_ratio)

    # pred_labels = torch.zeros(len(train_dataset), dtype =torch.long)

    # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    # existing_valid_representation = obtain_representations_for_valid_set(args, validset, net)

    

    # if not args.cluster_method_two:
    #     valid_ids, new_valid_representations = get_representative_valid_ids(trainloader, args, net, valid_count, cached_sample_weights = cached_sample_weights)
    #     valid_ids = determine_new_valid_ids(valid_ids, new_valid_representations, existing_valid_representation, valid_count)
    # else:

    #     valid_ids, new_valid_representations = get_representative_valid_ids2(trainloader, args, net, valid_count, cached_sample_weights = cached_sample_weights)
    #     valid_ids = determine_new_valid_ids(valid_ids, new_valid_representations, existing_valid_representation, valid_count)
        # valid_ids, new_valid_representations = get_representative_valid_ids(trainloader, args, net, valid_count - len(validset), cached_sample_weights = cached_sample_weights, existing_valid_representation = existing_valid_representation, existing_valid_set=validset)

    # torch.save(valid_ids, os.path.join(args.save_path, "valid_dataset_ids"))
    update_train_ids = torch.ones(len(train_dataset))
    if not args.include_valid_set_in_training:
        update_train_ids[valid_ids] = 0
    update_train_ids = update_train_ids.nonzero().view(-1)
    
    train_set, meta_set = split_train_valid_set_by_ids(args, train_dataset, origin_labels, valid_ids, update_train_ids)

    remaining_origin_labels = origin_labels[update_train_ids]

    return train_set, valid_set, meta_set, remaining_origin_labels



    # valid_set = Subset(train_loader.dataset, valid_ids)
    # valid_set = new_mnist_dataset2(train_dataset.data[valid_ids].clone(), train_dataset.targets[valid_ids].clone())

    # meta_set = new_mnist_dataset2(train_dataset.data[valid_ids].clone(), train_dataset.targets[valid_ids].clone())




    # origin_train_labels = train_dataset.targets.clone()

    # test(train_loader, net, args)

    # # if args.flip_labels:

    # #     logging.info("add errors to train set")

    # #     train_dataset, _ = random_flip_labels_on_training(train_loader.dataset, ratio = args.err_label_ratio)

    # flipped_labels = None
    # if args.flip_labels:

    #     logging.info("add errors to train set")

    #     # train_dataset, _ = random_flip_labels_on_training(train_loader.dataset, ratio = args.err_label_ratio)
    #     flipped_labels = obtain_flipped_labels(train_dataset, args)


    

    # train_dataset, _, _ = partition_train_valid_dataset_by_ids(train_loader.dataset, origin_train_labels, flipped_labels, valid_ids)


    # # train_dataset, _, _ = partition_train_valid_dataset_by_ids(train_dataset, origin_train_labels, valid_ids)

    # train_loader, valid_loader, meta_loader, _ = create_data_loader(train_dataset, valid_set, meta_set, None, args)

    # test(valid_loader, net, args)
    # test(train_loader, net, args)

    # return train_loader, valid_loader, meta_loader

def get_dataloader_for_post_evaluations(args):
    trainset, valid_set, meta_set, remaining_origin_labels = load_train_valid_set(args)
    train_sampler = DistributedSampler(
        trainset,
        num_replicas=args.world_size,
        rank=args.local_rank,
        shuffle = False
    )
    meta_sampler = DistributedSampler(
        meta_set,
        num_replicas=args.world_size,
        rank=args.local_rank,
        shuffle = False
    )

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    metaloader = DataLoader(
        meta_set,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=meta_sampler,
    )
    # validloader = torch.utils.data.DataLoader(validset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)

    return trainloader, metaloader



def cache_train_valid_set(args, train_set, valid_set, meta_set, remaining_origin_labels):
    if train_set is not None:
        torch.save(train_set, os.path.join(args.save_path, "cached_train_set"))
    if valid_set is not None:
        torch.save(valid_set, os.path.join(args.save_path, "cached_valid_set"))
    if meta_set is not None:
        torch.save(meta_set, os.path.join(args.save_path, "cached_meta_set"))
    if remaining_origin_labels is not None:
        torch.save(remaining_origin_labels, os.path.join(args.save_path, "cached_train_origin_labels"))

def cache_test_set(args, test_set):
    test_set_path = os.path.join(args.save_path, "cached_test_set")
    if test_set is not None:
        torch.save(test_set, test_set_path)

def load_test_set(args):
    test_set_path = os.path.join(args.prev_save_path, "cached_test_set")
    test_set = None
    if os.path.exists(test_set_path):
        test_set = torch.load(test_set_path)
    return test_set
    
def load_train_valid_set(args):
    train_set_path = os.path.join(args.prev_save_path, "cached_train_set")
    valid_set_path = os.path.join(args.prev_save_path, "cached_valid_set")
    meta_set_path = os.path.join(args.prev_save_path, "cached_meta_set")
    origin_label_path = os.path.join(args.prev_save_path, "cached_train_origin_labels")
    if os.path.exists(train_set_path):
        train_set = torch.load(train_set_path)
    else:
        train_set = None
    
    if os.path.exists(valid_set_path):
        valid_set = torch.load(valid_set_path)
    else:
        valid_set = None
    
    if os.path.exists(meta_set_path):
        meta_set = torch.load(meta_set_path)
    else:
        meta_set = None

    if os.path.exists(origin_label_path):
        remaining_origin_labels = torch.load(origin_label_path)
    else:
        remaining_origin_labels = None
    return train_set, valid_set, meta_set, remaining_origin_labels

def evaluate_dataset_with_basic_models(args, dataset, model, criterion):
    
    curr_model_state = model.state_dict()

    basic_model_state = torch.load(os.path.join(args.data_dir, args.dataset + "_basic_model"), map_location=torch.device('cpu'))
    model.load_state_dict(basic_model_state)
    if args.cuda:
        model = model.cuda()

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    test(dataloader, model, criterion, args, prefix='Train')

    model.load_state_dict(curr_model_state)

def randomly_produce_valid_set(testset, rate = 0.1):
    rand_test_ids = torch.randperm(testset.__len__())

    selected_valid_count = int(len(rand_test_ids)*rate)

    selected_valid_ids = rand_test_ids[0:selected_valid_count]

    selected_test_ids = rand_test_ids[selected_valid_count:]

    # print("selected_valid_ids::", selected_valid_ids)
    # print("test targets::", testset.targets)

    validset = testset.get_subset_dataset(testset, selected_valid_ids)
    testset = testset.get_subset_dataset(testset, selected_test_ids)

    # selected_valid_data = testset.data[selected_valid_ids]

    

    # selected_test_data = testset.data[selected_test_ids]

    # if type(testset.targets) is torch.Tensor:

    #     selected_valid_labels = testset.targets[selected_valid_ids]

    #     selected_test_labels = testset.targets[selected_test_ids]
    # else:
    #     if type(testset.targets) is list:
    #         selected_valid_labels = [testset.targets[idx] for idx  in selected_valid_ids]

    #         selected_test_labels = [testset.targets[idx] for idx  in selected_test_ids]


    # validset = dataset_wrapper(selected_valid_data, selected_valid_labels, transform_test)

    # testset = dataset_wrapper(selected_test_data, selected_test_labels, transform_test)

    return validset, testset

def experiment_tag(args):
    if args.bias_classes:
        return args.dataset + "_imb_" + str(args.imb_factor)
    elif args.flip_labels and args.biased_flip:
        return args.dataset + "_biased_noise_" + str(args.err_label_ratio)
    elif args.flip_labels and args.adversarial_flip:
        return args.dataset + "_adversarial_noise_" + str(args.err_label_ratio)
    elif args.flip_labels:
        return args.dataset + "_unif_noise_" + str(args.err_label_ratio)
    else:
        return ""

def generate_class_biased_dataset(trainset, args, logger, testset, origin_labels):
    if not args.load_dataset:
        imb_trainset = exp_datasets.ImbalanceDataset(trainset, args.imb_factor)
        trainset = trainset.get_subset_dataset(trainset, torch.nonzero(imb_trainset.mask).view(-1))
        origin_labels = origin_labels[imb_trainset.mask]
        logger.info(f"Total number of training samples: {trainset.data.shape[0]}")
        logger.info(f"Total number of testing samples: {testset.data.shape[0]}")
        assert trainset.data.shape[0] == len(trainset)
        torch.save(trainset, os.path.join(
            args.data_dir,
            experiment_tag(args) + "_bias_class_dataset"),
        )
        torch.save(origin_labels, os.path.join(
            args.data_dir,
            experiment_tag(args) + "_bias_class_origin_labels"),
        )
    else:
        trainset = torch.load(os.path.join(
            args.data_dir,
            experiment_tag(args) + "_bias_class_dataset"),
        )
        origin_labels = torch.load(os.path.join(
            args.data_dir,
            experiment_tag(args) + "_bias_class_origin_labels"),
        )
    return trainset, origin_labels


def generate_noisy_dataset(args, trainset, logger):
    flipped_labels = None
    logger.info("add errors to train set")

    # train_dataset, _ = random_flip_labels_on_training(train_loader.dataset, ratio = args.err_label_ratio)
    # flipped_labels = obtain_flipped_labels(train_dataset, args)
    if not args.load_dataset:
        logger.info("Not loading dataset")
        if args.adversarial_flip:
            flipped_labels = adversarial_flip_labels(trainset, ratio=args.err_label_ratio)
        elif args.biased_flip or args.dataset == 'retina':
            flipped_labels = random_flip_labels_on_training3(trainset, ratio=args.err_label_ratio)
        else:
            flipped_labels = random_flip_labels_on_training2(trainset, ratio=args.err_label_ratio)
        torch.save(flipped_labels, os.path.join(
            args.data_dir,
            experiment_tag(args) + "_flipped_labels"),
        )
    else:
        logger.info("Loading dataset")
        flipped_label_dir = os.path.join(
            args.data_dir,
            experiment_tag(args) + "_flipped_labels",
        )
        if not os.path.exists(flipped_label_dir):
            if args.adversarial_flip:
                flipped_labels = adversarial_flip_labels(trainset, ratio=args.err_label_ratio)
            elif args.biased_flip:
                flipped_labels = random_flip_labels_on_training3(trainset, ratio=args.err_label_ratio)
            else:
                flipped_labels = random_flip_labels_on_training2(trainset, ratio=args.err_label_ratio)
            torch.save(flipped_labels, flipped_label_dir)
        flipped_labels = torch.load(flipped_label_dir)
    # logger.info("Label accuracy: %f"%(torch.sum(flipped_labels == torch.tensor(trainset.df['level'].values)) / len(trainset.targets)))
    trainset.targets = flipped_labels
    return trainset


def get_dataloader_for_meta(
    criterion,
    optimizer,
    args,
    split_method,
    logger,
    pretrained_model=None,
    cached_sample_weights=None
):
    trainset = None
    validset = None
    testset = None
    if args.load_dataset:
        logger.info("Loading dataset")
        trainset, validset, metaset, origin_labels = load_train_valid_set(args)
        testset = load_test_set(args)    

    else:
        logger.info("Not loading dataset")
        if args.dataset == 'cifar10':
            transform_train_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]
            transform_train = transforms.Compose(transform_train_list)
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

            trainset = torchvision.datasets.CIFAR10(
                root=os.path.join(args.data_dir, 'CIFAR-10'),
                train=True,
                download=True,
                transform=transform_train,
            )
            testset = torchvision.datasets.CIFAR10(
                root=os.path.join(args.data_dir, 'CIFAR-10'),
                train=False,
                download=True,
                transform=transform_test,
            )
            args.pool_len = 4
            if type(trainset.data) is numpy.ndarray:
                trainset = dataset_wrapper(numpy.copy(trainset.data), numpy.copy(trainset.targets), transform_train)
                testset = dataset_wrapper(numpy.copy(testset.data), numpy.copy(testset.targets), transform_test)
                origin_labels = numpy.copy(trainset.targets)
            else:
                trainset = dataset_wrapper(trainset.data.clone(), trainset.targets.clone(), transform_train)
                testset = dataset_wrapper(testset.data.clone(), testset.targets.clone(), transform_test)
                origin_labels = trainset.targets.clone()

        elif args.dataset == 'cifar100':
            # CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            # CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
            norm_mean, norm_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
            transform_train_list = [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std)
                # transforms.RandomCrop(32, padding=4),
                # transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                # transforms.Normalize(CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD),
            ]
            transform_train = transforms.Compose(transform_train_list)
            transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(norm_mean, norm_std),
                ])
            trainset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'CIFAR-100'), train=True, download=True, transform=transform_train)

            testset = torchvision.datasets.CIFAR100(root=os.path.join(args.data_dir, 'CIFAR-100'), train=False, download=True, transform=transform_test)
            args.pool_len = 4
            if type(trainset.data) is numpy.ndarray:
                trainset = dataset_wrapper(numpy.copy(trainset.data), numpy.copy(trainset.targets), transform_train)
                testset = dataset_wrapper(numpy.copy(testset.data), numpy.copy(testset.targets), transform_test)
                origin_labels = numpy.copy(trainset.targets)
            else:
                trainset = dataset_wrapper(trainset.data.clone(), trainset.targets.clone(), transform_train)
                testset = dataset_wrapper(testset.data.clone(), testset.targets.clone(), transform_test)
                origin_labels = trainset.targets.clone()

        elif args.dataset == 'retina':
            # df = pd.read_csv(args.data_dir + "/diabetic-retinopathy-detection/trainLabels.csv")
            # df['image'] = df['image'].apply(lambda x: args.data_dir + "/diabetic-retinopathy-detection/train/" + x + ".jpeg")
            # df['eye'] = df['image'].map(lambda x: 1 if x.split('_')[-1]=='left' else 0)
            # train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['level'])

            # trainset = RetinaDataset(train_df)
            # testset = RetinaDataset(test_df)

            trainset = torch.load(os.path.join(args.data_dir, "transformed_train_dataset"))
            testset = torch.load(os.path.join(args.data_dir, "transformed_test_dataset"))
            if len(trainset.targets.unique()) == 5:
                trainset.targets = (trainset.targets >= 2).type(torch.long).view(-1)
                testset.targets = (testset.targets >= 2).type(torch.long).view(-1)
            
            
            origin_labels = trainset.targets.clone()

        elif args.dataset == 'imagenet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            trainset = torch.load(os.path.join(args.data_dir, "train_processed_dataset"))
            testset = torch.load(os.path.join(args.data_dir, "test_processed_dataset"))
            # trainset = ImageNetDataset(torchvision.datasets.ImageNet(
            #     args.data_dir,
            #     "train",
            #     transform=transforms.Compose([
            #         transforms.RandomResizedCrop(224),
            #         transforms.RandomHorizontalFlip(),
            #         transforms.ToTensor(),
            #         normalize,
            #     ])))

            # testset = ImageNetDataset(torchvision.datasets.ImageNet(
            #     args.data_dir,
            #     "val",
            #     transform=transforms.Compose([
            #         transforms.Resize(256),
            #         transforms.CenterCrop(224),
            #         transforms.ToTensor(),
            #         normalize,
            #     ])))
            origin_labels = trainset.targets.clone()
        elif args.dataset == 'MNIST':
            transform_train = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])
            trainset = torchvision.datasets.MNIST(
                args.data_dir,
                train=True,
                download=True,
                transform=transform_train,
            )

            transform_test = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.1307,), (0.3081,))
            ])

            testset = torchvision.datasets.MNIST(
                args.data_dir,
                train=False,
                download=True,
                transform=transform_test,
            )
            args.pool_len = 4

            if type(trainset.data) is numpy.ndarray:
                trainset = dataset_wrapper(numpy.copy(trainset.data), numpy.copy(trainset.targets), transform_train)
                testset = dataset_wrapper(numpy.copy(testset.data), numpy.copy(testset.targets), transform_test)
                origin_labels = numpy.copy(trainset.targets)
            else:
                trainset = dataset_wrapper(trainset.data.clone(), trainset.targets.clone(), transform_train)
                testset = dataset_wrapper(testset.data.clone(), testset.targets.clone(), transform_test)
                origin_labels = trainset.targets.clone()

        elif args.dataset.startswith('sst'):
            label_list = None
            if args.dataset.startswith('sst2'):
                sstprocess = SST2Processor(args.data_dir)
                label_list = sstprocess.get_labels()
            elif args.dataset.startswith('sst5'):
                sstprocess = SST5Processor(args.data_dir)
                label_list = sstprocess.get_labels()
            trainset, validset, testset = create_train_valid_test_set(sstprocess, label_list, pretrained_model._tokenizer)
            origin_labels = trainset.targets.clone()

        elif args.dataset.startswith('imdb'):
            label_list = None
            
            sstprocess = imdb_Processor(args.data_dir)
            label_list = sstprocess.get_labels()
            trainset, validset, testset = create_train_valid_test_set(sstprocess, label_list, pretrained_model._tokenizer)
            origin_labels = trainset.targets.clone()

        elif args.dataset.startswith("trec"):
            label_list = None
            
            sstprocess = trec_Processor(args.data_dir)
            label_list = sstprocess.get_labels()
            trainset, validset, testset = create_train_valid_test_set(sstprocess, label_list, pretrained_model._tokenizer)
            origin_labels = trainset.targets.clone()

        if args.low_data:
            trainset = trainset.subsampling_dataset_by_class(trainset, num_per_class=args.low_data_num_samples_per_class)
            if type(trainset.targets) is numpy.ndarray:
                origin_labels = numpy.copy(trainset.targets)
            else:
                origin_labels = trainset.targets.clone()

    metaset = None
        
    assert trainset is not None, "Training set was not initialized"
    assert testset is not None, "Test set was not initialized"

    if args.valid_count is None:
        valid_ratio = args.valid_ratio
        valid_count = int(len(trainset)*valid_ratio)
        args.valid_count = valid_count
    else:
        valid_count = args.valid_count
        args.valid_count = valid_count

    remaining_origin_labels = []

    if args.clustering_by_class:
        selection_method = random_partition_train_valid_dataset0_by_class
    else:
        selection_method = random_partition_train_valid_dataset0
    if split_method == 'cluster':
        if args.clustering_by_class:
            selection_method = find_representative_samples0_by_class
        else:
            selection_method = find_representative_samples0
    elif split_method == 'uncertainty':
        selection_method = uncertainty_sample
    elif split_method == 'certainty':
        selection_method = certainty_sample
    elif args.ta_vaal_train:
        selection_method = random_partition_train_valid_dataset0
    # elif split_method == 'craige':
        

    remaining_origin_labels = origin_labels

    if args.do_train:
        logger.info("Do train")
        if args.bias_classes:
            trainset, remaining_origin_labels = generate_class_biased_dataset(trainset,
                    args, logger, testset, remaining_origin_labels)
        if args.real_noise:
            print("use real noise in cifar")
            if args.dataset.lower()== 'cifar10':
                trainset.targets = torch.load(os.path.join(args.data_dir, "CIFAR-N/CIFAR-10_human.pt"))['worse_label']
            elif args.dataset.lower() == 'cifar100':
                trainset.targets = torch.load(os.path.join(args.data_dir, "CIFAR-N/CIFAR-100_human.pt"))['noisy_label']
        elif args.flip_labels:
            trainset = generate_noisy_dataset(args, trainset, logger)
    else:
        if args.continue_label:
            trainset, validset, metaset, origin_labels = load_train_valid_set(args)

        if metaset is not None and args.total_valid_sample_count > 0 and args.total_valid_sample_count <= len(metaset):
            args.logger.info("already collect enough samples, exit!!!!")
            sys.exit(1)

        if metaset is not None and len(metaset) + args.valid_count > args.total_valid_sample_count:
            args.valid_count = args.total_valid_sample_count - len(metaset)


        if not args.ta_vaal_train:
            if not split_method == 'craige':
                if args.dataset == 'retina':
                    testset.targets = testset.targets.float()
                    trainset.targets = trainset.targets.float()
                    validset.targets = validset.targets.float()
                    if metaset is not None:
                        metaset.targets = metaset.targets.float()
                trainset, new_metaset, remaining_origin_labels = selection_method(
                    criterion,
                    optimizer,
                    pretrained_model,
                    trainset,
                    metaset,
                    args,
                    remaining_origin_labels,
                    cached_sample_weights=cached_sample_weights,
                )
            else:
                if metaset is not None:
                    active_strategy = CRAIGActive(metaset.data, metaset.targets, trainset.data, pretrained_model, torch.nn.CrossEntropyLoss(),  dataset_wrapper_X, dataset_wrapper, args.num_class, args.lr, "Supervised",  True, validset.transform, {"lr": args.lr, "batch_size": args.batch_size})
                else:
                    active_strategy = CRAIGActive(None, None, trainset.data, pretrained_model, torch.nn.CrossEntropyLoss(),  dataset_wrapper_X, dataset_wrapper, args.num_class, args.lr, "Supervised",  True, validset.transform, {"lr": args.lr, "batch_size": args.batch_size})

                valid_ids = active_strategy.select(args.valid_count)

                update_train_ids = torch.ones(len(trainset))
                if not args.include_valid_set_in_training:
                    update_train_ids[valid_ids] = 0
                update_train_ids = update_train_ids.nonzero().view(-1)
                trainset, new_metaset = split_train_valid_set_by_ids(args, trainset, remaining_origin_labels, valid_ids, update_train_ids)
                remaining_origin_labels = origin_labels[update_train_ids]

            if args.continue_label:
                metaset = metaset.concat_validset(metaset, new_metaset)
            else:
                metaset = new_metaset

        else:
            if metaset is None:
                trainset, metaset, remaining_origin_labels = selection_method(
                        criterion,
                        optimizer,
                        pretrained_model,
                        trainset,
                        None,
                        args,
                        remaining_origin_labels,
                        cached_sample_weights=cached_sample_weights,
                    )

                torch.save(metaset, os.path.join(args.prev_save_path, "cached_meta_set"))

        assert (metaset is not None), "Must use one of --continue_label or --ta_vaal_train"

        unique_labels_count = len(set(metaset.targets.tolist()))
        args.logger.info("unique label count in meta set::%d"%(unique_labels_count))
        
    if validset is None:
        validset, testset = randomly_produce_valid_set(testset, rate = 0.1)

    cache_train_valid_set(args, trainset, validset, metaset, remaining_origin_labels)
    cache_test_set(args, testset)

    if args.dataset == 'retina':
        testset.targets = testset.targets.float()
        trainset.targets = trainset.targets.float()
        validset.targets = validset.targets.float()
        if metaset is not None:
            metaset.targets = metaset.targets.float()

    train_sampler = DistributedSampler(
        trainset,
        num_replicas=args.world_size,
        rank=args.local_rank,
    )
    metaloader = None
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        num_workers=4*4, #args.num_workers,
        pin_memory=True,
        shuffle=False,
        sampler=train_sampler,
    )
    if metaset is not None:
        # meta_sampler = DistributedSampler(
        #     metaset,
        #     num_replicas=args.world_size,
        #     rank=args.local_rank,
        # )
        if not args.finetune:
            meta_sampler = RandomSampler(metaset, replacement=True, num_samples=args.epochs*len(trainloader)*args.batch_size*10)
            metaloader = DataLoader(
                metaset,
                batch_size=args.test_batch_size,
                num_workers=0,#args.num_workers,
                pin_memory=True,
                sampler=meta_sampler,
            )
        else:
            meta_sampler = DistributedSampler(
                metaset,
                num_replicas=args.world_size,
                rank=args.local_rank,
            )
            metaloader = DataLoader(
                metaset,
                batch_size=args.batch_size,
                num_workers=4*4, #args.num_workers,
                pin_memory=True,
                shuffle=False,
                sampler=meta_sampler,
            )       

    
    
    validloader = DataLoader(validset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)
    testloader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)

    return trainloader, validloader, metaloader, testloader, remaining_origin_labels
