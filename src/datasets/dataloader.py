
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import datasets
from torch.utils.data import Subset, Dataset, DataLoader
from PIL import ImageFilter
import random
from main.find_valid_set import *

from PIL import Image
import numpy
import os, sys

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

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
        if not type(img) is numpy.ndarray:
            img = Image.fromarray(img.numpy(), mode="L")
        else:
            img = Image.fromarray(img)
        if self.transform is not None:
            img1 = self.transform(img)

            if self.two_imgs:
                img2 = self.transform(img)
                return (img1, img2), target, index


            if self.three_imgs:
                img2 = self.transform(img)
                img3 = self.transform(img)
                return (img1, img2, img3), target, index

        return (index, img1, target)
        # image, target = super(new_mnist_dataset, self).__getitem__(index)

        # return (index, image,target)

    def __len__(self):
        return len(self.data)


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
            trainset = datasets.CIFAR10Instance(root=os.path.join(args.data_dir, 'CIFAR-10'), train=True, download=True, transform=transform_train, two_imgs=args.two_imgs, three_imgs=args.three_imgs)
        
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)



        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, sampler=train_sampler)

        testset = datasets.CIFAR10Instance(root=os.path.join(args.data_dir, 'CIFAR-10'), train=False, download=True, transform=transform_test)
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
            trainset = datasets.CIFAR100Instance(root=os.path.join(args.data_dir, 'CIFAR-100'), train=True, download=True, transform=transform_train, two_imgs=args.two_imgs, three_imgs=args.three_imgs)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, sampler=train_sampler)

        testset = datasets.CIFAR100Instance(root=os.path.join(args.data_dir, 'CIFAR-100'), train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)
        args.pool_len = 4
        ndata = trainset.__len__()

    elif args.dataset == 'stl10':
        trainset = datasets.STL10(root=os.path.join(args.data_dir, 'STL10'), split='train', download=True, transform=transform_train, two_imgs=args.two_imgs, three_imgs=args.three_imgs)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, sampler=train_sampler)

        testset = datasets.STL10(root=os.path.join(args.data_dir, 'STL10'), split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)
        args.pool_len = 7
        ndata = trainset.__len__()

    elif args.dataset == 'stl10-full':
        trainset = datasets.STL10(root=os.path.join(args.data_dir, 'STL10'), split='train+unlabeled', download=True, transform=transform_train, two_imgs=args.two_imgs, three_imgs=args.three_imgs)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, 
                            pin_memory=False, sampler=train_sampler)

        labeledTrainset = datasets.STL10(root=os.path.join(args.data_dir, 'STL10'), split='train', download=True, transform=transform_train, two_imgs=args.two_imgs)
        labeledTrain_sampler = torch.utils.data.distributed.DistributedSampler(labeledTrainset)
        labeledTrainloader = torch.utils.data.DataLoader(labeledTrainset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=2, pin_memory=False, sampler=labeledTrain_sampler)
        testset = datasets.STL10(root=os.path.join(args.data_dir, 'STL10'), split='test', download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)
        args.pool_len = 7
        ndata = labeledTrainset.__len__()

    elif args.dataset == 'kitchen':
        trainset = datasets.CIFARImageFolder(root=os.path.join(args.data_dir, 'Kitchen-HC/train'), train=True, transform=transform_train, two_imgs=args.two_imgs, three_imgs=args.three_imgs)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)
        testset = datasets.CIFARImageFolder(root=os.path.join(args.data_dir, 'Kitchen-HC/test'), train=False, transform=transform_test)
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

        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, sampler=train_sampler)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)
        args.pool_len = 4
        ndata = trainset.__len__()

    return trainloader, testloader, ndata

def concat_valid_set(valid_set, new_valid_set):
    valid_data_mat = valid_set.data
    valid_labels = valid_set.targets
    if type(valid_data_mat) is numpy.ndarray:
        valid_data_mat = numpy.concatenate((valid_data_mat, new_valid_set.data), axis = 0)
        valid_labels = numpy.concatenate((valid_labels, new_valid_set.targets), axis = 0)
        
    else:
        valid_data_mat = torch.cat([valid_data_mat, new_valid_set.data], dim = 0)
        valid_labels = torch.cat([valid_labels, new_valid_set.targets], dim = 0)
    valid_set = dataset_wrapper(valid_data_mat, valid_labels, valid_set.transform)

    return valid_set

def split_train_valid_set_by_ids(args, train_dataset, origin_labels, valid_ids, update_train_ids, transform):
    if not type(train_dataset.data) is numpy.ndarray:
        valid_data = train_dataset.data[valid_ids].clone()

        valid_labels = origin_labels[valid_ids].clone()

        meta_data = valid_data.clone()

        meta_labels = valid_labels.clone()

        train_data = train_dataset.data[update_train_ids].clone()

        train_labels = train_dataset.targets[update_train_ids].clone()
    else:
        valid_data = numpy.copy(train_dataset.data[valid_ids])

        valid_labels = numpy.copy(origin_labels[valid_ids])

        meta_data = numpy.copy(valid_data)

        meta_labels = numpy.copy(valid_labels)

        train_data = numpy.copy(train_dataset.data[update_train_ids])

        train_labels = numpy.copy(train_dataset.targets[update_train_ids])

    train_set = dataset_wrapper(train_data, train_labels, transform)

    valid_set = dataset_wrapper(valid_data, valid_labels, transform)

    meta_set = dataset_wrapper(meta_data, meta_labels, transform)

    # flipped_labels = None
    # if args.flip_labels:

    #     logging.info("add errors to train set")

    #     # train_dataset, _ = random_flip_labels_on_training(train_loader.dataset, ratio = args.err_label_ratio)
    #     # flipped_labels = obtain_flipped_labels(train_dataset, args)
    #     flipped_labels = random_flip_labels_on_training2(train_set, ratio = args.err_label_ratio)
    #     train_set.targets = flipped_labels

    return train_set, valid_set, meta_set

def random_partition_train_valid_datastet0(args, train_dataset, transform, origin_labels):
    # valid_ratio = args.valid_ratio

    train_ids = torch.tensor(list(range(len(train_dataset))))

    rand_train_ids = torch.randperm(len(train_ids))

    valid_size = args.valid_count#int(len(train_dataset)*valid_ratio)

    valid_ids = rand_train_ids[0:valid_size]

    update_train_ids = rand_train_ids[valid_size:]

    torch.save(valid_ids, os.path.join(args.data_dir, "valid_dataset_ids"))
    train_set, valid_set, meta_set = split_train_valid_set_by_ids(args, train_dataset, origin_labels, valid_ids, update_train_ids, transform)
    remaining_origin_labels = origin_labels[update_train_ids]
    return train_set, valid_set, meta_set, remaining_origin_labels



def obtain_representations_for_valid_set(args, valid_set, net):
    validloader = torch.utils.data.DataLoader(valid_set, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)

    sample_representation_ls = []

    with torch.no_grad():

        # all_sample_representations = [None]*len(train_loader.dataset)

        for batch_id, (sample_ids, data, labels) in enumerate(validloader):

            if args.cuda:
                data = data.cuda()
                # labels = labels.cuda()
            
            sample_representation = net.feature_forward(data)
            sample_representation_ls.append(sample_representation)

    return torch.cat(sample_representation_ls)


def determine_new_valid_ids(valid_ids, new_valid_representations, existing_valid_representations, valid_count):

    existing_new_dists = pairwise_distance(existing_valid_representations, new_valid_representations, device = new_valid_representations.device)

    nearset_new_valid_distance,_ = torch.min(existing_new_dists, dim = 0)

    sorted_min_distance, sorted_min_sample_ids = torch.sort(nearset_new_valid_distance, descending=True)

    remaining_valid_ids = valid_ids[sorted_min_sample_ids[0:valid_count - existing_valid_representations.shape[0]]]
    

    



    # nearset_new_valid_ids = torch.argmin(existing_new_dists, dim = 1)


    # unique_nearest_new_valid_ids = torch.unique(nearset_new_valid_ids)

    # remaining_new_valid_id_tensor =  torch.ones(new_valid_representations.shape[0])

    # remaining_new_valid_id_tensor[unique_nearest_new_valid_ids] = 0

    # remaining_valid_ids = valid_ids[remaining_new_valid_id_tensor.nonzero().view(-1)]

    # if len(remaining_valid_ids) > valid_count - existing_valid_representations.shape[0]:
    #     existing_new_dists = existing_new_dists[:, remaining_new_valid_id_tensor.nonzero().view(-1)]
    #     nearset_new_valid_distances, nearset_new_valid_ids = torch.min(existing_new_dists, dim = 0)

    #     sorted_nearset_new_valid_distances, sorted_nearset_new_valid_ids = torch.sort(nearset_new_valid_distances.view(-1), descending=False)

    #     remaining_valid_ids = remaining_valid_ids[nearset_new_valid_ids[sorted_nearset_new_valid_ids[len(remaining_valid_ids) -(valid_count - existing_valid_representations.shape[0]):]]]

        # nearset_new_valid_ids[sorted_nearset_new_valid_ids]


    return remaining_valid_ids

def init_sampling_valid_samples(net, train_dataset, train_transform, args, origin_labels):
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

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


    train_set, valid_set, meta_set = split_train_valid_set_by_ids(args, train_dataset, origin_labels, valid_ids, update_train_ids, train_transform)

    remaining_origin_labels = origin_labels[update_train_ids]

    return train_set, valid_set, meta_set, remaining_origin_labels
    




def find_representative_samples0(net, train_dataset,validset, train_transform, args, origin_labels, cached_sample_weights = None):
    # valid_ratio = args.valid_ratio
    prob_gap_ls = torch.zeros(len(train_dataset))


    valid_count = len(validset) + args.valid_count#int(len(train_dataset)*valid_ratio)

    pred_labels = torch.zeros(len(train_dataset), dtype =torch.long)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    existing_valid_representation = obtain_representations_for_valid_set(args, validset, net)

    

    if not args.cluster_method_two:
        valid_ids, new_valid_representations = get_representative_valid_ids(trainloader, args, net, valid_count, cached_sample_weights = cached_sample_weights)
        valid_ids = determine_new_valid_ids(valid_ids, new_valid_representations, existing_valid_representation, valid_count)
    else:

        valid_ids, new_valid_representations = get_representative_valid_ids2(trainloader, args, net, valid_count, cached_sample_weights = cached_sample_weights)
        valid_ids = determine_new_valid_ids(valid_ids, new_valid_representations, existing_valid_representation, valid_count)
        # valid_ids, new_valid_representations = get_representative_valid_ids(trainloader, args, net, valid_count - len(validset), cached_sample_weights = cached_sample_weights, existing_valid_representation = existing_valid_representation, existing_valid_set=validset)

    torch.save(valid_ids, os.path.join(args.save_path, "valid_dataset_ids"))
    update_train_ids = torch.ones(len(train_dataset))
    if not args.include_valid_set_in_training:
        update_train_ids[valid_ids] = 0
    update_train_ids = update_train_ids.nonzero().view(-1)
    
    train_set, valid_set, meta_set = split_train_valid_set_by_ids(args, train_dataset, origin_labels, valid_ids, update_train_ids, train_transform)

    remaining_origin_labels = origin_labels[update_train_ids]

    return train_set, valid_set, meta_set, remaining_origin_labels


def find_representative_samples1(net, train_dataset,validset, train_transform, args, origin_labels, cached_sample_weights = None):
    # valid_ratio = args.valid_ratio
    prob_gap_ls = torch.zeros(len(train_dataset))

    prev_w_array_delta_ls_tensor = torch.load(os.path.join(args.prev_save_path, "cached_w_array_delta_ls"), map_location=torch.device('cpu'))
    
    prev_w_array_total_delta_tensor = torch.load(os.path.join(args.prev_save_path, "cached_w_array_total_delta"), map_location=torch.device('cpu'))

    if args.cuda:
        prev_w_array_delta_ls_tensor = prev_w_array_delta_ls_tensor.cuda()
        prev_w_array_total_delta_tensor = prev_w_array_total_delta_tensor.cuda()

    sorted_prev_w_array_total_delta_tensor, sorted_prev_w_array_total_delta_tensor_idx = torch.sort(torch.abs(prev_w_array_total_delta_tensor), descending=False)

    all_sample_ids = torch.tensor(list(range(len(train_dataset))))

    valid_count = len(validset) + args.valid_count

    cluster_ids_x, cluster_centers = kmeans(
        X=prev_w_array_delta_ls_tensor, num_clusters=valid_count, distance='euclidean', device=prev_w_array_total_delta_tensor.device)

    sorted_prev_w_array_idx_cluster_idx = cluster_ids_x[sorted_prev_w_array_total_delta_tensor_idx]

    selected_count = 0

    covered_cluster_id_set = set()

    idx = 0

    valid_idx_ls = []

    while selected_count < args.valid_count:
        curr_cluster_idx = sorted_prev_w_array_idx_cluster_idx[idx].item()
        curr_sample_idx = sorted_prev_w_array_total_delta_tensor_idx[idx].item()
        idx += 1
        if curr_cluster_idx in covered_cluster_id_set:
            continue

        covered_cluster_id_set.add(curr_cluster_idx)

        valid_idx_ls.append(curr_sample_idx)

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

    torch.save(valid_ids, os.path.join(args.save_path, "valid_dataset_ids"))
    update_train_ids = torch.ones(len(train_dataset))
    if not args.include_valid_set_in_training:
        update_train_ids[valid_ids] = 0
    update_train_ids = update_train_ids.nonzero().view(-1)
    
    train_set, valid_set, meta_set = split_train_valid_set_by_ids(args, train_dataset, origin_labels, valid_ids, update_train_ids, train_transform)

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

def cache_train_valid_set(args, train_set, valid_set, meta_set, remaining_origin_labels):
    torch.save(train_set, os.path.join(args.save_path, "cached_train_set"))
    torch.save(valid_set, os.path.join(args.save_path, "cached_valid_set"))
    torch.save(meta_set, os.path.join(args.save_path, "cached_meta_set"))
    torch.save(remaining_origin_labels, os.path.join(args.save_path, "cached_train_origin_labels"))

def load_train_valid_set(args):
    train_set = torch.load(os.path.join(args.prev_save_path, "cached_train_set"))
    valid_set = torch.load(os.path.join(args.prev_save_path, "cached_valid_set"))
    meta_set = torch.load(os.path.join(args.prev_save_path, "cached_meta_set"))
    remaining_origin_labels = torch.load(os.path.join(args.prev_save_path, "cached_train_origin_labels"))
    return train_set, valid_set, meta_set, remaining_origin_labels

def evaluate_dataset_with_basic_models(args, dataset, model, criterion):
    
    curr_model_state = model.state_dict()

    basic_model_state = torch.load(os.path.join(args.data_dir, args.dataset + "_basic_model"), map_location=torch.device('cpu'))
    model.load_state_dict(basic_model_state)
    if args.cuda:
        model = model.cuda()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    test(dataloader, model, criterion, args, prefix='Train')

    model.load_state_dict(curr_model_state)

def get_dataloader_for_meta(args, criterion, split_method, pretrained_model=None, cached_sample_weights = None):
    if args.dataset == 'cifar10':
        # transform_train_list = [
        #         transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
        #         transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        #         transforms.RandomGrayscale(p=0.2),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #     ]
        transform_train_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        transform_train = transforms.Compose(transform_train_list)
        transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'CIFAR-10'), train=True, download=True, transform=transform_train)
        # trainset = datasets.CIFAR10Instance(root=os.path.join(args.data_dir, 'CIFAR-10'), train=True, download=True, transform=transform_train, two_imgs=args.two_imgs, three_imgs=args.three_imgs)

        # trainset, validset, metaset = split_train_valid_func(args, trainset, transform_train, kwargs)

        testset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'CIFAR-10'), train=False, download=True, transform=transform_test)
        # testset = datasets.CIFAR10Instance(root=os.path.join(args.data_dir, 'CIFAR-10'), train=False, download=True, transform=transform_test)
        args.pool_len = 4
        # ndata = trainset.__len__()

    elif args.dataset == 'MNIST':

        transform_train = torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])
        trainset = torchvision.datasets.MNIST(args.data_dir, train=True, download=True,
                                    transform=transform_train)

        transform_test = torchvision.transforms.Compose([
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(
                                            (0.1307,), (0.3081,))
                                        ])

        testset = torchvision.datasets.MNIST(args.data_dir, train=False, download=True,
                                        transform=transform_test)
        args.pool_len = 4

    valid_ratio = args.valid_ratio
    valid_count = int(len(trainset)*valid_ratio)
    args.valid_count = valid_count

    if split_method == 'random':
        logging.info("Split method: random")
        if args.continue_label:
            trainset, validset, metaset, origin_labels = load_train_valid_set(args)
            trainset, new_validset, new_metaset, remaining_origin_labels = random_partition_train_valid_datastet0(args, trainset, transform_train, origin_labels)
            validset = concat_valid_set(validset, new_validset)
            metaset = concat_valid_set(metaset, new_metaset)
            cache_train_valid_set(args, trainset, validset, metaset, remaining_origin_labels)
        else:

            if type(trainset.data) is numpy.ndarray:
                trainset = dataset_wrapper(numpy.copy(trainset.data), numpy.copy(trainset.targets), transform_train)
                origin_labels = numpy.copy(trainset.targets)
            else:
                trainset = dataset_wrapper(trainset.data.clone(), trainset.targets.clone(), transform_train)
                origin_labels = trainset.targets.clone()

            flipped_labels = None
            
            if args.flip_labels:

                logging.info("add errors to train set")

                # train_dataset, _ = random_flip_labels_on_training(train_loader.dataset, ratio = args.err_label_ratio)
                # flipped_labels = obtain_flipped_labels(train_dataset, args)
                if not args.load_dataset:
                    logging.info("Not loading dataset")
                    if args.adversarial_flip:
                        flipped_labels = adversarial_flip_labels(trainset, ratio=args.err_label_ratio)
                    else:
                        flipped_labels = random_flip_labels_on_training2(trainset, ratio=args.err_label_ratio)
                    torch.save(flipped_labels, os.path.join(args.data_dir, args.dataset + "_flipped_labels"))
                else:
                    logging.info("Loading dataset")
                    flipped_label_dir = os.path.join(args.data_dir, args.dataset + "_flipped_labels")
                    if not os.path.exists(flipped_label_dir):
                        if args.adversarial_flip:
                            flipped_labels = adversarial_flip_labels(trainset, ratio=args.err_label_ratio)
                        else:
                            flipped_labels = random_flip_labels_on_training2(trainset, ratio=args.err_label_ratio)
                        torch.save(flipped_labels, flipped_label_dir)
                    flipped_labels = torch.load(flipped_label_dir)
                trainset.targets = flipped_labels
            trainset, validset, metaset, remaining_origin_labels = random_partition_train_valid_datastet0(args, trainset, transform_train, origin_labels)

            cache_train_valid_set(args, trainset, validset, metaset, remaining_origin_labels)

    elif split_method == 'cluster':
        if args.continue_label:
            trainset, validset, metaset, origin_labels = load_train_valid_set(args)
            # evaluate_dataset_with_basic_models(args, trainset, pretrained_model, criterion)
            args.flip_labels = False
            trainset, new_validset, new_metaset, remaining_origin_labels = find_representative_samples0(pretrained_model, trainset, validset, transform_train, args, origin_labels, cached_sample_weights = cached_sample_weights)


            validset = concat_valid_set(validset, new_validset)
            metaset = concat_valid_set(metaset, new_metaset)
            cache_train_valid_set(args, trainset, validset, metaset, remaining_origin_labels)
            # evaluate_dataset_with_basic_models(args, trainset, pretrained_model, criterion)
            # evaluate_dataset_with_basic_models(args, validset, pretrained_model, criterion)

            # print()
        else:
            if type(trainset.data) is numpy.ndarray:
                trainset = dataset_wrapper(numpy.copy(trainset.data), numpy.copy(trainset.targets), transform_train)
                origin_labels = numpy.copy(trainset.targets)
            else:
                trainset = dataset_wrapper(trainset.data.clone(), trainset.targets.clone(), transform_train)
                origin_labels = trainset.targets.clone()

            flipped_labels = None
            
            if args.flip_labels:

                logging.info("add errors to train set")

                # train_dataset, _ = random_flip_labels_on_training(train_loader.dataset, ratio = args.err_label_ratio)
                # flipped_labels = obtain_flipped_labels(train_dataset, args)
                if not args.load_dataset:
                    flipped_labels = random_flip_labels_on_training2(trainset, ratio = args.err_label_ratio)
                    torch.save(flipped_labels, os.path.join(args.data_dir, args.dataset + "_flipped_labels"))
                else:
                    flipped_label_dir = os.path.join(args.data_dir, args.dataset + "_flipped_labels")
                    if not os.path.exists(flipped_label_dir):
                        flipped_labels = random_flip_labels_on_training2(trainset, ratio = args.err_label_ratio)
                        torch.save(flipped_labels, flipped_label_dir)
                    flipped_labels = torch.load(flipped_label_dir)
                trainset.targets = flipped_labels
            # 
            # if args.init_cluster_by_confident:
            #     trainset, validset, metaset, remaining_origin_labels = init_sampling_valid_samples(pretrained_model, trainset, transform_train, args, origin_labels)
            # else:
            if args.cluster_method_three:
                trainset, validset, metaset, remaining_origin_labels = find_representative_samples1(pretrained_model, trainset, transform_train, args, origin_labels)
            else:
                trainset, validset, metaset, remaining_origin_labels = find_representative_samples0(pretrained_model, trainset, transform_train, args, origin_labels)
            cache_train_valid_set(args, trainset, validset, metaset, remaining_origin_labels)
        
    # if args.flip_labels:

    #     # train_dataset, _ = random_flip_labels_on_training(train_dataset, ratio = args.err_label_ratio)
    #     flipped_labels = obtain_flipped_labels(trainset, args)
    #     trainset.targets = flipped_labels.clone()
    # train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    # valid_sampler = torch.utils.data.distributed.DistributedSampler(validset)
    # meta_sampler = torch.utils.data.distributed.DistributedSampler(metaset)
    testset = dataset_wrapper(testset.data, testset.targets, transform_test)
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    # valid_sampler = torch.utils.data.distributed.DistributedSampler(validset)
    meta_sampler = torch.utils.data.distributed.DistributedSampler(metaset)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, sampler=train_sampler)

    
    metaloader = torch.utils.data.DataLoader(metaset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False, sampler=meta_sampler)

    validloader = torch.utils.data.DataLoader(validset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)





    return trainloader, validloader, metaloader, testloader

