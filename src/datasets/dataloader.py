
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
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from datasets.sst import *
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
        else:
            subset_data = numpy.copy(subset_data)
            subset_labels = numpy.copy(subset_labels)

        return dataset_wrapper(subset_data, subset_labels, transform, three_imgs, two_imgs)

    @staticmethod
    def concat_validset(dataset1, dataset2):
        valid_data_mat = dataset1.data
        valid_labels = dataset1.targets
        if type(valid_data_mat) is numpy.ndarray:
            valid_data_mat = numpy.concatenate((valid_data_mat, dataset2.data), axis = 0)
            valid_labels = numpy.concatenate((valid_labels, dataset2.targets), axis = 0)
            
        else:
            valid_data_mat = torch.cat([valid_data_mat, dataset2.data], dim = 0)
            valid_labels = torch.cat([valid_labels, dataset2.targets], dim = 0)
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
        clean_labels = train_dataset.targets.clone()
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

def random_partition_train_valid_dataset0(args, train_dataset, origin_labels):
    # valid_ratio = args.valid_ratio

    train_ids = torch.tensor(list(range(len(train_dataset))))

    rand_train_ids = torch.randperm(len(train_ids))

    valid_size = args.valid_count#int(len(train_dataset)*valid_ratio)

    valid_ids = rand_train_ids[0:valid_size]

    update_train_ids = rand_train_ids[valid_size:]

    torch.save(valid_ids, os.path.join(args.data_dir, "valid_dataset_ids"))
    train_set, meta_set = split_train_valid_set_by_ids(args, train_dataset, origin_labels, valid_ids, update_train_ids)
    remaining_origin_labels = origin_labels[update_train_ids]
    return train_set, meta_set, remaining_origin_labels



def obtain_representations_for_valid_set(args, valid_set, net, criterion, optimizer):
    validloader = torch.utils.data.DataLoader(valid_set, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)

    sample_representation_ls = []

    if not args.all_layer_grad:

        with torch.no_grad():

            # all_sample_representations = [None]*len(train_loader.dataset)

            for batch_id, (sample_ids, data, labels) in enumerate(validloader):

                if args.cuda:
                    data, labels = validloader.dataset.to_cuda(data, labels)
                    # data = data.cuda()
                    # labels = labels.cuda()
                
                sample_representation = net.feature_forward(data)
                sample_representation_ls.append(sample_representation)
        return torch.cat(sample_representation_ls)
    else:
        full_sample_representation_tensor, all_sample_ids = get_grad_by_example(args, validloader, net, criterion, optimizer)
        return full_sample_representation_tensor

    


def determine_new_valid_ids(args, valid_ids, new_valid_representations, existing_valid_representations, valid_count, cosine_dist = False, all_layer = False, is_cuda = False):

    if not args.all_layer_grad:

        if not cosine_dist:
            existing_new_dists = pairwise_distance(existing_valid_representations, new_valid_representations, is_cuda=is_cuda)
        else:
            if not all_layer:
                existing_new_dists = pairwise_cosine(existing_valid_representations, new_valid_representations, is_cuda=is_cuda)
            else:
                existing_new_dists = pairwise_cosine_ls(existing_valid_representations, new_valid_representations, is_cuda=is_cuda)
    
    else:
        existing_new_dists = pairwise_cosine2(existing_valid_representations, new_valid_representations, is_cuda=is_cuda)


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


    train_set, meta_set = split_train_valid_set_by_ids(args, train_dataset, origin_labels, valid_ids, update_train_ids)

    remaining_origin_labels = origin_labels[update_train_ids]

    return train_set, valid_set, meta_set, remaining_origin_labels
    




def find_representative_samples0(criterion, optimizer, net, train_dataset,validset,  args, origin_labels, cached_sample_weights = None):
    # valid_ratio = args.valid_ratio
    prob_gap_ls = torch.zeros(len(train_dataset))

    if validset is not None:
        valid_count = len(validset) + args.valid_count#int(len(train_dataset)*valid_ratio)
    else:
        valid_count = args.valid_count

    pred_labels = torch.zeros(len(train_dataset), dtype =torch.long)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    if validset is not None:
        existing_valid_representation = obtain_representations_for_valid_set(args, validset, net, criterion, optimizer)
    else:
        existing_valid_representation = None

    

    if not args.cluster_method_two:
        if args.cluster_method_three:
            valid_ids, new_valid_representations = get_representative_valid_ids3(trainloader, args, net, valid_count, cached_sample_weights = cached_sample_weights, existing_valid_representation = existing_valid_representation)
            
        else:
            valid_ids, new_valid_representations = get_representative_valid_ids(trainloader, args, net, valid_count, cached_sample_weights = cached_sample_weights)
            if existing_valid_representation is not None:
                valid_ids = determine_new_valid_ids(args, valid_ids, new_valid_representations, existing_valid_representation, valid_count, cosine_dist = args.cosin_dist, is_cuda=args.cuda)
    else:

        valid_ids, new_valid_representations = get_representative_valid_ids2(criterion, optimizer, trainloader, args, net, valid_count, cached_sample_weights = cached_sample_weights)
        if existing_valid_representation is not None:
            valid_ids = determine_new_valid_ids(args, valid_ids, new_valid_representations, existing_valid_representation, valid_count, cosine_dist = args.cosin_dist, is_cuda=args.cuda)
        # valid_ids, new_valid_representations = get_representative_valid_ids(trainloader, args, net, valid_count - len(validset), cached_sample_weights = cached_sample_weights, existing_valid_representation = existing_valid_representation, existing_valid_set=validset)

    torch.save(valid_ids, os.path.join(args.save_path, "valid_dataset_ids"))
    update_train_ids = torch.ones(len(train_dataset))
    if not args.include_valid_set_in_training:
        update_train_ids[valid_ids] = 0
    update_train_ids = update_train_ids.nonzero().view(-1)
    
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
        X=torch.transpose(prev_w_array_delta_ls_tensor,0,1), num_clusters=valid_count, distance='euclidean', device = prev_w_array_delta_ls_tensor.device)

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

    torch.save(valid_ids, os.path.join(args.save_path, "valid_dataset_ids"))
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

def randomly_produce_valid_set(testset, transform_test, rate = 0.1):
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

def get_dataloader_for_meta(
    criterion,
    optimizer,
    args,
    split_method,
    logger,
    pretrained_model=None,
    cached_sample_weights=None
):
    validset = None
    if args.dataset == 'cifar10':
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

    valid_ratio = args.valid_ratio
    valid_count = int(len(trainset)*valid_ratio)
    args.valid_count = valid_count

    remaining_origin_labels = []
    if split_method == 'random':
        logger.info("Split method: random")
        if args.continue_label:
            trainset, validset, metaset, origin_labels = load_train_valid_set(args)
            trainset, new_metaset, remaining_origin_labels = random_partition_train_valid_dataset0(args, trainset, origin_labels)
            # validset = concat_valid_set(validset, new_validset)
            metaset =metaset.concat_validset(metaset, new_metaset)
            # metaset = concat_valid_set(metaset, new_metaset)
        else:
            
            

            flipped_labels = None

            if args.bias_classes:
                trainset = datasets.ImbalanceDataset(trainset)
                origin_labels = origin_labels[trainset.mask]
                logger.info(f"Total number of training samples: {trainset.data.shape[0]}")
                logger.info(f"Total number of testing samples: {testset.data.shape[0]}")
            
            if args.flip_labels:
                logger.info("add errors to train set")

                if not args.load_dataset:
                    logger.info("Not loading dataset")
                    if args.adversarial_flip:
                        logger.info("Adding adversarial noise")
                        flipped_labels = adversarial_flip_labels(trainset, ratio=args.err_label_ratio)
                    else:
                        if args.biased_flip:
                            logger.info("Adding biased noise")
                            flipped_labels = random_flip_labels_on_training3(trainset, ratio=args.err_label_ratio)
                        else:
                            logger.info("Adding uniform noise")
                            flipped_labels = random_flip_labels_on_training4(trainset, ratio=args.err_label_ratio)
                    torch.save(flipped_labels, os.path.join(args.data_dir, args.dataset + "_flipped_labels"))
                else:
                    logger.info("Loading dataset")
                    flipped_label_dir = os.path.join(args.data_dir, args.dataset + "_flipped_labels")
                    if not os.path.exists(flipped_label_dir):
                        if args.adversarial_flip:
                            flipped_labels = adversarial_flip_labels(trainset, ratio=args.err_label_ratio)
                        else:
                            if args.biased_flip:
                                flipped_labels = random_flip_labels_on_training3(trainset, ratio=args.err_label_ratio)
                            else:
                                flipped_labels = random_flip_labels_on_training2(trainset, ratio=args.err_label_ratio)
                        torch.save(flipped_labels, flipped_label_dir)
                    flipped_labels = torch.load(flipped_label_dir)

                logger.info(torch.sum(torch.tensor(trainset.targets) == flipped_labels) / trainset.targets.shape[0])
                trainset.targets = flipped_labels

            trainset, metaset, remaining_origin_labels = random_partition_train_valid_dataset0(
                args,
                trainset,
                origin_labels,
            )
    elif split_method == 'cluster':
        if args.continue_label:
            trainset, validset, metaset, origin_labels = load_train_valid_set(args)
            # evaluate_dataset_with_basic_models(args, trainset, pretrained_model, criterion)
            # args.flip_labels = False
            # trainset, new_validset, new_metaset, remaining_origin_labels = find_representative_samples0(pretrained_model, trainset, validset, transform_train, args, origin_labels, cached_sample_weights = cached_sample_weights)


            # if args.cluster_method_three:
            #     # net, train_dataset,validset, train_transform, args, origin_labels
            #     trainset, new_validset, new_metaset, remaining_origin_labels = find_representative_samples1(pretrained_model, trainset, validset, transform_train, args, origin_labels)
            # else:
            trainset, new_metaset, remaining_origin_labels = find_representative_samples0(criterion, optimizer, pretrained_model, trainset, metaset, args, origin_labels, cached_sample_weights = cached_sample_weights)

            # metaset = concat_valid_set(metaset, new_metaset)
            metaset = metaset.concat_validset(metaset, new_metaset)
        else:
            # if type(trainset.data) is numpy.ndarray:
            #     trainset = dataset_wrapper(numpy.copy(trainset.data), numpy.copy(trainset.targets), transform_train)
            #     origin_labels = numpy.copy(trainset.targets)
            # else:
            #     trainset = dataset_wrapper(trainset.data.clone(), trainset.targets.clone(), transform_train)
            #     origin_labels = trainset.targets.clone()

            flipped_labels = None

            if args.bias_classes:
                trainset = datasets.ImbalanceDataset(trainset)
                origin_labels = origin_labels[trainset.mask]
                logger.info(f"Total number of training samples: {trainset.data.shape[0]}")
                logger.info(f"Total number of testing samples: {testset.data.shape[0]}")
            
            if args.flip_labels:

                logger.info("add errors to train set")

                # train_dataset, _ = random_flip_labels_on_training(train_loader.dataset, ratio = args.err_label_ratio)
                # flipped_labels = obtain_flipped_labels(train_dataset, args)
                if not args.load_dataset:
                    # flipped_labels = random_flip_labels_on_training2(trainset, ratio = args.err_label_ratio)
                    if args.biased_flip:
                        flipped_labels = random_flip_labels_on_training3(trainset, ratio=args.err_label_ratio)
                    else:
                        flipped_labels = random_flip_labels_on_training2(trainset, ratio=args.err_label_ratio)
                    torch.save(flipped_labels, os.path.join(args.data_dir, args.dataset + "_flipped_labels"))
                else:
                    flipped_label_dir = os.path.join(args.data_dir, args.dataset + "_flipped_labels")
                    if not os.path.exists(flipped_label_dir):
                        if args.biased_flip:
                            flipped_labels = random_flip_labels_on_training3(trainset, ratio=args.err_label_ratio)
                        else:
                            flipped_labels = random_flip_labels_on_training2(trainset, ratio=args.err_label_ratio)


                        # flipped_labels = random_flip_labels_on_training2(trainset, ratio = args.err_label_ratio)
                        torch.save(flipped_labels, flipped_label_dir)
                    flipped_labels = torch.load(flipped_label_dir)
                trainset.targets = flipped_labels
            # 
            # if args.init_cluster_by_confident:
            #     trainset, validset, metaset, remaining_origin_labels = init_sampling_valid_samples(pretrained_model, trainset, transform_train, args, origin_labels)
            # else:
            # if args.cluster_method_three:
            #     trainset, validset, metaset, remaining_origin_labels = find_representative_samples1(pretrained_model, trainset, transform_train, args, origin_labels)
            # else:
            trainset, metaset, remaining_origin_labels = find_representative_samples0(criterion, optimizer, pretrained_model, trainset, None, args, origin_labels, cached_sample_weights = cached_sample_weights)
        
    # if args.flip_labels:

    #     # train_dataset, _ = random_flip_labels_on_training(train_dataset, ratio = args.err_label_ratio)
    #     flipped_labels = obtain_flipped_labels(trainset, args)
    #     trainset.targets = flipped_labels.clone()
    # train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    # valid_sampler = torch.utils.data.distributed.DistributedSampler(validset)
    # meta_sampler = torch.utils.data.distributed.DistributedSampler(metaset)
    if validset is None:
        validset, testset = randomly_produce_valid_set(testset, transform_test, rate = 0.1)
    cache_train_valid_set(args, trainset, validset, metaset, remaining_origin_labels)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset,
        num_replicas=args.world_size,
        rank=args.local_rank,
    )
    meta_sampler = torch.utils.data.distributed.DistributedSampler(
        metaset,
        num_replicas=args.world_size,
        rank=args.local_rank,
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    metaloader = torch.utils.data.DataLoader(
        metaset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=meta_sampler,
    )
    validloader = torch.utils.data.DataLoader(validset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)

    return trainloader, validloader, metaloader, testloader, remaining_origin_labels
