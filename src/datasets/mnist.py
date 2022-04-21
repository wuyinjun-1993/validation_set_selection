import torch
import torchvision
from torch.utils.data import Subset, Dataset, DataLoader
import math
import os,sys
import logging
import numpy as np
from clustering_method.k_means import kmeans
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# from main.find_valid_set import *

class new_mnist_dataset(torchvision.datasets.MNIST):
    def __init__(self, *args, **kwargs):

        super(new_mnist_dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        image, target = super(new_mnist_dataset, self).__getitem__(index)

        return (index, image,target)

class new_mnist_dataset2(Dataset):
    def __init__(self, data_tensor, label_tensor):

        # super(new_mnist_dataset, self).__init__(*args, **kwargs)
        self.data = data_tensor
        self.targets = label_tensor

    def __getitem__(self, index):
        return (index, self.data[index], self.targets[index])
        # image, target = super(new_mnist_dataset, self).__getitem__(index)

        # return (index, image,target)

    def __len__(self):
        return len(self.data)




def pre_processing_mnist(train_loader, args, prefix = "train"):

    train_data_ls = []
    train_label_ls = []

    for (data,labels) in train_loader:
        train_data_ls.append(data)
        train_label_ls.append(labels)

    train_data_tensor = torch.cat(train_data_ls)
    train_label_tensor = torch.cat(train_label_ls)

    torch.save(train_data_tensor, os.path.join(args.data_dir, prefix + "_data_tensor"))
    torch.save(train_label_tensor, os.path.join(args.data_dir, prefix  + "_label_tensor"))


def pre_processing_mnist_main(args):
    train_dataset = torchvision.datasets.MNIST(args.data_dir, train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ]))

    test_dataset = torchvision.datasets.MNIST(args.data_dir, train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ]))

    train_loader, _, _, test_loader = create_data_loader(train_dataset, None, None, test_dataset, args, False)

    logging.info("start preprocessing train dataset")
    pre_processing_mnist(train_loader, args, 'train')
    logging.info("start preprocessing test dataset")
    pre_processing_mnist(test_loader, args, 'test')


def create_data_loader(train_dataset, valid_dataset, meta_dataset, test_dataset, args, train_shuffle = True):
    if train_dataset is not None:
        train_loader = torch.utils.data.DataLoader(train_dataset ,batch_size=args.batch_size, shuffle=train_shuffle)
    else:
        train_loader = None
    if valid_dataset is not None:
        valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=args.test_batch_size, shuffle=train_shuffle)
    else:
        valid_loader = None
    if meta_dataset is not None:
        meta_loader = torch.utils.data.DataLoader(meta_dataset,batch_size=args.test_batch_size, shuffle=train_shuffle)
    else:
        meta_loader = None

    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.test_batch_size, shuffle=False)
    else:
        test_loader = None
    return train_loader, valid_loader, meta_loader, test_loader



def mnist_to_device(inputs, args):
    index, data, targets = inputs
    if args.cuda:
        data = data.cuda()
        targets = targets.cuda()
        index = index.cuda()
    return (index, data, targets)


def partition_train_valid_dataset_by_ids(train_dataset, origin_train_labels, flipped_labels, valid_ids):
    if flipped_labels is not None:
        train_dataset.targets = flipped_labels.clone()
    all_train_ids = torch.ones(len(origin_train_labels))

    all_train_ids[valid_ids] = 0

    update_train_ids = torch.nonzero(all_train_ids).view(-1)
    valid_set = Subset(train_dataset, valid_ids)
    valid_set.targets = origin_train_labels[valid_ids]
    meta_set =Subset(train_dataset, valid_ids)
    meta_set.targets = origin_train_labels[valid_ids]
    train_set = Subset(train_dataset, update_train_ids)
    return train_set, valid_set, meta_set


def random_partition_train_valid_datastet(train_dataset, origin_train_labels, valid_ratio = 0.1):
    train_ids = torch.tensor(list(range(len(train_dataset))))

    rand_train_ids = torch.randperm(len(train_ids))

    valid_size = int(len(train_dataset)*valid_ratio)

    valid_ids = rand_train_ids[0:valid_size]

    update_train_ids = rand_train_ids[valid_size:]
    valid_set = Subset(train_dataset, valid_ids)
    valid_set.targets = origin_train_labels[valid_ids]
    meta_set =Subset(train_dataset, valid_ids)
    meta_set.targets = origin_train_labels[valid_ids]
    train_set = Subset(train_dataset, update_train_ids)
    return train_set, valid_set, meta_set


def random_partition_train_valid_dataset2(train_dataset, valid_ratio = 0.1):
    train_ids = torch.tensor(list(range(len(train_dataset))))

    rand_train_ids = torch.randperm(len(train_ids))

    valid_size = int(len(train_dataset)*valid_ratio)

    valid_ids = rand_train_ids[0:valid_size]

    update_train_ids = rand_train_ids[valid_size:]

    valid_set = new_mnist_dataset2(train_dataset.data[valid_ids].clone(), train_dataset.targets[valid_ids].clone())
    meta_set = new_mnist_dataset2(train_dataset.data[valid_ids].clone(), train_dataset.targets[valid_ids].clone())


    # valid_set = Subset(train_dataset, valid_ids)
    # valid_set.targets = origin_train_labels[valid_ids]
    # meta_set =Subset(train_dataset, valid_ids)
    # meta_set.targets = origin_train_labels[valid_ids]
    # train_set = new_mnist_dataset2(train_dataset.data[update_train_ids].clone(), train_dataset.targets[update_train_ids].clone())
    return update_train_ids, valid_set, meta_set


def get_mnist_dataset_without_valid_without_perturbations(args):
    test_dataset = new_mnist_dataset(args.data_dir, train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ]))

    train_dataset = new_mnist_dataset(args.data_dir, train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ]))

    train_loader, _,_,test_loader = create_data_loader(train_dataset, None, None, test_dataset, args)

    return train_loader, test_loader


def get_mnist_dataset_without_valid_without_perturbations2(args):
    train_data_tensor = torch.load(os.path.join(args.data_dir, "train_data_tensor"))

    train_label_tensor = torch.load(os.path.join(args.data_dir, "train_label_tensor"))

    test_data_tensor = torch.load(os.path.join(args.data_dir, "test_data_tensor"))

    test_label_tensor = torch.load(os.path.join(args.data_dir, "test_label_tensor"))

    test_dataset = new_mnist_dataset2(test_data_tensor, test_label_tensor)
    train_dataset = new_mnist_dataset2(train_data_tensor, train_label_tensor)
    # test_dataset = new_mnist_dataset(args.data_dir, train=False, download=True,
    #                                 transform=torchvision.transforms.Compose([
    #                                 torchvision.transforms.ToTensor(),
    #                                 torchvision.transforms.Normalize(
    #                                     (0.1307,), (0.3081,))
    #                                 ]))

    # train_dataset = new_mnist_dataset(args.data_dir, train=True, download=True,
    #                                 transform=torchvision.transforms.Compose([
    #                                 torchvision.transforms.ToTensor(),
    #                                 torchvision.transforms.Normalize(
    #                                     (0.1307,), (0.3081,))
    #                                 ]))

    train_loader, _,_,test_loader = create_data_loader(train_dataset, None, None, test_dataset, args)

    return train_loader, test_loader


def random_flip_labels_on_training(train_dataset, ratio = 0.5):
    full_ids = torch.tensor(list(range(len(train_dataset.targets))))

    full_rand_ids = torch.randperm(len(train_dataset.targets))

    err_label_count = int(len(full_rand_ids)*ratio)

    err_label_ids = full_rand_ids[0:err_label_count]

    correct_label_ids = full_rand_ids[err_label_count:]

    label_type_count = len(train_dataset.targets.unique())

    origin_labels = train_dataset.targets.clone()

    rand_err_labels = torch.randint(low=0,high=label_type_count-1, size=[len(err_label_ids)])

    # origin_err_labels = origin_labels[err_label_ids]

    # rand_err_labels[rand_err_labels == origin_err_labels] = (rand_err_labels[rand_err_labels == origin_err_labels] + 1)%label_type_count

    train_dataset.targets[err_label_ids] = rand_err_labels

    return train_dataset, origin_labels

def random_flip_labels_for_each_class(err_label_ids, origin_labels, label_type_count):
    origin_label_for_err_label_ids = origin_labels[err_label_ids]
    full_labels = set(list(range(label_type_count)))
    full_rand_err_label_ls = torch.zeros(len(err_label_ids), dtype = torch.long)

    for l in range(label_type_count):
        curr_sample_ids = (origin_label_for_err_label_ids == l).nonzero().view(-1)
        curr_remaining_labels = full_labels.difference(set([l]))
        curr_remaining_labels = torch.tensor(list(curr_remaining_labels))
        rand_err_label_ids = torch.randint(low=0,high=label_type_count-1, size=[len(curr_sample_ids)])
        rand_err_labels = curr_remaining_labels[rand_err_label_ids]
        full_rand_err_label_ls[curr_sample_ids] = rand_err_labels

    return full_rand_err_label_ls


def systematically_flip_labels(err_label_ids, origin_labels, label_type_count):
    origin_label_for_err_label_ids = origin_labels[err_label_ids]
    # full_labels = set(list(range(label_type_count)))
    rand_labels = torch.randperm(label_type_count)
    full_rand_err_label_ls = torch.zeros(len(err_label_ids), dtype = torch.long)

    for l in range(label_type_count):
        curr_sample_ids = (origin_label_for_err_label_ids == l).nonzero().view(-1)
        # curr_remaining_labels = full_labels.difference(set([l]))
        # curr_remaining_labels = torch.tensor(list(curr_remaining_labels))
        # rand_err_label_ids = torch.randint(low=0,high=label_type_count-1, size=[len(curr_sample_ids)])
        # rand_err_labels = curr_remaining_labels[rand_err_label_ids]
        full_rand_err_label_ls[curr_sample_ids] = rand_labels[l]

    return full_rand_err_label_ls


def uniform_mix_C(mixing_ratio, num_classes):
    '''
    returns a linear interpolation of a uniform matrix and an identity matrix
    '''
    return mixing_ratio * np.full((num_classes, num_classes), 1 / num_classes) + \
        (1 - mixing_ratio) * np.eye(num_classes)

def random_flip_labels_on_training2(train_dataset, ratio = 0.5):


    is_numpy = False

    if type(train_dataset.targets) is np.ndarray:
        is_numpy = True
        train_dataset.targets = torch.from_numpy(train_dataset.targets)

    label_type_count = len(train_dataset.targets.unique())

    C = uniform_mix_C(ratio, label_type_count)

    origin_labels = train_dataset.targets.clone()

    for i in range(len(origin_labels)):
        train_dataset.targets[i] = np.random.choice(label_type_count, p=C[origin_labels[i]])

    
    # full_ids = torch.tensor(list(range(len(train_dataset.targets))))

    # full_rand_ids = torch.randperm(len(train_dataset.targets))

    # err_label_count = int(len(full_rand_ids)*ratio)

    # err_label_ids = full_rand_ids[0:err_label_count]

    # correct_label_ids = full_rand_ids[err_label_count:]

    # label_type_count = len(train_dataset.targets.unique())

    # origin_labels = train_dataset.targets.clone()

    # rand_err_labels = random_flip_labels_for_each_class(err_label_ids, origin_labels, label_type_count)


    # # rand_err_labels = torch.randint(low=0,high=label_type_count-1, size=[len(err_label_ids)])

    # # origin_err_labels = origin_labels[err_label_ids]

    # # rand_err_labels[rand_err_labels == origin_err_labels] = (rand_err_labels[rand_err_labels == origin_err_labels] + 1)%label_type_count

    # origin_labels[err_label_ids] = rand_err_labels

    if is_numpy:
        train_dataset.targets = train_dataset.targets.numpy()

    return train_dataset.targets

def random_flip_labels_on_training3(train_dataset, ratio = 0.5):
    is_numpy = False

    if type(train_dataset.targets) is np.ndarray:
        is_numpy = True
        train_dataset.targets = torch.from_numpy(train_dataset.targets)
    full_ids = torch.tensor(list(range(len(train_dataset.targets))))

    full_rand_ids = torch.randperm(len(train_dataset.targets))

    err_label_count = int(len(full_rand_ids)*ratio)

    err_label_ids = full_rand_ids[0:err_label_count]

    correct_label_ids = full_rand_ids[err_label_count:]

    label_type_count = len(train_dataset.targets.unique())

    origin_labels = train_dataset.targets.clone()

    rand_err_labels = systematically_flip_labels(err_label_ids, origin_labels, label_type_count)


    # rand_err_labels = torch.randint(low=0,high=label_type_count-1, size=[len(err_label_ids)])

    # origin_err_labels = origin_labels[err_label_ids]

    # rand_err_labels[rand_err_labels == origin_err_labels] = (rand_err_labels[rand_err_labels == origin_err_labels] + 1)%label_type_count

    origin_labels[err_label_ids] = rand_err_labels

    if is_numpy:
        train_dataset.targets = train_dataset.targets.numpy()

    return origin_labels


def random_flip_labels_on_training4(train_dataset, ratio = 0.5):
    is_numpy = False

    if type(train_dataset.targets) is np.ndarray:
        is_numpy = True
        train_dataset.targets = torch.from_numpy(train_dataset.targets)

    full_rand_ids = torch.randperm(len(train_dataset.targets))
    err_label_count = int(len(full_rand_ids)*ratio)
    err_label_ids = full_rand_ids[0:err_label_count]

    label_type_count = len(train_dataset.targets.unique())
    origin_labels = train_dataset.targets.clone()

    rand_err_labels = torch.randint(0, label_type_count, err_label_ids.shape)

    origin_labels[err_label_ids] = rand_err_labels
    assert torch.all(origin_labels[err_label_ids] == rand_err_labels)

    if is_numpy:
        train_dataset.targets = train_dataset.targets.numpy()

    return origin_labels


def adversarial_flip_labels(dataset, ratio=0.5):
    print(dataset.data.shape)
    X = dataset.data.reshape((dataset.data.shape[0], -1))
    clusters, centers = kmeans(X.clone(), 10)
    a = torch.zeros(clusters.shape[0])
    b = torch.zeros(clusters.shape[0])
    for i in range(10):
        c_i = X[clusters == i].float()
        # Assign a as the mean distance between i and other points in cluster
        a[clusters == i] = ((1 / (c_i.shape[0] - 1))
            * torch.sum(torch.cdist(c_i[None], c_i[None])[0], dim=1))

        # Assign b as the smallest mean distance of i to all points in other
        # clusters
        for j in range(10):
            if i == j:
                continue
            c_j = X[clusters == j].float()
            b_tmp = (1 / c_j.shape[0]) * torch.sum(torch.cdist(c_i[None], c_j[None])[0],
                    dim=1)

            if (i == 0 and j == 1) or j == 0:
                b[clusters == i] = b_tmp
            else:
                for k in range(c_i.shape[0]):
                    if b_tmp[k] < b[clusters == i][k]:
                        index = torch.nonzero(clusters == i)[k]
                        b[index] = b_tmp[k]
                        assert (b[index] == b_tmp[k])

    silhouette_val = (b - a) / torch.max(torch.stack((a, b), dim=1), dim=1)[0]
    num_flip = math.ceil(dataset.data.shape[0] * ratio)
    adv_flip_index = torch.nonzero(silhouette_val < 0)
    rand_flip_index = torch.nonzero(silhouette_val >= 0)

    # Flip labels for adversarial samples and supplement with random samples if
    # there are not enough adversarial samples
    num_adv_flip = min(num_flip, adv_flip_index.shape[0])
    num_rand_flip = num_flip - num_adv_flip
    adv_flip_index = adv_flip_index[torch.randperm(adv_flip_index.shape[0])][:num_adv_flip]
    rand_flip_index = rand_flip_index[torch.randperm(rand_flip_index.shape[0])][:num_rand_flip]

    label_type_count = len(dataset.targets.unique())
    origin_labels = dataset.targets.clone()
    adv_err_labels = torch.randint(low=0,high=label_type_count-1,
            size=[len(adv_flip_index)])
    rand_err_labels = torch.randint(low=0,high=label_type_count-1,
            size=[len(rand_flip_index)])

    origin_labels[adv_flip_index.flatten()] = adv_err_labels
    origin_labels[rand_flip_index.flatten()] = rand_err_labels
    return origin_labels


def obtain_flipped_labels(train_dataset, args):
    if not args.load_dataset:

        flipped_labels = random_flip_labels_on_training2(train_dataset, ratio = args.err_label_ratio)

        torch.save(flipped_labels, os.path.join(args.data_dir, "flipped_labels"))

    else:
        flipped_labels = torch.load(os.path.join(args.data_dir, "flipped_labels"))
    return flipped_labels


def get_mnist_data_loader(args, partition_train_valid_dataset=random_partition_train_valid_datastet):

    test_dataset = new_mnist_dataset(args.data_dir, train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ]))

    if not args.load_dataset:

        train_dataset = new_mnist_dataset(args.data_dir, train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ]))
        origin_train_labels = train_dataset.targets.clone()
        flipped_labels = None
        if args.flip_labels:

            # train_dataset, _ = random_flip_labels_on_training(train_dataset, ratio = args.err_label_ratio)
            flipped_labels = obtain_flipped_labels(train_dataset, args)
            train_dataset.targets = flipped_labels.clone()

        train_dataset, valid_dataset, meta_dataset = partition_train_valid_dataset(train_dataset, origin_train_labels)

        
        # torch.save(origin_train_labels, )
        if not args.not_save_dataset:
            torch.save(train_dataset, os.path.join(args.data_dir, "train_dataset"))
            torch.save(valid_dataset, os.path.join(args.data_dir, "valid_dataset"))
            torch.save(meta_dataset, os.path.join(args.data_dir, "meta_dataset"))

    else:
        train_dataset = torch.load(os.path.join(args.data_dir, "train_dataset"))
        valid_dataset = torch.load(os.path.join(args.data_dir, "valid_dataset"))
        meta_dataset = torch.load(os.path.join(args.data_dir, "meta_dataset"))
    
    
    train_loader, valid_loader, meta_loader, test_loader = create_data_loader(train_dataset, valid_dataset, meta_dataset, test_dataset, args)

    return train_loader, valid_loader, meta_loader, test_loader


def get_mnist_data_loader2(args):

    train_data_tensor = torch.load(os.path.join(args.data_dir, "train_data_tensor"))

    train_label_tensor = torch.load(os.path.join(args.data_dir, "train_label_tensor"))

    test_data_tensor = torch.load(os.path.join(args.data_dir, "test_data_tensor"))

    test_label_tensor = torch.load(os.path.join(args.data_dir, "test_label_tensor"))

    test_dataset = new_mnist_dataset2(test_data_tensor, test_label_tensor)

    # test_dataset = new_mnist_dataset(args.data_dir, train=False, download=True,
    #                                 transform=torchvision.transforms.Compose([
    #                                 torchvision.transforms.ToTensor(),
    #                                 torchvision.transforms.Normalize(
    #                                     (0.1307,), (0.3081,))
    #                                 ]))

    # if not args.load_dataset:

    train_dataset = new_mnist_dataset2(train_data_tensor, train_label_tensor)
    # train_dataset = new_mnist_dataset(args.data_dir, train=True, download=True,
    #                             transform=torchvision.transforms.Compose([
    #                             torchvision.transforms.ToTensor(),
    #                             torchvision.transforms.Normalize(
    #                                 (0.1307,), (0.3081,))
    #                             ]))

    update_train_ids, valid_dataset, meta_dataset = random_partition_train_valid_dataset2(train_dataset, args.valid_ratio)
    # origin_train_labels = train_dataset.targets.clone()
    if args.flip_labels:

        # train_dataset, _ = random_flip_labels_on_training(train_dataset, ratio = args.err_label_ratio)
        flipped_labels = obtain_flipped_labels(train_dataset, args)
        train_dataset.targets = flipped_labels.clone()

    train_dataset = Subset(train_dataset, update_train_ids)
 
    
    # torch.save(origin_train_labels, )
    if not args.not_save_dataset:
        torch.save(train_dataset, os.path.join(args.data_dir, "train_dataset"))
        torch.save(valid_dataset, os.path.join(args.data_dir, "valid_dataset"))
        torch.save(meta_dataset, os.path.join(args.data_dir, "meta_dataset"))

    # else:
    #     train_dataset = torch.load(os.path.join(args.data_dir, "train_dataset"))
    #     valid_dataset = torch.load(os.path.join(args.data_dir, "valid_dataset"))
        # meta_dataset = torch.load(os.path.join(args.data_dir, "meta_dataset"))
    
    
    train_loader, valid_loader, meta_loader, test_loader = create_data_loader(train_dataset, valid_dataset, meta_dataset, test_dataset, args)

    return train_loader, valid_loader, meta_loader, test_loader