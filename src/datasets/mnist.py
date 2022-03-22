import torch
import torchvision
from torch.utils.data import Subset
import os


class new_mnist_dataset(torchvision.datasets.MNIST):
    def __init__(self, *args, **kwargs):

        super(new_mnist_dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        image, target = super(new_mnist_dataset, self).__getitem__(index)

        return (index, image,target)


def create_data_loader(train_dataset, valid_dataset, meta_dataset, test_dataset, args):
    if train_dataset is not None:
        train_loader = torch.utils.data.DataLoader(train_dataset ,batch_size=args.batch_size, shuffle=True)
    else:
        train_loader = None
    if valid_dataset is not None:
        valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=args.test_batch_size, shuffle=True)
    else:
        valid_loader = None
    if meta_dataset is not None:
        meta_loader = torch.utils.data.DataLoader(meta_dataset,batch_size=args.test_batch_size, shuffle=True)
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


def partition_train_valid_dataset_by_ids(train_dataset, origin_train_labels, valid_ids):

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
        if args.flip_labels:

            train_dataset, _ = random_flip_labels_on_training(train_dataset, ratio = args.err_label_ratio)

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
    