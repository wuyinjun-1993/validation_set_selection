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
    train_loader = torch.utils.data.DataLoader(train_dataset ,batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=args.test_batch_size, shuffle=True)
    meta_loader = torch.utils.data.DataLoader(valid_dataset,batch_size=args.test_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.test_batch_size, shuffle=False)
    return train_loader, valid_loader, meta_loader, test_loader



def mnist_to_device(inputs, args):
    index, data, targets = inputs
    if args.cuda:
        data = data.cuda()
        targets = targets.cuda()
        index = index.cuda()
    return (index, data, targets)


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


def random_flip_labels_on_training(train_dataset, ratio = 0.5):
    full_ids = torch.tensor(list(range(len(train_dataset.targets))))

    full_rand_ids = torch.randperm(len(train_dataset.targets))

    err_label_count = int(len(full_rand_ids)*ratio)

    err_label_ids = full_rand_ids[0:err_label_count]

    correct_label_ids = full_rand_ids[err_label_count:]

    label_type_count = len(train_dataset.targets.unique())

    origin_labels = train_dataset.targets.clone()

    rand_err_labels = torch.randint(low=0,high=label_type_count-1, size=[len(err_label_ids)])

    origin_err_labels = origin_labels[err_label_ids]

    rand_err_labels[rand_err_labels == origin_err_labels] = (rand_err_labels[rand_err_labels == origin_err_labels] + 1)%label_type_count

    train_dataset.targets[err_label_ids] = rand_err_labels

    return train_dataset, origin_labels


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

        torch.save(train_dataset, os.path.join(args.data_dir, "train_dataset"))
        torch.save(valid_dataset, os.path.join(args.data_dir, "valid_dataset"))
        torch.save(meta_dataset, os.path.join(args.data_dir, "meta_dataset"))

    else:
        train_dataset = torch.load(os.path.join(args.data_dir, "train_dataset"))
        valid_dataset = torch.load(os.path.join(args.data_dir, "valid_dataset"))
        meta_dataset = torch.load(os.path.join(args.data_dir, "meta_dataset"))
    
    
    train_loader, valid_loader, meta_loader, test_loader = create_data_loader(train_dataset, valid_dataset, meta_dataset, test_dataset, args)

    return train_loader, valid_loader, meta_loader, test_loader
    