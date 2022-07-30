import torch
import os,sys

from skimage import io, transform


sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from common.parse_args import *
from exp_datasets.mnist import *
from models.DNN import *
from common.utils import *
from tqdm.notebook import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import itertools
import torch_higher as higher
from main.find_valid_set import *
from main.meta_reweighting_rl import *
from exp_datasets.dataloader import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
from utils.logger import setup_logger
import models
from lib.NCECriterion import NCESoftmaxLoss
from lib.lr_scheduler import get_scheduler
from lib.BootstrappingLoss import SoftBootstrappingLoss, HardBootstrappingLoss
from models.ResNet import *
# from models.resnet3 import *
from models.bert import *
import collections
from models.LeNet5 import *
import models.TAVAAL
import pandas as pd

class rawDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        
        self.root_dir = root_dir
        self.transform = transform
        self.preprocess_csv_file_names()

    def preprocess_csv_file_names(self):
        file_name_ls = []
        img_name_ls = []
        label_ls = []
        for idx in range(self.landmarks_frame.shape[0]):
            file_name = self.landmarks_frame.iloc[idx, 0]
            landmarks = self.landmarks_frame.iloc[idx, 1]
            landmarks = torch.tensor([landmarks])
            img_name = os.path.join(self.root_dir, file_name) + ".jpeg"

            if os.path.exists(img_name):
                file_name_ls.append(file_name)
                img_name_ls.append(img_name)
                label_ls.append(landmarks)
        self.file_name_ls = file_name_ls
        self.img_name_ls = img_name_ls
        self.label_ls = torch.cat(label_ls)

    def __len__(self):
        return len(self.img_name_ls)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        img_name = self.img_name_ls[idx]
        image = io.imread(img_name)
        landmarks = self.label_ls[idx]
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        # sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            if not type(image) is numpy.ndarray:
                image = Image.fromarray(image.numpy(), mode="L")
            else:
                image = Image.fromarray(image)
            image = self.transform(image)

        return (idx, image, landmarks)


def obtain_features_single_dataset(trainloader, net):
    feature_out_array = []

    targets_array = []

    for _, (idx, features, targets) in tqdm(enumerate(trainloader)):
        features = features.cuda()
        # targets = targets.cuda()
        feature_out = features#net(features)

        feature_out_array.append(feature_out.detach().cpu())
        targets_array.append(targets)
    feature_out_array_tensor = torch.cat(feature_out_array)
    targets_array_tensor = torch.cat(targets_array)

    return feature_out_array_tensor, targets_array_tensor

def obtain_features(args, net, trainset, testset):
    train_sampler = DistributedSampler(
        trainset,
        num_replicas=args.world_size,
        rank=args.local_rank,
    )
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        num_workers=0, #args.num_workers,
        pin_memory=True,
        shuffle=False,
        sampler=train_sampler,
    )

    test_sampler = DistributedSampler(
        testset,
        num_replicas=args.world_size,
        rank=args.local_rank,
    )
    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        num_workers=0, #args.num_workers,
        pin_memory=True,
        shuffle=False,
        sampler=test_sampler,
    )


    test_feature_tensor, test_targets_tensor = obtain_features_single_dataset(testloader, net)
    transformed_test_dataset = dataset_wrapper(test_feature_tensor, test_targets_tensor, None)

    
    torch.save(transformed_test_dataset, os.path.join(args.save_path, "transformed_test_dataset"))

    train_feature_tensor, train_targets_tensor = obtain_features_single_dataset(trainloader, net)
    transformed_train_dataset = dataset_wrapper(train_feature_tensor, train_targets_tensor, None)
    
    torch.save(transformed_train_dataset, os.path.join(args.save_path, "transformed_train_dataset"))


def load_raw_set(args, csv_file_name, transform, is_train=True):
    if is_train:
        return rawDataset(os.path.join(os.path.join(args.data_dir, 'train'), csv_file_name), os.path.join(args.data_dir, 'train'), transform)
    else:
        return rawDataset(os.path.join(os.path.join(args.data_dir, 'test'), csv_file_name), os.path.join(args.data_dir, 'test'), transform)


def main(args, logger):
    logger.info("start")
    logger.info('==> Preparing data..')

    norm_mean, norm_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

    transform_train_list = [
        transforms.RandomResizedCrop((256,256)),
        transforms.RandomVerticalFlip(),
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
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])

    # args.data_dir = '/data1/wuyinjun/valid_set_selections/sample/'
    train_set = load_raw_set(args, "trainLabels.csv", transform_train, True)
    test_set = load_raw_set(args, "testLabels.csv", transform_test, False)

    net = resnet34_imagenet(pretrained=True, first = True, last = False)

    net.cuda()

    obtain_features(args, net, train_set, test_set)
    
    return



if __name__ == "__main__":
    args = parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    args.world_size = dist.get_world_size()

    cudnn.benchmark = True
    
    os.makedirs(args.save_path, exist_ok=True)
    logger = setup_logger(output=args.save_path, distributed_rank=dist.get_rank(), name="valid-selec")
    
    if dist.get_rank() == 0:
        path = os.path.join(args.save_path, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))
    args.device = torch.device("cuda", args.local_rank)
    args.logger = logger
    with logging_redirect_tqdm():
        main(args, logger)