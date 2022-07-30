import torch
import os,sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from common.parse_args import *
from exp_datasets.mnist import *
from models.DNN import *
from common.utils import *
from tqdm.notebook import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from main.find_valid_set import *
from main.meta_reweighting_rl import *
from exp_datasets.dataloader import *
import torch.distributed as dist
import json
from utils.logger import setup_logger
from models.ResNet import *
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def obtain_features_single_dataset(args, trainloader, net):
    feat_filename = os.path.join(args.save_path, "processed_data")
    label_filename = os.path.join(args.save_path, "processed_labels")
    processed = np.memmap(feat_filename, dtype='float32', mode='w+', shape=(len(trainloader), 3, 224, 224))
    targets = np.memmap(label_filename, dtype='float32', mode='w+', shape=(len(trainloader),))
    for idx, (features, targets) in tqdm(enumerate(trainloader)):
        processed[idx:idx+args.batch_size,:] = features
        targets[idx:idx+args.batch_size] = targets


def obtain_features(args, net, trainset, testset):
    train_sampler = DistributedSampler(
        trainset,
        num_replicas=args.world_size,
        rank=args.local_rank,
    )
    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        num_workers=16, #args.num_workers,
        pin_memory=False,
        shuffle=False,
        sampler=train_sampler,
    )

    # test_sampler = DistributedSampler(
    #     testset,
    #     num_replicas=args.world_size,
    #     rank=args.local_rank,
    # )
    # testloader = DataLoader(
    #     testset,
    #     batch_size=args.batch_size,
    #     num_workers=4, #args.num_workers,
    #     pin_memory=False,
    #     shuffle=False,
    #     sampler=test_sampler,
    # )


    # test_feature_tensor, test_targets_tensor = obtain_features_single_dataset(testloader, net)
    # transformed_test_dataset = dataset_wrapper(test_feature_tensor, test_targets_tensor, None)

    # 
    # torch.save(transformed_test_dataset, os.path.join(args.save_path, "transformed_test_dataset"))

    obtain_features_single_dataset(args, trainloader, net)


def main(args, logger):
    logger.info("start")
    logger.info('==> Preparing data..')
    args.logger = logger

    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_set = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_set = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))


    net = resnet34_imagenet(pretrained=True, first = True, last = False)

    net.cuda()

    obtain_features(args, net, train_set, val_set)
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
