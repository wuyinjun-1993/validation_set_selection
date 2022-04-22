import torch

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from common.parse_args import *
from datasets.mnist import *
from models.DNN import *
from common.utils import *
from tqdm.notebook import tqdm
import itertools
import torch_higher as higher
from main.find_valid_set import *
from main.meta_reweighting_rl import *
from datasets.dataloader import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
from utils.logger import setup_logger
import models
from lib.NCECriterion import NCESoftmaxLoss
from lib.lr_scheduler import get_scheduler
from lib.BootstrappingLoss import SoftBootstrappingLoss, HardBootstrappingLoss
from models.resnet import *
from models.bert import *
import collections



def load_models_from_cache(file_name_ls):

    model_state_ls = []

    for file_name in file_name_ls:
        # torch.load(file)
        cached_model_file_name = os.path.join(args.prev_save_path, file_name)
        if os.path.exists(cached_model_file_name):
            
            state = torch.load(cached_model_file_name, map_location=torch.device("cpu"))

            model_state_ls.append(state)

    return model_state_ls

def update_models_with_cached_state(state, model):

    if type(state) is collections.OrderedDict:
        model.load_state_dict(state)
    else:
        model.load_state_dict(state.state_dict())
    logging.info('==> Loading cached model successfully')
    del state

    return model


def get_optimizer_given_model(args, net):
    if args.dataset == 'MNIST':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    else:
        if args.dataset.startswith('cifar'):
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            optimizer.param_groups[0]['initial_lr'] = args.lr
        else:
            if args.dataset.startswith('sst2'):
                optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)# get_bert_optimizer(net, args.lr)
            else:
                if args.dataset.startswith('sst5'):
                    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)# get_bert_optimizer(net, args.lr)
                else:
                    raise NotImplementedError
    return optimizer

def do_clustering_main(args):

    if args.dataset == 'MNIST':
        net = DNN_three_layers(args.nce_k, low_dim=args.low_dim).cuda()
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    else:
        if args.dataset.startswith('cifar'):
            net = ResNet18().cuda()
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            optimizer.param_groups[0]['initial_lr'] = args.lr
        else:
            if args.dataset.startswith('sst2'):
                net = custom_Bert(2)
                # pretrained_rep_net = init_model_with_pretrained_model_weights(pretrained_rep_net)
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)# get_bert_optimizer(net, args.lr)
            else:
                if args.dataset.startswith('sst5'):
                    net = custom_Bert(5)
                    # pretrained_rep_net = init_model_with_pretrained_model_weights(pretrained_rep_net)
                    criterion = torch.nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)# get_bert_optimizer(net, args.lr)
                else:
                    raise NotImplementedError
        # pretrained_rep_net = ResNet18().cuda()
    criterion = torch.nn.CrossEntropyLoss()

    train_loader, valid_set = get_dataloader_for_post_evaluations(args)

    
    model_file_ls = ["refined_model_1", "refined_model_100"]

    model_state_ls = load_models_from_cache(model_file_ls)

    for model_state in model_state_ls:

        net = update_models_with_cached_state(model_state, net)

        optimizer = get_optimizer_given_model(args, net)

        get_representative_valid_ids2(criterion, optimizer, train_loader, args, net, len(valid_set), cached_sample_weights = None, existing_valid_representation = None, existing_valid_set = None, return_cluster_info=True)

if __name__ == "__main__":
    args = parse_args()