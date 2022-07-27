import torch

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from common.parse_args import *
from exp_datasets.mnist import *
from models.DNN import *
from common.utils import *
from tqdm.notebook import tqdm
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

    train_loader, metaloader = get_dataloader_for_post_evaluations(args)

    
    model_file_ls = ["cached_pretrain_model", "refined_model_1", "refined_model_400"]

    model_state_ls = load_models_from_cache(model_file_ls)

    all_cluster_ids_ls = []

    all_cluster_center_ls = []

    all_full_sim_mat_ls = []

    dist_ls = []

    full_train_sample_representation_tensor_ls = []

    full_meta_sample_representation_tensor_ls = []

    for model_state in model_state_ls:

        curr_net = update_models_with_cached_state(model_state, net)

        if args.cuda:
            curr_net = curr_net.cuda()

        optimizer = get_optimizer_given_model(args, curr_net)

        full_sample_representation_tensor = get_representative_valid_ids2(criterion, optimizer, train_loader, args, curr_net, len(metaloader.dataset), cached_sample_weights = None, existing_valid_representation = None, existing_valid_set = None, return_cluster_info=True, only_sample_representation=True)

        meta_sample_representation_tensor = get_representative_valid_ids2(criterion, optimizer, metaloader, args, curr_net, len(metaloader.dataset), cached_sample_weights = None, existing_valid_representation = None, existing_valid_set = None, return_cluster_info=True, only_sample_representation=True)

        if not args.cosin_dist:
            if not args.all_layer:
                pairwise_distance_function = pairwise_distance
            else:
                pairwise_distance_function = pairwise_distance_ls

        else:
            if not args.all_layer:
                pairwise_distance_function = pairwise_cosine
            else:
                pairwise_distance_function = pairwise_cosine_ls
        
        dis = pairwise_distance_function(full_sample_representation_tensor, meta_sample_representation_tensor,args.cuda)

        choice_cluster = torch.argmin(dis, dim=1)

        dist_ls.append(dis)
        full_sim_mat1,full_train_sample_representation_tensor, full_meta_sample_representation_tensor  = calculate_train_meta_grad_prod(args, train_loader, metaloader, net, criterion, optimizer)

        full_train_sample_representation_tensor_ls.append(full_train_sample_representation_tensor)

        full_meta_sample_representation_tensor_ls.append(full_meta_sample_representation_tensor)
        # if args.cosin_dist:
        #     full_sim_mat1 = pairwise_cosine_full(full_sample_representation_tensor, is_cuda=args.cuda)
        # else:
        #     # full_sim_mat1 = pairwise_l2_full(full_sample_representation_tensor, is_cuda=args.cuda)
        #     full_sim_mat1 = pairwise_distance_ls_full(full_sample_representation_tensor, full_sample_representation_tensor, is_cuda=args.cuda,  batch_size = 256)

        all_full_sim_mat_ls.append(full_sim_mat1)

        all_cluster_ids_ls.append(choice_cluster)

        # all_cluster_center_ls.append(cluster_centers)

        del curr_net

    print()

if __name__ == "__main__":
    args = parse_args()
    set_logger(args)
    dist.init_process_group(backend='nccl', init_method='env://')
    args.world_size = dist.get_world_size()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)

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
    do_clustering_main(args)