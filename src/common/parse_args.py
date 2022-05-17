import argparse
import torch
import os
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--biased_flip', action='store_true', help='use GPU')
    parser.add_argument('--noisy_valid', action='store_true', help='use the noisy labels in the validation set')
    parser.add_argument('--w_rectified_gaussian_init', action='store_true',
        help='initialize the sample weights with a rectified gaussian')

    parser.add_argument('--all_layer', action='store_true', help='use GPU')
    parser.add_argument('--get_representations', action='store_true', help='use GPU')
    parser.add_argument('--all_layer2', action='store_true', help='use GPU')
    parser.add_argument('--replace', action='store_true', help='use GPU')
    parser.add_argument('--all_layer_grad', action='store_true', help='use GPU')
    parser.add_argument('--all_layer_grad_greedy', action='store_true', help='use GPU')
    parser.add_argument('--all_layer_grad_no_full_loss', action='store_true', help='use GPU')
    parser.add_argument('--weight_by_norm', action='store_true', help='use GPU')
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')

    parser.add_argument('--grad_layer_depth', default=1, type=int, help='capture model_prov')
    parser.add_argument('--model_type', default="resnet34", type=str, help='capture model_prov')
    # sampled_param_count

    parser.add_argument('--sampled_param_count', default=0, type=int, help='capture model_prov')
    parser.add_argument('--all_layer_sim_agg', default="mean", type=str, help='capture model_prov')

    parser.add_argument('--full_model_out', action='store_true', help='use GPU')
    parser.add_argument('--cluster_method_two_sampling', action='store_true', help='use GPU')
    parser.add_argument('--cluster_method_two_sample_col_count', default=1000, type=int, help='capture model_prov')

    parser.add_argument('--cluster_no_reweighting', action='store_true', help='use GPU')
    parser.add_argument('--low_data', action='store_true', help='low data application')
    parser.add_argument('--low_data_num_samples_per_class', default=40, type=int, help='Create class bias')

    parser.add_argument('--flip_labels', action='store_true', help='flip labels')
    parser.add_argument('--adversarial_flip', action='store_true', help='flip labels')
    parser.add_argument('--bias_classes', action='store_true', help='Create class bias')
    parser.add_argument('--imb_factor', default=1.0, type=float, help='Create class bias')
    parser.add_argument('--l1_loss', action='store_true', help='Use the L1 loss for the basic learning step')
    parser.add_argument('--soft_bootstrapping_loss', action='store_true', help='Use the Bootstrapping loss for the basic learning step')
    parser.add_argument('--hard_bootstrapping_loss', action='store_true', help='Use the Bootstrapping loss for the basic learning step')
    parser.add_argument('--l1_meta_loss', action='store_true', help='Use the L1 loss for meta learning step')
    parser.add_argument('--load_dataset', action='store_true', help='load dataset')
    parser.add_argument('--continue_label', action='store_true', help='load dataset')
    parser.add_argument('--use_model_prov', action='store_true', help='capture model_prov')
    parser.add_argument('--model_prov_period', default=20, type=int, help='capture model_prov')
    parser.add_argument('--valid_count', default=None, type=int, help='capture model_prov')
    parser.add_argument('--qualitiative', action='store_true', help='load dataset')

    parser.add_argument('--total_valid_sample_count', default=-1, type=int, help='capture model_prov')
    parser.add_argument('--k_means_lr', default=0.0001, type=float, help='capture model_prov')
    parser.add_argument('--k_means_bz', default=128, type=int, help='capture model_prov')
    parser.add_argument('--k_means_epochs', default=200, type=int, help='capture model_prov')

    parser.add_argument('--inner_prod', action='store_true', help='not save dataset')
    parser.add_argument('--rand_init', action='store_true', help='not save dataset')
    parser.add_argument('--not_rescale_features', action='store_true', help='not save dataset')
    parser.add_argument('--not_save_dataset', action='store_true', help='not save dataset')
    parser.add_argument('--clustering_by_class', action='store_true', help='not save dataset')

    parser.add_argument('--select_valid_set', action='store_true', help='select valid set')
    parser.add_argument('--include_valid_set_in_training', action='store_true', help='select valid set')
    parser.add_argument('--cluster_method_two', action='store_true', help='select valid set')
    parser.add_argument('--cluster_method_two_plus', action='store_true', help='select valid set')

    parser.add_argument('--init_cluster_by_confident', action='store_true', help='select valid set')
    parser.add_argument('--resume_meta_train', action='store_true', help='resume meta training')
    parser.add_argument('--resume_train', action='store_true', help='resume training')
    parser.add_argument('--resumed_training_epoch',  default=0, type=int, help='start from epoch')

    parser.add_argument('--cluster_method_three', action='store_true', help='cache_loss_per_epoch')
    parser.add_argument('--cosin_dist', action='store_true', help='use cosine distance for k-means clustering')
    parser.add_argument('--prev_save_path', default=None, type=str)

    parser.add_argument('--unsup_rep', action='store_true', help='unsupervised representation usage')
    parser.add_argument('--no_sample_weights_k_means', action='store_true', help='unsupervised representation usage')

    parser.add_argument('--add_under_rep_samples', action='store_true', help='add under represented samples')

    parser.add_argument('--do_train', action='store_true', help='do training')
    parser.add_argument('--finetune', action='store_true', help='finetune model on meta set')
    parser.add_argument('--active_learning', action='store_true', help='perform training with active learning')
    parser.add_argument('--uncertain_select', action='store_true', help='perform training with active learning')
    parser.add_argument('--certain_select', action='store_true', help='perform training with active learning')
    parser.add_argument('--load_cached_weights', action='store_true', help='load_cached_weights')
    parser.add_argument('--lr_decay', action='store_true', help='load_cached_weights')

    parser.add_argument('--use_pretrained_model', action='store_true', help='use pretrained models')
    parser.add_argument('--reduce_dimension_all_layer', action='store_true', help='use pretrained models')
    parser.add_argument('--err_label_ratio', default=0.2, type=float)
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--save_path', default=None, type=str)

    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--gpu_id', default=4, type=int)
    parser.add_argument('--lr', default=0.2, type=float)
    parser.add_argument('--meta_lr', default=0.2, type=float)
    parser.add_argument('--valid_ratio', default=0.1, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int)
    # parser.add_argument('--epochs', default=4, type=int)

    parser.add_argument('--norm_fn', choices=['bound', 'linear', 'softmax'])
    parser.add_argument("--w_decay", default=10., type=float)
    parser.add_argument("--w_init", default=0., type=float)

    parser.add_argument('--image_softmax_norm_temp', default=1., type=float)


    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to training')
    parser.add_argument('--crop', type=float, default=0.08, help='minimum crop')
    parser.add_argument('--aug', type=str, default='CJ', choices=['NULL', 'CJ'],
                        help="augmentation type: NULL for normal supervised aug, CJ for aug with ColorJitter")
    # parser.add_argument('--batch-size', type=int, default=128, help='batch_size')
    parser.add_argument('--num-workers', type=int, default=4, help='num of workers to use')

    # model and loss function
    parser.add_argument('--alpha', type=float, default=0.999, help='exponential moving average weight')
    parser.add_argument('--nce-k', type=int, default=4096, help='num negative sampler')
    parser.add_argument('--nce-t', type=float, default=0.1, help='NCE temperature')
    parser.add_argument('--low-dim', default=128, type=int,
                        metavar='D', help='feature dimension')
    # optimization
    parser.add_argument('--base-learning-rate', '--base-lr', type=float, default=0.03,
                        help='base learning when batch size = 128. final lr is determined by linear scale')
    parser.add_argument('--lr-scheduler', type=str, default='cosine',
                        choices=["step", "cosine"], help="learning rate scheduler")
    parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--lr-decay-epochs', type=int, default=[120, 160, 200], nargs='+',
                        help='for step scheduler. where to decay lr, can be a list')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--amp-opt-level', type=str, default='O0', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')

    parser.add_argument('--cached_model_name', type=str, default = None,
                        help='cached model name')

    parser.add_argument('--cached_sample_weights_name', type=str, default = None,
                        help='cached sample weights')
    
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')

    # io
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save-freq', type=int, default=10, help='save frequency')
    # parser.add_argument('--save-dir', type=str, default='./output', help='output director')

    # misc
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument("--rng-seed", type=int, default=0, help='manual seed')

    # CLD related arguments
    parser.add_argument('--clusters', default=10, type=int,
                        help='num of clusters for spectral clustering')
    parser.add_argument('--k-eigen', default=10, type=int,
                        help='num of eigenvectors for k-way normalized cuts')
    parser.add_argument('--cld_t', default=0.07, type=float,
                        help='temperature for spectral clustering')
    parser.add_argument('--use-kmeans', action='store_true', help='Whether use k-means for clustering. \
                        Use Normalized Cuts if it is False')
    parser.add_argument('--num-iters', default=20, type=int,
                        help='num of iters for clustering')
    parser.add_argument('--Lambda', default=1.0, type=float,
                        help='weight of mutual information loss')
    parser.add_argument('--two-imgs', action='store_true', help='Whether use two randomly processed views')
    parser.add_argument('--three-imgs', action='store_true', help='Whether use three randomly processed views')
    parser.add_argument('--normlinear', action='store_true', help='whether use normalization linear layer')
    parser.add_argument('--aug-plus', action='store_true', help='whether add strong augmentation')
    parser.add_argument('--erasing', action='store_true', help='whether add random erasing as an augmentation')


    # parser.add_argument('--obj_cl_hidden_layer_count', default=0, type=int)

    args,_ = parser.parse_known_args()
    # torch.cuda.set_device(args.gpu_id)

    # args.gpu_id  = os.environ["CUDA_VISIBLE_DEVICES"]
    # if not args.cuda:
    #     # if not is_GPU:
    #     args.device = torch.device("cpu")
    # else:    
        
    #     # GPU_ID = os.environ["CUDA_VISIBLE_DEVICE"]
    #     GPU_ID = args.gpu_id
    #     args.device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")
    #     print("device::", args.device)
    
    torch.manual_seed(args.rng_seed)
    torch.cuda.manual_seed_all(args.rng_seed)


    return args