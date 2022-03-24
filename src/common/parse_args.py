import argparse
import torch
import os
def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')
    parser.add_argument('--flip_labels', action='store_true', help='flip labels')
    parser.add_argument('--load_dataset', action='store_true', help='load dataset')
    parser.add_argument('--not_save_dataset', action='store_true', help='not save dataset')
    parser.add_argument('--select_valid_set', action='store_true', help='select valid set')
    parser.add_argument('--do_train', action='store_true', help='do training')
    parser.add_argument('--err_label_ratio', default=0.2, type=float)
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--gpu_id', default=4, type=int)
    parser.add_argument('--lr', default=0.2, type=float)
    parser.add_argument('--meta_lr', default=0.2, type=float)
    parser.add_argument('--valid_ratio', default=0.1, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=4, type=int)



    # parser.add_argument('--obj_cl_hidden_layer_count', default=0, type=int)

    args,_ = parser.parse_known_args()
    torch.cuda.set_device(args.gpu_id)

    # args.gpu_id  = os.environ["CUDA_VISIBLE_DEVICES"]
    # if not args.cuda:
    #     # if not is_GPU:
    #     args.device = torch.device("cpu")
    # else:    
        
    #     # GPU_ID = os.environ["CUDA_VISIBLE_DEVICE"]
    #     GPU_ID = args.gpu_id
    #     args.device = torch.device("cuda:"+str(GPU_ID) if torch.cuda.is_available() else "cpu")
    #     print("device::", args.device)
    
    return args