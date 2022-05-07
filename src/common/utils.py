import logging
import os
import torch

from torch.optim.lr_scheduler import _LRScheduler


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    # if args.do_train:
    log_file = os.path.join(args.save_path, 'train.log')
    # else:
    #     log_file = os.path.join(args.save_path, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def obtain_optimizer_scheduler(args, net, start_epoch = 0):
    if args.dataset == 'MNIST':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
        scheduler = None
    else:
        if args.dataset == 'cifar10':
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                    momentum=0.9, weight_decay=5e-4)
            optimizer.param_groups[0]['initial_lr'] = args.lr
            if args.do_train:
                mile_stones_epochs = [160, 180]
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=mile_stones_epochs,
                    last_epoch=start_epoch-1,
                    gamma=0.2,
                )#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

            else:
                if args.use_pretrained_model:
                    mile_stones_epochs = [100,180]
                else:
                    mile_stones_epochs = [100,180]
                if args.lr_decay:
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones=mile_stones_epochs,
                        last_epoch=start_epoch-1,
                        gamma=0.1,
                    )#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
                else:
                    scheduler = None
        else:
            if args.dataset == 'cifar100':
                optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
                optimizer.param_groups[0]['initial_lr'] = args.lr
                if args.do_train:
                    # mile_stones_epochs = [100, 200]
                    mile_stones_epochs = [150, 225]
                    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                    #                                             milestones=mile_stones_epochs, last_epoch=start_epoch-1, gamma = 0.1)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=mile_stones_epochs, gamma=0.2) #learning rate decay
                else:
                    if args.use_pretrained_model:
                        mile_stones_epochs = [150, 225]
                    else:
                        mile_stones_epochs = [150, 225]
                    if args.lr_decay:
                        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=mile_stones_epochs, last_epoch=start_epoch-1, gamma = 0.2)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
                    else:
                        scheduler = None
            else:
                if args.dataset.startswith('sst'):
                    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)# get_bert_optimizer(net, args.lr)
                    scheduler = None
                else:
                    if args.dataset.startswith('imdb'):
                        # pretrained_rep_net = custom_Bert(2)
                        # pretrained_rep_net = init_model_with_pretrained_model_weights(pretrained_rep_net)
                        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)# get_bert_optimizer(net, args.lr)
                        scheduler = None
                    else:
                        if args.dataset.startswith('trec'):
                            # pretrained_rep_net = custom_Bert(2)
                            # pretrained_rep_net = init_model_with_pretrained_model_weights(pretrained_rep_net)
                            optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)# get_bert_optimizer(net, args.lr)
                            scheduler = None


    return optimizer, scheduler