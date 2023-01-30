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
    mile_stones_epochs = None
    if args.dataset == 'MNIST':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
        scheduler = None
    elif args.dataset == 'cifar10':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                momentum=0.9, weight_decay=5e-4, nesterov=True)
        optimizer.param_groups[0]['initial_lr'] = args.lr
        if args.do_train:
            if args.bias_classes:
                mile_stones_epochs = [160, 180]
                gamma = 0.1
            else:
                mile_stones_epochs = [100, 110]
                gamma = 0.2
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=mile_stones_epochs,
                last_epoch=start_epoch-1,
                gamma=gamma,
            )

        else:
            if args.use_pretrained_model:
                if args.bias_classes:
                    mile_stones_epochs = [80, 90]
                    gamma = 0.1
                else:
                    mile_stones_epochs = [80,90]
                    gamma = 0.1
            else:
                mile_stones_epochs = [80,90]
                gamma = 0.1

            if args.lr_decay:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=mile_stones_epochs,
                    last_epoch=start_epoch-1,
                    gamma=gamma,
                )
            else:
                scheduler = None
    elif args.dataset == 'cifar100':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                momentum=0.9, weight_decay=5e-4, nesterov=True)
        optimizer.param_groups[0]['initial_lr'] = args.lr
        if args.do_train:
            if args.bias_classes:
                mile_stones_epochs = [160, 180]
            else:
                mile_stones_epochs = [150, 225]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=mile_stones_epochs,
                gamma=0.1,
            )
        else:
            if args.use_pretrained_model:
                if args.bias_classes:
                    mile_stones_epochs = [80, 90]
                    gamma = 0.1
                else:
                    mile_stones_epochs = [80, 90]
                    gamma = 0.2
            else:
                mile_stones_epochs = [80, 90]
                gamma = 0.1
            if args.lr_decay:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=mile_stones_epochs,
                    last_epoch=start_epoch-1,
                    gamma=gamma,
                )
            else:
                scheduler = None
    elif args.dataset == 'retina':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                momentum=0.9, weight_decay=5e-4, nesterov=True)
        optimizer.param_groups[0]['initial_lr'] = args.lr
        # if args.do_train:
        #     if args.bias_classes:
        #         mile_stones_epochs = [50, 60]
        #     else:
        #         mile_stones_epochs = [20, 40]
        #     scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #         optimizer,
        #         milestones=mile_stones_epochs,
        #         gamma=0.1,
        #     )
        # else:
        #     if args.use_pretrained_model:
        #         if args.bias_classes:
        #             mile_stones_epochs = [50, 60]
        #             gamma = 0.1
        #         else:
        #             mile_stones_epochs = [20, 40]
        #             gamma = 0.2
        #     else:
        #         mile_stones_epochs = [20, 40]
        #         gamma = 0.1
        #     if args.lr_decay:
        #         scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #             optimizer,
        #             milestones=mile_stones_epochs,
        #             last_epoch=start_epoch-1,
        #             gamma=gamma,
        #         )
        #     else:
        scheduler = None
    elif args.dataset == 'imagenet':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                momentum=0.9, weight_decay=5e-4, nesterov=True)
        optimizer.param_groups[0]['initial_lr'] = args.lr
        if args.do_train:
            if args.bias_classes:
                mile_stones_epochs = [10, 15]
            else:
                mile_stones_epochs = [10, 15]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=mile_stones_epochs,
                gamma=0.1,
            )
        else:
            if args.use_pretrained_model:
                if args.bias_classes:
                    mile_stones_epochs = [40]
                    gamma = 0.1
                else:
                    # mile_stones_epochs = [10, 15]
                    mile_stones_epochs = [40]
                    gamma = 0.2
            else:
                mile_stones_epochs = [40]
                gamma = 0.1
            if args.lr_decay:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=mile_stones_epochs,
                    last_epoch=start_epoch-1,
                    gamma=gamma,
                )
            else:
                scheduler = None
    elif args.dataset.startswith('sst'):
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)# get_bert_optimizer(net, args.lr)
        scheduler = None
    elif args.dataset.startswith('imdb'):
        # pretrained_rep_net = custom_Bert(2)
        # pretrained_rep_net = init_model_with_pretrained_model_weights(pretrained_rep_net)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)# get_bert_optimizer(net, args.lr)
        scheduler = None
    elif args.dataset.startswith('trec'):
        # pretrained_rep_net = custom_Bert(2)
        # pretrained_rep_net = init_model_with_pretrained_model_weights(pretrained_rep_net)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)# get_bert_optimizer(net, args.lr)
        scheduler = None
    else:
        raise NotImplementedError


    return optimizer, (scheduler, mile_stones_epochs)
