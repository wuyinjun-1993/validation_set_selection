import logging
import os
import torch

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
        if args.dataset.startswith('cifar'):
            optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
            optimizer.param_groups[0]['initial_lr'] = args.lr
            if args.do_train:
                mile_stones_epochs = [100, 150]
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=mile_stones_epochs, last_epoch=start_epoch-1)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

            else:
                if args.use_pretrained_model:
                    mile_stones_epochs = [100,150]
                else:
                    mile_stones_epochs = [120,160]
                if args.lr_decay:
                    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=mile_stones_epochs, last_epoch=start_epoch-1)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
                else:
                    scheduler = None
        else:
            if args.dataset.startswith('sst'):
                optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)# get_bert_optimizer(net, args.lr)
                scheduler = None

    return optimizer, scheduler