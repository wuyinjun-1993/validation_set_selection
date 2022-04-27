import torch

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from common.utils import *

def obtain_net_grad(net):
    grad_ls = []
    for param in net.parameters():
        grad_ls.append(param.grad.detach().cpu().clone())

    return grad_ls


def obtain_full_loss(output, target, is_cuda, loss):
    onehot_target = torch.zeros(output.shape[1])
    onehot_target[target] = 1
    if is_cuda:
        onehot_target = onehot_target.cuda()

    total_loss = loss + torch.sum(onehot_target.view(-1)*output.view(-1))

    return total_loss

def obtain_model_gradient_per_sample_ls(args, net, train_loader, criterion, optimizer):
    full_sample_grad_ls = [None]*len(train_loader.dataset)

    for batch_id, (sample_ids, data, labels) in enumerate(train_loader):
        args.logger.info("sample batch ids::%d"%(batch_id))
        if args.cuda:
            data, labels = train_loader.dataset.to_cuda(data, labels)
        output = net.forward(data)
        for idx in range(labels.shape[0]):
            optimizer.zero_grad()
            loss = criterion(output[idx:idx+1], labels[idx:idx+1])
            loss = obtain_full_loss(output[idx:idx+1], labels[idx:idx+1], args.cuda, loss)

            loss.backward(retain_graph = True)
            sample_id = sample_ids[idx]
            curr_sample_grad_ls = obtain_net_grad(net)

            full_sample_grad_ls[sample_id] = curr_sample_grad_ls

    return full_sample_grad_ls


def load_checkpoint_by_epoch(args, model, epoch):
    args.logger.info('==> Loading cached model at epoch %d'%(epoch))
    cached_model_name = "refined_model_" + str(epoch)
    if args.prev_save_path is not None:
        cached_model_file_name = os.path.join(args.prev_save_path, cached_model_name)

        if not os.path.exists(cached_model_file_name):
            cached_model_name = "model_" + str(epoch)
            cached_model_file_name = os.path.join(args.prev_save_path, cached_model_name)

            
        state = torch.load(cached_model_file_name, map_location=torch.device("cpu"))

        if type(state) is collections.OrderedDict:
            model.load_state_dict(state)
        else:
            model.load_state_dict(state.state_dict())
        args.logger.info('==> Loading cached model successfully')
        del state
            
        
    return model

def obtain_extra_model_gradient_per_sample_ls(args, net, train_loader, criterion, optimizer, full_sample_grad_ls_ls):
    start_epoch_id = 0
    if args.use_pretrained_model:
        start_epoch_id = torch.load(os.path.join(args.prev_save_path, "early_stopping_epoch"))
    args.logger.info("extra representation starting from epoch %d"%(start_epoch_id))

    for ep in range(start_epoch_id, args.epochs, args.model_prov_period):
        net = load_checkpoint_by_epoch(args, net, ep)
        optimizer, _=obtain_optimizer_scheduler(args, net, start_epoch = 0)

        curr_full_sample_grad_ls = obtain_model_gradient_per_sample_ls(args, net, train_loader, criterion, optimizer)

        full_sample_grad_ls_ls.append(curr_full_sample_grad_ls)


def obtain_full_model_gradient_per_sample_ls(args, net, train_loader, criterion, optimizer):
    full_sample_grad_ls = obtain_model_gradient_per_sample_ls(args, net, train_loader, criterion, optimizer)

    if args.use_model_prov:
        full_sample_grad_ls_ls = [full_sample_grad_ls]
        obtain_extra_model_gradient_per_sample_ls(args, net, train_loader, criterion, optimizer, full_sample_grad_ls_ls)
    
    return full_sample_grad_ls


