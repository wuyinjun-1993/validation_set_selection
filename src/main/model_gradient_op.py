import torch
import collections
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from common.utils import *
import numpy as np
def obtain_net_grad(net):
    grad_ls = []
    for param in net.parameters():
        grad_ls.append(param.grad.detach().cpu().clone())

    return grad_ls

def compute_net_grad_norm_ls(grad_ls):
    grad_norm_ls = []
    for grad in grad_ls:
        grad_norm_ls.append(torch.sum(grad*grad))

    return torch.tensor(grad_norm_ls)

def obtain_net_param_count_ls(net):
    param_count_ls = []
    for param in net.parameters():
        param_count_ls.append(param.numel()) 
    return param_count_ls

def obtain_net_grad_norm(net):
    grad_norm = 0
    for param in net.parameters():
        grad_norm += (torch.norm(param.grad)**2).detach().cpu()
        # grad_ls.append(param.grad.detach().cpu().clone())

    return torch.sqrt(grad_norm)

def merge_grad_by_layer(grad_ls, vec_grad_ls):
    if len(vec_grad_ls) <= 0:
        for grad in grad_ls:
            vec_grad_ls.append([grad.view(1,-1)])
    else:
        idx = 0
        for grad in grad_ls:
            # vec_grad_ls[idx] = torch.cat([vec_grad_ls[idx], grad.view(1,-1)])
            vec_grad_ls[idx].append(grad.view(1,-1))
            idx += 1

    return vec_grad_ls

def rand_sample_parameter(net, sampled_layer_count = 5, sampled_param_count = 1000, include_last_layer = True):
    module_ls = list(net._modules)

    selected_param_layer_ls = []

    if include_last_layer:

        last_layer_param = list(getattr(net, module_ls[-1]).parameters())

        # last_layer_param = [last_layer_param[k].view(-1) for k in range(len(last_layer_param))]

        curr_sampled_param_count = sum([last_layer_param[k].numel() for k in range(len(last_layer_param))])

        selected_param_layer_ls.extend(last_layer_param)

        if sampled_param_count - curr_sampled_param_count <= 0:
            return selected_param_layer_ls, None
    else:

        last_layer_param = []
        curr_sampled_param_count = 0

    sampled_param_count = sampled_param_count - curr_sampled_param_count
    full_param_ls = list(net.parameters())

    remaining_param_ls = [full_param_ls[k] for k in range(len(full_param_ls)) if k < len(full_param_ls) - len(last_layer_param)]

    remaining_param_count_ls = [remaining_param_ls[k].numel() for k in range(len(remaining_param_ls))]

    if sampled_layer_count >= len(remaining_param_ls):
        selected_layer_id_ls = list(range(len(remaining_param_ls)))
        selected_param_count_ls = remaining_param_count_ls

    else:    
        selected_layer_id_ls = np.random.choice(list(range(len(remaining_param_ls))), size = sampled_layer_count, p = np.array(remaining_param_count_ls)/sum(remaining_param_count_ls), replace = False)
        selected_param_count_ls = [remaining_param_count_ls[k] for k in selected_layer_id_ls]

    selected_param_num_by_layer_ls = np.array(selected_param_count_ls)
    selected_sampled_param_id_by_layer_ls = []
    if len(selected_param_layer_ls) > 0:
        selected_sampled_param_id_by_layer_ls.extend([None]*len(selected_param_layer_ls))
    selected_param_layer_ls.extend([remaining_param_ls[k] for k in selected_layer_id_ls])
    if sampled_param_count > sum(selected_param_num_by_layer_ls):
        return selected_param_layer_ls, None


    
    selected_sampling_param_num_by_layer_ls = selected_param_num_by_layer_ls/sum(selected_param_num_by_layer_ls)*sampled_param_count
    selected_sampling_param_num_by_layer_ls = selected_sampling_param_num_by_layer_ls.astype(int)

    
    for idx in range(len(selected_sampling_param_num_by_layer_ls)):
        sampling_param_count = selected_sampling_param_num_by_layer_ls[idx]
        param_count_curr_layer = selected_param_num_by_layer_ls[idx]
        if sampling_param_count >= param_count_curr_layer:
            selected_sampled_param_id_by_layer_ls.append(None)
        else:
            rand_sampled_param_ids_curr_layer = np.random.choice(list(range(param_count_curr_layer)), size = sampling_param_count, replace=False)
            selected_sampled_param_id_by_layer_ls.append(rand_sampled_param_ids_curr_layer)

    return selected_param_layer_ls, selected_sampled_param_id_by_layer_ls


def biased_rand_sample_parameter(net, avg_grad_norm_by_layer, sampled_layer_count = 5, sampled_param_count = 1000, include_last_layer = True, replace=False):
    prob_ls = avg_grad_norm_by_layer/torch.sum(avg_grad_norm_by_layer)

    net_param_ls = list(net.parameters())

    all_layer_id_ls = np.array(list(range(len(net_param_ls))))

    selected_layer_id_ls = np.random.choice(all_layer_id_ls, size = sampled_layer_count, replace = replace, p = prob_ls.numpy())
    print(selected_layer_id_ls)

    selected_layer_param_ls = [net_param_ls[k] for k in selected_layer_id_ls]
    selected_layer_prob_ls = torch.tensor([torch.sqrt(prob_ls[k]) for k in selected_layer_id_ls])
    return selected_layer_param_ls, selected_layer_prob_ls
    # module_ls = list(net._modules)

    # selected_param_layer_ls = []

    # if include_last_layer:

    #     last_layer_param = list(getattr(net, module_ls[-1]).parameters())

    #     # last_layer_param = [last_layer_param[k].view(-1) for k in range(len(last_layer_param))]

    #     curr_sampled_param_count = sum([last_layer_param[k].numel() for k in range(len(last_layer_param))])

    #     selected_param_layer_ls.extend(last_layer_param)

    #     if sampled_param_count - curr_sampled_param_count <= 0:
    #         return selected_param_layer_ls, None
    # else:

    #     last_layer_param = []
    #     curr_sampled_param_count = 0

    # sampled_param_count = sampled_param_count - curr_sampled_param_count
    # full_param_ls = list(net.parameters())

    # remaining_param_ls = [full_param_ls[k] for k in range(len(full_param_ls)) if k < len(full_param_ls) - len(last_layer_param) and k >= len(full_param_ls) - sampled_layer_count - len(last_layer_param)]

    # remaining_param_count_ls = [remaining_param_ls[k].numel() for k in range(len(remaining_param_ls))]

    # if sampled_layer_count >= len(remaining_param_ls):
    #     selected_layer_id_ls = list(range(len(remaining_param_ls)))
    #     selected_param_count_ls = remaining_param_count_ls

    # else:    
    #     selected_layer_id_ls = np.random.choice(list(range(len(remaining_param_ls))), size = sampled_layer_count, p = np.array(remaining_param_count_ls)/sum(remaining_param_count_ls), replace = False)
    #     selected_param_count_ls = [remaining_param_count_ls[k] for k in selected_layer_id_ls]

    # selected_param_num_by_layer_ls = np.array(selected_param_count_ls)
    # selected_sampled_param_id_by_layer_ls = []
    # if len(selected_param_layer_ls) > 0:
    #     selected_sampled_param_id_by_layer_ls.extend([None]*len(selected_param_layer_ls))
    # selected_param_layer_ls.extend([remaining_param_ls[k] for k in selected_layer_id_ls])
    # if sampled_param_count > sum(selected_param_num_by_layer_ls):
    #     return selected_param_layer_ls, None


    
    # selected_sampling_param_num_by_layer_ls = selected_param_num_by_layer_ls/sum(selected_param_num_by_layer_ls)*sampled_param_count
    # selected_sampling_param_num_by_layer_ls = selected_sampling_param_num_by_layer_ls.astype(int)

    
    # for idx in range(len(selected_sampling_param_num_by_layer_ls)):
    #     sampling_param_count = selected_sampling_param_num_by_layer_ls[idx]
    #     param_count_curr_layer = selected_param_num_by_layer_ls[idx]
    #     if sampling_param_count >= param_count_curr_layer:
    #         selected_sampled_param_id_by_layer_ls.append(None)
    #     else:
    #         rand_sampled_param_ids_curr_layer = np.random.choice(list(range(param_count_curr_layer)), size = sampling_param_count, replace=False)
    #         selected_sampled_param_id_by_layer_ls.append(rand_sampled_param_ids_curr_layer)

    # return selected_param_layer_ls, selected_sampled_param_id_by_layer_ls
    # remaining_param_cum_count_ls = []
    # cum_count = 0
    # for count in remaining_param_count_ls:
    #     remaining_param_cum_count_ls.append(cum_count)
    #     cum_count += count

    # total_remaining_param_count = sum(remaining_param_count_ls)

    # sampled_param_id_ls = torch.randperm(total_remaining_param_count)[0:sampled_param_count - curr_sampled_param_count]

    # sampled_param_id_ls = torch.sort(sampled_param_id_ls, descending=False)[0]

    # param_layer_id = 0



    # for param_id in sampled_param_id_ls:
    #     if param_layer_id < len(remaining_param_cum_count_ls) - 1:
    #         while param_id >= remaining_param_cum_count_ls[param_layer_id + 1]:
    #             param_layer_id += 1
            
    #     parameter_ls.append([remaining_param_ls[param_layer_id].view(-1)[remaining_param_cum_count_ls[param_layer_id] - param_id]])

    # return parameter_ls



def obtain_net_grad2(net, loss, depth=1):
    grad_ls = []
    module_ls = list(net._modules)

    last_layer_param_count = len(list(getattr(net, module_ls[-1]).parameters()))

    total_parameter_layer_count = len(list(net.parameters()))

    curr_depth = 0

    param_ls_to_grad = []

    for idx in range(total_parameter_layer_count - last_layer_param_count):
        curr_model_param = list(net.parameters())[total_parameter_layer_count - last_layer_param_count - idx]
        param_ls_to_grad.append(curr_model_param)
        if len(curr_model_param.shape) > 1:
            curr_depth += 1
            if curr_depth >= depth:
                break

    res_grad_ls = list(torch.autograd.grad(loss, param_ls_to_grad, retain_graph=True))

    for grad in res_grad_ls:
        grad_ls.append(grad.detach().cpu())

    # for idx in range(total_parameter_layer_count - last_layer_param_count):
    #     if idx >= depth:
    #         break
    #     curr_param_ls = list(net.parameters())[total_parameter_layer_count - last_layer_param_count - idx]

    #     curr_grad = torch.autograd.grad(loss, curr_param_ls, retain_graph=True)[0]

    #     grad_ls.append(curr_grad)
    # for idx in range(len(module_ls)-1):
    #     if idx >= depth:
    #         break

    #     module = module_ls[len(module_ls) - idx - 2]
    #     curr_param_ls = list(getattr(net, module).parameters())

    #     for param in curr_param_ls:
    #         grad_ls.append(param.grad.detach().cpu().clone())

    return grad_ls

def obtain_net_grad3(loss, sampled_net_param_layer_ls, sampled_net_param_ids_per_layer):
    grad_ls = []

    res_grad_ls = list(torch.autograd.grad(loss, sampled_net_param_layer_ls, retain_graph=True))

    for idx in range(len(res_grad_ls)):
        grad = res_grad_ls[idx]
        if sampled_net_param_ids_per_layer is None:
            grad_ls.append(grad.detach().cpu())
            continue
        param_ids_curr_layer = sampled_net_param_ids_per_layer[idx]
        if param_ids_curr_layer is None:
            grad_ls.append(grad.detach().cpu())
        else:
            grad_ls.append(grad.view(-1).detach().cpu()[param_ids_curr_layer])

    return grad_ls

def obtain_net_grad4(loss, sampled_net_param_layer_ls, sampled_layer_sqrt_prob_ls):
    grad_ls = []

    res_grad_ls = list(torch.autograd.grad(loss, sampled_net_param_layer_ls, retain_graph=True))

    for idx in range(len(res_grad_ls)):
        grad = res_grad_ls[idx]
        grad = grad/sampled_layer_sqrt_prob_ls[idx]
        grad_ls.append(grad.detach().cpu())

        # if sampled_net_param_ids_per_layer is None:
        #     grad_ls.append(grad.detach().cpu())
        #     continue
        # param_ids_curr_layer = sampled_net_param_ids_per_layer[idx]
        # if param_ids_curr_layer is None:
        #     grad_ls.append(grad.detach().cpu())
        # else:
        #     grad_ls.append(grad.view(-1).detach().cpu()[param_ids_curr_layer])

    return grad_ls


def obtain_full_loss(output, target, is_cuda, loss):
    if len(output.shape) > 1:
        onehot_target = torch.zeros([output.shape[0], output.shape[1]])

        sample_id_ls = torch.tensor(list(range(output.shape[0])))
        onehot_target[sample_id_ls,target] = 1
        if is_cuda:
            onehot_target = onehot_target.cuda()

        total_loss = loss + torch.sum(onehot_target.view(output.shape[0], -1)*output.view(output.shape[0], -1))
    else:
        

        # onehot_target = torch.zeros([output.shape[0], output.shape[1]])

        # sample_id_ls = torch.tensor(list(range(output.shape[0])))
        # onehot_target[sample_id_ls,target] = 1
        # if is_cuda:
        #     onehot_target = onehot_target.cuda()

        total_loss = loss + torch.sum(output*target + (1-output)*(1-target))#torch.sum(onehot_target.view(output.shape[0], -1)*output.view(output.shape[0], -1))

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

        if not os.path.exists(cached_model_file_name):
            args.logger.warning("Could not find cached model: %s"%(cached_model_file_name))
            return None
            
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


