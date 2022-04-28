import torch

import numpy as np

from tqdm import tqdm


def compute_norm_for_grad_ls(full_grad_ls):
    grad_norm_ls = []
    for grad_ls in full_grad_ls:
        curr_norm = 0
        for grad in grad_ls:
            curr_norm += torch.norm(grad)**2

        curr_norm = torch.sqrt(curr_norm)

        grad_norm_ls.append(curr_norm)
    return torch.tensor(grad_norm_ls)

def average_grad_vec(full_vec_ls, weight_ls = None, is_cuda = False):
    average_vec_ls = []

    for idx in range(len(full_vec_ls)):
        vec_ls = full_vec_ls[idx]

        curr_weight = None

        if weight_ls is not None:
            curr_weight = weight_ls[idx].item()

        if len(average_vec_ls) <= 0:

            for vec in vec_ls:
                               
                if curr_weight is not None:
                    vec = vec*curr_weight

                if is_cuda:
                    average_vec_ls.append(vec.clone().cuda())
                else:
                    average_vec_ls.append(vec.clone())

        else:
            for v_idx in range(len(vec_ls)):
                vec = vec_ls[v_idx]
                if is_cuda:
                    vec = vec.cuda()

                if curr_weight is not None:
                    vec = vec*curr_weight

                average_vec_ls[v_idx] += vec


    for idx in range(len(average_vec_ls)):

        average_vec_ls[v_idx] /= torch.sum(weight_ls)

        average_vec_ls[v_idx] = average_vec_ls[v_idx].cpu()

    return average_vec_ls


def obtain_full_loss_batch(output, target, is_cuda, loss):
    onehot_target = torch.zeros([output.shape[0], output.shape[1]])
    onehot_target[torch.tensor(list(range(len(target)))),target] = 1
    if is_cuda:
        onehot_target = onehot_target.cuda()

    total_loss = loss + torch.sum(onehot_target.view(onehot_target.shape[0], -1)*output.view(output.shape[0], -1), dim = 1)

    return total_loss

def obtain_loss_per_example(args, train_loader, net, criterion):

    criterion.reduction = 'none'

    loss_ls = torch.zeros(len(train_loader.dataset))

    with torch.no_grad():
        for batch_id, (sample_ids, data, labels) in enumerate(train_loader):
            # args.logger.info("sample batch ids::%d"%(batch_id))
            if args.cuda:
                data, labels = train_loader.dataset.to_cuda(data, labels)
            output = net.forward(data)
            loss = criterion(output, labels)
            loss = obtain_full_loss_batch(output, labels, args.cuda, loss)
            loss_ls[sample_ids] = loss.cpu()

    return loss_ls

def perturb_net_by_grad(net, eps, grad_ls = None):
    k = 0
    for param in net.parameters():
        if grad_ls is None:
            param.data = param.data + eps*param.grad.data
        else:
            param.data = param.data + eps*grad_ls[k]
        k += 1


def pairwise_cosine_full_for_grad_vec(args, train_loader, net, criterion, grad_norm_ls, data2, data2_norm_ls, is_cuda=False,  batch_size = 2048):
    
    B = data2

    baseloss_ls = obtain_loss_per_example(args, train_loader, net, criterion)

    full_sim_mat = torch.zeros([len(train_loader.dataset), len(data2)])

    eps = 1e-6

    for idx in range(len(data2)):

        perturb_net_by_grad(net, eps, grad_ls = data2[idx])

        loss_ls_with_update_net = obtain_loss_per_example(args, train_loader, net, criterion)

        full_sim_mat[:, idx] = 1 - torch.abs((loss_ls_with_update_net - baseloss_ls)/eps)/(grad_norm_ls*data2_norm_ls[idx])

    return full_sim_mat

def obtain_vectorized_grad(net):
    gradient_ls = []
    for param in net.parameters():
        curr_gradient = param.grad.detach().cpu().clone().view(-1)
        gradient_ls.append(curr_gradient)

    return torch.cat(gradient_ls)

def pairwise_cosine_full_for_grad_vec2(args, train_loader, net, criterion, grad_ls, grad_norm_ls, data2, data2_norm_ls, is_cuda=False,  batch_size = 2048):
    

    B_ls = []

    for data2_item in data2:
        B_ls.append(obtain_vectorized_grad(data2_item).view(-1))

    B = torch.stack(B_ls)

    baseloss_ls = obtain_loss_per_example(args, train_loader, net, criterion)

    full_sim_mat = torch.zeros([len(train_loader.dataset), len(data2)])

    eps = 1e-6

    for idx in range(len(data2)):

        perturb_net_by_grad(net, eps, grad_ls = data2[idx])

        loss_ls_with_update_net = obtain_loss_per_example(args, train_loader, net, criterion)

        full_sim_mat[:, idx] = 1 - torch.abs((loss_ls_with_update_net - baseloss_ls)/eps)/(grad_norm_ls*data2_norm_ls[idx])

    return full_sim_mat
    # full_dist_ls = []

    # full_cosin_mat = torch.zeros([len(data1), len(data2)])

    # for start_id in range(0, A.shape[0], batch_size):
    #     end_id = start_id + batch_size
    #     print("calculated sample ids::", start_id)
    #     if end_id > A.shape[0]:
    #         end_id = A.shape[0]

    #     curr_A = A[start_id: end_id]
    #     if is_cuda:
    #         curr_A = curr_A.cuda()
    #     curr_A_normalized = curr_A# / curr_A.norm(dim=-1, keepdim=True)

    #     for start_id2 in range(0, B.shape[0], batch_size):
    #         end_id2 = start_id2 + batch_size
    #         if end_id2 > B.shape[0]:
    #             end_id2 = B.shape[0]
            
    #         curr_B = B[start_id2: end_id2]
    #         if is_cuda:
    #             curr_B = curr_B.cuda()


    #         B_normalized = curr_B #/ curr_B.norm(dim=-1, keepdim=True)
    #         curr_cosine_dis = torch.mm(curr_A_normalized, torch.t(B_normalized))# curr_A_normalized * B_normalized    
    #         # curr_cosine_dis = (curr_cosine.sum(dim=-1)).squeeze(1)

    #         full_cosin_mat[start_id:end_id, start_id2:end_id2] = curr_cosine_dis.cpu()
    #         del B_normalized, curr_B, curr_cosine_dis

    #     del curr_A_normalized, curr_A

    # return full_cosin_mat/data1.shape[0]

def initialize_rand_grad(data_grad_ls, data_grad_norm_ls, num_clusters, is_cuda = False):
    num_samples = len(data_grad_ls)
    
    indices = np.random.choice(num_samples, num_clusters, replace=False)

    init_state_ls = []

    for idx in indices:
        if not is_cuda:
            state = [data_grad_ls[idx][k].clone() for k in range(len(data_grad_ls[idx]))]
        else:
            state = [data_grad_ls[idx][k].clone().cuda() for k in range(len(data_grad_ls[idx]))]

        init_state_ls.append(state)

    init_grad_norm_ls = data_grad_norm_ls[torch.from_numpy(indices)]

    return init_state_ls, init_grad_norm_ls


def compute_grad_center_shirt(initial_state, initial_state_pre):
    total_dist = 0

    for idx in range(len(initial_state)):
        
        curr_total_dist = 0
        for layer_idx in range(len(initial_state[idx])):
            curr_total_dist += torch.sum((initial_state[idx][layer_idx] - initial_state_pre[idx][layer_idx]) ** 2)

        curr_total_dist = torch.sqrt(curr_total_dist)

        total_dist += curr_total_dist

    return total_dist

def kmeans_with_grad_vec(
        args,
        net,
        train_loader,
        criterion,
        X_grad_ls,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        is_cuda=False,
        sample_weights = None,
        total_iter_count=1000,
        all_layer = False
):
    """
    perform kmeans
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param tol: (float) threshold [default: 0.0001]
    :param device: (torch.device) device [default: cpu]
    :return: (torch.tensor, torch.tensor) cluster ids, cluster centers
    """
    if is_cuda:
        print(f'running k-means on cuda..')
    else:
        print(f'running k-means on cpu..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_cosine_full_for_grad_vec

    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine_full_for_grad_vec

    else:
        raise NotImplementedError

    # convert to float
    # if not all_layer:
    #     X = X.float()
    # else:
    #     for idx in range(len(X)):
    #         X[idx] = X[idx].float()

    # transfer to device
    # if args.cuda:
    #     X = X.cuda()
    # if is_cuda:
    #     X = X.cuda()
    X_grad_norm_ls = compute_norm_for_grad_ls(X_grad_ls)
    # initialize
    # initial_state = initialize(X, num_clusters, all_layer = all_layer)
    initial_state, initial_grad_norm_ls = initialize_rand_grad(X_grad_ls, X_grad_norm_ls, num_clusters, is_cuda)

    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]')
    # while True:
    for k in range(0,total_iter_count):

        full_centroid_state = initial_state

        full_centroid_norm = initial_grad_norm_ls

        dis = pairwise_distance_function(args, train_loader, net, criterion, X_grad_norm_ls, full_centroid_state, full_centroid_norm, is_cuda=is_cuda)

        # dis = pairwise_distance_function(X, full_centroid_state,is_cuda)

        choice_cluster = torch.argmin(dis, dim=1)

        if not all_layer:
            initial_state_pre = [state.clone() for state in initial_state]
        # else:
            # initial_state_pre = [initial_state[k].clone() for k in range(len(initial_state))]

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze()
            if torch.sum(choice_cluster == index) <= 0:
                continue

             
            # if args.cuda:
            #     selected = selected.cuda()
            selected_sample_weights = None
            if sample_weights is not None:
                # selected_sample_weights = torch.index_select(sample_weights, 0, selected)
                selected_sample_weights = sample_weights[selected.cpu()]
                # if is_cuda:
                #     selected_sample_weights = selected_sample_weights.cuda()
            # selected = torch.index_select(X, 0, selected)
            if not all_layer:
                selected_grad_ls = X_grad_ls[selected]
                # selected_grad_norm_ls = X_grad_norm_ls[selected]

                initial_state[index] = average_grad_vec(selected_grad_ls, weight_ls = selected_sample_weights, is_cuda = is_cuda)
                # if is_cuda:
                #     selected = selected.cuda()

            #     if sample_weights is None:
            #         selected_state = selected.mean(dim=0)
            #     else:
            #         selected_state = torch.sum(selected*selected_sample_weights.view(-1,1), dim = 0)/torch.sum(selected_sample_weights)
            #     if is_cuda:
            #         selected_state = selected_state.cuda()

            #     initial_state[index] = selected_state
            # else:
            #     selected = select_samples_by_ls(X, selected, is_cuda)
            
            #     update_centroid_by_ls(selected, initial_state, selected_sample_weights, index)
            
        if not all_layer:
            initial_grad_norm_ls = compute_norm_for_grad_ls(initial_state)  


            center_shift = compute_grad_center_shirt(initial_state, initial_state_pre)

        # if not all_layer:
        #     center_shift = torch.sum(
        #         torch.sqrt(
        #             torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
        #         ))

        # else:
        #     center_shift = compute_center_shift_by_ls(initial_state, initial_state_pre)

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        tqdm_meter.set_postfix(
            iteration=f'{iteration}',
            center_shift=f'{center_shift ** 2:0.6f}',
            tol=f'{tol:0.6f}'
        )
        tqdm_meter.update()
        if center_shift ** 2 < tol:
            break
    if not all_layer:
        return choice_cluster.cpu(), initial_state.cpu()
    else:
        for k in range(len(initial_state)):
            initial_state[k] = initial_state[k].cpu()
        return choice_cluster.cpu(), initial_state