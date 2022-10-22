import numpy as np
import torch
from tqdm import tqdm
import sys


def initialize(X, num_clusters, all_layer = False):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    if not all_layer:
        num_samples = len(X)
    else:
        num_samples = len(X[0])
    indices = np.random.choice(num_samples, num_clusters, replace=False)
    if not all_layer:
        initial_state = X[indices]
    else:
        initial_state = [X[k][indices] for k in range(len(X))]
    return initial_state

def initialize_sample_ids(X, num_clusters, all_layer = False):
    """
    initialize cluster centers
    :param X: (torch.tensor) matrix
    :param num_clusters: (int) number of clusters
    :return: (np.array) initial state
    """
    # if not all_layer:
    #     num_samples = len(X)
    # else:
    #     num_samples = len(X[0])
    num_samples = len(X)
    indices = np.random.choice(num_samples, num_clusters, replace=False)

    # if not all_layer:
    #     initial_state = X[indices]
    # else:
    #     initial_state = [X[k][indices] for k in range(len(X))]

    cluster_sample_id_ls = []

    for k in range(num_clusters):
        cluster_sample_id_ls.append(torch.tensor(indices[k]).view(-1))

    return cluster_sample_id_ls, None
    # if not all_layer:
    #     initial_state = X[indices]
    # else:
    #     initial_state = [X[k][indices] for k in range(len(X))]
    # return initial_state


def calculate_silhouette_scores(X, cluster_ids, cluster_centroids, is_cuda, sample_weights = None, distance = 'euclidean', batch_size = 100):
    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    new_cluster_centroids = []

    for k in range(len(cluster_centroids)):
        curr_cluster_centroid = torch.mean(X[cluster_ids == k])
        new_cluster_centroids.append(curr_cluster_centroid.view(-1))

    cluster_centroids = torch.stack(new_cluster_centroids)

    pairwise_distance_tensor = pairwise_distance_function(X, cluster_centroids,is_cuda)

    sorted_distance_by_dim, sorted_idx_by_dim = torch.sort(pairwise_distance_tensor,dim=1, descending=False)

    nearest_nb_cluster_ids = sorted_idx_by_dim[:,1]

    selected_cluster_ids = sorted_idx_by_dim[:,0]

    # assert torch.max(torch.abs(selected_cluster_ids.cpu().view(-1) - cluster_ids.view(-1))).item() <= 0.0
    
    mean_s_value = 0

    s_value_count = 0

    s_value_ls = torch.zeros([X.shape[0]])

    if is_cuda:
        cluster_ids = cluster_ids.cuda()

    if is_cuda:
        X = X.cuda()
    intra_clust_dists_ls = []
    inter_clust_dists_ls = []
    
    label_freqs = torch.bincount(cluster_ids)
    for start_id in range(0, X.shape[0], batch_size):
        end_id = start_id + batch_size
        if end_id >= X.shape[0]:
            end_id = X.shape[0]

        curr_X = X[start_id:end_id]

        curr_X_and_X_distance = pairwise_distance_function(curr_X, X, is_cuda)

        clust_dists = torch.zeros((len(curr_X), cluster_centroids.shape[0]),
                           dtype=X.dtype)

        if is_cuda:
            clust_dists = clust_dists.cuda()

        for i in range(len(curr_X)):
            clust_dists[i] += torch.bincount(cluster_ids, weights=curr_X_and_X_distance[i],
                                        minlength=len(label_freqs))

        intra_index = cluster_ids[start_id:end_id]# (np.arange(len(curr_X)), labels[start:start + len(D_chunk)])
        # intra_clust_dists are averaged over cluster size outside this function
        intra_clust_dists = clust_dists[torch.tensor(list(range(clust_dists.shape[0]))), intra_index]
        # of the remaining distances we normalise and extract the minimum
        clust_dists[torch.tensor(list(range(clust_dists.shape[0]))), intra_index] = torch.inf
        clust_dists /= label_freqs
        inter_clust_dists = torch.min(clust_dists,dim=1)[0]# clust_dists.min(dim=1)
        intra_clust_dists_ls.append(intra_clust_dists.view(-1))
        inter_clust_dists_ls.append(inter_clust_dists.view(-1))

    intra_clust_dists_tensor = torch.cat(intra_clust_dists_ls)
    inter_clust_dists_tensor = torch.cat(inter_clust_dists_ls)


    denom = (label_freqs - 1).take(cluster_ids)
    intra_clust_dists_tensor = intra_clust_dists_tensor/denom
    sil_samples = inter_clust_dists_tensor - intra_clust_dists_tensor

    sil_samples /= torch.maximum(intra_clust_dists_tensor, inter_clust_dists_tensor)

    sil_samples = torch.nan_to_num(sil_samples)
    # for c_id1 in range(len(cluster_centroids)):
        
    #     curr_cluster_samples = X[cluster_ids == c_id1]

    #     for c_id2 in range(len(cluster_centroids)):
    #         curr_sample_ids = torch.logical_and((cluster_ids == c_id1), (nearest_nb_cluster_ids == c_id2))

    #         nearest_cluster_samples = X[cluster_ids == c_id2]

    #         curr_X = X[curr_sample_ids]

    #         if curr_X.shape[0] <= 0:
    #             continue

    #         curr_weight = None
    #         if sample_weights is not None:
    #             curr_weight = sample_weights[curr_sample_ids]


    #         a_value = pairwise_distance_function(curr_cluster_samples, curr_X,device).view(-1,curr_X.shape[0])
    #         a_value = torch.sum(a_value, dim = 0)/(len(curr_cluster_samples)-1)

    #         b_value = pairwise_distance_function(nearest_cluster_samples, curr_X, device).view(-1, curr_X.shape[0])
    #         b_value = torch.mean(b_value, dim = 0)

    #         a_b_value_arr = torch.stack([a_value, b_value], dim = 1)
    #         s_value = (b_value - a_value)/(torch.max(a_b_value_arr, dim=1)[0])

    #         if sample_weights is not None:
    #             s_value = s_value*curr_weight

    #         # s_value_ls.append(s_value.cpu())
    #         s_value_ls[curr_sample_ids] = s_value.cpu()

    #         mean_s_value += torch.sum(s_value).cpu()

    #         s_value_count += s_value.shape[0]

    # s_value_array = torch.cat(s_value_ls)
    if sample_weights is None:
        return torch.mean(sil_samples)
    else:
        return torch.sum(sil_samples.view(-1)*sample_weights.view(-1))/len(sil_samples)
    # return mean_s_value/X.shape[0]


def select_samples_by_ls(X, selected, is_cuda):

    selected_X = []

    for idx in range(len(X)):
        curr_X = X[idx][selected]
        # print("selected shape::", selected.shape)
        # print("curr_X shape::", curr_X.shape)
        if is_cuda:
            curr_X = curr_X.cuda()

        selected_X.append(curr_X)
    
    return selected_X

def solve_k_mean_centroids(selected, initial_state, selected_sample_weights, index, pairwise_distance_function, selected_sample_norm = None, iter = 1000, learning_rate = 0.001, batch_size = 100, is_gpu=False):
    curr_state = [initial_state[k][index].unsqueeze(0) for k in range(len(initial_state))]

    for k in range(len(curr_state)):
        curr_state[k].requires_grad = True
        if is_gpu:
            curr_state[k] = curr_state[k].cuda()
    
    for i in range(iter):
        if len(selected[0]) >=batch_size:
            sub_sample_ids = np.random.choice(len(selected[0]), batch_size, replace = False)
        else:
            sub_sample_ids = np.array(list(range(len(selected[0]))))
        sub_sample_ids = torch.from_numpy(sub_sample_ids)
        sub_selected = [selected[k][sub_sample_ids] for k in range(len(selected))]
        sub_sample_weights = None
        if selected_sample_weights is not None:
            sub_sample_weights = selected_sample_weights[sub_sample_ids]
        if is_gpu:
            sub_selected = [sub_selected[k].cuda() for k in range(len(sub_selected))]
        
        # if all_layer:
        dis = pairwise_distance_function(sub_selected, curr_state, is_gpu, inner_prod= True)

        if sub_sample_weights is not None:
            full_dis = torch.sum(dis.view(-1)**2*sub_sample_weights.view(-1))/torch.sum(sub_sample_weights)
        else:
            full_dis = torch.mean(dis**2)
        curr_state_grad = torch.autograd.grad(full_dis, curr_state)
        if i % 100 == 0:
            print("grad norm::", sum([curr_state_grad[k].norm()**2 for k in range(len(curr_state_grad))]))

        curr_state = [curr_state[k] - learning_rate*curr_state_grad[k] for k in range(len(curr_state))]


    for k in range(len(initial_state)):
        curr_state[k] = curr_state[k].detach().cpu()
        # curr_state[k].requires_grad = False
        # curr_state[k] = curr_state[k].cpu()
        initial_state[k][index] = curr_state[k]
        # initial_state[k].requires_grad = True

    return initial_state

def update_centroid_by_ls(selected, initial_state, selected_sample_weights, index, selected_sample_norm = None):
    if selected_sample_weights is None:
        for k in range(len(initial_state)):
            if selected_sample_norm is None:
                initial_state[k][index] = selected[k].mean(dim=0)
            else:
                initial_state[k][index] = torch.mean(selected[k]/selected_sample_norm.view(-1,1), dim=0)
    else:
        for k in range(len(initial_state)):
            if selected_sample_norm is None:
                initial_state[k][index] = torch.sum(selected[k]*selected_sample_weights.view(-1,1), dim = 0)/torch.sum(selected_sample_weights)
            else:
                initial_state[k][index] = torch.sum(selected[k]*selected_sample_weights.view(-1,1), dim = 0)/torch.sum(selected_sample_weights.view(-1)*selected_sample_norm.view(-1))
        # initial_state[index] = torch.sum(selected*selected_sample_weights.view(-1,1), dim = 0)/torch.sum(selected_sample_weights)


def compute_center_shift_by_ls(initial_state, initial_state_pre):
    center_shift = 0
    for idx in range(len(initial_state)):
        center_shift += torch.sum(
                torch.sqrt(
                    torch.sum((initial_state[idx] - initial_state_pre[idx]) ** 2, dim=1)
                ))
    return center_shift


def print_cluster_dist_info(dist):
    min_dist_values, min_dist_ids = torch.min(dist, dim=1)

    print("max dist sample to assigned cluster mean::", torch.max(min_dist_values).item())

    if dist.shape[1] >= 2:
        second_min_cluster_dist, second_min_cluster_ids = torch.topk(dist, k = 2, largest=False)

        print("min dist sample to other cluster mean::", torch.min(second_min_cluster_dist[:,1]).item())


def rescale_dist_by_cluster_mean_norm(dist, cluster_mean_ls, all_layer = False):
    if not all_layer:
        cluster_mean_norm_ls = torch.norm(cluster_mean_ls, dim = 1)
    else:
        cluster_mean_norm_ls = torch.sqrt(torch.sum(torch.stack([torch.norm(cluster_mean_ls[k], dim = 1)**2 for k in range(len(cluster_mean_ls))],dim=1),dim=1))

    cluster_mean_norm_ls = torch.max(cluster_mean_norm_ls)/cluster_mean_norm_ls

    dist = dist*cluster_mean_norm_ls.view(1,-1)

    return dist


def kmeans_init(data, num_clusters, distance, all_layer,is_cuda, weight_by_norm = False, inner_prod = False, ls_idx_range=None):
    centroids = []
    
    if not all_layer:
        num_samples = data.shape[0]
        rand_sampled_id = np.random.randint(
            data.shape[0])
        centroids.append(data[rand_sampled_id, :])
    else:
        num_samples = data[0].shape[0]
        rand_sampled_id = np.random.randint(
            data[0].shape[0])
        centroids.extend([data[k][rand_sampled_id:rand_sampled_id+1, :] for k in range(len(data))])
    # plot(data, np.array(centroids))
  
    ## compute remaining k - 1 centroids
    for c_id in range(num_clusters - 1):
         
        ## initialize a list to store distances of data
        ## points from nearest centroid
        # dist = []
        if all_layer:
            full_dist = distance(data, centroids,is_cuda, weight_by_norm=weight_by_norm, inner_prod= inner_prod, ls_idx_range=ls_idx_range)
        else:
            full_dist = distance(data, centroids,is_cuda, weight_by_norm=weight_by_norm)


        # full_dist = distance(data, centroids, is_cuda, weight_by_norm = weight_by_norm, inner_prod = inner_prod, ls_idx_range = ls_idx_range)
        dist = torch.min(full_dist, dim = 1)[0]
        # for i in range(num_samples):
        #     if not all_layer:
        #         point = data[i, :]
        #     else:
        #         point =[data[s_id][i, :] for s_id in range(len(data))]
        #     d = sys.maxsize

            
        #     ## compute distance of 'point' from each of the previously
        #     ## selected centroid and store the minimum distance
        #     for j in range(len(centroids)):
        #         temp_dist = distance(point, centroids[j])
        #         d = min(d, temp_dist)
        #     dist.append(d)
             
        # ## select data point with maximum distance as our next centroid
        # dist = np.array(dist)
        max_dis_sample_id = torch.argmax(dist)
        if not all_layer:
            next_centroid = data[max_dis_sample_id, :]
            centroids.append(next_centroid)
        else:
            next_centroid = [data[k][max_dis_sample_id:max_dis_sample_id+1, :] for k in range(len(data))]
            centroids = [torch.cat([centroids[k],next_centroid[k]]) for k in range(len(next_centroid))]


        dist = []
        # plot(data, np.array(centroids))
    if not all_layer:
        return torch.stack(centroids, dim=0)
    else:
        return centroids
   
def scale_and_extend_data_vector(X):
    X_norms = torch.sqrt(torch.stack([torch.norm(X[k],dim=1).view(-1)**2 for k in range(len(X))], dim =1))
    X_max_norm = torch.max(X_norms)*2

    scaled_X = [X[k]/X_max_norm for k in range(len(X))]

    scaled_X_norms = torch.sum(torch.stack([torch.norm(scaled_X[k],dim=1).view(-1)**2 for k in range(len(scaled_X))], dim =1),dim=1)

    extend_X = []

    for l in range(1,10):
        extend_X.append((0.5-scaled_X_norms**(l)).unsqueeze(1))

    scaled_X.extend(extend_X)

    return scaled_X


def kmeans(
        # args,
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        is_cuda=False,
        sample_weights = None,
        existing_cluster_mean_ls = None,
        total_iter_count=200,
        all_layer = False,
        agg_sim_array = 'mean',
        weight_by_norm = False,
        inner_prod = False,
        k_means_lr = 0.001,
        k_means_epochs = 100,
        k_means_bz = 128,
        origin_X_ls_lenth = -1,
        rand_init = False
):
    is_cuda = False
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
        if not all_layer:
            pairwise_distance_function = pairwise_distance
        else:
            pairwise_distance_function = pairwise_distance_ls
    elif distance == 'cosine':
        if not all_layer:
            pairwise_distance_function = pairwise_cosine
        else:
            pairwise_distance_function = pairwise_cosine_ls
    # elif distance == 'cross':
    #     if not all_layer:
    #         pairwise_distance_function = pairwise_cross_prod
    #     else:
    #         pairwise_distance_function = pairwise_cross_prod_ls
    else:
        raise NotImplementedError

    # convert to float
    if not all_layer:
        X = X.float()
    else:
        for idx in range(len(X)):
            X[idx] = X[idx].float()

    

    if weight_by_norm:
        if not all_layer:
            sample_norm_ls = torch.norm(X,dim=1)
        else:
            sample_norm_ls = torch.sqrt(torch.sum(torch.stack([torch.norm(X[k],dim=1)**2 for k in range(len(X))],dim=1),dim=1))

    else:
        sample_norm_ls = None
    curr_sample_weights = None
    if sample_weights is not None:
        curr_sample_weights = sample_weights.clone()

    # if weight_by_norm:
    #     if curr_sample_weights is None:
    #         curr_sample_weights = sample_norm_ls.clone()
    #     else:
    #         curr_sample_weights = (sample_norm_ls*sample_weights).clone()
    # transfer to device
    # if args.cuda:
    #     X = X.cuda()
    # if is_cuda:
    #     X = X.cuda()

    # initialize
    if rand_init:
        initial_state = initialize(X, num_clusters, all_layer = all_layer)
    else:
        initial_state = kmeans_init(X, num_clusters, pairwise_distance_function, all_layer, is_cuda, weight_by_norm=weight_by_norm, inner_prod=inner_prod, ls_idx_range=origin_X_ls_lenth)
    if is_cuda:
        if not all_layer:
            initial_state = initial_state.cuda()
        else:
            for idx in range(len(initial_state)):
                initial_state[idx] = initial_state[idx].cuda()

    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]')
    # while True:
    for k in range(0,total_iter_count):

        full_centroid_state = initial_state

        if existing_cluster_mean_ls is not None:
            full_centroid_state = torch.cat([existing_cluster_mean_ls, full_centroid_state], dim = 0)

        if all_layer:
            dis = pairwise_distance_function(X, full_centroid_state,is_cuda, agg = agg_sim_array, weight_by_norm=weight_by_norm, inner_prod= inner_prod, ls_idx_range=origin_X_ls_lenth)
        else:
            dis = pairwise_distance_function(X, full_centroid_state,is_cuda, weight_by_norm=weight_by_norm)

        # if weight_by_norm:
        #     dis = rescale_dist_by_cluster_mean_norm(dis, full_centroid_state, all_layer)

        choice_cluster = torch.argmin(dis, dim=1)

        if not all_layer:
            initial_state_pre = initial_state.clone()
        else:
            initial_state_pre = [initial_state[k].clone() for k in range(len(initial_state))]

        

        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).view(-1)
            selected_sample_norm = None
            if sample_norm_ls is not None:
                selected_sample_norm = sample_norm_ls[selected]

                if is_cuda:
                    selected_sample_norm = selected_sample_norm.cuda()
            if torch.sum(choice_cluster == index) <= 0:
                continue

             
            # if args.cuda:
            #     selected = selected.cuda()
            selected_sample_weights = None
            if curr_sample_weights is not None:
                # selected_sample_weights = torch.index_select(sample_weights, 0, selected)
                selected_sample_weights = curr_sample_weights[selected.cpu()]


                if is_cuda:
                    selected_sample_weights = selected_sample_weights.cuda()

            if inner_prod:
                selected = select_samples_by_ls(X, selected, is_cuda)
                initial_state = solve_k_mean_centroids(selected, initial_state, selected_sample_weights, index, pairwise_distance_function, selected_sample_norm = selected_sample_norm, iter = k_means_epochs, learning_rate = k_means_lr, batch_size = k_means_bz, is_gpu=is_cuda)
            else:
                # selected = torch.index_select(X, 0, selected)
                if not all_layer:
                    selected = X[selected]
                    if is_cuda:
                        selected = selected.cuda()
                        if selected_sample_weights is not None:
                            selected_sample_weights = selected_sample_weights.cuda()

                    if selected_sample_weights is None:
                        if selected_sample_norm is None:
                            selected_state = selected.mean(dim=0)
                        else:
                            selected_state = torch.mean(selected/selected_sample_norm.view(-1,1), dim=0)
                    else:
                        if selected_sample_norm is None:
                            selected_state = torch.sum(selected*selected_sample_weights.view(-1,1), dim = 0)/torch.sum(selected_sample_weights)
                        else:
                            selected_state = torch.sum(selected*selected_sample_weights.view(-1,1), dim = 0)/torch.sum(selected_sample_weights.view(-1)*selected_sample_norm.view(-1))
                    if is_cuda:
                        selected_state = selected_state.cuda()

                    initial_state[index] = selected_state
                else:
                    selected = select_samples_by_ls(X, selected, is_cuda)
                
                    update_centroid_by_ls(selected, initial_state, selected_sample_weights, index, selected_sample_norm)
            

        if not all_layer:
            center_shift = torch.sum(
                torch.sqrt(
                    torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
                ))

        else:
            center_shift = compute_center_shift_by_ls(initial_state, initial_state_pre)

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
    
    print_cluster_dist_info(dis)
    
    if all_layer:
        no_abs_dis = pairwise_distance_function(X, full_centroid_state,is_cuda, agg = agg_sim_array, weight_by_norm=weight_by_norm, inner_prod= inner_prod, ls_idx_range=origin_X_ls_lenth, no_abs = True)
    else:
        no_abs_dis = pairwise_distance_function(X, full_centroid_state,is_cuda, weight_by_norm=weight_by_norm, no_abs = True)


    if not all_layer:
        return choice_cluster.cpu(), initial_state.cpu()
    else:
        for k in range(len(initial_state)):
            initial_state[k] = initial_state[k].cpu()
        return choice_cluster.cpu(), initial_state


def select_samples_with_greedy_algorithm(full_x_cosin_sim, num_clusters):

    selected_sample_boolean_arrays = torch.ones(len(full_x_cosin_sim))

    first_select_sample_id = torch.argmax(torch.mean(torch.abs(full_x_cosin_sim), dim=1)[0])
    curr_cosin_sim_sum = full_x_cosin_sim[first_select_sample_id]
    print("curr mean cosin sim sum::", torch.mean(torch.abs(curr_cosin_sim_sum)))
    selected_sample_boolean_arrays[first_select_sample_id] = 0

    for k in range(num_clusters-1):
        remaining_x_cosin_sim = full_x_cosin_sim[selected_sample_boolean_arrays.nonzero().view(-1)]
        curr_select_sample_id = torch.argmax(torch.mean(torch.abs(curr_cosin_sim_sum.view(1,-1) + remaining_x_cosin_sim), dim = 1)[0])

        curr_select_sample_real_id = selected_sample_boolean_arrays.nonzero().view(-1)[curr_select_sample_id]

        curr_cosin_sim_sum = curr_cosin_sim_sum + full_x_cosin_sim[curr_select_sample_real_id]
        selected_sample_boolean_arrays[curr_select_sample_real_id] = 0
        print("curr mean cosin sim sum::", str(k+1), torch.mean(torch.abs(curr_cosin_sim_sum)).item())

    return torch.nonzero(selected_sample_boolean_arrays == 0).view(-1)

def kmeans_cosin(
        # args,
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        is_cuda=False,
        sample_weights = None,
        existing_cluster_mean_ls = None,
        total_iter_count=1000,
        all_layer = False,
        full_x_cosin_sim = None
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

    # if distance == 'euclidean':
    #     pairwise_distance_function = pairwise_distance
    # elif distance == 'cosine':
    #     if not all_layer:
    if full_x_cosin_sim is None:

        full_x_cosin_sim = pairwise_cosine_full(X, is_cuda=is_cuda)

    # max_cosin_sim = torch.abs(full_x_cosin_sim).max()

    # full_x_cosin_sim = full_x_cosin_sim/max_cosin_sim

    print('max distance::', full_x_cosin_sim.max())
    print('min distance::', full_x_cosin_sim.min())

    pairwise_distance_function = pairwise_cosine2
    #     else:
    #         pairwise_distance_function = pairwise_cosine_ls
    # else:
    #     raise NotImplementedError

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

    # initialize
    # initial_state0 = initialize(X, num_clusters, all_layer = all_layer)

    initial_state_sample_ids, _ = initialize_sample_ids(full_x_cosin_sim, num_clusters, all_layer = all_layer)

    if is_cuda:
        initial_state_sample_ids = [sample_ids.cuda() for sample_ids in initial_state_sample_ids]
        # if not all_layer:
        #     initial_state0 = initial_state0.cuda()
        # else:
        #     for idx in range(len(initial_state0)):
        #         initial_state0[idx] = initial_state0[idx].cuda()

    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]')
    # while True:
    for k in range(0,total_iter_count):

        # full_centroid_state = initial_state

        # if existing_cluster_mean_ls is not None:
        #     full_centroid_state = torch.cat([existing_cluster_mean_ls, full_centroid_state], dim = 0)

        # dis = pairwise_distance_function(X, initial_state0,is_cuda)

        full_dis = pairwise_cosine_full_by_sample_ids(full_x_cosin_sim, initial_state_sample_ids, is_cuda=is_cuda, sample_weights = sample_weights)

        choice_cluster = torch.argmin(full_dis, dim=1)

        # if not all_layer:
        #     initial_state_pre = initial_state0.clone()
        # else:
        #     initial_state_pre = [initial_state0[k].clone() for k in range(len(initial_state0))]

        initial_state_sample_ids_pre = [sample_ids.clone() for sample_ids in initial_state_sample_ids]
        
        initial_state_sample_ids = []
        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).view(-1)
            if torch.sum(choice_cluster == index) <= 0:
                initial_state_sample_ids.append(initial_state_sample_ids_pre[index])
            else:
                initial_state_sample_ids.append(selected)


        # for index in range(num_clusters):
        #     selected = torch.nonzero(choice_cluster == index).view(-1)
        #     if torch.sum(choice_cluster == index) <= 0:
        #         continue

             
        #     # if args.cuda:
        #     #     selected = selected.cuda()
        #     selected_sample_weights = None
        #     if sample_weights is not None:
        #         # selected_sample_weights = torch.index_select(sample_weights, 0, selected)
        #         selected_sample_weights = sample_weights[selected.cpu()]
        #         if is_cuda:
        #             selected_sample_weights = selected_sample_weights.cuda()
        #     # selected = torch.index_select(X, 0, selected)
        #     if not all_layer:
        #         selected = X[selected]
        #         if is_cuda:
        #             selected = selected.cuda()

        #         if sample_weights is None:
        #             initial_state0[index] = selected.mean(dim=0)
        #         else:
        #             initial_state0[index] = torch.sum(selected*selected_sample_weights.view(-1,1), dim = 0)/torch.sum(selected_sample_weights)
        #     else:
        #         selected = select_samples_by_ls(X, selected, is_cuda)
            
        #         update_centroid_by_ls(selected, initial_state0, selected_sample_weights, index)
            

        # if not all_layer:
        #     center_shift0 = torch.sum(
        #         torch.sqrt(
        #             torch.sum((initial_state0 - initial_state_pre) ** 2, dim=1)
        #         ))

        # else:
        #     center_shift0 = compute_center_shift_by_ls(initial_state0, initial_state_pre)
        center_shift = 0
        for k in range(len(initial_state_sample_ids_pre)):
            curr_sample_weights = None
            curr_sample_weights_pre = None
            if sample_weights is not None:
                curr_sample_weights = sample_weights[initial_state_sample_ids[k]]
                curr_sample_weights_pre = sample_weights[initial_state_sample_ids_pre[k]]

            curr_x_cosin_sim = full_x_cosin_sim[initial_state_sample_ids_pre[k]][:,initial_state_sample_ids[k]]
            if curr_sample_weights_pre is not None:
                curr_full_dis = torch.sum(curr_x_cosin_sim.view(curr_sample_weights_pre.shape[0],-1)*curr_sample_weights_pre.view(curr_sample_weights_pre.shape[0],1), dim = 0)
                curr_full_dis = curr_full_dis/torch.sum(curr_sample_weights_pre)
            else:
                curr_full_dis = torch.mean(curr_x_cosin_sim, dim = 0) #pairwise_cosine_full_by_sample_ids(curr_x_cosin_sim, [torch.tensor(list(range(curr_x_cosin_sim.shape[0])))], is_cuda=is_cuda, sample_weights = curr_sample_weights_pre)

            if curr_sample_weights is None:
                curr_full_dis = torch.mean(curr_full_dis)
            else:
                curr_full_dis = torch.sum(curr_full_dis.view(-1)*curr_sample_weights.view(-1))/torch.sum(curr_sample_weights)
            center_shift += torch.abs(curr_full_dis)
        
        center_shift = center_shift/len(initial_state_sample_ids_pre)

        # center_shift = sum([(len(set(initial_state_sample_ids_pre[k].tolist()).difference(set(initial_state_sample_ids[k].tolist()))) + len(set(initial_state_sample_ids[k].tolist()).difference(set(initial_state_sample_ids_pre[k].tolist()))))/(len(initial_state_sample_ids_pre[k]) + len(initial_state_sample_ids[k])) for k in range(num_clusters)])

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        tqdm_meter.set_postfix(
            iteration=f'{iteration}',
            center_shift=f'{center_shift:0.6f}',
            tol=f'{tol:0.6f}'
        )
        tqdm_meter.update()
        if center_shift < tol:
            break

    return choice_cluster.cpu(), full_x_cosin_sim



def kmeans_l2(
        # args,
        X,
        num_clusters,
        distance='euclidean',
        tol=1e-4,
        is_cuda=False,
        sample_weights = None,
        existing_cluster_mean_ls = None,
        total_iter_count=1000,
        all_layer = False,
        full_x_cosin_sim = None
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

    # if distance == 'euclidean':
    #     pairwise_distance_function = pairwise_distance
    # elif distance == 'cosine':
    #     if not all_layer:
    if full_x_cosin_sim is None:

        full_x_cosin_sim = pairwise_l2_full(X, is_cuda=is_cuda)

    # max_cosin_sim = torch.abs(full_x_cosin_sim).max()

    # full_x_cosin_sim = full_x_cosin_sim/max_cosin_sim

    print('max distance::', full_x_cosin_sim.max())
    print('min distance::', full_x_cosin_sim.min())

    # pairwise_distance_function = pairwise_cosine2
    #     else:
    #         pairwise_distance_function = pairwise_cosine_ls
    # else:
    #     raise NotImplementedError

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

    # initialize
    # initial_state0 = initialize(X, num_clusters, all_layer = all_layer)

    initial_state_sample_ids, _ = initialize_sample_ids(full_x_cosin_sim, num_clusters, all_layer = all_layer)

    if is_cuda:
        initial_state_sample_ids = [sample_ids.cuda() for sample_ids in initial_state_sample_ids]
        # if not all_layer:
        #     initial_state0 = initial_state0.cuda()
        # else:
        #     for idx in range(len(initial_state0)):
        #         initial_state0[idx] = initial_state0[idx].cuda()

    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]')
    # while True:
    for k in range(0,total_iter_count):

        # full_centroid_state = initial_state

        # if existing_cluster_mean_ls is not None:
        #     full_centroid_state = torch.cat([existing_cluster_mean_ls, full_centroid_state], dim = 0)

        # dis = pairwise_distance_function(X, initial_state0,is_cuda)

        full_dis = pairwise_cosine_full_by_sample_ids(full_x_cosin_sim, initial_state_sample_ids, is_cuda=is_cuda, sample_weights = sample_weights)

        choice_cluster = torch.argmin(full_dis, dim=1)

        # if not all_layer:
        #     initial_state_pre = initial_state0.clone()
        # else:
        #     initial_state_pre = [initial_state0[k].clone() for k in range(len(initial_state0))]

        initial_state_sample_ids_pre = [sample_ids.clone() for sample_ids in initial_state_sample_ids]
        
        initial_state_sample_ids = []
        for index in range(num_clusters):
            selected = torch.nonzero(choice_cluster == index).view(-1)
            if torch.sum(choice_cluster == index) <= 0:
                initial_state_sample_ids.append(initial_state_sample_ids_pre[index])
            else:
                initial_state_sample_ids.append(selected)


        # for index in range(num_clusters):
        #     selected = torch.nonzero(choice_cluster == index).view(-1)
        #     if torch.sum(choice_cluster == index) <= 0:
        #         continue

             
        #     # if args.cuda:
        #     #     selected = selected.cuda()
        #     selected_sample_weights = None
        #     if sample_weights is not None:
        #         # selected_sample_weights = torch.index_select(sample_weights, 0, selected)
        #         selected_sample_weights = sample_weights[selected.cpu()]
        #         if is_cuda:
        #             selected_sample_weights = selected_sample_weights.cuda()
        #     # selected = torch.index_select(X, 0, selected)
        #     if not all_layer:
        #         selected = X[selected]
        #         if is_cuda:
        #             selected = selected.cuda()

        #         if sample_weights is None:
        #             initial_state0[index] = selected.mean(dim=0)
        #         else:
        #             initial_state0[index] = torch.sum(selected*selected_sample_weights.view(-1,1), dim = 0)/torch.sum(selected_sample_weights)
        #     else:
        #         selected = select_samples_by_ls(X, selected, is_cuda)
            
        #         update_centroid_by_ls(selected, initial_state0, selected_sample_weights, index)
            

        # if not all_layer:
        #     center_shift0 = torch.sum(
        #         torch.sqrt(
        #             torch.sum((initial_state0 - initial_state_pre) ** 2, dim=1)
        #         ))

        # else:
        #     center_shift0 = compute_center_shift_by_ls(initial_state0, initial_state_pre)
        center_shift = 0
        for k in range(len(initial_state_sample_ids_pre)):
            curr_sample_weights = None
            curr_sample_weights_pre = None
            if sample_weights is not None:
                curr_sample_weights = sample_weights[initial_state_sample_ids[k]]
                curr_sample_weights_pre = sample_weights[initial_state_sample_ids_pre[k]]

            curr_x_cosin_sim = full_x_cosin_sim[initial_state_sample_ids_pre[k]][:,initial_state_sample_ids[k]]
            if curr_sample_weights_pre is not None:
                curr_full_dis = torch.sum(curr_x_cosin_sim.view(curr_sample_weights_pre.shape[0],-1)*curr_sample_weights_pre.view(curr_sample_weights_pre.shape[0],1), dim = 0)
                curr_full_dis = curr_full_dis/torch.sum(curr_sample_weights_pre)
            else:
                curr_full_dis = torch.mean(curr_x_cosin_sim, dim = 0) #pairwise_cosine_full_by_sample_ids(curr_x_cosin_sim, [torch.tensor(list(range(curr_x_cosin_sim.shape[0])))], is_cuda=is_cuda, sample_weights = curr_sample_weights_pre)

            if curr_sample_weights is None:
                curr_full_dis = torch.mean(curr_full_dis)
            else:
                curr_full_dis = torch.sum(curr_full_dis.view(-1)*curr_sample_weights.view(-1))/torch.sum(curr_sample_weights)
            center_shift += torch.abs(curr_full_dis)
        
        center_shift = center_shift/len(initial_state_sample_ids_pre)

        # center_shift = sum([(len(set(initial_state_sample_ids_pre[k].tolist()).difference(set(initial_state_sample_ids[k].tolist()))) + len(set(initial_state_sample_ids[k].tolist()).difference(set(initial_state_sample_ids_pre[k].tolist()))))/(len(initial_state_sample_ids_pre[k]) + len(initial_state_sample_ids[k])) for k in range(num_clusters)])

        # increment iteration
        iteration = iteration + 1

        # update tqdm meter
        tqdm_meter.set_postfix(
            iteration=f'{iteration}',
            center_shift=f'{center_shift:0.6f}',
            tol=f'{tol:0.6f}'
        )
        tqdm_meter.update()
        if center_shift < tol:
            break

    return choice_cluster.cpu(), full_x_cosin_sim
    # if not all_layer:
    #     return choice_cluster.cpu(), initial_state0.cpu()
    # else:
    #     for k in range(len(initial_state0)):
    #         initial_state0[k] = initial_state0[k].cpu()
    #     return choice_cluster.cpu(), initial_state0


def kmeans_predict(
        X,
        cluster_centers,
        distance='euclidean',
        device=torch.device('cpu')
):
    """
    predict using cluster centers
    :param X: (torch.tensor) matrix
    :param cluster_centers: (torch.tensor) cluster centers
    :param distance: (str) distance [options: 'euclidean', 'cosine'] [default: 'euclidean']
    :param device: (torch.device) device [default: 'cpu']
    :return: (torch.tensor) cluster ids
    """
    print(f'predicting on {device}..')

    if distance == 'euclidean':
        pairwise_distance_function = pairwise_distance
    elif distance == 'cosine':
        pairwise_distance_function = pairwise_cosine
    else:
        raise NotImplementedError

    # convert to float
    X = X.float()

    # transfer to device
    X = X.to(device)

    dis = pairwise_distance_function(X, cluster_centers)
    choice_cluster = torch.argmin(dis, dim=1)

    return choice_cluster.cpu()


def pairwise_distance_ls_full(data1_ls, data2_ls, is_cuda = False, batch_size = 128):
    
    curr_dist_ls = []

    for start_id in range(0, data2_ls[0].shape[0], batch_size):
        end_id = start_id + batch_size
        if end_id > data2_ls[0].shape[0]:
            end_id = data2_ls[0].shape[0]
        curr_B_ls = []

        for idx in range(len(data2_ls)):
            data2 = data2_ls[idx][start_id:end_id]
            # if is_cuda:
            #     data2 = data2.cuda()
            curr_B_ls.append(data2)

        curr_dist = pairwise_distance_ls(data1_ls, curr_B_ls, is_cuda=is_cuda,  batch_size = batch_size)    

        curr_dist_ls.append(curr_dist)

    return torch.cat(curr_dist_ls, dim = 1)

def pairwise_distance_ls(data1_ls, data2_ls, is_cuda=False,  batch_size = 128, agg = 'mean'):
    # transfer to device
    # data1, data2 = data1.to(device), data2.to(device)
    B_ls = []
    for idx in range(len(data2_ls)):
        data2 = data2_ls[idx].unsqueeze(dim=0)
        if is_cuda:
            data2 = data2.cuda()
        B_ls.append(data2)

    A_ls = []
    for idx in range(len(data1_ls)):
        data1 = data1_ls[idx].unsqueeze(dim=1)
        A_ls.append(data1)

    # if is_cuda:
    #     data2 = data2.cuda()

    # N*1*M
    # A = data1.unsqueeze(dim=1)

    # # 1*N*M
    # B = data2.unsqueeze(dim=0)

    full_dist_ls = []

    for start_id in range(0, A_ls[0].shape[0], batch_size):
        end_id = start_id + batch_size
        if end_id > A_ls[0].shape[0]:
            end_id = A_ls[0].shape[0]
        
        cosine_dis_ls = []
        for idx in range(len(A_ls)):
            curr_A = A_ls[idx][start_id: end_id]
            if is_cuda:
                curr_A = curr_A.cuda()
            # curr_A_normalized = curr_A / curr_A.norm(dim=-1, keepdim=True)
            B = B_ls[idx]
            # B_normalized = B / B.norm(dim=-1, keepdim=True)
            # curr_cosine = curr_A_normalized * B_normalized    
            # curr_dist = torch.sqrt(((curr_A - B) **2.0).sum(dim=-1).squeeze())
            curr_dist = ((curr_A - B) **2.0).sum(dim=-1)
            # curr_cosine_dis = torch.abs(curr_cosine.sum(dim=-1)).squeeze()
            cosine_dis_ls.append(curr_dist)


            del curr_A, B
        # final_cosine_dis = torch.min(torch.stack(cosine_dis_ls, dim = 1), dim = 1)[0]
        if agg == 'mean':
            final_cosine_dis = torch.mean(torch.stack(cosine_dis_ls, dim = 1), dim = 1)
        else:
            if agg == 'min':
                final_cosine_dis = torch.max(torch.stack(cosine_dis_ls, dim = 1), dim = 1)
            else:
                raise NotImplementedError
        for item in cosine_dis_ls:
            del item

        del cosine_dis_ls
        # final_cosine_dis = 1 - max_cosine_sim
        full_dist_ls.append(final_cosine_dis.cpu())

        del final_cosine_dis

    full_dist_tensor = torch.cat(full_dist_ls)

    for idx in range(len(B_ls)):
        data2 = B_ls[idx]
        del data2

    if is_cuda:
        torch.cuda.empty_cache()
        # if is_cuda:
        #     data2 = data2.cuda()
        # B_ls.append(data2)

    return full_dist_tensor

def pairwise_distance(data1, data2, is_cuda=False, batch_size = 128,
        weight_by_norm=False):
    # transfer to device
    # data1, data2 = data1.to(device), data2.to(device)
    if is_cuda:
        data2 = data2.cuda()
    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    full_dist_ls = []

    for start_id in range(0, A.shape[0], batch_size):
        end_id = start_id + batch_size
        if end_id > A.shape[0]:
            end_id = A.shape[0]

        curr_A = A[start_id: end_id]
        if is_cuda:
            curr_A = curr_A.cuda()

        curr_dist = torch.sqrt(((curr_A - B) **2.0).sum(dim=-1))
        full_dist_ls.append(curr_dist)
        del curr_A

    dis = torch.cat(full_dist_ls)

    # dis = (A - B) ** 2.0
    # # return N*N matrix for pairwise distance
    # dis = dis.sum(dim=-1).squeeze()
    return dis


def pairwise_cosine(data1, data2, is_cuda=False,  batch_size = 128, weight_by_norm = False):
    # transfer to device
    # data1, data2 = data1.to(device), data2.to(device)
    if is_cuda:
        data2 = data2.cuda()

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    full_dist_ls = []

    for start_id in range(0, A.shape[0], batch_size):
        end_id = start_id + batch_size
        if end_id > A.shape[0]:
            end_id = A.shape[0]

        curr_A = A[start_id: end_id]
        if is_cuda:
            curr_A = curr_A.cuda()
        curr_A_normalized = curr_A / curr_A.norm(dim=-1, keepdim=True)
        if not weight_by_norm:
            B_normalized = B / B.norm(dim=-1, keepdim=True)
            curr_cosine = curr_A_normalized * B_normalized    
        else:
            curr_cosine = curr_A_normalized * B
        curr_cosine_dis = 1 - torch.abs(curr_cosine.sum(dim=-1)).squeeze(1)
        full_dist_ls.append(curr_cosine_dis)

    full_dist_tensor = torch.cat(full_dist_ls)

    return full_dist_tensor


def pairwise_cosine_no_abs(data1, data2, is_cuda=False,  batch_size = 128, weight_by_norm = False):
    # transfer to device
    # data1, data2 = data1.to(device), data2.to(device)
    if is_cuda:
        data2 = data2.cuda()

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    full_dist_ls = []

    for start_id in range(0, A.shape[0], batch_size):
        end_id = start_id + batch_size
        if end_id > A.shape[0]:
            end_id = A.shape[0]

        curr_A = A[start_id: end_id]
        if is_cuda:
            curr_A = curr_A.cuda()
        curr_A_normalized = curr_A / curr_A.norm(dim=-1, keepdim=True)
        if not weight_by_norm:
            B_normalized = B / B.norm(dim=-1, keepdim=True)
            curr_cosine = curr_A_normalized * B_normalized    
        else:
            curr_cosine = curr_A_normalized * B
        curr_cosine_dis =  - curr_cosine.sum(dim=-1).squeeze(1)
        full_dist_ls.append(curr_cosine_dis)

    full_dist_tensor = torch.cat(full_dist_ls)

    return full_dist_tensor




def pairwise_cross_prod(data1, data2, is_cuda=False,  batch_size = 128):
    # transfer to device
    # data1, data2 = data1.to(device), data2.to(device)
    if is_cuda:
        data2 = data2.cuda()

    # N*1*M
    A = data1#.unsqueeze(dim=1)

    # 1*N*M
    B = data2#.unsqueeze(dim=0)

    full_dist_ls = []

    for start_id in range(0, A.shape[0], batch_size):
        end_id = start_id + batch_size
        if end_id > A.shape[0]:
            end_id = A.shape[0]

        curr_A = A[start_id: end_id]
        if is_cuda:
            curr_A = curr_A.cuda()
        curr_A_normalized = torch.abs(curr_A / curr_A.norm(dim=-1, keepdim=True))
        B_normalized = torch.abs(B / B.norm(dim=-1, keepdim=True))
        curr_A_normalized = curr_A_normalized.unsqueeze(-1).unsqueeze(1)
        B_normalized = B_normalized.unsqueeze(-2).unsqueeze(0)
        curr_cosine = torch.sum(curr_A_normalized*B_normalized, dim = (-1,-2))/(A.shape[1]*A.shape[1])


        # curr_cosine = torch.sum(torch.mm(torch.abs(curr_A_normalized).view(-1,1), torch.abs(B_normalized).view(1,-1)))
        curr_cosine_dis = 1 - curr_cosine#torch.abs(curr_cosine.sum(dim=-1)).squeeze(1)
        full_dist_ls.append(curr_cosine_dis)

    full_dist_tensor = torch.cat(full_dist_ls)

    return full_dist_tensor


def pairwise_cosine2(data1, data2, is_cuda=False,  batch_size = 128):
    # transfer to device
    # data1, data2 = data1.to(device), data2.to(device)
    if is_cuda:
        data2 = data2.cuda()

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    full_dist_ls = []

    for start_id in range(0, A.shape[0], batch_size):
        end_id = start_id + batch_size
        if end_id > A.shape[0]:
            end_id = A.shape[0]

        curr_A = A[start_id: end_id]
        if is_cuda:
            curr_A = curr_A.cuda()
        curr_A_normalized = curr_A #/ curr_A.norm(dim=-1, keepdim=True)
        B_normalized = B# / B.norm(dim=-1, keepdim=True)
        curr_cosine = curr_A_normalized * B_normalized    
        # curr_cosine_dis = 1/(1 + torch.abs(curr_cosine.sum(dim=-1))).squeeze(1)
        curr_cosine_dis = (1-torch.abs(curr_cosine.sum(dim=-1))).squeeze(1)
        full_dist_ls.append(curr_cosine_dis)

    full_dist_tensor = torch.cat(full_dist_ls)

    return full_dist_tensor




def pairwise_cosine_full(data1, is_cuda=False,  batch_size = 2048, data2 = None):
    # transfer to device
    # data1, data2 = data1.to(device), data2.to(device)
    # if is_cuda:
    #     data2 = data2.cuda()

    # N*1*M
    A = data1#.unsqueeze(dim=1)

    # 1*N*M
    if data2 is None:
        B = data1.clone()#.unsqueeze(dim=0)
    else:
        B = data2

    full_dist_ls = []

    full_cosin_mat = torch.zeros([data1.shape[0], data2.shape[0]])

    for start_id in range(0, A.shape[0], batch_size):
        end_id = start_id + batch_size
        print("calculated sample ids::", start_id)
        if end_id > A.shape[0]:
            end_id = A.shape[0]

        curr_A = A[start_id: end_id]
        if is_cuda:
            curr_A = curr_A.cuda()
        curr_A_normalized = curr_A# / curr_A.norm(dim=-1, keepdim=True)

        for start_id2 in range(0, B.shape[0], batch_size):
            end_id2 = start_id2 + batch_size
            if end_id2 > B.shape[0]:
                end_id2 = B.shape[0]
            
            curr_B = B[start_id2: end_id2]
            if is_cuda:
                curr_B = curr_B.cuda()


            B_normalized = curr_B #/ curr_B.norm(dim=-1, keepdim=True)
            curr_cosine_dis = torch.mm(curr_A_normalized, torch.t(B_normalized))# curr_A_normalized * B_normalized    
            # curr_cosine_dis = (curr_cosine.sum(dim=-1)).squeeze(1)

            full_cosin_mat[start_id:end_id, start_id2:end_id2] = curr_cosine_dis.cpu()
            del B_normalized, curr_B, curr_cosine_dis

        del curr_A_normalized, curr_A

    return full_cosin_mat/data1.shape[0]


def pairwise_l2_full(data1, is_cuda=False,  batch_size = 2048):
    # transfer to device
    # data1, data2 = data1.to(device), data2.to(device)
    # if is_cuda:
    #     data2 = data2.cuda()

    # N*1*M
    A = data1#.unsqueeze(dim=1)

    # 1*N*M
    B = data1.clone()#.unsqueeze(dim=0)

    full_dist_ls = []

    full_cosin_mat = torch.zeros([data1.shape[0], data1.shape[0]])

    for start_id in range(0, A.shape[0], batch_size):
        end_id = start_id + batch_size
        print("calculated sample ids::", start_id)
        if end_id > A.shape[0]:
            end_id = A.shape[0]

        curr_A = A[start_id: end_id]
        if is_cuda:
            curr_A = curr_A.cuda()
        curr_A_normalized = curr_A.unsqueeze(dim=0)# / curr_A.norm(dim=-1, keepdim=True)

        for start_id2 in range(0, B.shape[0], batch_size):
            end_id2 = start_id2 + batch_size
            if end_id2 > B.shape[0]:
                end_id2 = B.shape[0]
            
            curr_B = B[start_id2: end_id2]
            if is_cuda:
                curr_B = curr_B.cuda()


            B_normalized = curr_B.unsqueeze(dim=1) #/ curr_B.norm(dim=-1, keepdim=True)
            # curr_cosine_dis = torch.mm(curr_A_normalized, torch.t(B_normalized))# curr_A_normalized * B_normalized    
            curr_dist = torch.sqrt(((curr_A_normalized - B_normalized) **2.0).sum(dim=-1))
            # curr_cosine_dis = (curr_cosine.sum(dim=-1)).squeeze(1)

            full_cosin_mat[start_id:end_id, start_id2:end_id2] = curr_dist.cpu()
            del B_normalized, curr_B, curr_dist

        del curr_A_normalized, curr_A

    return full_cosin_mat


def pairwise_cosine_full_by_sample_ids(full_cosin_sim, sample_ids_ls, is_cuda=False, sample_weights = None):
    # transfer to device
    # data1, data2 = data1.to(device), data2.to(device)
    # if is_cuda:
    #     data2 = data2.cuda()

    # N*num_clusters
    full_dist_mat = torch.zeros([full_cosin_sim.shape[0], len(sample_ids_ls)])
    if is_cuda:
        full_dist_mat = full_dist_mat.cuda()
        # full_cosin_sim = full_cosin_sim.cuda()
    for idx in range(len(sample_ids_ls)):
        curr_sample_ids = sample_ids_ls[idx]
        if len(curr_sample_ids) <= 0:
            continue

        if is_cuda:
            curr_sample_ids = curr_sample_ids.cuda()
        curr_sample_weights = None
        if sample_weights is not None:
            curr_sample_weights = sample_weights[curr_sample_ids]

        if curr_sample_weights is not None:
            curr_cluster_to_full_data_cosin_sim = torch.sum(full_cosin_sim[curr_sample_ids].view(len(curr_sample_ids), -1)*curr_sample_weights.view(len(curr_sample_ids),1),dim=0)/torch.sum(curr_sample_weights)
        else:
            curr_cluster_to_full_data_cosin_sim = torch.mean(full_cosin_sim[curr_sample_ids].view(len(curr_sample_ids), -1), dim = 0)

        if is_cuda:
            curr_cluster_to_full_data_cosin_sim = curr_cluster_to_full_data_cosin_sim.cuda()

        full_dist_mat[:,idx] = 1-torch.abs(curr_cluster_to_full_data_cosin_sim)


    return full_dist_mat
    #     full_dist_ls.append(curr_cosine_dis)

    # full_dist_tensor = torch.cat(full_dist_ls)

    # return full_dist_tensor



def pairwise_cosine_ls(data1_ls, data2_ls, is_cuda=False,  batch_size = 32, agg = 'mean', ls_idx_range=-1, weight_by_norm=False, inner_prod = False, no_abs = False, full_inner_prod=False, flatten = False):

    if ls_idx_range < 0:
        ls_idx_range = len(data1_ls)
    # transfer to device
    # data1, data2 = data1.to(device), data2.to(device)
    B_ls = []
    vec_norm_ls2 = []
    for idx in range(ls_idx_range):
        data2 = data2_ls[idx].unsqueeze(dim=0)
        if is_cuda:
            data2 = data2.cuda()
        B_ls.append(data2)
        vec_norm_ls2.append(data2.norm(dim=-1)**2)

    A_ls = []
    for idx in range(ls_idx_range):
        data1 = data1_ls[idx].unsqueeze(dim=1)
        A_ls.append(data1)

    # A_norm_ls = torch.sum(torch.cat([(data1_ls[k].view(data1_ls[k].shape[0], -1).norm(dim=-1)).view(-1)**2 for k in range(len(data1_ls))]))
    A_norm_ls = torch.sum(torch.stack([torch.norm(data1_ls[k],dim=1).view(-1)**2 for k in range(len(data1_ls))], dim =1),dim=1)

    if inner_prod:
        A_max = 0
        A_min = np.inf

    # if is_cuda:
    #     data2 = data2.cuda()

    # N*1*M
    # A = data1.unsqueeze(dim=1)

    # # 1*N*M
    # B = data2.unsqueeze(dim=0)

    full_dist_ls = []

    for start_id in range(0, A_ls[0].shape[0], batch_size):
        end_id = start_id + batch_size
        if end_id > A_ls[0].shape[0]:
            end_id = A_ls[0].shape[0]
        
        inner_prod_ls = []
        vec_norm_ls1 = A_norm_ls[start_id:end_id]
        
        # cosine_dis_ls = []
        for idx in range(ls_idx_range):
            curr_A = A_ls[idx][start_id: end_id]
            if is_cuda:
                curr_A = curr_A.cuda()
                vec_norm_ls1 = vec_norm_ls1.cuda()

            B = B_ls[idx]
            curr_inner_prod = (curr_A*B).sum(dim=-1)
            inner_prod_ls.append(curr_inner_prod)
            # vec_norm_ls1.append(curr_A.norm(dim=-1)**2)
            


            # curr_A_normalized = curr_A / curr_A.norm(dim=-1, keepdim=True)
            
            # B_normalized = B / B.norm(dim=-1, keepdim=True)
            # curr_cosine = curr_A_normalized * B_normalized    
            # curr_cosine_dis = torch.abs(curr_cosine.sum(dim=-1))
            # cosine_dis_ls.append(curr_cosine_dis)
        if flatten:
            total_cosin_ls = torch.stack(inner_prod_ls, dim = 1)
        else:
            total_cosin_ls = torch.sum(torch.stack(inner_prod_ls, dim = 1), dim=1)

        if inner_prod:
            if no_abs:
                max_cosine_sim = total_cosin_ls
            else:
                max_cosine_sim = torch.abs(total_cosin_ls)
            A_max = max(A_max, torch.max(max_cosine_sim).detach().item())
            A_min = min(A_min, torch.min(max_cosine_sim).detach().item())
        else:

            if not weight_by_norm:
                # total_norm_ls = torch.sqrt(torch.sum(torch.stack(vec_norm_ls1, dim = 0), dim =0).view(-1).unsqueeze(1)*torch.sum(torch.stack(vec_norm_ls2, dim = 0), dim =0).view(-1).unsqueeze(0))
                total_norm_ls = torch.sqrt(vec_norm_ls1.view(-1).unsqueeze(1)*torch.sum(torch.stack(vec_norm_ls2, dim = 0), dim =0).view(-1).unsqueeze(0))
                if full_inner_prod:
                    max_cosine_sim = torch.abs(total_cosin_ls)
                else:
                    max_cosine_sim = torch.abs(total_cosin_ls)/total_norm_ls
            else:
                # total_norm_ls = torch.sqrt(torch.ones_like(vec_norm_ls1.view(-1)).unsqueeze(1)*torch.sum(torch.stack(vec_norm_ls2, dim = 0), dim =0).view(-1).unsqueeze(0))
                total_norm_ls = torch.sqrt(vec_norm_ls1.view(-1).unsqueeze(1)*torch.ones_like(torch.sum(torch.stack(vec_norm_ls2, dim = 0), dim =0).view(-1)).unsqueeze(0))

                if full_inner_prod:
                    max_cosine_sim = torch.abs(total_cosin_ls)
                else:
                    max_cosine_sim = torch.abs(total_cosin_ls)/total_norm_ls
        # if agg == 'mean':
        # max_cosine_sim = torch.mean(torch.stack(cosine_dis_ls, dim = 1), dim = 1)
        # else:
        #     if agg == 'min':
        #         max_cosine_sim = torch.min(torch.stack(cosine_dis_ls, dim = 1), dim = 1)[0]
        #     else:
        #         raise NotImplementedError
        if full_inner_prod:
            final_cosine_dis = max_cosine_sim
        else:
            if not inner_prod:
                final_cosine_dis = 1 - max_cosine_sim
            else:
                final_cosine_dis = max_cosine_sim
        full_dist_ls.append(final_cosine_dis)

    full_dist_tensor = torch.cat(full_dist_ls)
    # if inner_prod:
    #     full_dist_tensor = (A_max + full_dist_tensor)/(A_max-A_min+0.0001)

    return full_dist_tensor


def pairwise_cosine_ls_no_abs(data1_ls, data2_ls, is_cuda=False,  batch_size = 128, agg = 'mean', weight_by_norm=False):
    # transfer to device
    # data1, data2 = data1.to(device), data2.to(device)
    B_ls = []
    for idx in range(len(data2_ls)):
        data2 = data2_ls[idx].unsqueeze(dim=0)
        if is_cuda:
            data2 = data2.cuda()
        B_ls.append(data2)

    A_ls = []
    for idx in range(len(data1_ls)):
        data1 = data1_ls[idx].unsqueeze(dim=1)
        A_ls.append(data1)

    # if is_cuda:
    #     data2 = data2.cuda()

    # N*1*M
    # A = data1.unsqueeze(dim=1)

    # # 1*N*M
    # B = data2.unsqueeze(dim=0)

    full_dist_ls = []

    for start_id in range(0, A_ls[0].shape[0], batch_size):
        end_id = start_id + batch_size
        if end_id > A_ls[0].shape[0]:
            end_id = A_ls[0].shape[0]
        
        inner_prod_ls = []
        vec_norm_ls1 = []
        vec_norm_ls2 = []
        # cosine_dis_ls = []
        for idx in range(len(A_ls)):
            curr_A = A_ls[idx][start_id: end_id]
            if is_cuda:
                curr_A = curr_A.cuda()

            B = B_ls[idx]
            curr_inner_prod = (curr_A*B).sum(dim=-1)
            inner_prod_ls.append(curr_inner_prod)
            vec_norm_ls1.append(curr_A.norm(dim=-1)**2)
            vec_norm_ls2.append(B.norm(dim=-1)**2)


            # curr_A_normalized = curr_A / curr_A.norm(dim=-1, keepdim=True)
            
            # B_normalized = B / B.norm(dim=-1, keepdim=True)
            # curr_cosine = curr_A_normalized * B_normalized    
            # curr_cosine_dis = torch.abs(curr_cosine.sum(dim=-1))
            # cosine_dis_ls.append(curr_cosine_dis)
        total_cosin_ls = torch.sum(torch.stack(inner_prod_ls, dim = 1), dim=1)

        if not weight_by_norm:
            total_norm_ls = torch.sqrt(torch.sum(torch.stack(vec_norm_ls1, dim = 0), dim =0).view(-1).unsqueeze(1)*torch.sum(torch.stack(vec_norm_ls2, dim = 0), dim =0).view(-1).unsqueeze(0))
            max_cosine_sim = total_cosin_ls/total_norm_ls
        else:
            total_norm_ls = torch.sqrt(torch.sum(torch.stack(vec_norm_ls1, dim = 0), dim =0).view(-1).unsqueeze(1)*torch.ones_like(torch.sum(torch.stack(vec_norm_ls2, dim = 0), dim =0).view(-1)).unsqueeze(0))
            max_cosine_sim = total_cosin_ls/total_norm_ls
        # if agg == 'mean':
        # max_cosine_sim = torch.mean(torch.stack(cosine_dis_ls, dim = 1), dim = 1)
        # else:
        #     if agg == 'min':
        #         max_cosine_sim = torch.min(torch.stack(cosine_dis_ls, dim = 1), dim = 1)[0]
        #     else:
        #         raise NotImplementedError
        final_cosine_dis = - max_cosine_sim
        full_dist_ls.append(final_cosine_dis)

    full_dist_tensor = torch.cat(full_dist_ls)

    return full_dist_tensor




def pairwise_cross_prod_ls(data1_ls, data2_ls, is_cuda=False,  batch_size = 128):
    # transfer to device
    # data1, data2 = data1.to(device), data2.to(device)
    B_ls = []
    for idx in range(len(data2_ls)):
        data2 = data2_ls[idx]
        if is_cuda:
            data2 = data2.cuda()
        B_ls.append(data2)

    A_ls = []
    for idx in range(len(data1_ls)):
        data1 = data1_ls[idx]
        A_ls.append(data1)

    # if is_cuda:
    #     data2 = data2.cuda()

    # N*1*M
    # A = data1.unsqueeze(dim=1)

    # # 1*N*M
    # B = data2.unsqueeze(dim=0)

    full_dist_ls = []

    for start_id in range(0, A_ls[0].shape[0], batch_size):
        end_id = start_id + batch_size
        if end_id > A_ls[0].shape[0]:
            end_id = A_ls[0].shape[0]
        
        cosine_dis_ls = []
        for idx in range(len(A_ls)):
            curr_A = A_ls[idx][start_id: end_id]
            if is_cuda:
                curr_A = curr_A.cuda()
            curr_A_normalized = torch.abs(curr_A / curr_A.norm(dim=-1, keepdim=True))
            B = B_ls[idx]
            B_normalized = torch.abs(B / B.norm(dim=-1, keepdim=True))

            # curr_sim = torch.sum(torch.bmm(curr_A_normalized.view(curr_A_normalized.shape[0], -1,1), B_normalized.view(B_normalized.shape[0], 1,-1)), dim = (-2,-1)).view(-1)

            curr_A_normalized = curr_A_normalized.unsqueeze(-1).unsqueeze(1)
            B_normalized = B_normalized.unsqueeze(-2).unsqueeze(0)
            curr_cosine = torch.sum(curr_A_normalized*B_normalized, dim = (-1,-2))

            # curr_sim = torch.sum(torch.mm(torch.abs(curr_A_normalized).view(-1,1), torch.abs(B_normalized).view(1,-1)))

            # curr_cosine = curr_A_normalized * B_normalized    
            # curr_cosine_dis = torch.abs(curr_cosine.sum(dim=-1))
            cosine_dis_ls.append(curr_cosine)
        max_cosine_sim = torch.mean(torch.stack(cosine_dis_ls, dim = 1), dim = 1)
        final_cosine_dis = 1 - max_cosine_sim
        full_dist_ls.append(final_cosine_dis)

    full_dist_tensor = torch.cat(full_dist_ls)

    return full_dist_tensor

    # normalize the points  | [0.3, 0.4] -> [0.3/sqrt(0.09 + 0.16), 0.4/sqrt(0.09 + 0.16)] = [0.3/0.5, 0.4/0.5]
    # A_normalized = A / A.norm(dim=-1, keepdim=True)
    # B_normalized = B / B.norm(dim=-1, keepdim=True)

    # cosine = A_normalized * B_normalized

    # # return N*N matrix for pairwise distance
    # cosine_dis = 1 - cosine.sum(dim=-1).squeeze()
    # return cosine_dis

