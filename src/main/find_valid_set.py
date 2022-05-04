import torch
import numpy as np
# from kmeans_pytorch import kmeans


import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from datasets.mnist import *
from common.utils import *
from main.helper_func import *
from clustering_method.k_means import *
from sklearn import metrics
from main.model_gradient_op import *
from clustering_method.k_means_with_grad_vector import *
import collections


def test_s_scores(sample_representation_vec_ls, cluster_ids_x, cluster_centers, num_clusters, distance = 'euclidean', is_cuda=False):

    all_sample_ids = []
    for k in range(num_clusters):
        curr_sample_ids = torch.nonzero(cluster_ids_x == k)

        # rand_curr_sample_id_ids = torch.randperm(curr_sample_ids.shape[0])[0:10]
        rand_curr_sample_id_ids = torch.tensor(list(range(10)))

        all_sample_ids.append(curr_sample_ids[rand_curr_sample_id_ids])

    all_sample_ids_tensor = torch.cat(all_sample_ids).view(-1)

    selected_samples = sample_representation_vec_ls[all_sample_ids_tensor].view(all_sample_ids_tensor.shape[0],-1)
    selected_sample_cluster_ids = cluster_ids_x[all_sample_ids_tensor].view(-1)

    s_score1 = metrics.silhouette_score(selected_samples.cpu().numpy(), selected_sample_cluster_ids.cpu().numpy(), metric=distance)

    s_score2 = calculate_silhouette_scores(selected_samples, selected_sample_cluster_ids, cluster_centers, is_cuda=is_cuda, sample_weights = None, distance = distance)


    print()


def find_best_cluster_num(sample_representation_vec_ls, sample_weights, distance = 'euclidean', is_cuda=False, all_layer = False):

    s_score_ls = []
    for k in range(5, 100, 5):
        cluster_ids_x, cluster_centers = kmeans(
            X=sample_representation_vec_ls, num_clusters=k, distance=distance, is_cuda=is_cuda, sample_weights=sample_weights, existing_cluster_mean_ls=None, all_layer = all_layer)
        s_score2 = calculate_silhouette_scores(sample_representation_vec_ls, cluster_ids_x, cluster_centers, is_cuda=is_cuda, distance = distance, sample_weights = sample_weights)
        logging.info("s score for cluste count %d: %f" %(k, s_score2))
        s_score_ls.append(s_score2)

    print(s_score_ls)

def cluster_per_class(
    args,
    sample_representation_vec_ls,
    sample_id_ls,
    full_sim_mat=None,
    valid_count_per_class=10,
    num_clusters=4,
    sample_weights=None,
    existing_cluster_centroids=None,
    cosin_distance=False,
    is_cuda=False,
    all_layer=False,
    return_cluster_info=False,
):
    
    if num_clusters > 0:
        if not cosin_distance:
            cluster_ids_x, cluster_centers = kmeans(
                X=sample_representation_vec_ls,
                num_clusters=num_clusters,
                distance='euclidean',
                is_cuda=is_cuda,
                sample_weights=sample_weights,
                existing_cluster_mean_ls=existing_cluster_centroids,
                all_layer=all_layer,
            )
        else:
            # if not args.all_layer_grad:
            cluster_ids_x, cluster_centers = kmeans(
                X=sample_representation_vec_ls,
                num_clusters=num_clusters,
                distance='cosine',
                is_cuda=is_cuda,
                sample_weights=sample_weights,
                existing_cluster_mean_ls=existing_cluster_centroids,
                all_layer=all_layer,
            )
            # else:
            #     cluster_assignment_file_name = os.path.join(args.save_path, "cluster_assignments")

            #     sim_mat_file_name = os.path.join(args.save_path, "full_similarity_mat")
            #     if not (os.path.exists(cluster_assignment_file_name) and os.path.exists(sim_mat_file_name)):
            #         cluster_ids_x, full_x_cosin_sim = kmeans_cosin(
            #             X=sample_representation_vec_ls, num_clusters=num_clusters, distance='cosine', is_cuda=is_cuda, sample_weights=sample_weights, existing_cluster_mean_ls=existing_cluster_centroids, all_layer=all_layer, full_x_cosin_sim=full_sim_mat)

            #         torch.save(cluster_ids_x, cluster_assignment_file_name)
            #         torch.save(full_x_cosin_sim, sim_mat_file_name)
            #     else:
            #         cluster_ids_x = torch.load(cluster_assignment_file_name)
            #         full_x_cosin_sim = torch.load(sim_mat_file_name)
            # distance = 'cosine'

        if return_cluster_info:
            return cluster_ids_x, cluster_centers

    else:
        if not cosin_distance:
            find_best_cluster_num(sample_representation_vec_ls, sample_weights, distance = 'euclidean', all_layer = all_layer)
        else:
            find_best_cluster_num(sample_representation_vec_ls, sample_weights, distance = 'cosine', all_layer = all_layer)


    

    representive_id_ls = []
    representive_representation_ls = []

    if not cosin_distance:
        if not all_layer:
            pairwise_distance_function = pairwise_distance
        else:
            pairwise_distance_function = pairwise_distance_ls
    elif cosin_distance:
        if not all_layer:
            pairwise_distance_function = pairwise_cosine
        else:
            pairwise_distance_function = pairwise_cosine_ls
    else:
        raise NotImplementedError

    if not args.all_layer_grad:
        if is_cuda:
            if not all_layer:
                cluster_centers = cluster_centers.cuda()
            else:
                for idx in range(len(cluster_centers)):
                    cluster_centers[idx] = cluster_centers[idx].cuda()

    full_representative_representation_ls = None
    for cluster_id in range(num_clusters):
        # curr_cluster_center = cluster_centers[cluster_id]

        cluster_dist_ls = []

        min_cluster_distance = 0
        min_sample_id = -1
        if torch.sum(cluster_ids_x == cluster_id).item() <= 0:
            continue

        # if args.all_layer_grad:
        #     curr_sample_ids = torch.nonzero(cluster_ids_x == cluster_id).view(-1)
        #     curr_sample_weights = None
        #     if sample_weights is not None:
        #         curr_sample_weights = sample_weights[curr_sample_ids]

        #     curr_x_cosin_sim = full_x_cosin_sim[curr_sample_ids][:,curr_sample_ids]

        #     curr_sample_cluster_to_sample_dist = pairwise_cosine_full_by_sample_ids(curr_x_cosin_sim, [torch.tensor(list(range(curr_sample_ids.shape[0])))], is_cuda=is_cuda, sample_weights = curr_sample_weights)

        #     local_sample_ids = curr_sample_ids[torch.argmin(curr_sample_cluster_to_sample_dist)]
        #     curr_selected_sample_id = sample_id_ls[local_sample_ids:local_sample_ids +1]
        #     representive_id_ls.append(curr_selected_sample_id)
        #     representive_representation_ls.append(sample_representation_vec_ls[curr_selected_sample_id])
        # else:
            # for idx in range(num_clusters):

        if not all_layer:        
            curr_cluster_sample_representation = sample_representation_vec_ls[cluster_ids_x == cluster_id].view(torch.sum(cluster_ids_x == cluster_id), -1)

            if is_cuda:
                curr_cluster_sample_representation = curr_cluster_sample_representation.cuda()
            
            cluster_dist_ls_tensor = pairwise_distance_function(curr_cluster_sample_representation, cluster_centers[cluster_id].view(1,-1), is_cuda = is_cuda).view(-1)
            
            sorted_dist_tensor, sorted_sample_idx_tensor = torch.sort(cluster_dist_ls_tensor, descending=False)

            selected_count = int(valid_count_per_class/num_clusters)

            representive_id_ls.append(sample_id_ls[cluster_ids_x == cluster_id][sorted_sample_idx_tensor[0:selected_count]].cpu())
        
            representive_representation_ls.append(sample_representation_vec_ls[cluster_ids_x == cluster_id][sorted_sample_idx_tensor[0:selected_count]].cpu())
    
        else:
            curr_cluster_sample_representation_ls = []
            curr_cluster_center_ls = []
            for arr_idx in range(len(sample_representation_vec_ls)):
                curr_cluster_sample_representation = sample_representation_vec_ls[arr_idx][cluster_ids_x == cluster_id].view(torch.sum(cluster_ids_x == cluster_id), -1)
                curr_cluster_center = cluster_centers[arr_idx][cluster_id].view(1,-1)
                curr_cluster_center_ls.append(curr_cluster_center)
                if is_cuda:
                    curr_cluster_sample_representation = curr_cluster_sample_representation.cuda()
                curr_cluster_sample_representation_ls.append(curr_cluster_sample_representation)
            cluster_dist_ls_tensor = pairwise_distance_function(curr_cluster_sample_representation_ls, curr_cluster_center_ls, is_cuda = is_cuda).view(-1)

            sorted_dist_tensor, sorted_sample_idx_tensor = torch.sort(cluster_dist_ls_tensor, descending=False)

            selected_count = int(valid_count_per_class/num_clusters)

            representive_id_ls.append(sample_id_ls[cluster_ids_x == cluster_id][sorted_sample_idx_tensor[0:selected_count]].cpu())

            curr_representive_represetation = []

            for arr_idx in range(len(sample_representation_vec_ls)):

                curr_representive_represetation.append(sample_representation_vec_ls[arr_idx][cluster_ids_x == cluster_id][sorted_sample_idx_tensor[0:selected_count]].cpu())

            representive_representation_ls.append(curr_representive_represetation)

    if not all_layer:
        return torch.cat(representive_id_ls), torch.cat(representive_representation_ls)
    else:
        # res_representive_represetation_ls = []
        res_representative_representation_ls = []

        for idx1 in range(len(representive_representation_ls[0])):
            for idx2 in range(len(representive_representation_ls)):
                if idx2 == 0:
                    res_representative_representation_ls.append(representive_representation_ls[idx2][idx1])
                else:
                    res_representative_representation_ls[idx1] = torch.cat([res_representative_representation_ls[idx1], representive_representation_ls[idx2][idx1]])




        return torch.cat(representive_id_ls), res_representative_representation_ls

def cluster_per_class_on_grad_vec(args, net, train_loader, criterion, sample_representation_vec_ls, sample_id_ls, valid_count_per_class = 10, num_clusters = 4, sample_weights = None, existing_cluster_centroids = None, cosin_distance = False, is_cuda = False, all_layer = False, return_cluster_info = False):
    
    if num_clusters > 0:
        if not cosin_distance:
            cluster_ids_x, cluster_centers = kmeans_with_grad_vec(args, net, train_loader,  criterion, sample_representation_vec_ls, 
                num_clusters=num_clusters, distance='euclidean', is_cuda=is_cuda, sample_weights=sample_weights, all_layer=all_layer)


            # distance = 'euclidean'
            # test_s_scores(sample_representation_vec_ls, cluster_ids_x, cluster_centers, num_clusters, distance = 'euclidean')
        else:
            # if not args.all_layer_grad:
            cluster_ids_x, cluster_centers = kmeans_with_grad_vec(args, net, train_loader,  criterion, sample_representation_vec_ls, 
                num_clusters=num_clusters, distance='cosine', is_cuda=is_cuda, sample_weights=sample_weights, all_layer=all_layer)
            # cluster_ids_x, cluster_centers = kmeans_with_grad_vec(
            #     X=sample_representation_vec_ls, num_clusters=num_clusters, distance='cosine', is_cuda=is_cuda, sample_weights=sample_weights, existing_cluster_mean_ls=existing_cluster_centroids, all_layer=all_layer)
            # else:
            #     cluster_assignment_file_name = os.path.join(args.save_path, "cluster_assignments")

            #     sim_mat_file_name = os.path.join(args.save_path, "full_similarity_mat")
            #     if not (os.path.exists(cluster_assignment_file_name) and os.path.exists(sim_mat_file_name)):
            #         cluster_ids_x, full_x_cosin_sim = kmeans_cosin(
            #             X=sample_representation_vec_ls, num_clusters=num_clusters, distance='cosine', is_cuda=is_cuda, sample_weights=sample_weights, existing_cluster_mean_ls=existing_cluster_centroids, all_layer=all_layer, full_x_cosin_sim=full_sim_mat)

            #         torch.save(cluster_ids_x, cluster_assignment_file_name)
            #         torch.save(full_x_cosin_sim, sim_mat_file_name)
            #     else:
            #         cluster_ids_x = torch.load(cluster_assignment_file_name)
            #         full_x_cosin_sim = torch.load(sim_mat_file_name)
            # distance = 'cosine'

        if return_cluster_info:
            return cluster_ids_x, cluster_centers

    else:
        if not cosin_distance:
            find_best_cluster_num(sample_representation_vec_ls, sample_weights, distance = 'euclidean', all_layer = all_layer)
        else:
            find_best_cluster_num(sample_representation_vec_ls, sample_weights, distance = 'cosine', all_layer = all_layer)


    

    representive_id_ls = []
    representive_representation_ls = []

    if not cosin_distance:
        if not all_layer:
            pairwise_distance_function = pairwise_cosine_full_for_grad_vec
        else:
            pairwise_distance_function = pairwise_cosine_full_for_grad_vec
    elif cosin_distance:
        if not all_layer:
            pairwise_distance_function = pairwise_cosine_full_for_grad_vec
        else:
            pairwise_distance_function = pairwise_cosine_full_for_grad_vec
    else:
        raise NotImplementedError


    # if not args.all_layer_grad:
    #     if is_cuda:
    #         if not all_layer:
    #             cluster_centers = cluster_centers.cuda()
    #         else:
    #             for idx in range(len(cluster_centers)):
    #                 cluster_centers[idx] = cluster_centers[idx].cuda()

    full_representative_representation_ls = None
    for cluster_id in range(num_clusters):
        # curr_cluster_center = cluster_centers[cluster_id]

        cluster_dist_ls = []

        min_cluster_distance = 0
        min_sample_id = -1
        if torch.sum(cluster_ids_x == cluster_id).item() <= 0:
            continue

        # if args.all_layer_grad:
        #     curr_sample_ids = torch.nonzero(cluster_ids_x == cluster_id).view(-1)
        #     curr_sample_weights = None
        #     if sample_weights is not None:
        #         curr_sample_weights = sample_weights[curr_sample_ids]

        #     curr_x_cosin_sim = full_x_cosin_sim[curr_sample_ids][:,curr_sample_ids]

        #     curr_sample_cluster_to_sample_dist = pairwise_cosine_full_by_sample_ids(curr_x_cosin_sim, [torch.tensor(list(range(curr_sample_ids.shape[0])))], is_cuda=is_cuda, sample_weights = curr_sample_weights)

        #     local_sample_ids = curr_sample_ids[torch.argmin(curr_sample_cluster_to_sample_dist)]
        #     curr_selected_sample_id = sample_id_ls[local_sample_ids:local_sample_ids +1]
        #     representive_id_ls.append(curr_selected_sample_id)
        #     representive_representation_ls.append(sample_representation_vec_ls[curr_selected_sample_id])
        # else:
            # for idx in range(num_clusters):

        if not all_layer:        
            # curr_cluster_sample_representation = sample_representation_vec_ls[cluster_ids_x == cluster_id].view(torch.sum(cluster_ids_x == cluster_id), -1)
            curr_cluster_sample_representation = [sample_representation_vec_ls[sample_idx] for sample_idx in range(len(sample_representation_vec_ls)) if cluster_ids_x[sample_idx] == cluster_id]

            if is_cuda:
                curr_cluster_sample_representation = curr_cluster_sample_representation.cuda()
            
            cluster_dist_ls_tensor = pairwise_distance_function(curr_cluster_sample_representation, cluster_centers[cluster_id].view(1,-1), is_cuda = is_cuda).view(-1)
            
            sorted_dist_tensor, sorted_sample_idx_tensor = torch.sort(cluster_dist_ls_tensor, descending=False)

            selected_count = int(valid_count_per_class/num_clusters)

            representive_id_ls.append(sample_id_ls[cluster_ids_x == cluster_id][sorted_sample_idx_tensor[0:selected_count]].cpu())
        
            representive_representation_ls.append(sample_representation_vec_ls[cluster_ids_x == cluster_id][sorted_sample_idx_tensor[0:selected_count]].cpu())
    
        else:
            curr_cluster_sample_representation_ls = []
            curr_cluster_center_ls = []
            for arr_idx in range(len(sample_representation_vec_ls)):
                curr_cluster_sample_representation = sample_representation_vec_ls[arr_idx][cluster_ids_x == cluster_id].view(torch.sum(cluster_ids_x == cluster_id), -1)
                curr_cluster_center = cluster_centers[arr_idx][cluster_id].view(1,-1)
                curr_cluster_center_ls.append(curr_cluster_center)
                if is_cuda:
                    curr_cluster_sample_representation = curr_cluster_sample_representation.cuda()
                curr_cluster_sample_representation_ls.append(curr_cluster_sample_representation)
            cluster_dist_ls_tensor = pairwise_distance_function(curr_cluster_sample_representation_ls, curr_cluster_center_ls, is_cuda = is_cuda).view(-1)

            sorted_dist_tensor, sorted_sample_idx_tensor = torch.sort(cluster_dist_ls_tensor, descending=False)

            selected_count = int(valid_count_per_class/num_clusters)

            representive_id_ls.append(sample_id_ls[cluster_ids_x == cluster_id][sorted_sample_idx_tensor[0:selected_count]].cpu())

            curr_representive_represetation = []

            for arr_idx in range(len(sample_representation_vec_ls)):

                curr_representive_represetation.append(sample_representation_vec_ls[arr_idx][cluster_ids_x == cluster_id][sorted_sample_idx_tensor[0:selected_count]].cpu())

            representive_representation_ls.append(curr_representive_represetation)

    if not all_layer:
        return torch.cat(representive_id_ls), torch.cat(representive_representation_ls)
    else:
        # res_representive_represetation_ls = []
        res_representative_representation_ls = []

        for idx1 in range(len(representive_representation_ls[0])):
            for idx2 in range(len(representive_representation_ls)):
                if idx2 == 0:
                    res_representative_representation_ls.append(representive_representation_ls[idx2][idx1])
                else:
                    res_representative_representation_ls[idx1] = torch.cat([res_representative_representation_ls[idx1], representive_representation_ls[idx2][idx1]])




        return torch.cat(representive_id_ls), res_representative_representation_ls


def obtain_most_under_represent_samples(under_represent_count, full_sample_representations, full_sample_ids, full_valid_sample_representation_ls, is_cuda=False):
    full_distance = pairwise_distance(full_sample_representations, full_valid_sample_representation_ls, is_cuda=is_cuda)

    min_distance_by_sample,_ = torch.min(full_distance, dim = 1)

    most_under_representive_sample_dist, most_under_representive_sample_ids = torch.sort(min_distance_by_sample, descending=True)

    
    most_under_representive_sample_ids = full_sample_ids[most_under_representive_sample_ids[0:under_represent_count]]

    return most_under_representive_sample_ids

def random_obtain_other_samples(under_represent_count, full_sample_ids, valid_ids):
    full_sample_id_tensor = torch.ones(len(full_sample_ids))
    full_sample_id_tensor[valid_ids] = 0

    remaining_sample_ids = full_sample_ids[full_sample_id_tensor.nonzero()]

    rand_remaining_sample_ids = torch.randperm(len(remaining_sample_ids))

    selected_sample_ids = remaining_sample_ids[rand_remaining_sample_ids[0:under_represent_count]]

    return selected_sample_ids

def sort_prob_gap_by_class0(prob_gap_ls, select_count, existing_valid_ids):
    # boolean_id_arrs = torch.logical_and((label_ls == class_id).view(-1), (pred_labels == label_ls).view(-1))
    # boolean_id_arrs = (label_ls == class_id).view(-1)

    # sample_id_with_curr_class = torch.nonzero(boolean_id_arrs).view(-1)

    prob_gap_ls_curr_class = prob_gap_ls#[boolean_id_arrs]

    sorted_probs, sorted_idx = torch.sort(prob_gap_ls_curr_class, dim = 0, descending = False)

    sorted_idx = sorted_idx[torch.all(sorted_idx.view(-1,1) != existing_valid_ids.view(1, -1), dim = 1).nonzero().view(-1)]

    # selected_sub_ids = (sorted_probs < 0.05).nonzero()


    # # # selected_sub_ids = (sorted_probs > 0.999).nonzero()
    # select_count = min(select_count, len(selected_sub_ids))

    selected_sample_indx = sorted_idx[0:select_count]

    selected_prob_gap_values = sorted_probs[0:select_count]

    return selected_sample_indx


def get_boundary_valid_ids0(train_loader, net, args, valid_count, existing_valid_ids):
    pred_labels = torch.zeros(len(train_loader.dataset), dtype =torch.long)

    pred_correct_count = 0

    prob_gap_ls = torch.zeros(len(train_loader.dataset))

    label_ls = torch.zeros(len(train_loader.dataset), dtype =torch.long)

    for batch_id, (sample_ids, data, labels) in enumerate(train_loader):

        if args.cuda:
            data = data.cuda()
            # labels = labels.cuda()

        out_probs = torch.exp(net(data))
        sorted_probs, sorted_indices = torch.sort(out_probs, dim = 1, descending = True)

        prob_gap = sorted_probs[:,0] - sorted_probs[:,1]

        prob_gap_ls[sample_ids] = prob_gap.detach().cpu()

        label_ls[sample_ids] = labels

        curr_pred_labels = sorted_indices[:,0].detach().cpu()

        pred_labels[sample_ids] = curr_pred_labels

        pred_correct_count += torch.sum(labels.view(-1) == curr_pred_labels.view(-1))

    pred_accuracy = pred_correct_count*1.0/len(train_loader.dataset)

    logging.info("training accuracy is %f"%(pred_accuracy.item()))

    unique_label_ls = label_ls.unique()

    selected_valid_ids_ls = []

    # for label_id in unique_label_ls:
    selected_valid_ids = sort_prob_gap_by_class0(prob_gap_ls, valid_count, existing_valid_ids)
    selected_valid_ids_ls.append(selected_valid_ids)

    valid_ids = torch.cat(selected_valid_ids_ls)
    return valid_ids


def get_representative_valid_ids(train_loader, args, net, valid_count, cached_sample_weights = None, existing_valid_representation = None, existing_valid_set = None):

    if args.add_under_rep_samples:
        under_represent_count = int(valid_count/2)
        main_represent_count = valid_count - under_represent_count
    else:
        under_represent_count = 0
        main_represent_count = valid_count


    sample_representation_vec_ls_by_class = dict()
    sample_id_ls_by_class = dict()


    with torch.no_grad():

        # all_sample_representations = [None]*len(train_loader.dataset)

        for batch_id, (sample_ids, data, labels) in enumerate(train_loader):

            if args.cuda:
                data = data.cuda()
                # labels = labels.cuda()
            if not args.all_layer:
                sample_representation = net.feature_forward(data)
            else:
                sample_representation = net.feature_forward(data, all_layer=args.all_layer)

            for idx in range(len(labels)):
                curr_label = labels[idx].item()
                sample_id = sample_ids[idx]
                if curr_label not in sample_representation_vec_ls_by_class:
                    sample_representation_vec_ls_by_class[curr_label] = []
                    sample_id_ls_by_class[curr_label] = []
                sample_representation_vec_ls_by_class[curr_label].append(sample_representation[idx])
                sample_id_ls_by_class[curr_label].append(sample_id)

    valid_ids_ls = []
    valid_sample_representation_ls = []
    full_sample_representation_ls = []
    full_sample_id_ls = []

    for label in sample_representation_vec_ls_by_class:
        sample_representation_vec_ls_by_class[label] = torch.stack(sample_representation_vec_ls_by_class[label])
    
        sample_representation_vec_ls = sample_representation_vec_ls_by_class[label]

        sample_id_ls = torch.tensor(sample_id_ls_by_class[label])

        curr_cached_sample_weights = None
        if cached_sample_weights is not None:
            curr_cached_sample_weights = cached_sample_weights[sample_id_ls]

        if existing_valid_representation is not None and existing_valid_set is not None:
            valid_ids, valid_sample_representation = cluster_per_class(args, sample_representation_vec_ls, sample_id_ls, valid_count_per_class = int(main_represent_count/len(sample_representation_vec_ls_by_class)), num_clusters = int(main_represent_count/len(sample_representation_vec_ls_by_class)), sample_weights=curr_cached_sample_weights, existing_cluster_centroids=existing_valid_representation[existing_valid_set.targets == label], cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=args.all_layer)    

        else:
            valid_ids, valid_sample_representation = cluster_per_class(args, sample_representation_vec_ls, sample_id_ls, valid_count_per_class = int(main_represent_count/len(sample_representation_vec_ls_by_class)), num_clusters = int(main_represent_count/len(sample_representation_vec_ls_by_class)), sample_weights=curr_cached_sample_weights, cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=args.all_layer)

        valid_ids_ls.append(valid_ids)
        valid_sample_representation_ls.append(valid_sample_representation)
        full_sample_representation_ls.append(sample_representation_vec_ls)
        full_sample_id_ls.append(sample_id_ls.view(-1))
    
    valid_ids = torch.cat(valid_ids_ls)
    valid_sample_representation_tensor = torch.cat(valid_sample_representation_ls)
    # under_represent_count = valid_count - len(valid_ids)
    # if under_represent_count > 0 and args.add_under_rep_samples:
    #     # under_represent_valid_ids = obtain_most_under_represent_samples(under_represent_count, torch.cat(full_sample_representation_ls), torch.cat(full_sample_id_ls), torch.cat(valid_sample_representation_ls))
    #     # under_represent_valid_ids = random_obtain_other_samples(under_represent_count, torch.cat(full_sample_id_ls), valid_ids)
    #     under_represent_valid_ids = get_boundary_valid_ids0(train_loader, net, args, under_represent_count, valid_ids)
    #     valid_ids = torch.cat([valid_ids.view(-1), under_represent_valid_ids.view(-1)])

    return valid_ids, valid_sample_representation_tensor


def concat_sample_representation_for_all_layer(sample_representation_vec_ls):
    sample_representation_vec_tensor_ls = []
    for idx in range(len(sample_representation_vec_ls[0])):
        sample_representation_vec_tensor_ls.append(sample_representation_vec_ls[0][idx].clone())

    for sample_idx in range(len(sample_representation_vec_ls)):
        for idx in range(len(sample_representation_vec_ls[0])):
            curr_sample_reprentation_vec_tensor = sample_representation_vec_tensor_ls[idx]
            sample_representation_vec_tensor_ls[idx] = torch.cat([curr_sample_reprentation_vec_tensor, sample_representation_vec_ls[sample_idx][idx]])
            # sample_representation_vec_tensor_ls.append(torch.cat([sample_representation_vec_ls[0][idx], ]))
    return sample_representation_vec_tensor_ls


def reduce_dimension_for_feature_representations(sample_representation_vec_ls, retained_dim=100):
    for idx in range(len(sample_representation_vec_ls)):
        sample_representation_vec = sample_representation_vec_ls[idx]
        s,v,d = torch.svd(sample_representation_vec)




# def obtain_net_grad(net):
#     gradient_ls = []
#     for param in net.parameters():
#         curr_gradient = param.grad.detach().cpu().clone().view(-1)
#         gradient_ls.append(curr_gradient)

#     return gradient_ls






def get_all_grad_by_example(args, train_loader, net, criterion, optimizer):
    full_sample_representation_tensor_ls = []
    full_sample_representation_tensor, all_sample_ids = get_grad_by_example(args, train_loader, net, criterion, optimizer)

    
    if args.use_model_prov:
        full_sample_representation_tensor_ls.append(full_sample_representation_tensor)
        get_extra_gradient_layer(args, train_loader, criterion, net, full_sample_representation_tensor_ls)
        return full_sample_representation_tensor_ls, all_sample_ids
    else:
        return full_sample_representation_tensor, all_sample_ids



def get_grad_by_example(args, train_loader, net, criterion, optimizer, vectorize_grad = False):
    vec_grad_by_example_ls = []
    sample_id_ls = []

    for batch_id, (sample_ids, data, labels) in tqdm(enumerate(train_loader)):
        args.logger.info("sample batch ids::%d"%(batch_id))
        if args.cuda:
            data, labels = train_loader.dataset.to_cuda(data, labels)
        output = net.forward(data)
        for idx in range(labels.shape[0]):
            optimizer.zero_grad()
            loss = criterion(output[idx:idx+1], labels[idx:idx+1])
            if not args.all_layer_grad_no_full_loss:
                loss = obtain_full_loss(output[idx:idx+1], labels[idx:idx+1], args.cuda, loss)
            loss.backward(retain_graph = True)
            if not vectorize_grad:
                vec_grad_by_example_ls.append(obtain_net_grad(net))
            else:
                vec_grad_by_example_ls.append(obtain_vectorized_grad(net).view(-1))
        sample_id_ls.append(sample_ids)
            # optimizer.step()
    if not vectorize_grad:
        return vec_grad_by_example_ls, torch.cat(sample_id_ls)
    else:
        return torch.stack(vec_grad_by_example_ls), torch.cat(sample_id_ls)




# def compute_ap


def compute_grad_prod_sample_with_cluster_centroids(args, net, criterion, optimizer, initial_state_ids, train_loader, baseloss_ls):
    eps = 1e-6
    for curr_state_ids in initial_state_ids:

        curr_data_ls = []
        curr_labels_ls = []

        for batch_id, (sample_ids, data, labels) in enumerate(train_loader):
            args.logger.info("sample batch ids::%d"%(batch_id))


            matched_sample_ids = torch.sum(sample_ids.view(-1,1) == curr_state_ids.view(1,-1), dim = 1).bool()

            if torch.sum(matched_sample_ids) <= 0:
                continue
            
            curr_data_ls.append(data[matched_sample_ids])
            curr_labels_ls.append(labels[matched_sample_ids])

        

            if args.cuda:
                data, labels = train_loader.dataset.to_cuda(data, labels)


            
            output = net.forward(data)
            for idx in range(labels.shape[0]):
                optimizer.zero_grad()
                loss = criterion(output[idx:idx+1], labels[idx:idx+1])
                loss.backward(retain_graph = True)

                grad_ls = obtain_net_grad(net)          
                perturb_net_by_grad(net, eps)

                loss_ls_with_update_net = obtain_loss_per_example(args, train_loader, net, criterion)

                full_sim_mat[sample_ids] = (loss_ls_with_update_net - baseloss_ls)/eps


def kmeans_cosin2(args, train_loader, net, criterion, optimizer, num_clusters, total_iter_count = 200, sample_weights = None, tol = 0.0001):

    initial_state_ids = initialize_sample_ids(train_loader.dataset, num_clusters)

    if args.cuda:
        initial_state_ids = [state_ids.cuda() for state_ids in initial_state_ids]

    iteration = 0
    tqdm_meter = tqdm(desc='[running kmeans]')
    # while True:
    for k in range(0,total_iter_count):

        dis = pairwise_distance_function(X, full_centroid_state,args.cuda)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = [state_ids.cuda() for state_ids in initial_state_ids]
        

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
                if args.cuda:
                    selected_sample_weights = selected_sample_weights.cuda()
            
            selected = X[selected]
            if args.cuda:
                selected = selected.cuda()

            if sample_weights is None:
                initial_state[index] = selected.mean(dim=0)
            else:
                initial_state[index] = torch.sum(selected*selected_sample_weights.view(-1,1), dim = 0)/torch.sum(selected_sample_weights)
        

        
        center_shift = torch.sum(
            torch.sqrt(
                torch.sum((initial_state - initial_state_pre) ** 2, dim=1)
            ))

        

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
    
    return choice_cluster.cpu(), initial_state.cpu()
    



def full_approx_grad_prod(args, train_loader, net, criterion, optimizer):

    full_sim_mat, sample_id_ls = get_approx_grad_prod(args, train_loader, net, criterion, optimizer)

    full_sim_mat_ls = []
    if args.use_model_prov:
        full_sim_mat_ls.append(full_sim_mat)
        full_sim_mat_ls = get_extra_approx_grad_prod(args, train_loader, criterion, net, optimizer, full_sim_mat_ls)    
        return torch.sum(torch.stack(torch.abs(full_sim_mat_ls)),dim=0), sample_id_ls

    else:
        return full_sim_mat, sample_id_ls


def full_approx_grad_prod2(args, train_loader, net, criterion, optimizer):

    full_sim_mat, sample_id_ls = get_approx_grad_prod(args, train_loader, net, criterion, optimizer)

    full_sim_mat_ls = []
    if args.use_model_prov:
        full_sim_mat_ls.append(full_sim_mat)
        full_sim_mat_ls = get_extra_approx_grad_prod(args, train_loader, criterion, net, optimizer, full_sim_mat_ls)    
        return torch.sum(torch.stack(torch.abs(full_sim_mat_ls)),dim=0), sample_id_ls

    else:
        return full_sim_mat, sample_id_ls


def get_extra_approx_grad_prod(args, train_loader, criterion, net, optimizer, full_sim_mat_ls):
    start_epoch_id = 0
    if args.use_pretrained_model:
        start_epoch_id = torch.load(os.path.join(args.prev_save_path, "early_stopping_epoch"))
    args.logger.info("extra representation starting from epoch %d"%(start_epoch_id))

        # start_epoch_id = int(args.epochs/2)
    for ep in range(start_epoch_id, args.epochs, args.model_prov_period):
        net = load_checkpoint_by_epoch(args, net, ep)
        optimizer, _=obtain_optimizer_scheduler(args, net, start_epoch = 0)

        full_sim_mat, sample_id_ls = get_approx_grad_prod(args, train_loader, net, criterion, optimizer)

        full_sim_mat_ls.append(full_sim_mat)

    return full_sim_mat_ls

def get_approx_grad_prod(args, train_loader, net, criterion, optimizer):
    sample_id_ls = []
    eps = 1e-6

    baseloss_ls = obtain_loss_per_example(args, train_loader, net, criterion)

    full_sim_mat = torch.zeros([len(train_loader.dataset), len(train_loader.dataset)])

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

            # grad_ls = obtain_net_grad(net)          
            perturb_net_by_grad(net, eps)

            loss_ls_with_update_net = obtain_loss_per_example(args, train_loader, net, criterion)

            full_sim_mat[sample_ids[idx]] = (loss_ls_with_update_net - baseloss_ls)/eps

            # vec_grad_by_example_ls.append(obtain_vectorized_grad(net))
        sample_id_ls.append(sample_ids)

    return full_sim_mat, sample_id_ls


def get_grad_prod(args, grad_ls):

    full_sim_mat = torch.zeros([len(grad_ls), len(grad_ls)])

    for start_id1 in tqdm(range(0, len(grad_ls), args.batch_size)):
        end_id1 = start_id1 + args.batch_size
        if end_id1 >= len(grad_ls):
            end_id1 = len(grad_ls)

        selected_grad_mat1 = grad_ls[start_id1: end_id1]
        if args.cuda:
            selected_grad_mat1 = selected_grad_mat1.cuda()

        for start_id2 in range(0, len(grad_ls), args.batch_size):
            end_id2 = start_id2 + args.batch_size
            if end_id2 >= len(grad_ls):
                end_id2 = len(grad_ls)

            selected_grad_mat2 = grad_ls[start_id2: end_id2]
            if args.cuda:
                selected_grad_mat2 = selected_grad_mat2.cuda()


            curr_sim = torch.mm(selected_grad_mat1, torch.t(selected_grad_mat2))
            full_sim_mat[start_id1:end_id1, start_id2:end_id2] = curr_sim

    return full_sim_mat
            

    # sample_id_ls = []
    # eps = 1e-6

    # # baseloss_ls = obtain_loss_per_example(args, train_loader, net, criterion)

    # full_sim_mat = torch.zeros([len(train_loader.dataset), len(train_loader.dataset)])

    # for batch_id, (sample_ids, data, labels) in enumerate(train_loader):
    #     args.logger.info("sample batch ids::%d"%(batch_id))
    #     if args.cuda:
    #         data, labels = train_loader.dataset.to_cuda(data, labels)
    #     output = net.forward(data)
    #     for idx in range(labels.shape[0]):
    #         optimizer.zero_grad()
    #         loss = criterion(output[idx:idx+1], labels[idx:idx+1])
    #         loss = obtain_full_loss(output[idx:idx+1], labels[idx:idx+1], args.cuda, loss)

    #         loss.backward(retain_graph = True)

    #         # grad_ls = obtain_net_grad(net)          
    #         perturb_net_by_grad(net, eps)

    #         loss_ls_with_update_net = obtain_loss_per_example(args, train_loader, net, criterion)

    #         full_sim_mat[sample_ids[idx]] = (loss_ls_with_update_net - baseloss_ls)/eps

    #         # vec_grad_by_example_ls.append(obtain_vectorized_grad(net))
    #     sample_id_ls.append(sample_ids)

    # return full_sim_mat, sample_id_ls

            # optimizer.step()
    # return torch.stack(vec_grad_by_example_ls), torch.cat(sample_id_ls)



def obtain_sample_representation_grad_last_layer(net, sample_representation, labels, criterion, optimizer, is_cuda = False):
    sample_representation_grad_ls = []
    for k in range(sample_representation.shape[0]):
        optimizer.zero_grad()
        sample_representation_grad = net.obtain_gradient_last_full_layer(sample_representation[k:k+1], labels[k:k+1], criterion, is_cuda = is_cuda).view(-1).cpu()
        sample_representation_grad_ls.append(sample_representation_grad)
    return torch.stack(sample_representation_grad_ls)

def calculate_train_meta_grad_prod(args, train_loader, meta_loader, net, criterion, optimizer):
    full_train_sample_representation_tensor, all_train_sample_ids = get_grad_by_example(args, train_loader, net, criterion, optimizer)

    full_meta_sample_representation_tensor, all_meta_sample_ids = get_grad_by_example(args, meta_loader, net, criterion, optimizer)


    # if args.cosin_dist:
    full_sim_mat1 = pairwise_cosine_full(full_train_sample_representation_tensor, is_cuda=args.cuda, data2 = full_meta_sample_representation_tensor)


    # else:
    #     # full_sim_mat1 = pairwise_l2_full(full_sample_representation_tensor, is_cuda=args.cuda)
    #     full_sim_mat1 = pairwise_distance_ls_full(full_train_sample_representation_tensor, full_meta_sample_representation_tensor, is_cuda=args.cuda,  batch_size = 256)


    return full_sim_mat1, full_train_sample_representation_tensor, full_meta_sample_representation_tensor





def obtain_representations_last_layer_given_model(args, train_loader, net, criterion, optimizer):
    sample_representation_vec_ls = []

    sample_id_ls = []
    # with torch.no_grad():

        # all_sample_representations = [None]*len(train_loader.dataset)


    for batch_id, (sample_ids, data, labels) in enumerate(train_loader):

        if args.cuda:
            data, labels = train_loader.dataset.to_cuda(data, labels)
            # labels = labels.cuda()
        
        sample_representation = net.feature_forward(data, all_layer=False)
        if args.all_layer:
            # sample_representation_grad = net.obtain_gradient_last_full_layer(sample_representation, labels, criterion)
            sample_representation_grad = obtain_sample_representation_grad_last_layer(net, sample_representation, labels, criterion, optimizer, is_cuda=args.cuda)


        if not args.all_layer:
            sample_representation_vec_ls.append(sample_representation.detach().cpu())
        else:
            if batch_id == 0:
                sample_representation_vec_ls.extend([sample_representation.detach().cpu(), sample_representation_grad.detach().cpu()])
            else:
                sample_representation_vec_ls[0] = torch.cat([sample_representation_vec_ls[0].detach().cpu(), sample_representation.detach().cpu()])
                sample_representation_vec_ls[1] = torch.cat([sample_representation_vec_ls[1].detach().cpu(), sample_representation_grad.detach().cpu()])
                # for arr_idx in range(len(sample_representation_vec_ls)):
                #     sample_representation_vec_ls[arr_idx] = torch.cat([sample_representation_vec_ls[arr_idx].detach().cpu(), sample_representation[arr_idx].detach().cpu()])

        sample_id_ls.append(sample_ids)
    if args.all_layer:
        return sample_representation_vec_ls, sample_id_ls
    else:
        return torch.cat(sample_representation_vec_ls), sample_id_ls


def get_extra_representations_last_layer(args, train_loader, criterion, net, full_sample_representation_vec_ls):
    
    start_epoch_id = 0
    if args.use_pretrained_model:
        start_epoch_id = torch.load(os.path.join(args.prev_save_path, "early_stopping_epoch"))
    args.logger.info("extra representation starting from epoch %d"%(start_epoch_id))

        # start_epoch_id = int(args.epochs/2)
    if not args.all_layer:
        full_sample_representation_vec_ls = [full_sample_representation_vec_ls]

    for ep in range(start_epoch_id, args.epochs, args.model_prov_period):
        net = load_checkpoint_by_epoch(args, net, ep)
        optimizer, _=obtain_optimizer_scheduler(args, net, start_epoch = 0)
        sample_representation_vec_ls, _ = obtain_representations_last_layer_given_model(args, train_loader, net, criterion, optimizer)

        if args.all_layer:
            full_sample_representation_vec_ls.extend(sample_representation_vec_ls)
        else:
            full_sample_representation_vec_ls.append(sample_representation_vec_ls)

    return full_sample_representation_vec_ls


def get_extra_gradient_layer(args, train_loader, criterion, net, full_sample_representation_vec_ls):
    
    start_epoch_id = 0
    if args.use_pretrained_model:
        start_epoch_id = torch.load(os.path.join(args.prev_save_path, "early_stopping_epoch"))
    args.logger.info("extra representation starting from epoch %d"%(start_epoch_id))

        # start_epoch_id = int(args.epochs/2)
    

    for ep in range(start_epoch_id, args.epochs, args.model_prov_period):
        net = load_checkpoint_by_epoch(args, net, ep)
        optimizer, _=obtain_optimizer_scheduler(args, net, start_epoch = 0)
        full_sample_representation_tensor, all_sample_ids = get_grad_by_example(args, train_loader, net, criterion, optimizer)


        # sample_representation_vec_ls, _ = obtain_representations_last_layer_given_model(args, train_loader, net, criterion, optimizer)

        # if args.all_layer:
        #     full_sample_representation_vec_ls.extend(sample_representation_vec_ls)
        # else:
        full_sample_representation_vec_ls.append(full_sample_representation_tensor)

    return full_sample_representation_vec_ls



def get_representations_last_layer(args, train_loader, criterion, optimizer, net):

    sample_representation_vec_ls, sample_id_ls = obtain_representations_last_layer_given_model(args, train_loader, net, criterion, optimizer)
    if args.use_model_prov:
        sample_representation_vec_ls = get_extra_representations_last_layer(args, train_loader, criterion, net, sample_representation_vec_ls)
    # sample_representation_vec_ls = []

    # sample_id_ls = []
    # # with torch.no_grad():

    #     # all_sample_representations = [None]*len(train_loader.dataset)


    # for batch_id, (sample_ids, data, labels) in enumerate(train_loader):

    #     if args.cuda:
    #         data, labels = train_loader.dataset.to_cuda(data, labels)
    #         # labels = labels.cuda()
        
    #     sample_representation = net.feature_forward(data, all_layer=False)
    #     if args.all_layer:
    #         # sample_representation_grad = net.obtain_gradient_last_full_layer(sample_representation, labels, criterion)
    #         sample_representation_grad = obtain_sample_representation_grad_last_layer(net, sample_representation, labels, criterion, optimizer, is_cuda=args.cuda)


    #     if not args.all_layer:
    #         sample_representation_vec_ls.append(sample_representation.detach().cpu())
    #     else:
    #         if batch_id == 0:
    #             sample_representation_vec_ls.extend([sample_representation.detach().cpu(), sample_representation_grad.detach().cpu()])
    #         else:
    #             sample_representation_vec_ls[0] = torch.cat([sample_representation_vec_ls[0].detach().cpu(), sample_representation.detach().cpu()])
    #             sample_representation_vec_ls[1] = torch.cat([sample_representation_vec_ls[1].detach().cpu(), sample_representation_grad.detach().cpu()])
    #             # for arr_idx in range(len(sample_representation_vec_ls)):
    #             #     sample_representation_vec_ls[arr_idx] = torch.cat([sample_representation_vec_ls[arr_idx].detach().cpu(), sample_representation[arr_idx].detach().cpu()])

    #     sample_id_ls.append(sample_ids)
            # for idx in range(len(labels)):
            #     curr_label = labels[idx].item()
            #     sample_id = sample_ids[idx]
            #     if curr_label not in sample_representation_vec_ls_by_class:
            #         sample_representation_vec_ls_by_class[curr_label] = []
            #         sample_id_ls_by_class[curr_label] = []
            #     sample_representation_vec_ls_by_class[curr_label].append(sample_representation[idx])
            #     sample_id_ls_by_class[curr_label].append(sample_id)

    # if args.all_layer:
    #     full_sample_representation_tensor = concat_sample_representation_for_all_layer(sample_representation_vec_ls)
    # else:
    # if not args.all_layer and not args.use_model_prov:
    #     full_sample_representation_tensor = torch.cat(sample_representation_vec_ls)
    # else:

        # if args.reduce_dimension_all_layer:
        #     reduce_dimension_for_feature_representations(sample_representation_vec_ls)

    full_sample_representation_tensor = sample_representation_vec_ls

    all_sample_ids = torch.cat(sample_id_ls)

    return full_sample_representation_tensor, all_sample_ids

def get_representative_valid_ids2_3(criterion, optimizer, train_loader, args, net, valid_count, cached_sample_weights = None, existing_valid_representation = None, existing_valid_set = None, return_cluster_info = False, only_sample_representation = False):

    if args.add_under_rep_samples:
        under_represent_count = int(valid_count/2)
        main_represent_count = valid_count - under_represent_count
    else:
        under_represent_count = 0
        main_represent_count = valid_count


    full_sample_representation_tensor, all_sample_ids = get_all_grad_by_example(args, train_loader, net, criterion, optimizer)


    if cached_sample_weights is not None:
            valid_ids, valid_sample_representation_tensor = cluster_per_class_on_grad_vec(args, net, train_loader, criterion,  full_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=cached_sample_weights[all_sample_ids], cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=args.all_layer | args.use_model_prov, return_cluster_info = return_cluster_info)  
    else:
        valid_ids, valid_sample_representation_tensor = cluster_per_class_on_grad_vec(args, net, train_loader, criterion,  full_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=None, cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=args.all_layer | args.use_model_prov, return_cluster_info = return_cluster_info)  

    if not return_cluster_info:
        return valid_ids, valid_sample_representation_tensor
    else:
        return valid_ids, valid_sample_representation_tensor, full_sample_representation_tensor




def get_representative_valid_ids2(criterion, optimizer, train_loader, args, net, valid_count, cached_sample_weights = None, existing_valid_representation = None, existing_valid_set = None, return_cluster_info = False, only_sample_representation = False):

    if args.add_under_rep_samples:
        under_represent_count = int(valid_count/2)
        main_represent_count = valid_count - under_represent_count
    else:
        under_represent_count = 0
        main_represent_count = valid_count


    # sample_representation_vec_ls_by_class = dict()
    # sample_id_ls_by_class = dict()
    full_sim_mat1 = None
    if not args.all_layer_grad:
        full_sample_representation_tensor, all_sample_ids = get_representations_last_layer(args, train_loader, criterion, optimizer, net)
    else:

        full_sample_representation_tensor, all_sample_ids = get_grad_by_example(args, train_loader, net, criterion, optimizer, vectorize_grad=True)

        
        # full_sim_mat1, sample_id_ls = full_approx_grad_prod(args, train_loader, net, criterion, optimizer)
        # sim_mat_file_name = os.path.join(args.save_path, "full_similarity_mat")
        # if not os.path.exists(sim_mat_file_name):
        #     full_sim_mat1 = pairwise_cosine_full(full_sample_representation_tensor, is_cuda=args.cuda)
        #     torch.save(full_sim_mat1, sim_mat_file_name)
        # else:
        #     full_sim_mat1 = torch.load(sim_mat_file_name)



        print()
    # valid_ids_ls = []
    # valid_sample_representation_ls = []
    # full_sample_representation_ls = []
    # full_sample_id_ls = []

    # sample_representation_vec_ls = sample_representation_vec_ls_by_class[label]

    if args.all_layer_grad_greedy:
        full_sim_mat1 = get_grad_prod(args, full_sample_representation_tensor)
        valid_ids = select_samples_with_greedy_algorithm(full_sim_mat1, main_represent_count)
        valid_sample_representation_tensor = None
        return valid_ids, valid_sample_representation_tensor
    else:
        if only_sample_representation:
            return full_sample_representation_tensor



        if args.cluster_no_reweighting:
            logging.info("no reweighting for k-means")
            cached_sample_weights = None

    
    if cached_sample_weights is not None:
        valid_ids, valid_sample_representation_tensor = cluster_per_class(
            args,
            full_sample_representation_tensor,
            all_sample_ids,
            valid_count_per_class=main_represent_count,
            num_clusters=main_represent_count,
            sample_weights=cached_sample_weights[all_sample_ids],
            cosin_distance=args.cosin_dist,
            is_cuda=args.cuda,
            all_layer=args.all_layer | args.use_model_prov,
            full_sim_mat=full_sim_mat1,
            return_cluster_info=return_cluster_info,
        )  
    else:
        valid_ids, valid_sample_representation_tensor = cluster_per_class(
            args,
            full_sample_representation_tensor,
            all_sample_ids,
            valid_count_per_class=main_represent_count,
            num_clusters=main_represent_count,
            sample_weights=None,
            cosin_distance=args.cosin_dist,
            is_cuda=args.cuda,
            all_layer=args.all_layer | args.use_model_prov,
            full_sim_mat=full_sim_mat1,
            return_cluster_info=return_cluster_info,
        )  

    if not return_cluster_info:
        return valid_ids, valid_sample_representation_tensor
    else:
        return valid_ids, valid_sample_representation_tensor, full_sample_representation_tensor

def get_representative_valid_ids3(train_loader, args, net, valid_count, cached_sample_weights = None, existing_valid_representation = None):

    if args.add_under_rep_samples:
        under_represent_count = int(valid_count/2)
        main_represent_count = valid_count - under_represent_count
    else:
        under_represent_count = 0
        main_represent_count = valid_count


    # sample_representation_vec_ls_by_class = dict()
    # sample_id_ls_by_class = dict()
    sample_representation_vec_ls = []

    sample_id_ls = []
    with torch.no_grad():

        # all_sample_representations = [None]*len(train_loader.dataset)


        for batch_id, (sample_ids, data, labels) in enumerate(train_loader):

            if args.cuda:
                data = data.cuda()
                # labels = labels.cuda()
            
            sample_representation = net.feature_forward(data)

            sample_representation_vec_ls.append(sample_representation)

            sample_id_ls.append(sample_ids)
    
    
    all_sample_ids = torch.cat(sample_id_ls)
    sample_weights=cached_sample_weights[all_sample_ids]
    sample_representation_vec_tensor = torch.cat(sample_representation_vec_ls)

    num_clusters = main_represent_count

    if not args.cosin_dist:
        cluster_ids_x, cluster_centers = kmeans(
            X=sample_representation_vec_tensor, num_clusters=num_clusters, distance='euclidean', is_cuda=args.cuda, sample_weights=sample_weights, all_layer=args.all_layer)
    else:
        cluster_ids_x, cluster_centers = kmeans(
            X=sample_representation_vec_tensor, num_clusters=num_clusters, distance='cosine', is_cuda=args.cuda, sample_weights=sample_weights, all_layer=args.all_layer)


    if not args.cosin_dist:
        existing_valid_training_dists = pairwise_distance(sample_representation_vec_tensor,existing_valid_representation, is_cuda=args.cuda)
    else:
        existing_valid_training_dists = pairwise_cosine(sample_representation_vec_tensor, existing_valid_representation, is_cuda=args.cuda)

    min_existing_valid_training_dists,_ = torch.min(existing_valid_training_dists, dim = 1)

    sorted_min_existing_valid_training_dists, sorted_min_existing_valid_training_ids  = torch.sort(min_existing_valid_training_dists, descending=True)

    
    covered_cluster_id_set = set()

    idx = 0

    selected_count = 0

    valid_idx_ls = []

    while selected_count < args.valid_count:
        curr_sample_idx = sorted_min_existing_valid_training_ids[idx].item()
        curr_cluster_idx = cluster_ids_x[curr_sample_idx].item()
        
        idx += 1
        if curr_cluster_idx in covered_cluster_id_set:
            continue

        covered_cluster_id_set.add(curr_cluster_idx)

        valid_idx_ls.append(curr_sample_idx)

        selected_count += 1

    valid_ids = torch.tensor(valid_idx_ls)


    valid_sample_representation_tensor = sample_representation_vec_tensor[valid_ids]
    
    # valid_ids, valid_sample_representation_tensor = cluster_per_class(, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, , cosin_distance=args.cosin_dist)  


    return valid_ids, valid_sample_representation_tensor



def find_representative_samples(net, train_loader, args, valid_ratio = 0.1):
    prob_gap_ls = torch.zeros(len(train_loader.dataset))

    label_ls = torch.zeros(len(train_loader.dataset), dtype =torch.long)

    valid_count = int(len(train_loader.dataset)*valid_ratio)

    pred_labels = torch.zeros(len(train_loader.dataset), dtype =torch.long)

    valid_ids,_ = get_representative_valid_ids(train_loader, args, net, valid_count)

    # valid_set = Subset(train_loader.dataset, valid_ids)
    valid_set = new_mnist_dataset2(train_loader.dataset.data[valid_ids].clone(), train_loader.dataset.targets[valid_ids].clone())

    meta_set = new_mnist_dataset2(train_loader.dataset.data[valid_ids].clone(), train_loader.dataset.targets[valid_ids].clone())




    origin_train_labels = train_loader.dataset.targets.clone()

    test(train_loader, net, args)

    # if args.flip_labels:

    #     logging.info("add errors to train set")

    #     train_dataset, _ = random_flip_labels_on_training(train_loader.dataset, ratio = args.err_label_ratio)

    flipped_labels = None
    if args.flip_labels:

        logging.info("add errors to train set")

        # train_dataset, _ = random_flip_labels_on_training(train_loader.dataset, ratio = args.err_label_ratio)
        flipped_labels = obtain_flipped_labels(train_loader.dataset, args)


    

    train_dataset, _, _ = partition_train_valid_dataset_by_ids(train_loader.dataset, origin_train_labels, flipped_labels, valid_ids)


    # train_dataset, _, _ = partition_train_valid_dataset_by_ids(train_dataset, origin_train_labels, valid_ids)

    train_loader, valid_loader, meta_loader, _ = create_data_loader(train_dataset, valid_set, meta_set, None, args)

    test(valid_loader, net, args)
    test(train_loader, net, args)

    return train_loader, valid_loader, meta_loader
    # origin_train_labels = train_loader.dataset.targets.clone()

    # if args.flip_labels:

    #     logging.info("add errors to train set")

    #     train_dataset, _ = random_flip_labels_on_training(train_loader.dataset, ratio = args.err_label_ratio)


    # valid_ids = torch.cat(valid_ids_ls)

    # train_dataset, valid_dataset, meta_dataset = partition_train_valid_dataset_by_ids(train_dataset, origin_train_labels, valid_ids)

    # train_loader, valid_loader, meta_loader, _ = create_data_loader(train_dataset, valid_dataset, meta_dataset, None, args)

    # return train_loader, valid_loader, meta_loader


