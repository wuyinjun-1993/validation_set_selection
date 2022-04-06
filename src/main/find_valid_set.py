import torch
import numpy as np
# from kmeans_pytorch import kmeans


import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from datasets.mnist import *
from common.utils import *
from main.helper_func import *
from clustering_method.k_means import *


def cluster_per_class(sample_representation_vec_ls, sample_id_ls, valid_count_per_class = 10, num_clusters = 4, sample_weights = None, existing_cluster_centroids = None, cosin_distance = False):
    if not cosin_distance:
        cluster_ids_x, cluster_centers = kmeans(
            X=sample_representation_vec_ls, num_clusters=num_clusters, distance='euclidean', device=sample_representation_vec_ls.device, sample_weights=sample_weights, existing_cluster_mean_ls=existing_cluster_centroids)
    else:
        cluster_ids_x, cluster_centers = kmeans(
            X=sample_representation_vec_ls, num_clusters=num_clusters, distance='cosine', device=sample_representation_vec_ls.device, sample_weights=sample_weights, existing_cluster_mean_ls=existing_cluster_centroids)

    representive_id_ls = []
    representive_representation_ls = []
    cluster_centers = cluster_centers.to(sample_representation_vec_ls.device)
    for cluster_id in range(len(cluster_centers)):
        curr_cluster_center = cluster_centers[cluster_id]

        cluster_dist_ls = []

        min_cluster_distance = 0
        min_sample_id = -1

        # for idx in range(num_clusters):
        cluster_dist_ls_tensor = torch.norm(sample_representation_vec_ls[cluster_ids_x == cluster_id].view(torch.sum(cluster_ids_x == cluster_id), -1) - cluster_centers[cluster_id].view(1,-1), dim = 1)


        # for idx in range(len(cluster_ids_x[cluster_id])):
        #     curr_idx = cluster_ids_x[cluster_id][idx]
        #     curr_dist = torch.norm(sample_representation_vec_ls[curr_idx] - curr_cluster_center)
        #     cluster_dist_ls.append(curr_dist.item())

        # cluster_dist_ls_tensor = torch.tensor(cluster_dist_ls)

        sorted_dist_tensor, sorted_sample_idx_tensor = torch.sort(cluster_dist_ls_tensor, descending=False)

        selected_count = int(valid_count_per_class/num_clusters)

        representive_id_ls.append(sample_id_ls[cluster_ids_x == cluster_id][sorted_sample_idx_tensor[0:selected_count]])
    
        representive_representation_ls.append(sample_representation_vec_ls[cluster_ids_x == cluster_id][sorted_sample_idx_tensor[0:selected_count]])

    return torch.cat(representive_id_ls), torch.cat(representive_representation_ls)


def obtain_most_under_represent_samples(under_represent_count, full_sample_representations, full_sample_ids, full_valid_sample_representation_ls):
    full_distance = pairwise_distance(full_sample_representations, full_valid_sample_representation_ls, device=torch.device('cpu'))

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
            
            sample_representation = net.feature_forward(data)

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
            valid_ids, valid_sample_representation = cluster_per_class(sample_representation_vec_ls, sample_id_ls, valid_count_per_class = int(main_represent_count/len(sample_representation_vec_ls_by_class)), num_clusters = int(main_represent_count/len(sample_representation_vec_ls_by_class)), sample_weights=curr_cached_sample_weights, existing_cluster_centroids=existing_valid_representation[existing_valid_set.targets == label], cosin_distance=args.cosin_dist)    

        else:
            valid_ids, valid_sample_representation = cluster_per_class(sample_representation_vec_ls, sample_id_ls, valid_count_per_class = int(main_represent_count/len(sample_representation_vec_ls_by_class)), num_clusters = int(main_represent_count/len(sample_representation_vec_ls_by_class)), sample_weights=curr_cached_sample_weights, cosin_distance=args.cosin_dist)

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


def get_representative_valid_ids2(train_loader, args, net, valid_count, cached_sample_weights = None, existing_valid_representation = None, existing_valid_set = None):

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
            # for idx in range(len(labels)):
            #     curr_label = labels[idx].item()
            #     sample_id = sample_ids[idx]
            #     if curr_label not in sample_representation_vec_ls_by_class:
            #         sample_representation_vec_ls_by_class[curr_label] = []
            #         sample_id_ls_by_class[curr_label] = []
            #     sample_representation_vec_ls_by_class[curr_label].append(sample_representation[idx])
            #     sample_id_ls_by_class[curr_label].append(sample_id)

    # valid_ids_ls = []
    # valid_sample_representation_ls = []
    # full_sample_representation_ls = []
    # full_sample_id_ls = []

    # sample_representation_vec_ls = sample_representation_vec_ls_by_class[label]

    all_sample_ids = torch.cat(sample_id_ls)
    valid_ids, valid_sample_representation_tensor = cluster_per_class(torch.cat(sample_representation_vec_ls), all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=cached_sample_weights[all_sample_ids], cosin_distance=args.cosin_dist)  


    # for label in sample_representation_vec_ls_by_class:
    #     sample_representation_vec_ls_by_class[label] = torch.stack(sample_representation_vec_ls_by_class[label])
    
        

    #     sample_id_ls = torch.tensor(sample_id_ls_by_class[label])

    #     curr_cached_sample_weights = None
    #     if cached_sample_weights is not None:
    #         curr_cached_sample_weights = cached_sample_weights[sample_id_ls]

    #     if existing_valid_representation is not None and existing_valid_set is not None:
    #         valid_ids, valid_sample_representation = cluster_per_class(sample_representation_vec_ls, sample_id_ls, valid_count_per_class = int(main_represent_count/len(sample_representation_vec_ls_by_class)), num_clusters = int(main_represent_count/len(sample_representation_vec_ls_by_class)), sample_weights=curr_cached_sample_weights, existing_cluster_centroids=existing_valid_representation[existing_valid_set.targets == label])    

    #     else:
    #         valid_ids, valid_sample_representation = cluster_per_class(sample_representation_vec_ls, sample_id_ls, valid_count_per_class = int(main_represent_count/len(sample_representation_vec_ls_by_class)), num_clusters = int(main_represent_count/len(sample_representation_vec_ls_by_class)), sample_weights=curr_cached_sample_weights)

    #     valid_ids_ls.append(valid_ids)
    #     valid_sample_representation_ls.append(valid_sample_representation)
    #     full_sample_representation_ls.append(sample_representation_vec_ls)
    #     full_sample_id_ls.append(sample_id_ls.view(-1))
    
    # valid_ids = torch.cat(valid_ids_ls)
    # valid_sample_representation_tensor = torch.cat(valid_sample_representation_ls)
    # under_represent_count = valid_count - len(valid_ids)
    # if under_represent_count > 0 and args.add_under_rep_samples:
    #     # under_represent_valid_ids = obtain_most_under_represent_samples(under_represent_count, torch.cat(full_sample_representation_ls), torch.cat(full_sample_id_ls), torch.cat(valid_sample_representation_ls))
    #     # under_represent_valid_ids = random_obtain_other_samples(under_represent_count, torch.cat(full_sample_id_ls), valid_ids)
    #     under_represent_valid_ids = get_boundary_valid_ids0(train_loader, net, args, under_represent_count, valid_ids)
    #     valid_ids = torch.cat([valid_ids.view(-1), under_represent_valid_ids.view(-1)])

    return valid_ids, valid_sample_representation_tensor

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
            X=sample_representation_vec_tensor, num_clusters=num_clusters, distance='euclidean', device=sample_representation_vec_ls.device, sample_weights=sample_weights)
    else:
        cluster_ids_x, cluster_centers = kmeans(
            X=sample_representation_vec_tensor, num_clusters=num_clusters, distance='cosine', device=sample_representation_vec_ls.device, sample_weights=sample_weights)


    if not args.cosin_dist:
        existing_valid_training_dists = pairwise_distance(sample_representation_vec_tensor,existing_valid_representation, device = sample_representation_vec_tensor.device)
    else:
        existing_valid_training_dists = pairwise_cosine(sample_representation_vec_tensor, existing_valid_representation, device = sample_representation_vec_tensor.device)

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


