import torch
import numpy as np
from kmeans_pytorch import kmeans

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from datasets.mnist import *
from common.utils import *

def cluster_per_class(sample_representation_vec_ls, sample_id_ls, valid_count_per_class = 10, num_clusters = 4):
    cluster_ids_x, cluster_centers = kmeans(
        X=sample_representation_vec_ls, num_clusters=num_clusters, distance='euclidean', device=sample_representation_vec_ls.device)

    representive_id_ls = []
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
    
    return torch.cat(representive_id_ls)


def random_flip_labels_on_training(train_dataset, ratio = 0.5):
    full_ids = torch.tensor(list(range(len(train_dataset.targets))))

    full_rand_ids = torch.randperm(len(train_dataset.targets))

    err_label_count = int(len(full_rand_ids)*ratio)

    err_label_ids = full_rand_ids[0:err_label_count]

    correct_label_ids = full_rand_ids[err_label_count:]

    label_type_count = len(train_dataset.targets.unique())

    origin_labels = train_dataset.targets.clone()

    rand_err_labels = torch.randint(low=0,high=label_type_count-1, size=[len(err_label_ids)])

    origin_err_labels = origin_labels[err_label_ids]

    rand_err_labels[rand_err_labels == origin_err_labels] = (rand_err_labels[rand_err_labels == origin_err_labels] + 1)%label_type_count

    train_dataset.targets[err_label_ids] = rand_err_labels

    return train_dataset, origin_labels


def find_representative_samples(net, train_loader, args, valid_ratio = 0.1):
    prob_gap_ls = torch.zeros(len(train_loader.dataset))

    label_ls = torch.zeros(len(train_loader.dataset), dtype =torch.long)

    valid_count = int(len(train_loader.dataset)*valid_ratio)

    pred_labels = torch.zeros(len(train_loader.dataset), dtype =torch.long)

    sample_representation_vec_ls_by_class = dict()
    sample_id_ls_by_class = dict()

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

    for label in sample_representation_vec_ls_by_class:
        sample_representation_vec_ls_by_class[label] = torch.stack(sample_representation_vec_ls_by_class[label])
    
        sample_representation_vec_ls = sample_representation_vec_ls_by_class[label]

        sample_id_ls = torch.tensor(sample_id_ls_by_class[label])

        valid_ids = cluster_per_class(sample_representation_vec_ls, sample_id_ls, valid_count_per_class = int(valid_count/len(sample_representation_vec_ls_by_class)), num_clusters = 10)

        valid_ids_ls.append(valid_ids)

    
    origin_train_labels = train_loader.dataset.targets.clone()

    if args.flip_labels:

        logging.info("add errors to train set")

        train_dataset, _ = random_flip_labels_on_training(train_loader.dataset, ratio = args.err_label_ratio)


    valid_ids = torch.cat(valid_ids_ls)

    train_dataset, valid_dataset, meta_dataset = partition_train_valid_dataset_by_ids(train_dataset, origin_train_labels, valid_ids)

    train_loader, valid_loader, meta_loader, _ = create_data_loader(train_dataset, valid_dataset, meta_dataset, None, args)

    return train_loader, valid_loader, meta_loader


