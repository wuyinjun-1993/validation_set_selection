import torch
import numpy as np
# from kmeans_pytorch import kmeans


import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from exp_datasets.mnist import *
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
            X=sample_representation_vec_ls, num_clusters=k, distance=distance, is_cuda=is_cuda, sample_weights=sample_weights, existing_cluster_mean_ls=None, all_layer = all_layer, rand_init=args.rand_init)
        s_score2 = calculate_silhouette_scores(sample_representation_vec_ls, cluster_ids_x, cluster_centers, is_cuda=is_cuda, distance = distance, sample_weights = sample_weights)
        logging.info("s score for cluste count %d: %f" %(k, s_score2))
        s_score_ls.append(s_score2)

    print(s_score_ls)

def handle_outliers(args, sample_ids, sample_representation_vec_ls, valid_sample_representation_tensor, sample_weights=None, threshold=0.8):
    dis = compute_distance(args, args.cosin_dist, True, sample_representation_vec_ls, valid_sample_representation_tensor, args.cuda)
    outlier_ids = torch.nonzero(torch.min(dis, dim=1)[0] > threshold).view(-1)
    if len(outlier_ids) > 0:
        subset_outlier = [sample_representation_vec_ls[k][outlier_ids] for k in range(len(sample_representation_vec_ls))]
        subset_outlier_sample_ids = sample_ids[outlier_ids]
        sub_sample_weights = None
        if sample_weights is not None:
            sub_sample_weights = sample_weights[outlier_ids]

        if args.valid_count > valid_sample_representation_tensor[0].shape[0]:
            extra_cluster_count = args.valid_count - valid_sample_representation_tensor[0].shape[0]
        else:
            return None, None
        
        extra_valid_ids, extra_valid_sample_representation_tensor = cluster_per_class(args, subset_outlier, subset_outlier_sample_ids, valid_count_per_class = extra_cluster_count, num_clusters = extra_cluster_count, sample_weights=sub_sample_weights, cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, existing_cluster_centroids = None, handles_outlier=False)  
            # do_cluster(args, extra_cluster, subset_outlier, args.cuda, args.cosin_dist, sub_sample_weights, None, all_layer = True)



        return extra_valid_ids, extra_valid_sample_representation_tensor
    else:
        return None, None


def handle_outliers2(args, sample_ids, sample_representation_vec_ls, valid_sample_representation_tensor, sample_weights=None, threshold=0.8):
    if args.valid_count <= valid_sample_representation_tensor[0].shape[0]:
        return None, None
    
    dis = compute_distance(args, args.cosin_dist, True, sample_representation_vec_ls, valid_sample_representation_tensor, args.cuda)
    outlier_ids = torch.nonzero(torch.min(dis, dim=1)[0] > threshold).view(-1)

    full_extra_valid_ids = None

    full_extra_valid_sample_representation_tensor_ls = None
    
    while len(outlier_ids) > 0:
        subset_outlier = [sample_representation_vec_ls[k][outlier_ids] for k in range(len(sample_representation_vec_ls))]
        subset_outlier_sample_ids = sample_ids[outlier_ids]
        sub_sample_weights = None
        if sample_weights is not None:
            sub_sample_weights = sample_weights[outlier_ids]

        if full_extra_valid_sample_representation_tensor_ls is not None:
            if args.valid_count > full_extra_valid_sample_representation_tensor_ls[0].shape[0]:
                extra_cluster_count = args.valid_count - full_extra_valid_sample_representation_tensor_ls[0].shape[0]
        else:
            extra_cluster_count = args.valid_count - valid_sample_representation_tensor[0].shape[0]
        # else:
        #     return None, None
        
        extra_valid_ids, extra_valid_sample_representation_tensor = cluster_per_class(args, subset_outlier, subset_outlier_sample_ids, valid_count_per_class = extra_cluster_count, num_clusters = extra_cluster_count, sample_weights=sub_sample_weights, cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, existing_cluster_centroids = None, handles_outlier=False)  
            # do_cluster(args, extra_cluster, subset_outlier, args.cuda, args.cosin_dist, sub_sample_weights, None, all_layer = True)
        if full_extra_valid_ids is None:
            full_extra_valid_ids = extra_valid_ids
            full_extra_valid_sample_representation_tensor_ls = extra_valid_sample_representation_tensor
        else:
            full_extra_valid_ids = torch.cat([full_extra_valid_ids.view(-1), extra_valid_ids.view(-1)])
            full_extra_valid_sample_representation_tensor_ls = [torch.cat([full_extra_valid_sample_representation_tensor_ls[k], extra_valid_sample_representation_tensor[k]]) for k in range(len(extra_valid_sample_representation_tensor))]

        if len(full_extra_valid_ids) + valid_sample_representation_tensor[0].shape[0] >= args.valid_count:
            break

        full_extra_dis = compute_distance(args, args.cosin_dist, True, sample_representation_vec_ls, full_extra_valid_sample_representation_tensor_ls, args.cuda)
        full_dis = torch.cat([full_extra_dis, dis], dim=1)
        outlier_ids = torch.nonzero(torch.min(full_dis, dim=1)[0] > threshold).view(-1)



    return full_extra_valid_ids, full_extra_valid_sample_representation_tensor_ls
    # else:
    #     return None, None


def do_cluster(args, num_clusters, sample_representation_vec_ls, is_cuda, cosin_distance, sample_weights, existing_cluster_centroids = None, all_layer = False):
    if args.full_model_out:
        dist_metric = 'cross'
    elif cosin_distance:
        dist_metric = 'cosine'
    else:
        dist_metric = 'euclidean'

    cluster_ids_x, cluster_centers = kmeans(
        X=sample_representation_vec_ls,
        num_clusters=num_clusters,
        distance=dist_metric,
        is_cuda=is_cuda,
        sample_weights=sample_weights,
        existing_cluster_mean_ls=existing_cluster_centroids,
        all_layer=all_layer,
        agg_sim_array=args.all_layer_sim_agg,
        weight_by_norm=args.weight_by_norm,
        k_means_bz=args.k_means_bz,
        k_means_lr=args.k_means_lr,
        k_means_epochs=args.k_means_epochs,
        inner_prod=args.inner_prod,
        origin_X_ls_lenth = args.origin_X_ls_lenth,
        rand_init=args.rand_init
        # no_abs_cluster_sim = args.no_abs_cluster_sim
    )
    return cluster_ids_x, cluster_centers


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
    handles_outlier=False
):
    if num_clusters > 0:
        cluster_ids_x, cluster_centers = do_cluster(
            args,
            num_clusters,
            sample_representation_vec_ls,
            is_cuda,
            cosin_distance,
            sample_weights,
            existing_cluster_centroids=existing_cluster_centroids,
            all_layer=all_layer,
        )

        unique_cluster_count = len(cluster_ids_x.unique())
        args.logger.info("cluster count before and after:(%d,%d)"%(num_clusters, unique_cluster_count))
        if args.remove_empty_clusters:
            if unique_cluster_count < num_clusters:
                while(True):
                    cluster_ids_x, cluster_centers = do_cluster(
                        args,
                        unique_cluster_count,
                        sample_representation_vec_ls,
                        is_cuda,
                        cosin_distance,
                        sample_weights,
                        existing_cluster_centroids=existing_cluster_centroids,
                        all_layer=all_layer,
                    )

                    new_unique_cluster_count = len(cluster_ids_x.unique())
                    args.logger.info("cluster count before and after:(%d,%d)"%(unique_cluster_count, new_unique_cluster_count))

                    if new_unique_cluster_count >= unique_cluster_count:
                        break
                    unique_cluster_count = new_unique_cluster_count

                unique_cluster_count = new_unique_cluster_count
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

    elif not cosin_distance:
        find_best_cluster_num(sample_representation_vec_ls, sample_weights, distance = 'euclidean', all_layer = all_layer)
    else:
        find_best_cluster_num(sample_representation_vec_ls, sample_weights, distance = 'cosine', all_layer = all_layer)


    

    representive_id_ls = []
    representive_representation_ls = []

    if args.full_model_out:
        if not all_layer:
            pairwise_distance_function = pairwise_cross_prod
        else:
            pairwise_distance_function = pairwise_cross_prod_ls

    else:
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
    dist_to_cluster_centroid_ls = []
    dist_to_other_cluster_centroid_ls = []
    args.logger.info("unique cluster count::%d"%(unique_cluster_count))

    for cluster_id in range(unique_cluster_count):
        # curr_cluster_center = cluster_centers[cluster_id]

        cluster_dist_ls = []

        min_cluster_distance = 0
        min_sample_id = -1
        # if torch.sum(cluster_ids_x == cluster_id).item() <= 0:
        #     continue

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
            # curr_cluster_sample_representation = sample_representation_vec_ls[cluster_ids_x == cluster_id].view(
            #     torch.sum(cluster_ids_x == cluster_id), -1,
            # )

            curr_cluster_sample_representation = sample_representation_vec_ls

            if is_cuda:
                curr_cluster_sample_representation = curr_cluster_sample_representation.cuda()
            
            cluster_dist_ls_tensor = pairwise_distance_function(
                curr_cluster_sample_representation,
                cluster_centers[cluster_id].view(1,-1),
                is_cuda=is_cuda,
                weight_by_norm=args.weight_by_norm,
                inner_prod=args.inner_prod,
                ls_idx_range=args.origin_X_ls_lenth,
                full_inner_prod=False
            )
            
            # if args.weight_by_norm:
            #     cluster_dist_ls_tensor = rescale_dist_by_cluster_mean_norm(cluster_dist_ls_tensor, cluster_centers[cluster_id].view(1,-1), all_layer)
            cluster_dist_ls_tensor = cluster_dist_ls_tensor.view(-1)

            sorted_dist_tensor, sorted_sample_idx_tensor = torch.sort(cluster_dist_ls_tensor, descending=False)

            selected_count = int(valid_count_per_class/num_clusters)
            args.logger.info("Validation samples to select: %s"%(selected_count))

            # representive_id_ls.append(
            #     sample_id_ls[cluster_ids_x == cluster_id][sorted_sample_idx_tensor[0:selected_count]].cpu())
            representive_id_ls.append(
                sample_id_ls[sorted_sample_idx_tensor[0:selected_count]].cpu())
        
            # representive_representation_ls.append(
            #     sample_representation_vec_ls[cluster_ids_x == cluster_id][sorted_sample_idx_tensor[0:selected_count]].cpu())

            representive_representation_ls.append(
                sample_representation_vec_ls[sorted_sample_idx_tensor[0:selected_count]].cpu())
    
        else:
            curr_cluster_sample_representation_ls = []
            curr_cluster_center_ls = []
            for arr_idx in range(len(sample_representation_vec_ls)):
                # curr_cluster_sample_representation = sample_representation_vec_ls[arr_idx][cluster_ids_x == cluster_id].view(torch.sum(cluster_ids_x == cluster_id), -1)
                curr_cluster_sample_representation = sample_representation_vec_ls[arr_idx]
                curr_cluster_center = cluster_centers[arr_idx][cluster_id].view(1,-1)
                curr_cluster_center_ls.append(curr_cluster_center.cpu())
                # if is_cuda:
                #     curr_cluster_sample_representation = curr_cluster_sample_representation.cuda()
                curr_cluster_sample_representation_ls.append(curr_cluster_sample_representation)

            cluster_dist_ls_tensor = pairwise_distance_function(
                curr_cluster_sample_representation_ls,
                curr_cluster_center_ls,
                is_cuda=False,
                weight_by_norm=args.weight_by_norm,
                inner_prod=args.inner_prod,
                ls_idx_range=args.origin_X_ls_lenth,
                full_inner_prod=False
            )

            # if args.weight_by_norm:
            #     cluster_dist_ls_tensor = rescale_dist_by_cluster_mean_norm(cluster_dist_ls_tensor, curr_cluster_center_ls,all_layer)

            cluster_dist_ls_tensor = cluster_dist_ls_tensor.view(-1)

            sorted_dist_tensor, sorted_sample_idx_tensor = torch.sort(cluster_dist_ls_tensor, descending=False)

            selected_count = int(valid_count_per_class/num_clusters)

            # representive_id_ls.append(sample_id_ls[cluster_ids_x == cluster_id][sorted_sample_idx_tensor[0:selected_count]].cpu())
            representive_id_ls.append(sample_id_ls[sorted_sample_idx_tensor[0:selected_count]].cpu())

            curr_representive_represetation = []

            for arr_idx in range(len(sample_representation_vec_ls)):

                # curr_representive_represetation.append(
                #     sample_representation_vec_ls[arr_idx][cluster_ids_x == cluster_id][sorted_sample_idx_tensor[0:selected_count]].cpu())
                curr_representive_represetation.append(
                    sample_representation_vec_ls[arr_idx][sorted_sample_idx_tensor[0:selected_count]].cpu())

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

        valid_ids, valid_sample_representation_tensor =  torch.cat(representive_id_ls), res_representative_representation_ls

        if handles_outlier:

            extra_valid_ids, extra_valid_sample_representation_tensor = handle_outliers2(args, sample_id_ls, sample_representation_vec_ls, res_representative_representation_ls, sample_weights)

            if extra_valid_ids is not None and extra_valid_sample_representation_tensor is not None:
                valid_ids = torch.cat([valid_ids.view(-1), extra_valid_ids.view(-1)])

                valid_sample_representation_tensor = [torch.cat([valid_sample_representation_tensor[k], extra_valid_sample_representation_tensor[k]]) for k in range(len(extra_valid_sample_representation_tensor))]



        return valid_ids, valid_sample_representation_tensor

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

    
    # if args.use_model_prov:
    full_sample_representation_tensor_ls.append(full_sample_representation_tensor)
    get_extra_gradient_layer(args, train_loader, criterion, net, full_sample_representation_tensor_ls)
    return full_sample_representation_tensor_ls, all_sample_ids
    # else:
    #     return full_sample_representation_tensor, all_sample_ids



def get_grad_by_example_per_batch(args, labels, output, net, criterion, optimizer, vec_grad_by_example_ls, merge_grad = False, depth=1):
    for idx in range(labels.shape[0]):
        optimizer.zero_grad()
        loss = criterion(output[idx:idx+1], labels[idx:idx+1])
        if not args.all_layer_grad_no_full_loss:
            loss = obtain_full_loss(output[idx:idx+1], labels[idx:idx+1], args.cuda, loss)
        # loss.backward(retain_graph = True)
        # if not vectorize_grad:
        if not merge_grad:
            vec_grad_by_example_ls.append(obtain_net_grad2(net, loss, depth=depth))
        else:
            curr_sample_grad = obtain_net_grad2(net, loss, depth=depth)

            vec_grad_by_example_ls = merge_grad_by_layer(curr_sample_grad, vec_grad_by_example_ls)

    res_grad_by_example_ls = []

    for idx in range(len(vec_grad_by_example_ls)):
        res_grad_by_example_ls.append(torch.cat(vec_grad_by_example_ls[idx]).detach().cpu())

    return res_grad_by_example_ls

def get_grad_by_example_per_batch2(args, labels, output, net, criterion, optimizer, vec_grad_by_example_ls, sampled_net_param_layer_ls, sampled_layer_sqrt_prob_ls, merge_grad = False):
    for idx in range(labels.shape[0]):
        # args, output, net, optimizer, target, criterion
        # obtain_class_wise_grad_ratio_wrt_full_grad(args, output[idx:idx+1], net, optimizer, labels[idx:idx+1], criterion)
        optimizer.zero_grad()
        loss = criterion(output[idx:idx+1], labels[idx:idx+1])
        if not args.all_layer_grad_no_full_loss:
            loss = obtain_full_loss(output[idx:idx+1], labels[idx:idx+1], args.cuda, loss)
        # loss.backward(retain_graph = True)
        # if not vectorize_grad:
        if not merge_grad:
            vec_grad_by_example_ls.append(obtain_net_grad4(loss, sampled_net_param_layer_ls, sampled_layer_sqrt_prob_ls))
        else:
            curr_sample_grad = obtain_net_grad4(loss, sampled_net_param_layer_ls, sampled_layer_sqrt_prob_ls)

            vec_grad_by_example_ls = merge_grad_by_layer(curr_sample_grad, vec_grad_by_example_ls)

    res_grad_by_example_ls = []

    for idx in range(len(vec_grad_by_example_ls)):
        res_grad_by_example_ls.append(torch.cat(vec_grad_by_example_ls[idx]).detach().cpu())

    return res_grad_by_example_ls





def get_grad_by_example(args, train_loader, net, criterion, optimizer, vectorize_grad = False):
    vec_grad_by_example_ls = []
    sample_id_ls = []

    for batch_id, (sample_ids, data, labels) in tqdm(enumerate(train_loader)):
        args.logger.info("sample batch ids::%d"%(batch_id))
        if args.cuda:
            data, labels = train_loader.dataset.to_cuda(data, labels)
        output = net.forward(data)
        vec_grad_by_example_ls = get_grad_by_example_per_batch(args, labels, output, net, criterion, optimizer, vec_grad_by_example_ls, merge_grad = False, depth=1)

        # for idx in range(labels.shape[0]):
        #     optimizer.zero_grad()
        #     loss = criterion(output[idx:idx+1], labels[idx:idx+1])
        #     if not args.all_layer_grad_no_full_loss:
        #         loss = obtain_full_loss(output[idx:idx+1], labels[idx:idx+1], args.cuda, loss)
        #     loss.backward(retain_graph = True)
        #     # if not vectorize_grad:
        #     vec_grad_by_example_ls.append(obtain_net_grad(net))
            # else:
            #     vec_grad_by_example_ls.append(obtain_vectorized_grad(net).view(-1))
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
    # if args.use_model_prov:
    full_sim_mat_ls.append(full_sim_mat)
    full_sim_mat_ls = get_extra_approx_grad_prod(args, train_loader, criterion, net, optimizer, full_sim_mat_ls)    
    return torch.sum(torch.stack(torch.abs(full_sim_mat_ls)),dim=0), sample_id_ls

    # else:
    #     return full_sim_mat, sample_id_ls


def full_approx_grad_prod2(args, train_loader, net, criterion, optimizer):

    full_sim_mat, sample_id_ls = get_approx_grad_prod(args, train_loader, net, criterion, optimizer)

    full_sim_mat_ls = []
    # if args.use_model_prov:
    full_sim_mat_ls.append(full_sim_mat)
    full_sim_mat_ls = get_extra_approx_grad_prod(args, train_loader, criterion, net, optimizer, full_sim_mat_ls)    
    return torch.sum(torch.stack(torch.abs(full_sim_mat_ls)),dim=0), sample_id_ls

    # else:
    #     return full_sim_mat, sample_id_ls


def get_extra_approx_grad_prod(args, train_loader, criterion, net, optimizer, full_sim_mat_ls):
    start_epoch_id = 0
    if args.use_pretrained_model:
        start_epoch_id = torch.load(os.path.join(args.prev_save_path, "early_stopping_epoch"))
    args.logger.info("extra representation starting from epoch %d"%(start_epoch_id))

        # start_epoch_id = int(args.epochs/2)
    for ep in range(start_epoch_id, args.epochs, args.model_prov_period):
        net = load_checkpoint_by_epoch(args, net, ep)
        if net is None:
            continue
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
        for idx in tqdm(range(labels.shape[0])):
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

def obtain_sample_representation_grad_last_layer2(net, sample_representation, labels, criterion, optimizer, is_cuda = False):
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



def print_norm_range_of_representations(args, sample_representation_vec_ls):
    if type(sample_representation_vec_ls) is list:
        for idx in range(len(sample_representation_vec_ls)):
            sample_rep = sample_representation_vec_ls[idx]
            norm_per_sample = torch.norm(sample_rep, dim = 1)
            max_norm = torch.max(norm_per_sample).item()
            min_norm = torch.min(norm_per_sample).item()
            args.logger.info("max norm of representation number %d:%f"%(idx, max_norm))
            args.logger.info("min norm of representation number %d:%f"%(idx, min_norm))

    else:
        sample_rep = sample_representation_vec_ls
        norm_per_sample = torch.norm(sample_rep, dim = 1)
        max_norm = torch.max(norm_per_sample).item()
        min_norm = torch.min(norm_per_sample).item()
        args.logger.info("max norm of the representation:%f"%(max_norm))
        args.logger.info("min norm of the representation:%f"%(min_norm))

def obtain_representations_last_layer_given_model(args, train_loader, net, criterion, optimizer, sampled_col_ids = None, origin_label = None):
    sample_representation_vec_ls = []

    sample_id_ls = []
    # with torch.no_grad():

        # all_sample_representations = [None]*len(train_loader.dataset)


    for batch_id, (sample_ids, data, labels) in tqdm(enumerate(train_loader)):

        if args.cuda:
            data, labels = train_loader.dataset.to_cuda(data, labels)
        if origin_label is not None:
            labels = torch.tensor(origin_label[sample_ids])
            if args.cuda:
                labels = labels.cuda()
            # labels = labels.cuda()
        if not args.cluster_method_two_plus:
            sample_representation = net.feature_forward(data, all_layer=False)
        else:
            # sample_representation = F.softmax(net.forward(data),dim=1)
            sample_representation = net.feature_forward2(data, all_layer_grad_no_full_loss=args.all_layer_grad_no_full_loss, labels = labels)

        if not args.all_layer and not args.all_layer2:
            sample_representation_vec_ls.append(sample_representation.detach().cpu())

        else:
            if args.all_layer:
                sample_representation_grad = obtain_sample_representation_grad_last_layer(net, sample_representation, labels, criterion, optimizer, is_cuda=args.cuda)
                if batch_id == 0:
                    sample_representation_vec_ls.extend([sample_representation.detach().cpu(), sample_representation_grad.detach().cpu()])
                else:
                    sample_representation_vec_ls[0] = torch.cat([sample_representation_vec_ls[0].detach().cpu(), sample_representation.detach().cpu()])
                    sample_representation_vec_ls[1] = torch.cat([sample_representation_vec_ls[1].detach().cpu(), sample_representation_grad.detach().cpu()])
            else:

                if args.all_layer2:
                    # sample_representation_grad = net.obtain_gradient_last_full_layer(sample_representation, labels, criterion)
                    output = net.forward(data)
                    vec_grad_by_example_ls = []
                    vec_grad_by_example_ls = get_grad_by_example_per_batch(args, labels, output, net, criterion, optimizer, vec_grad_by_example_ls, merge_grad = True, depth=args.grad_layer_depth)
                    # sample_representation_grad = obtain_sample_representation_grad_last_layer(net, sample_representation, labels, criterion, optimizer, is_cuda=args.cuda)


                
                # else:
                    # if batch_id == 0:
                    #     sample_representation_vec_ls.extend([sample_representation.detach().cpu(), sample_representation_grad.detach().cpu()])
                    # else:
                    #     sample_representation_vec_ls[0] = torch.cat([sample_representation_vec_ls[0].detach().cpu(), sample_representation.detach().cpu()])
                    #     sample_representation_vec_ls[1] = torch.cat([sample_representation_vec_ls[1].detach().cpu(), sample_representation_grad.detach().cpu()])

                    if batch_id == 0:
                        sample_representation_vec_ls.append(sample_representation.detach().cpu())
                        sample_representation_vec_ls.extend(vec_grad_by_example_ls)
                    else:

                        sample_representation_vec_ls[0] = torch.cat([sample_representation_vec_ls[0], sample_representation.detach().cpu()])
                        for sample_arr_id in range(len(sample_representation_vec_ls)-1):
                            sample_representation_vec_ls[sample_arr_id+1] = torch.cat([sample_representation_vec_ls[sample_arr_id+1].detach().cpu(), vec_grad_by_example_ls[sample_arr_id].detach().cpu()])



                # for arr_idx in range(len(sample_representation_vec_ls)):
                #     sample_representation_vec_ls[arr_idx] = torch.cat([sample_representation_vec_ls[arr_idx].detach().cpu(), sample_representation[arr_idx].detach().cpu()])

        sample_id_ls.append(sample_ids)
    if args.all_layer or args.all_layer2:
        print_norm_range_of_representations(args, sample_representation_vec_ls)
        return sample_representation_vec_ls, sample_id_ls
    else:
        sample_representation_vec_ls = torch.cat(sample_representation_vec_ls)
        if args.cluster_method_two_sampling:
            if sampled_col_ids is not None:
                sample_representation_vec_ls = sample_representation_vec_ls[:, sampled_col_ids]
            else:
                if args.cluster_method_two_sample_col_count >= sample_representation_vec_ls.shape[1]:
                    sampled_col_ids = torch.tensor(list(range(sample_representation_vec_ls.shape[1])))
                else:
                    sampled_col_ids = np.random.choice(sample_representation_vec_ls.shape[1], size = args.cluster_method_two_sample_col_count, replace=False)
                    sampled_col_ids = torch.from_numpy(sampled_col_ids)

                sample_representation_vec_ls = sample_representation_vec_ls[:, sampled_col_ids]
        
        print_norm_range_of_representations(args, sample_representation_vec_ls)
        return sample_representation_vec_ls, sample_id_ls, sampled_col_ids


def obtain_class_wise_gradient(class_digit, output, net, optimizer):
    optimizer.zero_grad()
    loss = torch.sum(output.view(-1)*class_digit.view(-1))
    loss.backward(retain_graph=True)
    net_norm_sum = obtain_net_grad_norm(net)
    return net_norm_sum


def obtain_class_wise_grad_ratio_wrt_full_grad(args, output, net, optimizer, target, criterion):
    class_digit = torch.ones(output.shape[-1])
    if args.cuda:
        class_digit = class_digit.cuda()
    optimizer.zero_grad()
    class_digit = torch.zeros(output.shape[-1])*0.5

    if args.cuda:
        class_digit = class_digit.cuda()
    loss = torch.mean(F.log_softmax(output).view(-1)*class_digit.view(-1))
    loss = criterion(output, target)
    loss.backward(retain_graph=True)
    # full_loss = obtain_full_loss(output, target, args.cuda, loss)
    # full_loss.backward(retain_graph=True) 
    full_net_norm_sum = obtain_net_grad_norm(net)
    # full_class_wise_grad_norm =  obtain_class_wise_gradient(class_digit, output, net, optimizer)
    for k in range(output.shape[-1]):
        optimizer.zero_grad()
        # class_digit = torch.zeros(output.shape[-1])
        # class_digit[k]=1
        class_digit = torch.tensor([k])
        if args.cuda:
            class_digit = class_digit.cuda()
        loss = criterion(output, class_digit)
        loss.backward(retain_graph=True)

        curr_class_wise_grad_norm =  obtain_net_grad_norm(net)#obtain_class_wise_gradient(class_digit, output, net, optimizer)

        norm_ratio = curr_class_wise_grad_norm/full_net_norm_sum
        args.logger.info("norm ratio for class %d:%f"%(k, norm_ratio.item()))
        print()
        

def obtain_norms_for_each_layer(args, train_dataset, net, criterion, optimizer):
    # train_loader = 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    net_param_count_ls = obtain_net_param_count_ls(net)
    full_net_grad_norm_ls = 0
    for batch_id, (sample_ids, data, labels) in tqdm(enumerate(train_loader)):

        if args.cuda:
            data, labels = train_loader.dataset.to_cuda(data, labels)

        output = net.forward(data)

        loss = criterion(output, labels)
        if not args.all_layer_grad_no_full_loss:
            loss = obtain_full_loss(output, labels, args.cuda, loss)
        loss.backward()
        net_grad_ls = obtain_net_grad(net)
        net_grad_norm_ls = compute_net_grad_norm_ls(net_grad_ls)
        full_net_grad_norm_ls += net_grad_norm_ls*labels.shape[0]
    full_net_grad_norm_ls = full_net_grad_norm_ls/len(train_loader.dataset)
    return full_net_grad_norm_ls, net_param_count_ls
        
def obtain_representations_for_validset(valid_set, args, net, criterion, optimizer, sampled_net_param_layer_ls, sampled_layer_sqrt_prob_ls, sampled_col_ids_ls = None):
    validloader = torch.utils.data.DataLoader(valid_set, batch_size=args.test_batch_size, shuffle=False, num_workers=2, pin_memory=False)
    sample_representation_vec_ls = []
    for batch_id, (_, data, labels) in tqdm(enumerate(validloader)):
    
        if args.cuda:
            data, labels = validloader.dataset.to_cuda(data, labels)
            # labels = labels.cuda()
        # if not args.full_model_out:
        output = net.forward(data)
        # else:
        #     sample_representation = F.softmax(net.forward(data),dim=1)

        # if not args.all_layer and not args.all_layer2:
        #     sample_representation_vec_ls.append(sample_representation.detach().cpu())
        vec_grad_by_example_ls = []
        vec_grad_by_example_ls = get_grad_by_example_per_batch2(args, labels, output, net, criterion, optimizer, vec_grad_by_example_ls, sampled_net_param_layer_ls, sampled_layer_sqrt_prob_ls, merge_grad = True)

        if batch_id == 0:
            # sample_representation_vec_ls.append(sample_representation.detach().cpu())
            sample_representation_vec_ls.extend(vec_grad_by_example_ls)
        else:

            # sample_representation_vec_ls[0] = torch.cat([sample_representation_vec_ls[0], sample_representation.detach().cpu()])
            for sample_arr_id in range(len(sample_representation_vec_ls)):
                sample_representation_vec_ls[sample_arr_id] = torch.cat([sample_representation_vec_ls[sample_arr_id].detach().cpu(), vec_grad_by_example_ls[sample_arr_id].detach().cpu()])


    return sample_representation_vec_ls


def obtain_sampled_representations_cluster_method_three(sample_representation_vec_ls, args, sampled_col_ids_ls = None):
    sampled_sample_representation_vec_ls = []
    sampled_sampled_col_ids = []

    for layer_idx in range(len(sample_representation_vec_ls)):
        represetion_vec = sample_representation_vec_ls[layer_idx]
        sampled_col_ids = None
        if sampled_col_ids_ls is not None:
            sampled_col_ids = sampled_col_ids_ls[layer_idx]

        if sampled_col_ids is not None:
            represetion_vec = represetion_vec[:, sampled_col_ids]
        else:
            if args.cluster_method_three_sample_col_count >= represetion_vec.shape[1]:
                sampled_col_ids = torch.tensor(list(range(represetion_vec.shape[1])))
            else:
                sampled_col_ids = np.random.choice(represetion_vec.shape[1], size = args.cluster_method_three_sample_col_count, replace=False)
                sampled_col_ids = torch.from_numpy(sampled_col_ids)

            represetion_vec = represetion_vec[:, sampled_col_ids]

        sampled_sample_representation_vec_ls.append(represetion_vec)
        sampled_sampled_col_ids.append(sampled_col_ids)
    sample_representation_vec_ls = sampled_sample_representation_vec_ls
    sampled_col_ids_ls  = sampled_sampled_col_ids

    return sample_representation_vec_ls, sampled_col_ids_ls

def obtain_representations_last_layer_given_model2(train_dataset, args, train_loader, net, criterion, optimizer, validset = None, sampled_col_ids_ls = None):
    sample_representation_vec_ls = []

    sample_id_ls = []
    # with torch.no_grad():

        # all_sample_representations = [None]*len(train_loader.dataset)
    grad_norm_by_layer_ls, net_param_count_ls = obtain_norms_for_each_layer(args, train_dataset, net, criterion, optimizer)
    avg_grad_norm_by_layer = grad_norm_by_layer_ls/torch.tensor(net_param_count_ls)
    sampled_net_param_layer_ls,sampled_layer_sqrt_prob_ls = biased_rand_sample_parameter(net, avg_grad_norm_by_layer, sampled_param_count = args.sampled_param_count, include_last_layer = True, replace=True)#args.replace)

    for batch_id, (sample_ids, data, labels) in tqdm(enumerate(train_loader)):

        if args.cuda:
            data, labels = train_loader.dataset.to_cuda(data, labels)
            # labels = labels.cuda()
        # if not args.full_model_out:
        output = net.forward(data)
        # else:
        #     sample_representation = F.softmax(net.forward(data),dim=1)

        # if not args.all_layer and not args.all_layer2:
        #     sample_representation_vec_ls.append(sample_representation.detach().cpu())
        vec_grad_by_example_ls = []
        vec_grad_by_example_ls = get_grad_by_example_per_batch2(args, labels, output, net, criterion, optimizer, vec_grad_by_example_ls, sampled_net_param_layer_ls, sampled_layer_sqrt_prob_ls, merge_grad = True)

        if batch_id == 0:
            # sample_representation_vec_ls.append(sample_representation.detach().cpu())
            sample_representation_vec_ls.extend(vec_grad_by_example_ls)
        else:

            # sample_representation_vec_ls[0] = torch.cat([sample_representation_vec_ls[0], sample_representation.detach().cpu()])
            for sample_arr_id in range(len(sample_representation_vec_ls)):
                sample_representation_vec_ls[sample_arr_id] = torch.cat([sample_representation_vec_ls[sample_arr_id].detach().cpu(), vec_grad_by_example_ls[sample_arr_id].detach().cpu()])



        # else:
        #     if args.all_layer:
        #         sample_representation_grad = obtain_sample_representation_grad_last_layer(net, sample_representation, labels, criterion, optimizer, is_cuda=args.cuda)
        #         if batch_id == 0:
        #             sample_representation_vec_ls.extend([sample_representation.detach().cpu(), sample_representation_grad.detach().cpu()])
        #         else:
        #             sample_representation_vec_ls[0] = torch.cat([sample_representation_vec_ls[0].detach().cpu(), sample_representation.detach().cpu()])
        #             sample_representation_vec_ls[1] = torch.cat([sample_representation_vec_ls[1].detach().cpu(), sample_representation_grad.detach().cpu()])
        #     else:

        #         if args.all_layer2:
        #             # sample_representation_grad = net.obtain_gradient_last_full_layer(sample_representation, labels, criterion)
        #             output = net.forward(data)
        #             vec_grad_by_example_ls = []
        #             vec_grad_by_example_ls = get_grad_by_example_per_batch(args, labels, output, net, criterion, optimizer, vec_grad_by_example_ls, merge_grad = True, depth=args.grad_layer_depth)
                    # sample_representation_grad = obtain_sample_representation_grad_last_layer(net, sample_representation, labels, criterion, optimizer, is_cuda=args.cuda)


                
                # else:
                    # if batch_id == 0:
                    #     sample_representation_vec_ls.extend([sample_representation.detach().cpu(), sample_representation_grad.detach().cpu()])
                    # else:
                    #     sample_representation_vec_ls[0] = torch.cat([sample_representation_vec_ls[0].detach().cpu(), sample_representation.detach().cpu()])
                    #     sample_representation_vec_ls[1] = torch.cat([sample_representation_vec_ls[1].detach().cpu(), sample_representation_grad.detach().cpu()])

                    


                # for arr_idx in range(len(sample_representation_vec_ls)):
                #     sample_representation_vec_ls[arr_idx] = torch.cat([sample_representation_vec_ls[arr_idx].detach().cpu(), sample_representation[arr_idx].detach().cpu()])

        sample_id_ls.append(sample_ids)
    # if args.all_layer or args.all_layer2:
    print_norm_range_of_representations(args, sample_representation_vec_ls)
    valid_sample_representation_ls = None

    if args.cluster_method_three_sampling:
        sample_representation_vec_ls, sampled_col_ids_ls =  obtain_sampled_representations_cluster_method_three(sample_representation_vec_ls, args, sampled_col_ids_ls = sampled_col_ids_ls)

    if validset is not None:
        valid_sample_representation_ls = obtain_representations_for_validset(validset, args, net, criterion, optimizer, sampled_net_param_layer_ls, sampled_layer_sqrt_prob_ls, sampled_col_ids_ls)
        if args.cluster_method_three_sampling:
            valid_sample_representation_ls, _ =  obtain_sampled_representations_cluster_method_three(valid_sample_representation_ls, args, sampled_col_ids_ls = sampled_col_ids_ls)

    return sample_representation_vec_ls, sample_id_ls, valid_sample_representation_ls
    # else:
    #     sample_representation_vec_ls = torch.cat(sample_representation_vec_ls)
    #     print_norm_range_of_representations(args, sample_representation_vec_ls)
    #     return sample_representation_vec_ls, sample_id_ls



def get_extra_representations_last_layer(args, train_loader, criterion, net, full_sample_representation_vec_ls, valid_sample_representation_vec_ls, validloader = None, full_sample_representation_vec_ls2 = None, valid_sample_representation_vec_ls2 = None, qualitiative = False, origin_label = None):
    
    start_epoch_id = 0
    if args.use_pretrained_model:
        start_epoch_id = torch.load(os.path.join(args.prev_save_path, "early_stopping_epoch"))
    args.logger.info("extra representation starting from epoch %d"%(start_epoch_id))

        # start_epoch_id = int(args.epochs/2)
    if not args.all_layer and not args.all_layer2:
        full_sample_representation_vec_ls = [full_sample_representation_vec_ls]
        if validloader is not None:
            full_valid_sample_representation_vec_ls = [valid_sample_representation_vec_ls]
        else:
            full_valid_sample_representation_vec_ls = None

    if origin_label is not None:
        full_sample_representation_vec_ls2, full_sample_representation_vec_ls3 = full_sample_representation_vec_ls2
        full_sample_representation_vec_ls3 = [full_sample_representation_vec_ls3]
    else:
        full_sample_representation_vec_ls3 = None

    if full_sample_representation_vec_ls2 is not None:
        full_sample_representation_vec_ls2 = [full_sample_representation_vec_ls2]
    else:
        full_sample_representation_vec_ls2 = None
    if valid_sample_representation_vec_ls2 is not None:
        full_valid_sample_representation_vec_ls2 = [valid_sample_representation_vec_ls2]
    else:
        full_valid_sample_representation_vec_ls2 = None

    

    # for ep in range(start_epoch_id, args.epochs, args.model_prov_period):
    #     net = load_checkpoint_by_epoch(args, net, ep)
    max_net_prov_count = 5 
    for k in range(0, max_net_prov_count):
    # for ep in range(start_epoch_id, args.epochs, args.model_prov_period):
        ep = start_epoch_id + k*args.model_prov_period
        # if epoch_count > 5:
        #     break
        net = load_checkpoint_by_epoch(args, net, ep)
        if net is None:
            continue
        optimizer, _=obtain_optimizer_scheduler(args, net, start_epoch = 0)
        sample_representation_vec_ls, _,sampled_col_ids = obtain_representations_last_layer_given_model(args, train_loader, net, criterion, optimizer)
        if qualitiative:
            args.all_layer_grad_no_full_loss = True
            sample_representation_vec_ls2, _,_ = obtain_representations_last_layer_given_model(args, train_loader, net, criterion, optimizer, sampled_col_ids=sampled_col_ids)
            if origin_label is not None:
                sample_representation_vec_ls3, _,_ = obtain_representations_last_layer_given_model(args, train_loader, net, criterion, optimizer, sampled_col_ids=sampled_col_ids, origin_label = origin_label)
            args.all_layer_grad_no_full_loss = False

        if validloader is not None:
            curr_valid_sample_representation_vec_ls, _,_ = obtain_representations_last_layer_given_model(args, validloader, net, criterion, optimizer, sampled_col_ids=sampled_col_ids)
            if qualitiative:
                args.all_layer_grad_no_full_loss = True
                curr_valid_sample_representation_vec_ls2, _,_ = obtain_representations_last_layer_given_model(args, validloader, net, criterion, optimizer, sampled_col_ids=sampled_col_ids)
                args.all_layer_grad_no_full_loss = False

        if args.all_layer or args.all_layer2:
            full_sample_representation_vec_ls.extend(sample_representation_vec_ls)
            if full_sample_representation_vec_ls2 is not None:
                full_sample_representation_vec_ls2.extend(sample_representation_vec_ls2)
            if full_sample_representation_vec_ls3 is not None:
                full_sample_representation_vec_ls3.extend(sample_representation_vec_ls3)
            
            if validloader is not None:
                full_valid_sample_representation_vec_ls.extend(curr_valid_sample_representation_vec_ls)
                if full_valid_sample_representation_vec_ls2 is not None:
                    full_valid_sample_representation_vec_ls2.extend(curr_valid_sample_representation_vec_ls2)
        else:
            full_sample_representation_vec_ls.append(sample_representation_vec_ls)
            if full_sample_representation_vec_ls2 is not None:
                full_sample_representation_vec_ls2.append(sample_representation_vec_ls2)
            if full_sample_representation_vec_ls3 is not None:
                    full_sample_representation_vec_ls3.append(sample_representation_vec_ls3)
            if validloader is not None:
                full_valid_sample_representation_vec_ls.append(curr_valid_sample_representation_vec_ls)
                if full_valid_sample_representation_vec_ls2 is not None:
                    full_valid_sample_representation_vec_ls2.append(curr_valid_sample_representation_vec_ls2)

    if not qualitiative:
        return full_sample_representation_vec_ls, full_valid_sample_representation_vec_ls
    else:
        if origin_label is None:
            return full_sample_representation_vec_ls, full_valid_sample_representation_vec_ls, full_sample_representation_vec_ls2, full_valid_sample_representation_vec_ls2
        else:
            return full_sample_representation_vec_ls, full_valid_sample_representation_vec_ls, (full_sample_representation_vec_ls2, full_sample_representation_vec_ls3), full_valid_sample_representation_vec_ls2


def obtain_grad_last_layer(trainloader, args, net):
    output_probs_ls = []

    output_origin_probs_ls = []

    for batch_id, (sample_ids, data, labels) in tqdm(enumerate(trainloader)):

        if args.cuda:
            data, labels = trainloader.dataset.to_cuda(data, labels)

        output_probs = F.softmax(net.forward(data),dim=1)

        label_info = F.one_hot(labels.view(-1), num_classes=output_probs.shape[1])

        output_probs_ls.append((output_probs.detach() - label_info).cpu())

        output_origin_probs_ls.append(output_probs.detach().cpu())

    output_probs_tensor = torch.cat(output_probs_ls)

    output_origin_probs_tensor = torch.cat(output_origin_probs_ls)

    return output_probs_tensor, output_origin_probs_tensor

def get_extra_output_prob_ls(args, train_loader, net, full_output_probs_tensor_ls, full_output_origin_probs_tensor_ls):
    
    start_epoch_id = 0
    if args.use_pretrained_model:
        start_epoch_id = torch.load(os.path.join(args.prev_save_path, "early_stopping_epoch"))
        # start_epoch_id += 1
    args.logger.info("extra representation starting from epoch %d"%(start_epoch_id))

        # start_epoch_id = int(args.epochs/2)
    # if not args.all_layer and not args.all_layer2:
    #     full_output_probs_tensor_ls = [full_output_probs_tensor_ls]
    #     full_output_origin_probs_tensor_ls = [full_output_origin_probs_tensor_ls]
    #     if full_valid_sample_representation_vec_ls is not None:
    #         full_valid_sample_representation_vec_ls = [full_valid_sample_representation_vec_ls]

    epoch_count = 0
    max_net_prov_count = 5 
    for k in range(0, max_net_prov_count):
    # for ep in range(start_epoch_id, args.epochs, args.model_prov_period):
        ep = start_epoch_id + k*args.model_prov_period
        # if epoch_count > 5:
        #     break
        net = load_checkpoint_by_epoch(args, net, ep)
        if net is None:
            continue


        output_probs_tensor, output_origin_probs_tensor = obtain_grad_last_layer(train_loader, args, net)
        full_output_probs_tensor_ls.append(output_probs_tensor)
        full_output_origin_probs_tensor_ls.append(output_origin_probs_tensor)


def get_extra_representations_last_layer2(train_dataset, args, train_loader, criterion, net, full_sample_representation_vec_ls, validset = None, full_valid_sample_representation_vec_ls = None):
    
    start_epoch_id = 0
    if args.use_pretrained_model:
        start_epoch_id = torch.load(os.path.join(args.prev_save_path, "early_stopping_epoch"))
        # start_epoch_id += 1
    args.logger.info("extra representation starting from epoch %d"%(start_epoch_id))

        # start_epoch_id = int(args.epochs/2)
    if not args.all_layer and not args.all_layer2:
        full_sample_representation_vec_ls = [full_sample_representation_vec_ls]
        if full_valid_sample_representation_vec_ls is not None:
            full_valid_sample_representation_vec_ls = [full_valid_sample_representation_vec_ls]

    epoch_count = 0
    max_net_prov_count = 5 
    for k in range(0, max_net_prov_count):
    # for ep in range(start_epoch_id, args.epochs, args.model_prov_period):
        ep = start_epoch_id + k*args.model_prov_period
        # if epoch_count > 5:
        #     break
        net = load_checkpoint_by_epoch(args, net, ep)
        if net is None:
            continue
        optimizer, _=obtain_optimizer_scheduler(args, net, start_epoch = 0)
        sample_representation_vec_ls, _, curr_valid_sample_representation_ls  = obtain_representations_last_layer_given_model2(train_dataset, args, train_loader, net, criterion, optimizer, validset = validset)

        if args.all_layer or args.all_layer2:
            full_sample_representation_vec_ls.extend(sample_representation_vec_ls)
            if full_valid_sample_representation_vec_ls is not None:
                full_valid_sample_representation_vec_ls.extend(curr_valid_sample_representation_ls)
        else:
            full_sample_representation_vec_ls.append(sample_representation_vec_ls)
            if full_valid_sample_representation_vec_ls is not None:
                full_valid_sample_representation_vec_ls.append(curr_valid_sample_representation_ls)

        epoch_count += 1

    return full_sample_representation_vec_ls, full_valid_sample_representation_vec_ls

def get_extra_gradient_layer(args, train_loader, criterion, net, full_sample_representation_vec_ls):
    
    start_epoch_id = 0
    if args.use_pretrained_model:
        start_epoch_id = torch.load(os.path.join(args.prev_save_path, "early_stopping_epoch"))
    args.logger.info("extra representation starting from epoch %d"%(start_epoch_id))

        # start_epoch_id = int(args.epochs/2)
    

    # for ep in range(start_epoch_id, args.epochs, args.model_prov_period):
    max_net_prov_count = 5 
    for k in range(0, max_net_prov_count):
    # for ep in range(start_epoch_id, args.epochs, args.model_prov_period):
        ep = start_epoch_id + k*args.model_prov_period
        net = load_checkpoint_by_epoch(args, net, ep)
        if net is None:
            continue
        optimizer, _=obtain_optimizer_scheduler(args, net, start_epoch = 0)
        full_sample_representation_tensor, all_sample_ids = get_grad_by_example(args, train_loader, net, criterion, optimizer)


        # sample_representation_vec_ls, _ = obtain_representations_last_layer_given_model(args, train_loader, net, criterion, optimizer)

        # if args.all_layer:
        #     full_sample_representation_vec_ls.extend(sample_representation_vec_ls)
        # else:
        full_sample_representation_vec_ls.append(full_sample_representation_tensor)

    return full_sample_representation_vec_ls



def get_representations_last_layer(args, train_loader, criterion, optimizer, net, validset = None, qualitiative = False, origin_label = None):

    sample_representation_vec_ls, sample_id_ls, sampled_col_ids = obtain_representations_last_layer_given_model(args, train_loader, net, criterion, optimizer)
    sample_representation_vec_ls2 = None
    sample_representation_vec_ls3 = None
    valid_sample_representation_vec_ls2 = None
    if qualitiative:
        args.all_layer_grad_no_full_loss = True
        sample_representation_vec_ls2, sample_id_ls, _ = obtain_representations_last_layer_given_model(args, train_loader, net, criterion, optimizer, sampled_col_ids = sampled_col_ids)
        if origin_label is not None:
            sample_representation_vec_ls3, _, _ = obtain_representations_last_layer_given_model(args, train_loader, net, criterion, optimizer, sampled_col_ids = sampled_col_ids, origin_label = origin_label)
            sample_representation_vec_ls2 = (sample_representation_vec_ls2, sample_representation_vec_ls3)

        args.all_layer_grad_no_full_loss = False

    if validset is not None:
        validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=True)
        valid_sample_representation_vec_ls, _, _ = obtain_representations_last_layer_given_model(args, validloader, net, criterion, optimizer, sampled_col_ids=sampled_col_ids)
        if qualitiative:
            args.all_layer_grad_no_full_loss = True
            valid_sample_representation_vec_ls2, _, _ = obtain_representations_last_layer_given_model(args, validloader, net, criterion, optimizer, sampled_col_ids=sampled_col_ids)
            args.all_layer_grad_no_full_loss = False
    else:
        validloader = None
        valid_sample_representation_vec_ls = None

    

    # if args.use_model_prov:
    if not qualitiative:
        sample_representation_vec_ls, valid_sample_representation_vec_ls = get_extra_representations_last_layer(args, train_loader, criterion, net, sample_representation_vec_ls, valid_sample_representation_vec_ls, validloader = validloader, full_sample_representation_vec_ls2 = sample_representation_vec_ls2, valid_sample_representation_vec_ls2 = valid_sample_representation_vec_ls2, qualitiative = qualitiative)
    else:
        sample_representation_vec_ls, valid_sample_representation_vec_ls, sample_representation_vec_ls2, valid_sample_representation_vec_ls2 = get_extra_representations_last_layer(args, train_loader, criterion, net, sample_representation_vec_ls, valid_sample_representation_vec_ls, validloader = validloader, full_sample_representation_vec_ls2 = sample_representation_vec_ls2, valid_sample_representation_vec_ls2 = valid_sample_representation_vec_ls2, qualitiative = qualitiative, origin_label = origin_label)
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

    if qualitiative:
        return (full_sample_representation_tensor, sample_representation_vec_ls2), all_sample_ids, (valid_sample_representation_vec_ls, valid_sample_representation_vec_ls2)
    else:
        return full_sample_representation_tensor, all_sample_ids, valid_sample_representation_vec_ls

def get_representations_last_layer2(train_dataset, args, train_loader, criterion, optimizer, net, validset = None):

    sample_representation_vec_ls, sample_id_ls, valid_sample_representation_ls = obtain_representations_last_layer_given_model2(train_dataset, args, train_loader, net, criterion, optimizer, validset = validset)
    # if args.use_model_prov:
    args.all_layer = True
    sample_representation_vec_ls, valid_sample_representation_ls = get_extra_representations_last_layer2(train_dataset, args, train_loader, criterion, net, sample_representation_vec_ls, validset, valid_sample_representation_ls)


    full_sample_representation_tensor = sample_representation_vec_ls

    all_sample_ids = torch.cat(sample_id_ls)

    return full_sample_representation_tensor, all_sample_ids, valid_sample_representation_ls

def get_representative_valid_ids2(criterion, optimizer, train_loader, args, net, valid_count, cached_sample_weights = None, existing_valid_set = None, return_cluster_info = False, only_sample_representation = False, validset=None, qualitiative = False, origin_label = None):

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
        full_sample_representation_tensor, all_sample_ids, existing_valid_representation = get_representations_last_layer(args, train_loader, criterion, optimizer, net, validset = validset, qualitiative=qualitiative, origin_label = origin_label)
    else:

        full_sample_representation_tensor, all_sample_ids = get_grad_by_example(args, train_loader, net, criterion, optimizer, vectorize_grad=True)

        
        full_sim_mat1, sample_id_ls = full_approx_grad_prod(args, train_loader, net, criterion, optimizer)
        # sim_mat_file_name = os.path.join(args.save_path, "full_similarity_mat")
        # if not os.path.exists(sim_mat_file_name):
        #     full_sim_mat1 = pairwise_cosine_full(full_sample_representation_tensor, is_cuda=args.cuda)
        #     torch.save(full_sim_mat1, sim_mat_file_name)
        # else:
        #     full_sim_mat1 = torch.load(sim_mat_file_name)



        print()
    
    if qualitiative:
        full_sample_representation_tensor, (full_sample_representation_tensor2, full_sample_representation_tensor3) = full_sample_representation_tensor
        existing_valid_representation, existing_valid_representation2 = existing_valid_representation

    origin_X_ls_lenth = len(full_sample_representation_tensor)

    if args.get_representations:
        torch.save(full_sample_representation_tensor, os.path.join(args.save_path,
            "sample_representation"))

    args.origin_X_ls_lenth = origin_X_ls_lenth
    # if not args.not_rescale_features:
    #     full_sample_representation_tensor = scale_and_extend_data_vector(full_sample_representation_tensor)

    # args.origin_X_ls_lenth = origin_X_ls_lenth

    # full_sample_representation_tensor = scale_and_extend_data_vector(full_sample_representation_tensor)
    # valid_ids_ls = []
    # valid_sample_representation_ls = []
    # full_sample_representation_ls = []
    # full_sample_id_ls = []
    if only_sample_representation:
        return full_sample_representation_tensor, existing_valid_representation
    # sample_representation_vec_ls = sample_representation_vec_ls_by_class[label]
    if args.cluster_no_reweighting:
        logging.info("no reweighting for k-means")
        cached_sample_weights = None
    # extra_valid_ids, extra_valid_sample_representation_tensor = None, None
    if existing_valid_representation is None:
        if cached_sample_weights is not None:
            valid_ids, valid_sample_representation_tensor = cluster_per_class(args, full_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=cached_sample_weights[all_sample_ids], cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None, handles_outlier=False)  
            # if args.inner_prod:
            #     extra_valid_ids, extra_valid_sample_representation_tensor = handle_outliers(args, all_sample_ids, full_sample_representation_tensor, valid_sample_representation_tensor, cached_sample_weights[all_sample_ids])
        else:
            valid_ids, valid_sample_representation_tensor = cluster_per_class(args, full_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=None, cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None, handles_outlier=False)  
        #     if args.inner_prod:
        #         extra_valid_ids, extra_valid_sample_representation_tensor = handle_outliers(args, all_sample_ids, full_sample_representation_tensor, valid_sample_representation_tensor)
        # if extra_valid_ids is not None and extra_valid_sample_representation_tensor is not None:
        #     valid_ids = torch.cat([valid_ids.view(-1), extra_valid_ids.view(-1)])

        #     valid_sample_representation_tensor = [torch.cat([valid_sample_representation_tensor[k], extra_valid_sample_representation_tensor[k]]) for k in range(len(extra_valid_sample_representation_tensor))]
        far_sample_representation_tensor = full_sample_representation_tensor
    # while len(valid_ids) < main_represent_count:
    #     far_sample_representation_tensor, far_sample_ids = obtain_farthest_training_samples(args, args.cosin_dist, True, far_sample_representation_tensor, valid_sample_representation_tensor, args.cuda)
    #     all_sample_ids = all_sample_ids[far_sample_ids]

    #     curr_representation_count = valid_count - len(valid_ids)
    #     if cached_sample_weights is not None:
    #         new_valid_ids, new_valid_sample_representation_tensor = cluster_per_class(args, far_sample_representation_tensor, all_sample_ids, valid_count_per_class = curr_representation_count, num_clusters = curr_representation_count, sample_weights=cached_sample_weights[far_sample_ids], cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None)  
    #     else:
    #         new_valid_ids, new_valid_sample_representation_tensor = cluster_per_class(args, far_sample_representation_tensor, all_sample_ids, valid_count_per_class = curr_representation_count, num_clusters = curr_representation_count, sample_weights=None, cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None)  


    #     uncovered_valid_ids, uncovered_valid_sample_ids, _ = get_uncovered_new_valid_ids(args, new_valid_ids, new_valid_sample_representation_tensor, valid_sample_representation_tensor, cosine_dist = args.cosin_dist, all_layer = True, is_cuda = args.cuda)
        
    #     args.logger.info("new valid sample count::%d"%(len(uncovered_valid_sample_ids)))
    #     if len(uncovered_valid_sample_ids) > 0:
    #         uncovered_valid_sample_representations = [new_valid_sample_representation_tensor[k][uncovered_valid_ids] for k in range(len(new_valid_sample_representation_tensor))]
    #         full_dists = compute_distance(args, args.cosin_dist, True, full_sample_representation_tensor, valid_sample_representation_tensor, args.cuda)
    #         print(full_dists[uncovered_valid_sample_ids])
    #         valid_ids = torch.cat([uncovered_valid_sample_ids, valid_ids])
    #         valid_sample_representation_tensor = [torch.cat([valid_sample_representation_tensor[k], uncovered_valid_sample_representations[k]]) for k in range(len(valid_sample_representation_tensor))]
    # print()
    # if existing_valid_representation is not None:
    #     uncovered_valid_ids, uncovered_valid_sample_ids, max_existing_to_valid_dist = get_uncovered_new_valid_ids(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, cosine_dist = args.cosin_dist, all_layer = True, is_cuda = args.cuda)
    #     valid_ids = uncovered_valid_sample_ids
    #     valid_sample_representation_tensor = [valid_sample_representation_tensor[k][uncovered_valid_ids] for k in range(len(valid_sample_representation_tensor))]
    

    # if existing_valid_representation is not None:
    else:
        # remaining_valid_ids, remaining_local_valid_ids = determine_new_valid_ids(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, len(valid_ids), cosine_dist=args.cosin_dist, all_layer = True, is_cuda = args.cuda)
        # full_dists = compute_distance(args, args.cosin_dist, True, full_sample_representation_tensor, existing_valid_representation, args.cuda)
        # cluster_max_radius = torch.max(torch.min(full_dists, dim = 1)[0])/2
        
        # uncovered_valid_ids, uncovered_valid_sample_ids, max_existing_to_valid_dist = get_uncovered_new_valid_ids(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, cluster_max_radius, cosine_dist = args.cosin_dist, all_layer = True, is_cuda = args.cuda)# get_uncovered_new_valid_ids(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, cosine_dist = args.cosin_dist, all_layer = True, is_cuda = args.cuda)
        far_sample_representation_tensor = full_sample_representation_tensor
        # while len(remaining_valid_ids) <= 0:
            

        far_sample_representation_tensor, far_sample_ids = obtain_farthest_training_samples(args, args.cosin_dist, True, far_sample_representation_tensor, existing_valid_representation, args.cuda)
        all_sample_ids = all_sample_ids[far_sample_ids]
        main_represent_count = valid_count - existing_valid_representation[0].shape[0]
        # if args.no_sample_weights_k_means:
        valid_ids, valid_sample_representation_tensor = cluster_per_class(args, far_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=None, cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None, handles_outlier=False)  
        # else:
        #     if cached_sample_weights is not None:
        #         valid_ids, valid_sample_representation_tensor = cluster_per_class(args, far_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=cached_sample_weights[all_sample_ids], cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None, handles_outlier=False)  
        #     else:
        #         valid_ids, valid_sample_representation_tensor = cluster_per_class(args, far_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=None, cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None, handles_outlier=False)  

        curr_total_valid_count = len(valid_ids) + existing_valid_representation[0].shape[0]
        # if curr_total_valid_count
        if args.total_valid_sample_count > 0 and args.total_valid_sample_count < curr_total_valid_count:
            remaining_valid_ids, remaining_valid_sample_ids, _ = get_uncovered_new_valid_ids2(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, args.total_valid_sample_count- existing_valid_representation[0].shape[0], cosine_dist = args.cosin_dist, all_layer = True, is_cuda = args.cuda)
            valid_ids = remaining_valid_sample_ids
            valid_sample_representation_tensor = [valid_sample_representation_tensor[k][remaining_valid_ids] for k in range(len(valid_sample_representation_tensor))]

        #     selected_sample_ids = sorted_min_sample_ids[0:args.total_valid_sample_count - existing_valid_count]
        # else:
        #     selected_sample_ids = sorted_min_sample_ids[0:valid_count - existing_valid_count]

            # uncovered_valid_ids, uncovered_valid_sample_ids, max_existing_to_valid_dist = get_uncovered_new_valid_ids(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, cluster_max_radius, cosine_dist = args.cosin_dist, all_layer = True, is_cuda = args.cuda)#get_uncovered_new_valid_ids(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, cosine_dist = args.cosin_dist, all_layer = True, is_cuda = args.cuda)
        #     remaining_valid_ids, remaining_local_valid_ids = determine_new_valid_ids(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, len(valid_ids), cosine_dist=args.cosin_dist, all_layer = True, is_cuda = args.cuda)
        #     print()

        # valid_ids = remaining_valid_ids
        # valid_sample_representation_tensor = [valid_sample_representation_tensor[k][remaining_local_valid_ids] for k in range(len(valid_sample_representation_tensor))]

        if qualitiative:
            args.logger.info("obtain D values:")
            full_distance0 = compute_distance(args, args.cosin_dist, True, far_sample_representation_tensor, valid_sample_representation_tensor, args.cuda, inner_prod=True, no_abs=True)
            ratio = torch.abs(torch.sum(full_distance0, dim=1))/torch.sum(torch.abs(full_distance0),dim=1)
            D_value = (ratio + 1)/(1-ratio)
            min_D_value = torch.min(D_value).item()
            quant_val = torch.quantile(D_value, q = 0.001).item()
            final_ratio = (D_value - 1)/(D_value + 1)
            args.logger.info("min D value::%f"%(min_D_value))
            args.logger.info("0.1%% quantile value::%f"%(quant_val))
            # print(final_ratio)

            args.logger.info("obtain D_0,D_1,D_2 values:")
            valid_sample_representation_tensor2 = [full_sample_representation_tensor3[k][valid_ids] for k in range(len(full_sample_representation_tensor2))]
            far_sample_representation_tensor2 = [full_sample_representation_tensor2[k][all_sample_ids] for k in range(len(full_sample_representation_tensor2))]

            full_distance = compute_distance(args, args.cosin_dist, True, far_sample_representation_tensor, valid_sample_representation_tensor, args.cuda)
            full_distance2 = compute_distance(args, args.cosin_dist, True, far_sample_representation_tensor2, valid_sample_representation_tensor2, args.cuda)
            cluster_centroid1 = torch.argmin(full_distance, dim=1)
            cluster_centroid2 = torch.argmin(full_distance2, dim=1)
            matched_cluster_centroid_count = torch.sum(cluster_centroid1 == cluster_centroid2)
            total_sample_count = cluster_centroid1.shape[0]
            args.logger.info("cluster centroid match count %d out of %d::"%(matched_cluster_centroid_count, total_sample_count))
            print()


    if not return_cluster_info:
        return valid_ids, valid_sample_representation_tensor
    else:
        return valid_ids, valid_sample_representation_tensor, full_sample_representation_tensor

    # if args.all_layer_grad_greedy:
    #     # full_sim_mat1 = get_grad_prod(args, full_sample_representation_tensor)
    #     valid_ids = select_samples_with_greedy_algorithm(full_sim_mat1, main_represent_count)
    #     valid_sample_representation_tensor = None
    #     return valid_ids, valid_sample_representation_tensor
    # else:
    #     if only_sample_representation:
    #         return full_sample_representation_tensor



    #     if args.cluster_no_reweighting:
    #         logging.info("no reweighting for k-means")
    #         cached_sample_weights = None

    
    # if cached_sample_weights is not None:
    #     valid_ids, valid_sample_representation_tensor = cluster_per_class(
    #         args,
    #         full_sample_representation_tensor,
    #         all_sample_ids,
    #         valid_count_per_class=main_represent_count,
    #         num_clusters=main_represent_count,
    #         sample_weights=cached_sample_weights[all_sample_ids],
    #         cosin_distance=args.cosin_dist,
    #         is_cuda=args.cuda,
    #         all_layer=args.all_layer | args.all_layer2 | args.use_model_prov,
    #         full_sim_mat=full_sim_mat1,
    #         return_cluster_info=return_cluster_info,
    #     )  
    # else:
    #     valid_ids, valid_sample_representation_tensor = cluster_per_class(
    #         args,
    #         full_sample_representation_tensor,
    #         all_sample_ids,
    #         valid_count_per_class=main_represent_count,
    #         num_clusters=main_represent_count,
    #         sample_weights=None,
    #         cosin_distance=args.cosin_dist,
    #         is_cuda=args.cuda,
    #         all_layer=args.all_layer | args.all_layer2 | args.use_model_prov,
    #         full_sim_mat=full_sim_mat1,
    #         return_cluster_info=return_cluster_info,
    #     )  

    # if not return_cluster_info:
    #     return valid_ids, valid_sample_representation_tensor
    # else:
    #     return valid_ids, valid_sample_representation_tensor, full_sample_representation_tensor


def get_uncovered_new_valid_ids(args, valid_ids, new_valid_representations, existing_valid_representations, cluster_max_radius, cosine_dist = False, all_layer = False, is_cuda = False):
    
    existing_valid_count = 0
    if all_layer:
        existing_valid_count = existing_valid_representations[0].shape[0]
        valid_count = new_valid_representations[0].shape[0]
    else:
        existing_valid_count = existing_valid_representations.shape[0]
        valid_count = new_valid_representations.shape[0]
    if not args.all_layer_grad:

        if not cosine_dist:
            if not all_layer:
                existing_new_dists = pairwise_distance(new_valid_representations, existing_valid_representations, is_cuda=is_cuda)
            else:
                existing_new_dists = pairwise_distance_ls(new_valid_representations, existing_valid_representations , is_cuda=is_cuda)
        else:
            if not all_layer:
                existing_new_dists = pairwise_cosine(new_valid_representations, existing_valid_representations, is_cuda=is_cuda, weight_by_norm = args.weight_by_norm)
            else:
                existing_new_dists = pairwise_cosine_ls(new_valid_representations, existing_valid_representations, is_cuda=is_cuda, weight_by_norm = args.weight_by_norm, inner_prod=args.inner_prod, ls_idx_range=args.origin_X_ls_lenth, full_inner_prod=False)
    
    else:
        existing_new_dists = pairwise_cosine2(new_valid_representations, existing_valid_representations, is_cuda=is_cuda)

    uncovered_new_valid_ids = torch.nonzero(torch.sum(existing_new_dists > cluster_max_radius, dim = 1) >= existing_new_dists.shape[1])
    uncovered_new_valid_ids = uncovered_new_valid_ids.view(-1)
    # nearset_new_valid_distance,nearest_covered_new_valid_ids = torch.min(existing_new_dists, dim = 0)
    # nearest_covered_new_valid_ids = nearest_covered_new_valid_ids.unique()
    # uncovered_new_valid_ids = set(list(range(valid_count))).difference(set(nearest_covered_new_valid_ids.tolist()))
    # uncovered_new_valid_ids = torch.tensor(list(uncovered_new_valid_ids))
    if len(uncovered_new_valid_ids) <= 0:
        return [], [], existing_new_dists
    else:
        return uncovered_new_valid_ids, valid_ids[uncovered_new_valid_ids], existing_new_dists



def get_uncovered_new_valid_ids2(args, valid_ids, new_valid_representations, existing_valid_representations, selected_count, cosine_dist = False, all_layer = False, is_cuda = False):

    if not args.all_layer_grad:

        if not cosine_dist:
            if not all_layer:
                existing_new_dists = pairwise_distance(new_valid_representations, existing_valid_representations, is_cuda=is_cuda)
            else:
                existing_new_dists = pairwise_distance_ls(new_valid_representations, existing_valid_representations , is_cuda=is_cuda)
        else:
            if not all_layer:
                existing_new_dists = pairwise_cosine(new_valid_representations, existing_valid_representations, is_cuda=is_cuda, weight_by_norm = args.weight_by_norm)
            else:
                existing_new_dists = pairwise_cosine_ls(new_valid_representations, existing_valid_representations, is_cuda=is_cuda, weight_by_norm = args.weight_by_norm, inner_prod=args.inner_prod, ls_idx_range=args.origin_X_ls_lenth,full_inner_prod=False)
    
    else:
        existing_new_dists = pairwise_cosine2(new_valid_representations, existing_valid_representations, is_cuda=is_cuda)

    min_new_valid_to_existing_dist = torch.min(existing_new_dists, dim = 1)[0]

    _, sorted_sample_ids = torch.sort(min_new_valid_to_existing_dist, descending=True)

    uncovered_new_valid_ids = sorted_sample_ids[0:selected_count]

    # uncovered_new_valid_ids = torch.nonzero(torch.sum(existing_new_dists > cluster_max_radius, dim = 1) >= existing_new_dists.shape[1])
    # uncovered_new_valid_ids = uncovered_new_valid_ids.view(-1)
    # nearset_new_valid_distance,nearest_covered_new_valid_ids = torch.min(existing_new_dists, dim = 0)
    # nearest_covered_new_valid_ids = nearest_covered_new_valid_ids.unique()
    # uncovered_new_valid_ids = set(list(range(valid_count))).difference(set(nearest_covered_new_valid_ids.tolist()))
    # uncovered_new_valid_ids = torch.tensor(list(uncovered_new_valid_ids))
    if len(uncovered_new_valid_ids) <= 0:
        return [], [], existing_new_dists
    else:
        return uncovered_new_valid_ids, valid_ids[uncovered_new_valid_ids], existing_new_dists

def determine_new_valid_ids(args, valid_ids, new_valid_representations, existing_valid_representations, valid_count, cosine_dist = False, all_layer = False, is_cuda = False):
    existing_valid_count = 0
    if all_layer:
        existing_valid_count = existing_valid_representations[0].shape[0]
    else:
        existing_valid_count = existing_valid_representations.shape[0]

    if existing_valid_count > valid_count:
        return [],[]
    if not args.all_layer_grad:

        if not cosine_dist:
            if not all_layer:
                existing_new_dists = pairwise_distance(existing_valid_representations, new_valid_representations, is_cuda=is_cuda)
            else:
                existing_new_dists = pairwise_distance_ls(existing_valid_representations, new_valid_representations, is_cuda=is_cuda)
        else:
            if not all_layer:
                existing_new_dists = pairwise_cosine(existing_valid_representations, new_valid_representations, is_cuda=is_cuda, weight_by_norm = args.weight_by_norm)
            else:
                existing_new_dists = pairwise_cosine_ls(existing_valid_representations, new_valid_representations, is_cuda=is_cuda, weight_by_norm = args.weight_by_norm, inner_prod=args.inner_prod, ls_idx_range=args.origin_X_ls_lenth, full_inner_prod=False)
    
    else:
        existing_new_dists = pairwise_cosine2(existing_valid_representations, new_valid_representations, is_cuda=is_cuda)


    nearset_new_valid_distance,_ = torch.min(existing_new_dists, dim = 0)

    sorted_min_distance, sorted_min_sample_ids = torch.sort(nearset_new_valid_distance, descending=True)

    if args.total_valid_sample_count > 0 and args.total_valid_sample_count < valid_count:
        selected_sample_ids = sorted_min_sample_ids[0:args.total_valid_sample_count - existing_valid_count]
    else:
        selected_sample_ids = sorted_min_sample_ids[0:valid_count - existing_valid_count]

    remaining_valid_ids = valid_ids[selected_sample_ids]


    return remaining_valid_ids, selected_sample_ids

# def get_uncovered_new_valid_ids2(args, valid_ids, new_valid_representations, existing_valid_representations, cluster_max_radius, cosine_dist = False, all_layer = False, is_cuda = False):
    
#     existing_valid_count = 0
#     if all_layer:
#         existing_valid_count = existing_valid_representations[0].shape[0]
#         valid_count = new_valid_representations[0].shape[0]
#     else:
#         existing_valid_count = existing_valid_representations.shape[0]
#         valid_count = new_valid_representations.shape[0]
#     if not args.all_layer_grad:

#         if not cosine_dist:
#             if not all_layer:
#                 existing_new_dists = pairwise_distance(existing_valid_representations, new_valid_representations, is_cuda=is_cuda)
#             else:
#                 existing_new_dists = pairwise_distance_ls(existing_valid_representations, new_valid_representations, is_cuda=is_cuda)
#         else:
#             if not all_layer:
#                 existing_new_dists = pairwise_cosine(existing_valid_representations, new_valid_representations, is_cuda=is_cuda, weight_by_norm = args.weight_by_norm)
#             else:
#                 existing_new_dists = pairwise_cosine_ls(existing_valid_representations, new_valid_representations, is_cuda=is_cuda, weight_by_norm = args.weight_by_norm)
    
#     else:
#         existing_new_dists = pairwise_cosine2(existing_valid_representations, new_valid_representations, is_cuda=is_cuda)


#     torch.min(existing_new_dists, dim = 0)

#     uncovered_new_valid_ids = torch.nonzero(torch.sum(existing_new_dists > cluster_max_radius, dim = 1) >= existing_new_dists.shape[1])
#     uncovered_new_valid_ids = uncovered_new_valid_ids.view(-1)
#     # nearset_new_valid_distance,nearest_covered_new_valid_ids = torch.min(existing_new_dists, dim = 0)
#     # nearest_covered_new_valid_ids = nearest_covered_new_valid_ids.unique()
#     # uncovered_new_valid_ids = set(list(range(valid_count))).difference(set(nearest_covered_new_valid_ids.tolist()))
#     # uncovered_new_valid_ids = torch.tensor(list(uncovered_new_valid_ids))
#     if len(uncovered_new_valid_ids) <= 0:
#         return [], [], existing_new_dists
#     else:
#         return uncovered_new_valid_ids, valid_ids[uncovered_new_valid_ids], existing_new_dists




def compute_distance(args, cosine_dist, all_layer, full_sample_representation_tensor, valid_sample_representation_tensor, is_cuda, inner_prod=False, no_abs=False, flatten = False):
    if not cosine_dist:
        if not all_layer:
            existing_new_dists = pairwise_distance(full_sample_representation_tensor, valid_sample_representation_tensor, is_cuda=is_cuda)
        else:
            existing_new_dists = pairwise_distance_ls(full_sample_representation_tensor, valid_sample_representation_tensor, is_cuda=is_cuda)
    else:
        if not all_layer:
            existing_new_dists = pairwise_cosine(full_sample_representation_tensor, valid_sample_representation_tensor, is_cuda=is_cuda, weight_by_norm = args.weight_by_norm)
        else:
            existing_new_dists = pairwise_cosine_ls(full_sample_representation_tensor, valid_sample_representation_tensor, is_cuda=is_cuda, weight_by_norm = args.weight_by_norm, inner_prod=inner_prod, ls_idx_range=args.origin_X_ls_lenth, no_abs=no_abs, flatten = flatten)
    return existing_new_dists

def obtain_farthest_training_samples(args, cosine_dist, all_layer, full_sample_representation_tensor, valid_sample_representation_tensor, is_cuda):
    
    existing_new_dists = compute_distance(args, cosine_dist, all_layer, full_sample_representation_tensor, valid_sample_representation_tensor, is_cuda)

    min_dist_per_sample = torch.min(existing_new_dists, dim = 1)[0]

    _, sorted_dist_sample_ids = torch.sort(min_dist_per_sample, descending = True)

    far_sample_count = int(len(sorted_dist_sample_ids)/2)

    far_sample_ids = sorted_dist_sample_ids[0:far_sample_count]

    return [full_sample_representation_tensor[k][far_sample_ids] for k in range(len(full_sample_representation_tensor))], far_sample_ids



def get_representative_valid_ids2_3(train_dataset, criterion, optimizer, train_loader, args, net, valid_count, cached_sample_weights = None, validset = None, existing_valid_set = None, return_cluster_info = False, only_sample_representation = False):

    if args.add_under_rep_samples:
        under_represent_count = int(valid_count/2)
        main_represent_count = valid_count - under_represent_count
    else:
        under_represent_count = 0
        main_represent_count = valid_count


    # sample_representation_vec_ls_by_class = dict()
    # sample_id_ls_by_class = dict()
    full_sim_mat1 = None

    full_sample_representation_tensor, all_sample_ids, existing_valid_representation = get_representations_last_layer2(train_dataset, args, train_loader, criterion, optimizer, net, validset = validset)

    
    if only_sample_representation:
        return full_sample_representation_tensor



    if args.cluster_no_reweighting:
        logging.info("no reweighting for k-means")
        cached_sample_weights = None

    
    if cached_sample_weights is not None:
        valid_ids, valid_sample_representation_tensor = cluster_per_class(args, full_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=cached_sample_weights[all_sample_ids], cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None)  
    else:
        valid_ids, valid_sample_representation_tensor = cluster_per_class(args, full_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=None, cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None)  
    

    far_sample_representation_tensor = full_sample_representation_tensor
    full_dists = compute_distance(args, args.cosin_dist, True, full_sample_representation_tensor, valid_sample_representation_tensor, args.cuda)
    cluster_max_radius = torch.max(torch.min(full_dists, dim = 1)[0])/3*2
    
    while len(valid_ids) < main_represent_count - 1:
        far_sample_representation_tensor, far_sample_ids = obtain_farthest_training_samples(args, args.cosin_dist, True, far_sample_representation_tensor, valid_sample_representation_tensor, args.cuda)
        all_sample_ids = all_sample_ids[far_sample_ids]

        curr_representation_count = valid_count - len(valid_ids)
        if cached_sample_weights is not None:
            new_valid_ids, new_valid_sample_representation_tensor = cluster_per_class(args, far_sample_representation_tensor, all_sample_ids, valid_count_per_class = curr_representation_count, num_clusters = curr_representation_count, sample_weights=cached_sample_weights[far_sample_ids], cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None)  
        else:
            new_valid_ids, new_valid_sample_representation_tensor = cluster_per_class(args, far_sample_representation_tensor, all_sample_ids, valid_count_per_class = curr_representation_count, num_clusters = curr_representation_count, sample_weights=None, cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None)  


        uncovered_valid_ids, uncovered_valid_sample_ids, existing_new_dists = get_uncovered_new_valid_ids(args, new_valid_ids, new_valid_sample_representation_tensor, valid_sample_representation_tensor, cluster_max_radius, cosine_dist = args.cosin_dist, all_layer = True, is_cuda = args.cuda)
        
        args.logger.info("new valid sample count::%d + %d"%(len(uncovered_valid_sample_ids), len(valid_ids)))
        if len(uncovered_valid_sample_ids) > 0:
            uncovered_valid_sample_representations = [new_valid_sample_representation_tensor[k][uncovered_valid_ids] for k in range(len(new_valid_sample_representation_tensor))]
            
            print(torch.norm(full_dists[uncovered_valid_sample_ids] - existing_new_dists[uncovered_valid_ids, 0:full_dists.shape[1]]))
            valid_ids = torch.cat([uncovered_valid_sample_ids, valid_ids])
            valid_sample_representation_tensor = [torch.cat([valid_sample_representation_tensor[k], uncovered_valid_sample_representations[k]]) for k in range(len(valid_sample_representation_tensor))]
    print()
    
    # if existing_valid_representation is not None:
    #     uncovered_valid_ids, uncovered_valid_sample_ids, max_existing_to_valid_dist = get_uncovered_new_valid_ids(args, valid_ids, existing_valid_representation, valid_sample_representation_tensor, cluster_max_radius,  cosine_dist = args.cosin_dist, all_layer = True, is_cuda = args.cuda)
    #     valid_ids = uncovered_valid_sample_ids
    #     valid_sample_representation_tensor = [valid_sample_representation_tensor[k][uncovered_valid_ids] for k in range(len(valid_sample_representation_tensor))]

    # if existing_valid_representation is not None:
    #     uncovered_valid_ids, uncovered_valid_sample_ids, max_existing_to_valid_dist = get_uncovered_new_valid_ids(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, cosine_dist = args.cosin_dist, all_layer = True, is_cuda = args.cuda)
    #     far_sample_representation_tensor = full_sample_representation_tensor
    #     while len(uncovered_valid_ids) <= 0:
            

    #         far_sample_representation_tensor, far_sample_ids = obtain_farthest_training_samples(args, args.cosin_dist, True, far_sample_representation_tensor, existing_valid_representation, args.cuda)
    #         all_sample_ids = all_sample_ids[far_sample_ids]
    #         main_represent_count = valid_count - existing_valid_representation[0].shape[0]
    #         if cached_sample_weights is not None:
    #             valid_ids, valid_sample_representation_tensor = cluster_per_class(args, far_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=cached_sample_weights[all_sample_ids], cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None)  
    #         else:
    #             valid_ids, valid_sample_representation_tensor = cluster_per_class(args, far_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=None, cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None)  

    #         uncovered_valid_ids, uncovered_valid_sample_ids, max_existing_to_valid_dist = get_uncovered_new_valid_ids(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, cosine_dist = args.cosin_dist, all_layer = True, is_cuda = args.cuda)

    #         print()

    #     valid_ids = uncovered_valid_sample_ids
    #     valid_sample_representation_tensor = [valid_sample_representation_tensor[k][uncovered_valid_ids] for k in range(len(valid_sample_representation_tensor))]

    if not return_cluster_info:
        return valid_ids, valid_sample_representation_tensor
    else:
        return valid_ids, valid_sample_representation_tensor, full_sample_representation_tensor


def get_representative_valid_ids2_4(train_dataset, criterion, optimizer, train_loader, args, net, valid_count, cached_sample_weights = None, validset = None, existing_valid_set = None, return_cluster_info = False, only_sample_representation = False):
    
    if args.add_under_rep_samples:
        under_represent_count = int(valid_count/2)
        main_represent_count = valid_count - under_represent_count
    else:
        under_represent_count = 0
        main_represent_count = valid_count


    # sample_representation_vec_ls_by_class = dict()
    # sample_id_ls_by_class = dict()
    full_sim_mat1 = None

    full_sample_representation_tensor, all_sample_ids, existing_valid_representation = get_representations_last_layer2(train_dataset, args, train_loader, criterion, optimizer, net, validset = validset)

    origin_X_ls_lenth = len(full_sample_representation_tensor)

    args.origin_X_ls_lenth = origin_X_ls_lenth
    # if not args.not_rescale_features:
    #     full_sample_representation_tensor = scale_and_extend_data_vector(full_sample_representation_tensor)

    if only_sample_representation:
        return full_sample_representation_tensor, existing_valid_representation



    if args.cluster_no_reweighting:
        logging.info("no reweighting for k-means")
        cached_sample_weights = None
    # extra_valid_ids, extra_valid_sample_representation_tensor = None, None
    if existing_valid_representation is None:
        if cached_sample_weights is not None:
            valid_ids, valid_sample_representation_tensor = cluster_per_class(args, full_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=cached_sample_weights[all_sample_ids], cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None, handles_outlier=False)  
            # if args.inner_prod:
            #     extra_valid_ids, extra_valid_sample_representation_tensor = handle_outliers(args, all_sample_ids, full_sample_representation_tensor, valid_sample_representation_tensor, cached_sample_weights[all_sample_ids])
        else:
            valid_ids, valid_sample_representation_tensor = cluster_per_class(args, full_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=None, cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None, handles_outlier=False)  
        #     if args.inner_prod:
        #         extra_valid_ids, extra_valid_sample_representation_tensor = handle_outliers(args, all_sample_ids, full_sample_representation_tensor, valid_sample_representation_tensor)
        # if extra_valid_ids is not None and extra_valid_sample_representation_tensor is not None:
        #     valid_ids = torch.cat([valid_ids.view(-1), extra_valid_ids.view(-1)])

        #     valid_sample_representation_tensor = [torch.cat([valid_sample_representation_tensor[k], extra_valid_sample_representation_tensor[k]]) for k in range(len(extra_valid_sample_representation_tensor))]
        far_sample_representation_tensor = full_sample_representation_tensor
    # while len(valid_ids) < main_represent_count:
    #     far_sample_representation_tensor, far_sample_ids = obtain_farthest_training_samples(args, args.cosin_dist, True, far_sample_representation_tensor, valid_sample_representation_tensor, args.cuda)
    #     all_sample_ids = all_sample_ids[far_sample_ids]

    #     curr_representation_count = valid_count - len(valid_ids)
    #     if cached_sample_weights is not None:
    #         new_valid_ids, new_valid_sample_representation_tensor = cluster_per_class(args, far_sample_representation_tensor, all_sample_ids, valid_count_per_class = curr_representation_count, num_clusters = curr_representation_count, sample_weights=cached_sample_weights[far_sample_ids], cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None)  
    #     else:
    #         new_valid_ids, new_valid_sample_representation_tensor = cluster_per_class(args, far_sample_representation_tensor, all_sample_ids, valid_count_per_class = curr_representation_count, num_clusters = curr_representation_count, sample_weights=None, cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None)  


    #     uncovered_valid_ids, uncovered_valid_sample_ids, _ = get_uncovered_new_valid_ids(args, new_valid_ids, new_valid_sample_representation_tensor, valid_sample_representation_tensor, cosine_dist = args.cosin_dist, all_layer = True, is_cuda = args.cuda)
        
    #     args.logger.info("new valid sample count::%d"%(len(uncovered_valid_sample_ids)))
    #     if len(uncovered_valid_sample_ids) > 0:
    #         uncovered_valid_sample_representations = [new_valid_sample_representation_tensor[k][uncovered_valid_ids] for k in range(len(new_valid_sample_representation_tensor))]
    #         full_dists = compute_distance(args, args.cosin_dist, True, full_sample_representation_tensor, valid_sample_representation_tensor, args.cuda)
    #         print(full_dists[uncovered_valid_sample_ids])
    #         valid_ids = torch.cat([uncovered_valid_sample_ids, valid_ids])
    #         valid_sample_representation_tensor = [torch.cat([valid_sample_representation_tensor[k], uncovered_valid_sample_representations[k]]) for k in range(len(valid_sample_representation_tensor))]
    # print()
    # if existing_valid_representation is not None:
    #     uncovered_valid_ids, uncovered_valid_sample_ids, max_existing_to_valid_dist = get_uncovered_new_valid_ids(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, cosine_dist = args.cosin_dist, all_layer = True, is_cuda = args.cuda)
    #     valid_ids = uncovered_valid_sample_ids
    #     valid_sample_representation_tensor = [valid_sample_representation_tensor[k][uncovered_valid_ids] for k in range(len(valid_sample_representation_tensor))]
    

    # if existing_valid_representation is not None:
    else:
        # remaining_valid_ids, remaining_local_valid_ids = determine_new_valid_ids(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, len(valid_ids), cosine_dist=args.cosin_dist, all_layer = True, is_cuda = args.cuda)
        # full_dists = compute_distance(args, args.cosin_dist, True, full_sample_representation_tensor, existing_valid_representation, args.cuda)
        # cluster_max_radius = torch.max(torch.min(full_dists, dim = 1)[0])/2
        
        # uncovered_valid_ids, uncovered_valid_sample_ids, max_existing_to_valid_dist = get_uncovered_new_valid_ids(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, cluster_max_radius, cosine_dist = args.cosin_dist, all_layer = True, is_cuda = args.cuda)# get_uncovered_new_valid_ids(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, cosine_dist = args.cosin_dist, all_layer = True, is_cuda = args.cuda)
        far_sample_representation_tensor = full_sample_representation_tensor
        # while len(remaining_valid_ids) <= 0:
            

        far_sample_representation_tensor, far_sample_ids = obtain_farthest_training_samples(args, args.cosin_dist, True, far_sample_representation_tensor, existing_valid_representation, args.cuda)
        all_sample_ids = all_sample_ids[far_sample_ids]
        main_represent_count = valid_count - existing_valid_representation[0].shape[0]
        # if args.no_sample_weights_k_means:
        valid_ids, valid_sample_representation_tensor = cluster_per_class(args, far_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=None, cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None, handles_outlier=False)  
        # else:
        #     if cached_sample_weights is not None:
        #         valid_ids, valid_sample_representation_tensor = cluster_per_class(args, far_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=cached_sample_weights[all_sample_ids], cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None, handles_outlier=False)  
        #     else:
        #         valid_ids, valid_sample_representation_tensor = cluster_per_class(args, far_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=None, cosin_distance=args.cosin_dist, is_cuda=args.cuda, all_layer=True, full_sim_mat=full_sim_mat1, return_cluster_info = return_cluster_info, existing_cluster_centroids = None, handles_outlier=False)  

        curr_total_valid_count = len(valid_ids) + existing_valid_representation[0].shape[0]
        # if curr_total_valid_count
        if args.total_valid_sample_count > 0 and args.total_valid_sample_count < curr_total_valid_count:
            remaining_valid_ids, remaining_valid_sample_ids, _ = get_uncovered_new_valid_ids2(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, args.total_valid_sample_count- existing_valid_representation[0].shape[0], cosine_dist = args.cosin_dist, all_layer = True, is_cuda = args.cuda)
            valid_ids = remaining_valid_sample_ids
            valid_sample_representation_tensor = [valid_sample_representation_tensor[k][remaining_valid_ids] for k in range(len(valid_sample_representation_tensor))]

        #     selected_sample_ids = sorted_min_sample_ids[0:args.total_valid_sample_count - existing_valid_count]
        # else:
        #     selected_sample_ids = sorted_min_sample_ids[0:valid_count - existing_valid_count]

            # uncovered_valid_ids, uncovered_valid_sample_ids, max_existing_to_valid_dist = get_uncovered_new_valid_ids(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, cluster_max_radius, cosine_dist = args.cosin_dist, all_layer = True, is_cuda = args.cuda)#get_uncovered_new_valid_ids(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, cosine_dist = args.cosin_dist, all_layer = True, is_cuda = args.cuda)
        #     remaining_valid_ids, remaining_local_valid_ids = determine_new_valid_ids(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, len(valid_ids), cosine_dist=args.cosin_dist, all_layer = True, is_cuda = args.cuda)
        #     print()

        # valid_ids = remaining_valid_ids
        # valid_sample_representation_tensor = [valid_sample_representation_tensor[k][remaining_local_valid_ids] for k in range(len(valid_sample_representation_tensor))]

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
            X=sample_representation_vec_tensor, num_clusters=num_clusters, distance='euclidean', is_cuda=args.cuda, sample_weights=sample_weights, all_layer=args.all_layer, rand_init=args.rand_init)
    else:
        cluster_ids_x, cluster_centers = kmeans(
            X=sample_representation_vec_tensor, num_clusters=num_clusters, distance='cosine', is_cuda=args.cuda, sample_weights=sample_weights, all_layer=args.all_layer, rand_init=args.rand_init)


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


