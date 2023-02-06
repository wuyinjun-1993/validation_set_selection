import torch
import numpy as np
# from kmeans_pytorch import kmeans


import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from exp_datasets.mnist import *
from common.utils import *
from clustering_method.k_means import *
from sklearn import metrics
from main.model_gradient_op import *

def do_cluster_both(args, num_clusters, sample_representation_vec_ls, is_cuda, cosin_distance, sample_weights, all_layer = False):
    # if args.full_model_out:
    #     dist_metric = 'cross'
    if cosin_distance:
        dist_metric = 'cosine'
    else:
        dist_metric = 'euclidean'

    cluster_ids_x, cluster_centers = kmeans(
        X=sample_representation_vec_ls,
        num_clusters=num_clusters,
        distance=dist_metric,
        is_cuda=is_cuda,
        sample_weights=sample_weights,
        all_layer=all_layer,
        weight_by_norm=args.weight_by_norm,
        origin_X_ls_lenth = args.origin_X_ls_lenth,

    )
    return cluster_ids_x, cluster_centers


def cluster_per_class_both(
    args,
    sample_representation_vec_ls,
    sample_id_ls,
    valid_count_per_class=10,
    num_clusters=4,
    sample_weights=None,
    cosin_distance=True,
    is_cuda=False,
    all_layer=False,
    return_cluster_info=False,
):
    assert num_clusters > 0
    # if num_clusters > 0:
    cluster_ids_x, cluster_centers = do_cluster_both(
        args,
        num_clusters,
        sample_representation_vec_ls,
        is_cuda,
        cosin_distance,
        sample_weights,
        all_layer=all_layer,
    )

    unique_cluster_count = len(cluster_ids_x.unique())
    args.logger.info("cluster count before and after:(%d,%d)"%(num_clusters, unique_cluster_count))
    # if args.remove_empty_clusters:
    if unique_cluster_count < num_clusters:
        while(True):
            cluster_ids_x, cluster_centers = do_cluster_both(
                args,
                unique_cluster_count,
                sample_representation_vec_ls,
                is_cuda,
                cosin_distance,
                sample_weights,
                all_layer=all_layer,
            )

            new_unique_cluster_count = len(cluster_ids_x.unique())
            args.logger.info("cluster count before and after:(%d,%d)"%(unique_cluster_count, new_unique_cluster_count))

            if new_unique_cluster_count >= unique_cluster_count:
                break
            unique_cluster_count = new_unique_cluster_count

        unique_cluster_count = new_unique_cluster_count


    if return_cluster_info:
        return cluster_ids_x, cluster_centers


    

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

    # if not args.all_layer_grad:
    if is_cuda:
        if not all_layer:
            cluster_centers = cluster_centers.cuda()
        else:
            for idx in range(len(cluster_centers)):
                cluster_centers[idx] = cluster_centers[idx].cuda()

    # full_representative_representation_ls = None
    # dist_to_cluster_centroid_ls = []
    # dist_to_other_cluster_centroid_ls = []
    args.logger.info("unique cluster count::%d"%(unique_cluster_count))

    # for cluster_id in range(num_clusters):
    for cluster_id in list(cluster_ids_x.unique()):
        
        if not all_layer:        

            curr_cluster_sample_representation = sample_representation_vec_ls

            if is_cuda:
                curr_cluster_sample_representation = curr_cluster_sample_representation.cuda()
            
            cluster_dist_ls_tensor = pairwise_distance_function(
                curr_cluster_sample_representation,
                cluster_centers[cluster_id].view(1,-1),
                is_cuda=is_cuda,
                weight_by_norm=args.weight_by_norm,
                # inner_prod=args.inner_prod,
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
            # data1_ls, data2_ls, is_cuda=False,  batch_size = 32, ls_idx_range=-1, weight_by_norm=False, flatten = False
            cluster_dist_ls_tensor = pairwise_distance_function(
                curr_cluster_sample_representation_ls,
                curr_cluster_center_ls,
                is_cuda=is_cuda,
                weight_by_norm=args.weight_by_norm,
                # inner_prod=args.inner_prod,
                ls_idx_range=args.origin_X_ls_lenth,
                # full_inner_prod=False
            )

            # if args.weight_by_norm:
            #     cluster_dist_ls_tensor = rescale_dist_by_cluster_mean_norm(cluster_dist_ls_tensor, curr_cluster_center_ls,all_layer)

            cluster_dist_ls_tensor = cluster_dist_ls_tensor.view(-1).cpu()

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

        return valid_ids, valid_sample_representation_tensor


def get_grad_by_example_per_batch_gbc(args, labels, output, net, criterion, optimizer, vec_grad_by_example_ls, sampled_net_param_layer_ls, sampled_layer_sqrt_prob_ls, merge_grad = False):
    for idx in range(labels.shape[0]):
        # args, output, net, optimizer, target, criterion
        # obtain_class_wise_grad_ratio_wrt_full_grad(args, output[idx:idx+1], net, optimizer, labels[idx:idx+1], criterion)
        optimizer.zero_grad()
        loss = criterion(output[idx:idx+1], labels[idx:idx+1])
        if not args.label_aware:
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

def obtain_representations_given_model_rbc(args, train_loader, net, criterion, optimizer, sampled_col_ids = None, origin_label = None):
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
        # if not args.cluster_method_two_plus:
        #     sample_representation = net.feature_forward(data, all_layer=False)
        # else:
            # sample_representation = F.softmax(net.forward(data),dim=1)
        sample_representation = net.feature_forward2(data, label_aware=args.label_aware, labels = labels)

        # if not args.all_layer and not args.all_layer2:
        sample_representation_vec_ls.append(sample_representation.detach().cpu())

        
        sample_id_ls.append(sample_ids)
    # if args.all_layer or args.all_layer2:
    #     print_norm_range_of_representations(args, sample_representation_vec_ls)
    #     return sample_representation_vec_ls, sample_id_ls
    # else:
    sample_representation_vec_ls = torch.cat(sample_representation_vec_ls)
         
    print_norm_range_of_representations(args, sample_representation_vec_ls)
    return sample_representation_vec_ls, sample_id_ls, sampled_col_ids
        

def obtain_norms_for_each_layer_gbc(args, train_dataset, net, criterion, optimizer):
    # train_loader = 
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    net_param_count_ls = obtain_net_param_count_ls(net)
    full_net_grad_norm_ls = 0
    for batch_id, (sample_ids, data, labels) in tqdm(enumerate(train_loader)):

        if args.cuda:
            data, labels = train_loader.dataset.to_cuda(data, labels)

        output = net.forward(data)

        loss = criterion(output, labels)
        if not args.label_aware:
            loss = obtain_full_loss(output, labels, args.cuda, loss)
        loss.backward()
        net_grad_ls = obtain_net_grad(net)
        net_grad_norm_ls = compute_net_grad_norm_ls(net_grad_ls)
        full_net_grad_norm_ls += net_grad_norm_ls*labels.shape[0]
    full_net_grad_norm_ls = full_net_grad_norm_ls/len(train_loader.dataset)
    return full_net_grad_norm_ls, net_param_count_ls
        
def obtain_representations_for_validset_gbc(valid_set, args, net, criterion, optimizer, sampled_net_param_layer_ls, sampled_layer_sqrt_prob_ls, sampled_col_ids_ls = None):
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
        vec_grad_by_example_ls = get_grad_by_example_per_batch_gbc(args, labels, output, net, criterion, optimizer, vec_grad_by_example_ls, sampled_net_param_layer_ls, sampled_layer_sqrt_prob_ls, merge_grad = True)

        if batch_id == 0:
            # sample_representation_vec_ls.append(sample_representation.detach().cpu())
            sample_representation_vec_ls.extend(vec_grad_by_example_ls)
        else:

            # sample_representation_vec_ls[0] = torch.cat([sample_representation_vec_ls[0], sample_representation.detach().cpu()])
            for sample_arr_id in range(len(sample_representation_vec_ls)):
                sample_representation_vec_ls[sample_arr_id] = torch.cat([sample_representation_vec_ls[sample_arr_id].detach().cpu(), vec_grad_by_example_ls[sample_arr_id].detach().cpu()])


    return sample_representation_vec_ls

def obtain_representations_given_model_gbc(train_dataset, args, train_loader, net, criterion, optimizer, validset = None, sampled_col_ids_ls = None):
    sample_representation_vec_ls = []

    sample_id_ls = []
    # with torch.no_grad():

        # all_sample_representations = [None]*len(train_loader.dataset)
    grad_norm_by_layer_ls, net_param_count_ls = obtain_norms_for_each_layer_gbc(args, train_dataset, net, criterion, optimizer)
    avg_grad_norm_by_layer = grad_norm_by_layer_ls/torch.tensor(net_param_count_ls)
    sampled_net_param_layer_ls,sampled_layer_sqrt_prob_ls = biased_rand_sample_parameter(net, avg_grad_norm_by_layer, replace=True)#args.replace)

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
        vec_grad_by_example_ls = get_grad_by_example_per_batch_gbc(args, labels, output, net, criterion, optimizer, vec_grad_by_example_ls, sampled_net_param_layer_ls, sampled_layer_sqrt_prob_ls, merge_grad = True)

        if batch_id == 0:
            # sample_representation_vec_ls.append(sample_representation.detach().cpu())
            sample_representation_vec_ls.extend(vec_grad_by_example_ls)
        else:

            # sample_representation_vec_ls[0] = torch.cat([sample_representation_vec_ls[0], sample_representation.detach().cpu()])
            for sample_arr_id in range(len(sample_representation_vec_ls)):
                sample_representation_vec_ls[sample_arr_id] = torch.cat([sample_representation_vec_ls[sample_arr_id].detach().cpu(), vec_grad_by_example_ls[sample_arr_id].detach().cpu()])

        sample_id_ls.append(sample_ids)
    # if args.all_layer or args.all_layer2:
    print_norm_range_of_representations(args, sample_representation_vec_ls)
    valid_sample_representation_ls = None

    # if args.cluster_method_three_sampling:
    #     sample_representation_vec_ls, sampled_col_ids_ls =  obtain_sampled_representations_cluster_method_three(sample_representation_vec_ls, args, sampled_col_ids_ls = sampled_col_ids_ls)

    if validset is not None:
        valid_sample_representation_ls = obtain_representations_for_validset_gbc(validset, args, net, criterion, optimizer, sampled_net_param_layer_ls, sampled_layer_sqrt_prob_ls, sampled_col_ids_ls)
        # if args.cluster_method_three_sampling:
        #     valid_sample_representation_ls, _ =  obtain_sampled_representations_cluster_method_three(valid_sample_representation_ls, args, sampled_col_ids_ls = sampled_col_ids_ls)

    return sample_representation_vec_ls, sample_id_ls, valid_sample_representation_ls
 
def get_extra_representations_given_model_rbc(args, optimizer, train_loader, criterion, net, full_sample_representation_vec_ls, valid_sample_representation_vec_ls, validloader = None, full_sample_representation_vec_ls2 = None, valid_sample_representation_vec_ls2 = None, qualitiative = False, origin_label = None):
    
    start_epoch_id = 0
    if args.use_pretrained_model:
        start_epoch_id = torch.load(os.path.join(args.prev_save_path, "early_stopping_epoch"))
    args.logger.info("extra representation starting from epoch %d"%(start_epoch_id))

        # start_epoch_id = int(args.epochs/2)
    if not args.all_layer:# and not args.all_layer2:
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
        # optimizer, _=obtain_optimizer_scheduler(args, net, start_epoch = 0)
        sample_representation_vec_ls, _,sampled_col_ids = obtain_representations_given_model_rbc(args, train_loader, net, criterion, optimizer)
        if qualitiative:
            args.label_aware = True
            sample_representation_vec_ls2, _,_ = obtain_representations_given_model_rbc(args, train_loader, net, criterion, optimizer, sampled_col_ids=sampled_col_ids)
            if origin_label is not None:
                sample_representation_vec_ls3, _,_ = obtain_representations_given_model_rbc(args, train_loader, net, criterion, optimizer, sampled_col_ids=sampled_col_ids, origin_label = origin_label)
            args.label_aware = False

        if validloader is not None:
            curr_valid_sample_representation_vec_ls, _,_ = obtain_representations_given_model_rbc(args, validloader, net, criterion, optimizer, sampled_col_ids=sampled_col_ids)
            if qualitiative:
                args.label_aware = True
                curr_valid_sample_representation_vec_ls2, _,_ = obtain_representations_given_model_rbc(args, validloader, net, criterion, optimizer, sampled_col_ids=sampled_col_ids)
                args.label_aware = False

        if args.all_layer:# or args.all_layer2:
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

def get_extra_representations_given_model_gbc(train_dataset, args, train_loader, criterion, net, full_sample_representation_vec_ls, validset = None, full_valid_sample_representation_vec_ls = None):
    
    start_epoch_id = 0
    if args.use_pretrained_model:
        start_epoch_id = torch.load(os.path.join(args.prev_save_path, "early_stopping_epoch"))
        # start_epoch_id += 1
    args.logger.info("extra representation starting from epoch %d"%(start_epoch_id))

        # start_epoch_id = int(args.epochs/2)
    if not args.all_layer:# and not args.all_layer2:
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
        sample_representation_vec_ls, _, curr_valid_sample_representation_ls  = obtain_representations_given_model_gbc(train_dataset, args, train_loader, net, criterion, optimizer, validset = validset)

        if args.all_layer:# or args.all_layer2:
            full_sample_representation_vec_ls.extend(sample_representation_vec_ls)
            if full_valid_sample_representation_vec_ls is not None:
                full_valid_sample_representation_vec_ls.extend(curr_valid_sample_representation_ls)
        else:
            full_sample_representation_vec_ls.append(sample_representation_vec_ls)
            if full_valid_sample_representation_vec_ls is not None:
                full_valid_sample_representation_vec_ls.append(curr_valid_sample_representation_ls)

        epoch_count += 1

    return full_sample_representation_vec_ls, full_valid_sample_representation_vec_ls



def get_all_sample_representations_rbc(args, train_loader, criterion, optimizer, net, validset = None, qualitiative = False, origin_label = None):

    sample_representation_vec_ls, sample_id_ls, sampled_col_ids = obtain_representations_given_model_rbc(args, train_loader, net, criterion, optimizer)
    sample_representation_vec_ls2 = None
    sample_representation_vec_ls3 = None
    valid_sample_representation_vec_ls2 = None
    if validset is not None:
        validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size, shuffle=True)
        valid_sample_representation_vec_ls, _, _ = obtain_representations_given_model_rbc(args, validloader, net, criterion, optimizer, sampled_col_ids=sampled_col_ids)
        # if qualitiative:
        #     args.label_aware = True
        #     valid_sample_representation_vec_ls2, _, _ = obtain_representations_last_layer_given_model(args, validloader, net, criterion, optimizer, sampled_col_ids=sampled_col_ids)
        #     args.label_aware = False
    else:
        validloader = None
        valid_sample_representation_vec_ls = None

    

    # if args.use_model_prov:
    # if not qualitiative:
    sample_representation_vec_ls, valid_sample_representation_vec_ls = get_extra_representations_given_model_rbc(args, optimizer, train_loader, criterion, net, sample_representation_vec_ls, valid_sample_representation_vec_ls, validloader = validloader, full_sample_representation_vec_ls2 = sample_representation_vec_ls2, valid_sample_representation_vec_ls2 = valid_sample_representation_vec_ls2, qualitiative = qualitiative)
 
    full_sample_representation_tensor = sample_representation_vec_ls

    all_sample_ids = torch.cat(sample_id_ls)
    return full_sample_representation_tensor, all_sample_ids, valid_sample_representation_vec_ls

def get_all_sample_representations_gbc(train_dataset, args, train_loader, criterion, optimizer, net, validset = None):

    sample_representation_vec_ls, sample_id_ls, valid_sample_representation_ls = obtain_representations_given_model_gbc(train_dataset, args, train_loader, net, criterion, optimizer, validset = validset)
    # if args.use_model_prov:
    args.all_layer = True
    sample_representation_vec_ls, valid_sample_representation_ls = get_extra_representations_given_model_gbc(train_dataset, args, train_loader, criterion, net, sample_representation_vec_ls, validset, valid_sample_representation_ls)


    full_sample_representation_tensor = sample_representation_vec_ls

    all_sample_ids = torch.cat(sample_id_ls)

    return full_sample_representation_tensor, all_sample_ids, valid_sample_representation_ls

def get_representative_valid_ids_rbc(criterion, optimizer, train_loader, args, net, valid_count, cached_sample_weights = None, return_cluster_info = False, only_sample_representation = False, validset=None, qualitiative = False, origin_label = None):

    main_represent_count = valid_count

    full_sim_mat1 = None
    # if not args.all_layer_grad:
    full_sample_representation_tensor, all_sample_ids, existing_valid_representation = get_all_sample_representations_rbc(args, train_loader, criterion, optimizer, net, validset = validset, qualitiative=qualitiative, origin_label = origin_label)

    origin_X_ls_lenth = len(full_sample_representation_tensor)

    # if args.get_representations:
    #     torch.save(full_sample_representation_tensor, os.path.join(args.save_path,
    #         "sample_representation"))

    args.origin_X_ls_lenth = origin_X_ls_lenth

    if only_sample_representation:
        return full_sample_representation_tensor, existing_valid_representation
    # sample_representation_vec_ls = sample_representation_vec_ls_by_class[label]
    # if args.cluster_no_reweighting:
    #     logging.info("no reweighting for k-means")
    #     cached_sample_weights = None
    # extra_valid_ids, extra_valid_sample_representation_tensor = None, None
    far_sample_representation_tensor = full_sample_representation_tensor
    global_far_sample_ids = all_sample_ids
    all_valid_ids = None
    
    # if existing_valid_representation is None:
    all_valid_sample_representation_tensor = []
    
    cluster_count = main_represent_count
    while True:
        
        if all_valid_ids is not None:
            cluster_count = cluster_count - len(all_valid_ids)
        
        if cached_sample_weights is not None:
            valid_ids, valid_sample_representation_tensor = cluster_per_class_both(args, far_sample_representation_tensor, global_far_sample_ids, valid_count_per_class = cluster_count, num_clusters = cluster_count, sample_weights=cached_sample_weights[all_sample_ids], cosin_distance=True, is_cuda=args.cuda, all_layer=True, return_cluster_info = return_cluster_info)  
            # if args.inner_prod:
            #     extra_valid_ids, extra_valid_sample_representation_tensor = handle_outliers(args, all_sample_ids, full_sample_representation_tensor, valid_sample_representation_tensor, cached_sample_weights[all_sample_ids])
        else:
            valid_ids, valid_sample_representation_tensor = cluster_per_class_both(args, far_sample_representation_tensor, global_far_sample_ids, valid_count_per_class = cluster_count, num_clusters = cluster_count, sample_weights=None, cosin_distance=True, is_cuda=args.cuda, all_layer=True, return_cluster_info = return_cluster_info)  
        
        if all_valid_ids is None:            
            all_valid_ids = valid_ids
        else:
            all_valid_ids = torch.cat([all_valid_ids, valid_ids])
        
        if len(all_valid_sample_representation_tensor) <= 0:
            all_valid_sample_representation_tensor.extend(valid_sample_representation_tensor)
        else:
            for k in range(len(all_valid_sample_representation_tensor)):
                all_valid_sample_representation_tensor[k] = torch.cat([all_valid_sample_representation_tensor[k], valid_sample_representation_tensor[k]])
        
        if len(all_valid_ids) >= args.total_valid_sample_count:
            break
        assert torch.norm(torch.stack([full_sample_representation_tensor[0][id] for id in all_valid_ids]) - all_valid_sample_representation_tensor[0]) <= 0

        far_sample_representation_tensor, far_sample_ids = obtain_farthest_training_samples_both(args, True, True, far_sample_representation_tensor, all_valid_sample_representation_tensor, args.cuda)

        if global_far_sample_ids is None:
            global_far_sample_ids = far_sample_ids
        else:
            global_far_sample_ids = global_far_sample_ids[far_sample_ids]
        
        
    
    valid_sample_representation_tensor = all_valid_sample_representation_tensor
    valid_ids = all_valid_ids
    # if args.total_valid_sample_count > 0 and args.total_valid_sample_count < len(valid_ids):
    #     remaining_valid_ids, remaining_valid_sample_ids, _ = get_uncovered_new_valid_ids_both(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, args.total_valid_sample_count- existing_valid_representation[0].shape[0], cosine_dist = True, all_layer = True, is_cuda = args.cuda)
    #     valid_ids = remaining_valid_sample_ids
    #     valid_sample_representation_tensor = [valid_sample_representation_tensor[k][remaining_valid_ids] for k in range(len(valid_sample_representation_tensor))]


    # else:
    #     far_sample_representation_tensor = full_sample_representation_tensor
            

    #     far_sample_representation_tensor, far_sample_ids = obtain_farthest_training_samples_both(args, True, True, far_sample_representation_tensor, existing_valid_representation, args.cuda)
    #     all_sample_ids = all_sample_ids[far_sample_ids]
    #     main_represent_count = valid_count - existing_valid_representation[0].shape[0]
    #     # if args.no_sample_weights_k_means:
    #     valid_ids, valid_sample_representation_tensor = cluster_per_class_both(args, far_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=None, cosin_distance=True, is_cuda=args.cuda, all_layer=True, return_cluster_info = return_cluster_info)  

    #     curr_total_valid_count = len(valid_ids) + existing_valid_representation[0].shape[0]
    #     # if curr_total_valid_count
    #     if args.total_valid_sample_count > 0 and args.total_valid_sample_count < curr_total_valid_count:
    #         remaining_valid_ids, remaining_valid_sample_ids, _ = get_uncovered_new_valid_ids_both(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, args.total_valid_sample_count- existing_valid_representation[0].shape[0], cosine_dist = True, all_layer = True, is_cuda = args.cuda)
    #         valid_ids = remaining_valid_sample_ids
    #         valid_sample_representation_tensor = [valid_sample_representation_tensor[k][remaining_valid_ids] for k in range(len(valid_sample_representation_tensor))]

    if not return_cluster_info:
        return valid_ids, valid_sample_representation_tensor
    else:
        return valid_ids, valid_sample_representation_tensor, full_sample_representation_tensor


def get_uncovered_new_valid_ids_both(args, valid_ids, new_valid_representations, existing_valid_representations, selected_count, cosine_dist = False, all_layer = False, is_cuda = False):

    # if not args.all_layer_grad:

    if not cosine_dist:
        if not all_layer:
            existing_new_dists = pairwise_distance(new_valid_representations, existing_valid_representations, is_cuda=is_cuda)
        else:
            existing_new_dists = pairwise_distance_ls(new_valid_representations, existing_valid_representations , is_cuda=is_cuda)
    else:
        if not all_layer:
            existing_new_dists = pairwise_cosine(new_valid_representations, existing_valid_representations, is_cuda=is_cuda, weight_by_norm = args.weight_by_norm)
        else:
            existing_new_dists = pairwise_cosine_ls(new_valid_representations, existing_valid_representations, is_cuda=is_cuda, weight_by_norm = args.weight_by_norm, ls_idx_range=args.origin_X_ls_lenth)
    
    # else:
    #     existing_new_dists = pairwise_cosine2(new_valid_representations, existing_valid_representations, is_cuda=is_cuda)

    min_new_valid_to_existing_dist = torch.min(existing_new_dists, dim = 1)[0]

    _, sorted_sample_ids = torch.sort(min_new_valid_to_existing_dist, descending=True)

    uncovered_new_valid_ids = sorted_sample_ids[0:selected_count]

    if len(uncovered_new_valid_ids) <= 0:
        return [], [], existing_new_dists
    else:
        return uncovered_new_valid_ids, valid_ids[uncovered_new_valid_ids], existing_new_dists


def compute_distance_both(args, cosine_dist, all_layer, full_sample_representation_tensor, valid_sample_representation_tensor, is_cuda, flatten = False):
    if not cosine_dist:
        if not all_layer:
            existing_new_dists = pairwise_distance(full_sample_representation_tensor, valid_sample_representation_tensor, is_cuda=is_cuda)
        else:
            existing_new_dists = pairwise_distance_ls(full_sample_representation_tensor, valid_sample_representation_tensor, is_cuda=is_cuda)
    else:
        if not all_layer:
            existing_new_dists = pairwise_cosine(full_sample_representation_tensor, valid_sample_representation_tensor, is_cuda=is_cuda, weight_by_norm = args.weight_by_norm)
        else:
            existing_new_dists = pairwise_cosine_ls(full_sample_representation_tensor, valid_sample_representation_tensor, is_cuda=is_cuda, weight_by_norm = args.weight_by_norm, ls_idx_range=args.origin_X_ls_lenth, flatten = flatten)
    return existing_new_dists

def obtain_farthest_training_samples_both(args, cosine_dist, all_layer, full_sample_representation_tensor, valid_sample_representation_tensor, is_cuda):
    
    existing_new_dists = compute_distance_both(args, cosine_dist, all_layer, full_sample_representation_tensor, valid_sample_representation_tensor, is_cuda)

    existing_new_dists = existing_new_dists.cpu()

    min_dist_per_sample = torch.min(existing_new_dists, dim = 1)[0]

    _, sorted_dist_sample_ids = torch.sort(min_dist_per_sample, descending = True)

    far_sample_count = int(len(sorted_dist_sample_ids)/2)

    far_sample_ids = sorted_dist_sample_ids[0:far_sample_count]

    return [full_sample_representation_tensor[k][far_sample_ids] for k in range(len(full_sample_representation_tensor))], far_sample_ids


def get_representative_valid_ids_gbc(train_dataset, criterion, optimizer, train_loader, args, net, valid_count, cached_sample_weights = None, validset = None, existing_valid_set = None, return_cluster_info = False, only_sample_representation = False):
    
    main_represent_count = valid_count

    full_sim_mat1 = None

    full_sample_representation_tensor, all_sample_ids, existing_valid_representation = get_all_sample_representations_gbc(train_dataset, args, train_loader, criterion, optimizer, net, validset = validset)

    origin_X_ls_lenth = len(full_sample_representation_tensor)

    args.origin_X_ls_lenth = origin_X_ls_lenth

    if only_sample_representation:
        return full_sample_representation_tensor, existing_valid_representation



    # if args.cluster_no_reweighting:
    #     logging.info("no reweighting for k-means")
    #     cached_sample_weights = None
    # extra_valid_ids, extra_valid_sample_representation_tensor = None, None
    
    
    far_sample_representation_tensor = full_sample_representation_tensor
    global_far_sample_ids = all_sample_ids
    all_valid_ids = None
    
    # if existing_valid_representation is None:
    all_valid_sample_representation_tensor = []
    
    cluster_count = main_represent_count
    
    while True:
        
        if all_valid_ids is not None:
            cluster_count = cluster_count - len(all_valid_ids)
        
        if cached_sample_weights is not None:
            valid_ids, valid_sample_representation_tensor = cluster_per_class_both(args, far_sample_representation_tensor, global_far_sample_ids, valid_count_per_class = cluster_count, num_clusters = cluster_count, sample_weights=cached_sample_weights[all_sample_ids], cosin_distance=True, is_cuda=args.cuda, all_layer=True, return_cluster_info = return_cluster_info)  
            # if args.inner_prod:
            #     extra_valid_ids, extra_valid_sample_representation_tensor = handle_outliers(args, all_sample_ids, full_sample_representation_tensor, valid_sample_representation_tensor, cached_sample_weights[all_sample_ids])
        else:
            valid_ids, valid_sample_representation_tensor = cluster_per_class_both(args, far_sample_representation_tensor, global_far_sample_ids, valid_count_per_class = cluster_count, num_clusters = cluster_count, sample_weights=None, cosin_distance=True, is_cuda=args.cuda, all_layer=True, return_cluster_info = return_cluster_info)  
        
        if all_valid_ids is None:            
            all_valid_ids = valid_ids
        else:
            all_valid_ids = torch.cat([all_valid_ids, valid_ids])
        
        if len(all_valid_sample_representation_tensor) <= 0:
            all_valid_sample_representation_tensor.extend(valid_sample_representation_tensor)
        else:
            for k in range(len(all_valid_sample_representation_tensor)):
                all_valid_sample_representation_tensor[k] = torch.cat([all_valid_sample_representation_tensor[k], valid_sample_representation_tensor[k]])
        
        if len(all_valid_ids) >= args.total_valid_sample_count:
            break
        assert torch.norm(torch.stack([full_sample_representation_tensor[0][id] for id in all_valid_ids]) - all_valid_sample_representation_tensor[0]) <= 0

        far_sample_representation_tensor, far_sample_ids = obtain_farthest_training_samples_both(args, True, True, far_sample_representation_tensor, all_valid_sample_representation_tensor, args.cuda)

        if global_far_sample_ids is None:
            global_far_sample_ids = far_sample_ids
        else:
            global_far_sample_ids = global_far_sample_ids[far_sample_ids]
        
        
    
    valid_sample_representation_tensor = all_valid_sample_representation_tensor
    valid_ids = all_valid_ids
    # if args.total_valid_sample_count > 0 and args.total_valid_sample_count < len(valid_ids):
    #     remaining_valid_ids, remaining_valid_sample_ids, _ = get_uncovered_new_valid_ids_both(args, valid_ids, valid_sample_representation_tensor, None, args.total_valid_sample_count, cosine_dist = True, all_layer = True, is_cuda = args.cuda)
    #     valid_ids = remaining_valid_sample_ids
    #     valid_sample_representation_tensor = [valid_sample_representation_tensor[k][remaining_valid_ids] for k in range(len(valid_sample_representation_tensor))]

    # if existing_valid_representation is None:
    #     if cached_sample_weights is not None:
    #         valid_ids, valid_sample_representation_tensor = cluster_per_class_both(args, full_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=cached_sample_weights[all_sample_ids], cosin_distance=True, is_cuda=args.cuda, all_layer=True, return_cluster_info = return_cluster_info)  
    #         # if args.inner_prod:
    #         #     extra_valid_ids, extra_valid_sample_representation_tensor = handle_outliers(args, all_sample_ids, full_sample_representation_tensor, valid_sample_representation_tensor, cached_sample_weights[all_sample_ids])
    #     else:
    #         valid_ids, valid_sample_representation_tensor = cluster_per_class_both(args, full_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=None, cosin_distance=True, is_cuda=args.cuda, all_layer=True, return_cluster_info = return_cluster_info)  
    #     far_sample_representation_tensor = full_sample_representation_tensor
    # else:
    #     far_sample_representation_tensor = full_sample_representation_tensor
    #     # while len(remaining_valid_ids) <= 0:
            

    #     far_sample_representation_tensor, far_sample_ids = obtain_farthest_training_samples_both(args, True, True, far_sample_representation_tensor, existing_valid_representation, args.cuda)
    #     all_sample_ids = all_sample_ids[far_sample_ids]
    #     main_represent_count = valid_count - existing_valid_representation[0].shape[0]
    #     # if args.no_sample_weights_k_means:
    #     valid_ids, valid_sample_representation_tensor = cluster_per_class_both(args, far_sample_representation_tensor, all_sample_ids, valid_count_per_class = main_represent_count, num_clusters = main_represent_count, sample_weights=None, cosin_distance=True, is_cuda=args.cuda, all_layer=True, return_cluster_info = return_cluster_info)  
 
    #     curr_total_valid_count = len(valid_ids) + existing_valid_representation[0].shape[0]
    #     # if curr_total_valid_count
    #     if args.total_valid_sample_count > 0 and args.total_valid_sample_count < curr_total_valid_count:
    #         remaining_valid_ids, remaining_valid_sample_ids, _ = get_uncovered_new_valid_ids_both(args, valid_ids, valid_sample_representation_tensor, existing_valid_representation, args.total_valid_sample_count- existing_valid_representation[0].shape[0], cosine_dist = True, all_layer = True, is_cuda = args.cuda)
    #         valid_ids = remaining_valid_sample_ids
    #         valid_sample_representation_tensor = [valid_sample_representation_tensor[k][remaining_valid_ids] for k in range(len(valid_sample_representation_tensor))]

    if not return_cluster_info:
        return valid_ids, valid_sample_representation_tensor
    else:
        return valid_ids, valid_sample_representation_tensor, full_sample_representation_tensor
