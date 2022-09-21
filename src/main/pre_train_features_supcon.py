import torch


import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from common.parse_args import *
from exp_datasets.mnist import *
from models.DNN import *
from common.utils import *
from tqdm.notebook import tqdm
import itertools
import torch_higher as higher
from main.find_valid_set import *
from main.meta_reweighting_rl import *
from exp_datasets.dataloader import *
import torch.distributed as dist
import json
from utils.logger import setup_logger
import models

def calculate_entity_embedding_sim0_sample_wise(cl_temp, features, device, labels=None, mask=None, pos_in_deno = False, cl_loss_sqr = False, sample_weights = None):
    """Compute loss for model. If both `labels` and `mask` are None,
    it degenerates to SimCLR unsupervised loss:
    https://arxiv.org/pdf/2002.05709.pdf

    Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
            has the same class as sample i. Can be asymmetric.
    Returns:
        A loss scalar.
    """
    # device = (torch.device('cuda')
    #           if features.is_cuda
    #           else torch.device('cpu'))

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    # if self.contrast_mode == 'one':
    anchor_feature = features[:, 0]
    anchor_count = 1
    # elif self.contrast_mode == 'all':
    #     anchor_feature = contrast_feature
    #     anchor_count = contrast_count
    # else:
    #     raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

    # compute logits

    anchor_feature = anchor_feature/torch.norm(anchor_feature, dim=1).view(-1,1)
    contrast_feature = contrast_feature/torch.norm(contrast_feature, dim = 1).view(-1,1)
    anchor_dot_contrast = torch.div(
        torch.matmul(anchor_feature, contrast_feature.T),
        cl_temp,
    )

    # logging.info("cl_temp::%f", (cl_temp))
    # for numerical stability
    # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    # logits = anchor_dot_contrast - logits_max.detach()
    # norm_prod = torch.mm(, torch.norm(contrast_feature,dim=1).view(1,-1))
    logits = anchor_dot_contrast#/norm_prod

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )

    inverse_mask = ~(mask.bool())
    inverse_mask = inverse_mask.int()*logits_mask
    mask = mask * logits_mask

    curr_mb_sim_weight = sample_weights*mask

    curr_mb_dissim_weight = sample_weights*inverse_mask
    if curr_mb_sim_weight is not None:
        curr_mb_sim_weight = curr_mb_sim_weight * logits_mask
        assert torch.norm((curr_mb_sim_weight > 0).float() - mask).item() == 0.0
    if curr_mb_dissim_weight is not None:
        curr_mb_dissim_weight = curr_mb_dissim_weight * logits_mask
        
    # if separate_cl_loss:
    #     loss = self.compute_full_cl_loss2(logits, mask, inverse_mask, logits_mask, anchor_count, extra_similarity_batch_wise, extra_similar_count, extra_dissimilarity_batch_wise, weight1, weight2, pos_in_deno)
    # else:
    # logits, mask, inverse_mask, logits_mask, anchor_count, pos_in_deno = False, curr_mb_sim_weight = None, curr_mb_dissim_weight = None
    # loss = compute_full_cl_loss0(logits, mask, inverse_mask, logits_mask, anchor_count, pos_in_deno, curr_mb_sim_weight = curr_mb_sim_weight, curr_mb_dissim_weight = curr_mb_dissim_weight, extra_similar_count=extra_similar_count, extra_similarity_batch_wise=extra_similarity_batch_wise, extra_dissimilarity_batch_wise=extra_dissimilarity_batch_wise, rand_perm = rand_perm)
    ######################################################Updates from add one if statement for filtering out a corner case################################
    if (curr_mb_dissim_weight is None or torch.norm(curr_mb_dissim_weight) > 0) and torch.norm(inverse_mask) > 0:
        # print("compute cl loss")
        loss = compute_full_cl_loss0(labels, logits, mask, inverse_mask, logits_mask, anchor_count, pos_in_deno, curr_mb_sim_weight = curr_mb_sim_weight, curr_mb_dissim_weight = curr_mb_dissim_weight, cl_loss_sqr = cl_loss_sqr, sample_weights = sample_weights)
    else:
        loss = 0

    return loss

def compute_full_cl_loss0(labels, logits, mask, inverse_mask, logits_mask, anchor_count, pos_in_deno = False, curr_mb_sim_weight = None, curr_mb_dissim_weight = None, extra_similar_count = None, cl_loss_sqr = False, sample_weights = None, use_neg_in_cl  =False):
        if pos_in_deno:        
            exp_logits = torch.exp(logits) * logits_mask
        else:
            exp_logits = torch.exp(logits) * logits_mask*inverse_mask

        if curr_mb_dissim_weight is not None:
            # if rand_perm:
            #     values = curr_mb_dissim_weight[curr_mb_dissim_weight !=0]
            #     curr_mb_dissim_weight[curr_mb_dissim_weight!=0] = values[torch.randperm(values.shape[0])]
            exp_logits = exp_logits * curr_mb_dissim_weight
        # if extra_dissimilarity_batch_wise is None:
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # else:
        #     log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).view(-1) + extra_dissimilarity_batch_wise.view(-1))
        # if selected_tuple_id_ls is not None:
        #     tuple_id_tensor = torch.tensor(selected_tuple_id_ls)
        #     log_prob = log_prob[tuple_id_tensor]
        #     mask = mask[tuple_id_tensor]

        # if extra_similarity_batch_wise is not None:
        #     if extra_dissimilarity_batch_wise is None:
        #         extra_log_prob = extra_similarity_batch_wise-torch.log(exp_logits.sum(1, keepdim=True)).view(-1)
        #     else:
        #         extra_log_prob = extra_similarity_batch_wise-torch.log(exp_logits.sum(1, keepdim=True).view(-1) + extra_dissimilarity_batch_wise.view(-1))

        if curr_mb_sim_weight is not None:
            # if rand_perm:
            #     values = curr_mb_sim_weight[curr_mb_sim_weight != 0]
            #     curr_mb_sim_weight[curr_mb_sim_weight != 0] = values[torch.randperm(values.shape[0])]
        # else:
            mask = curr_mb_sim_weight


        # if extra_similarity_batch_wise is None:
        total_mask = (mask != 0).sum(1)
        # else:
        #     total_mask = mask.sum(1).view(-1) + extra_similar_count.view(-1)

        # total_mask = (mask != 0).sum(1)
        
        # if extra_similarity_batch_wise is None:
        mean_log_prob_pos = ((mask * log_prob).sum(1))[total_mask != 0] / total_mask[total_mask != 0]
        # else:
        #     mean_log_prob_pos = ((mask * log_prob).sum(1) + extra_log_prob)[total_mask != 0] / total_mask[total_mask != 0]
        # if extra_similarity_batch_wise is None:
        #     total_mask = (mask != 0).sum(1)
        # else:
        #     total_mask = mask.sum(1).view(-1) + extra_similar_count.view(-1)
        # compute mean of log-likelihood over positive
        # if extra_similarity_batch_wise is None:
        #     mean_log_prob_pos = weight1*((mask * log_prob).sum(1))[total_mask != 0] / total_mask[total_mask != 0]
        # else:
        #     mean_log_prob_pos = (weight1*(mask * log_prob).sum(1) + weight2*extra_log_prob)[total_mask != 0] / total_mask[total_mask != 0]

        loss = -mean_log_prob_pos
        # loss = - (self.cl_temp) * mean_log_prob_pos
        # loss = loss.view(anchor_count, batch_size).mean()

        labels = labels[total_mask != 0].view(-1)
        if torch.sum(labels != 0) <= 0:
            return 0.0

        if sample_weights is not None:
            sample_weights = sample_weights[total_mask != 0]
        if not cl_loss_sqr:
            if sample_weights is None:
                # if not use_neg_in_cl:
                    
                #     loss = loss.view(anchor_count, torch.sum(total_mask != 0))[:,labels !=0].mean()
                # else:
                #     logging.info("use neg in cl")
                loss = loss.view(anchor_count, torch.sum(total_mask != 0)).mean()
            else:
                # if not use_neg_in_cl:
                #     loss = ((loss.view(anchor_count, torch.sum(total_mask != 0))*sample_weights.view(-1,torch.sum(total_mask != 0)))[:,labels !=0]).mean()
                # else:
                #     logging.info("use neg in cl")
                loss = ((loss.view(anchor_count, torch.sum(total_mask != 0))*sample_weights.view(-1,torch.sum(total_mask != 0)))).mean()
        else:
            if sample_weights is None:
                # if not use_neg_in_cl:
                #     loss = (loss*loss).view(anchor_count, torch.sum(total_mask != 0))[:,labels !=0].mean()
                # else:
                # logging.info("use neg in cl")
                loss = (loss*loss).view(anchor_count, torch.sum(total_mask != 0)).mean()
            else:
                # if not use_neg_in_cl:
                #     loss = (((loss*loss).view(anchor_count, torch.sum(total_mask != 0))*sample_weights.view(-1,torch.sum(total_mask != 0)))[:,labels !=0]).mean()
                # else:
                # logging.info("use neg in cl")
                loss = (((loss*loss).view(anchor_count, torch.sum(total_mask != 0))*sample_weights.view(-1,torch.sum(total_mask != 0)))).mean()

        # loss = loss.view(anchor_count, torch.sum(total_mask != 0)).mean()

        if torch.isinf(loss):
                print()
        if torch.isnan(loss):
            print()

        return loss




def train_features_with_contrastivel_learning(args, train_loader, sample_weights, network, optimizer, scheduler = None, cl_temp=0.1):
    network.train()
    
    for epoch in range(args.epochs):

        full_loss = 0

        for batch_idx, (sample_idx, data, target) in enumerate(train_loader):
            curr_sample_weights = sample_weights[sample_idx]
            optimizer.zero_grad()
            if args.cuda:
                data = data.cuda()
                target = target.cuda()
            features = network.feature_forward(data).unsqueeze(1)



            loss = calculate_entity_embedding_sim0_sample_wise(cl_temp, features, data.device, labels=target, mask=None, pos_in_deno = False, cl_loss_sqr = False, sample_weights = curr_sample_weights)
            # loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            full_loss += loss.cpu().detach().item()*data.shape[0]
            # if batch_idx % log_interval == 0:
        # logging.info("Train Epoch: %d [{}/{} ({:.0f}%)]\tLoss: {:.6f}", (
        # epoch, batch_idx * len(data), len(train_loader.dataset),
        # 100. * batch_idx / len(train_loader), loss.item()))

        full_loss = full_loss/len(train_loader.dataset)

        logging.info("Train Epoch: %d \tLoss: %f"%(epoch, full_loss))
        # epoch, loss.item()))
        torch.save(network.state_dict(), os.path.join(args.save_path, "pretrained_model_" + str(epoch)))
        # logging.info("train performance at epoch %d"%(epoch))
        # test(train_loader,network, args)
        # logging.info("valid performance at epoch %d"%(epoch))
        # test(valid_loader,network, criterion, args, "valid")
        # logging.info("test performance at epoch %d"%(epoch))
        # test(test_loader,network, criterion,args, "test")



def main(args):

    set_logger(args)
    
    logging.info("start")

    train_loader, test_loader, ndata = get_dataloader(args, add_erasing=args.erasing, aug_plus=args.aug_plus)

    if args.load_cached_weights:
        cached_sample_weights = torch.load(os.path.join(args.prev_save_path, args.cached_sample_weights_name))
        logging.info("load sample weights successfully")


    net = DNN_three_layers(args.nce_k, low_dim=args.low_dim).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
    
    train_features_with_contrastivel_learning(args, train_loader, cached_sample_weights, net, optimizer, scheduler = None, cl_temp=0.1)





if __name__ == '__main__':
    opt = parse_args()

    torch.cuda.set_device(opt.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    os.makedirs(opt.save_path, exist_ok=True)
    logger = setup_logger(output=opt.save_path, distributed_rank=dist.get_rank(), name="moco+cld")
    if dist.get_rank() == 0:
        path = os.path.join(opt.save_path, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(opt), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    main(opt)
