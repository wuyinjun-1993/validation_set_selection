import torch

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from common.parse_args import *
from datasets.mnist import *
from models.DNN import *
from common.utils import *
from tqdm.notebook import tqdm
import itertools
import torch_higher as higher
from main.find_valid_set import *


def vary_learning_rate(current_learning_rate, eps, args, model=None):
    current_learning_rate = current_learning_rate / 2
    logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, eps))

    optimizer = None

    if model is not None:
        # optimizer = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, model.parameters()), 
        #     lr=current_learning_rate


        # )

        optimizer = torch.optim.SGD(model.parameters(), lr=current_learning_rate)

    return optimizer, current_learning_rate

def meta_learning_model(args, model, opt, criterion, train_loader, meta_loader, valid_loader, test_loader, to_device, cached_w_array = None, target_id = None):
    
    # train_loader = DataLoader(train_dataset, batch_size=args['bs'], shuffle=True, num_workers=2, pin_memory=True)
    # test_loader =  DataLoader(test_dataset, batch_size=args['bs'], shuffle=False, num_workers=2, pin_memory=True)
    # meta_loader = DataLoader(meta_dataset, batch_size=args['bs'], shuffle=True, pin_memory=True)
    
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # test_loader =  DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    # meta_loader = DataLoader(meta_dataset, batch_size=args.test_batch_size, shuffle=True)
    # train_loader, meta_loader, valid_loader, test_loader = create_data_loader(train_dataset, meta_dataset, test_dataset, args)
    
    meta_loader = itertools.cycle(meta_loader)
    
    if args.cuda:
        device = torch.device("cuda:" + str(torch.cuda.current_device()))
    else:
        device = torch.device("cpu")

    if cached_w_array is None:
        w_array =torch.rand(len(train_loader.dataset), requires_grad=True, device = device)
    else:
        cached_w_array.requires_grad = False
        w_array = cached_w_array.clone()

    # if args.cuda:
        w_array = w_array.to(device)

        w_array.requires_grad = True
    
    # with torch.no_grad():
    #     metrics = model.test(model, valid_loader, args)
        
    #     log_metrics('Valid', 0, metrics)
            
    #     metrics = model.test(model, test_loader, args)
        
    #     log_metrics('Test', 0, metrics)   
    warm_up_steps = 200

    total_iter_count = 1

    # warm_up_steps = args.warm_up_steps

    # if warm_up_steps is None:
    #     warm_up_steps = 10000# args.max_steps//2

    curr_learning_rate = args.lr

    curr_ilp_learning_rate = args.meta_lr

    model.train()
    for ep in tqdm(range(1, args.epochs+1)):
        
        train_loss, train_acc = 0, 0
        for idx, inputs in enumerate(train_loader):
            # inputs, labels = inputs.to(device=args['device'], non_blocking=True),\
                                # labels.to(device=args['device'], non_blocking=True)
            train_ids = (train_loader.dataset.indices.view(1,-1) == inputs[0].view(-1,1)).nonzero()[:,1]

            assert len(torch.nonzero(train_loader.dataset.__getitem__(train_ids[0])[1] - inputs[1][0])) == 0

            
            inputs = to_device(inputs, args)
            
            w_array.requires_grad = True
            
            opt.zero_grad()
            with higher.innerloop_ctx(model, opt) as (meta_model, meta_opt):
                
                # 1. Update meta model on training data
                # if type(inputs) is tuple:
                #     eps = torch.zeros(inputs[0].size()[0], requires_grad=True, device=args.device)
                # else:
                #     eps = torch.zeros(inputs.size()[0], requires_grad=True, device=args.device)
                
                eps = w_array[train_ids]
                
                
                # if args.cuda:
                # eps = eps.to(args.device)
                
                # print('eps shape::', eps.shape, eps)
                
                if criterion is not None:
                    criterion.reduction = 'none'
                
                meta_train_outputs = meta_model(inputs[1])

                meta_train_loss = torch.mean(criterion(meta_train_outputs, inputs[2])*eps)
                
                # 
                # meta_train_loss = criterion(meta_train_outputs, labels.type_as(meta_train_outputs))
                #
                # meta_train_loss = torch.sum(eps * meta_train_loss)
                
                meta_opt.step(meta_train_loss)
    
                # 2. Compute grads of eps on meta validation data
                # meta_inputs, meta_labels =  next(meta_loader)
                # meta_inputs, meta_labels = meta_inputs.to(device=args['device'], non_blocking=True),\
                #                  meta_labels.to(device=args['device'], non_blocking=True)
                meta_inputs =  next(meta_loader)
                
                # print(meta_inputs)
    
                meta_inputs = to_device(meta_inputs, args)
                
                if criterion is not None:
                    criterion.reduction = 'mean'
                
                # meta_val_loss,meta_valid_numbers = loss_func(meta_inputs, meta_model, criterion, no_agg = False)
                
                meta_val_loss = criterion(meta_model(meta_inputs[1]), meta_inputs[2])

                # meta_val_outputs = meta_model(meta_inputs)
                #
                # meta_val_loss = criterion(meta_val_outputs, meta_labels.type_as(meta_train_outputs))
                # print(meta_val_loss, eps)
                
                eps_grads = torch.autograd.grad(meta_val_loss, eps)[0].detach()
    
            # 3. Compute weights for current training batch
            
            # output_model_param(model)
            #
            # output_model_param(meta_model)
            
            w_array.requires_grad = False
            
            w_array[train_ids] =  w_array[train_ids]-curr_ilp_learning_rate*eps_grads
            
            w_array[train_ids] = torch.clamp(w_array[train_ids], max=1, min=1e-7) #torch.relu(w_array[train_ids])

            


            # if args.learning_decay and total_iter_count%warm_up_steps == 0:
            #         opt, curr_learning_rate = vary_learning_rate(curr_learning_rate, ep, args, model=model)

            #         _, curr_ilp_learning_rate = vary_learning_rate(curr_ilp_learning_rate, ep, args, model=None)

            #         logging.info('Iteration number at step %d: %d' % (ep, total_iter_count))

            #         logging.info('warm up iteration at step %d: %d' % (ep, warm_up_steps))

            #         warm_up_steps*=3

            # l1_norm = torch.sum(w_array[train_ids])
            # if l1_norm != 0:
            #     w_array[train_ids] = w_array[train_ids] / l1_norm
            # else:
            #     w = w_array[train_ids]
    
            # 4. Train model on weighted batch
            if criterion is not None:
                criterion.reduction = 'none'
            
            minibatch_loss = torch.mean(criterion(model(inputs[1]), inputs[2])*w_array[train_ids])
            # minibatch_loss,_ = loss_func(inputs, model, criterion, w_array[train_ids])
            
            # logging.info("Training Loss at the iteration %d: %f" %(int(idx), float(minibatch_loss.item())))
            
            # logging.info("Validation Loss at the iteration %d: %f" %(int(idx), float(meta_val_loss.item())))
            
            # logging.info("meta_train_loss at the iteration %d: %f" %(int(idx), float(meta_train_loss.item())))

            # logging.info("learning rate at the iteration %d: %f" %(int(idx), float(curr_learning_rate)))

            # logging.info("meta learning rate at the iteration %d: %f" %(int(idx), float(curr_ilp_learning_rate)))
        
            # if torch.sum(torch.isnan(minibatch_loss)) > 0:
                
            #     print("epsilon at the iteration", str(idx), eps)
                
            #     print("epsilon grad at the iteration", str(idx), eps_grads)
                
            #     print("w at the iteration", str(idx), w_array[train_ids])
                
            #     print("train ids::", str(idx), train_ids)
                
            #     for number in meta_numbers:
            #         print('number::', number)
            #         # del number
                
            #     for number in meta_valid_numbers:
            #         print('valid number::', number)
            #         # del number
                
            #     output_model_param(model)
                
            #     output_model_param(meta_model)
            
            # del meta_numbers, meta_valid_numbers
            
            # outputs = model(inputs)
            # minibatch_loss = criterion(outputs, labels.type_as(outputs))
            # minibatch_loss = torch.sum(w * minibatch_loss)
            minibatch_loss.backward()
            opt.step()

            total_iter_count += 1

            # if total_iter_count%warm_up_steps == 0:
            #     warm_up_steps*=3
            #     curr_ilp_learning_rate = curr_ilp_learning_rate/10
            #     opt, curr_learning_rate = vary_learning_rate(curr_learning_rate, ep, args, model=model)
    
            # keep track of epoch loss/accuracy
            # if type(inputs) is tuple:
            #     train_loss += minibatch_loss.item()*inputs[0].shape[0]
            # else:
            #     train_loss += minibatch_loss.item()*inputs.shape[0]
            # pred_labels = (F.sigmoid(outputs) > 0.5).int()
            # train_acc += torch.sum(torch.eq(pred_labels, labels)).item()
    
        # inference after epoch
        with torch.no_grad():
            logging.info("valid performance at epoch %d"%(ep))
            test(valid_loader, model, args, "valid")
            logging.info("test performance at epoch %d"%(ep))
            test(test_loader, model, args, "test")

        torch.save(model, os.path.join(args.save_path, 'refined_model_' + str(ep)))
        torch.save(w_array, os.path.join(args.save_path, 'sample_weights_' + str(ep)))





def basic_train(train_loader, valid_loader, test_loader, args, network, optimizer):

    network.train()
    
    for epoch in range(args.epochs):

        for batch_idx, (_, data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if args.cuda:
                data = data.cuda()
                target = target.cuda()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:
        # logging.info("Train Epoch: %d [{}/{} ({:.0f}%)]\tLoss: {:.6f}", (
        # epoch, batch_idx * len(data), len(train_loader.dataset),
        # 100. * batch_idx / len(train_loader), loss.item()))

        # logging.info("Train Epoch: %d \tLoss: %f", (
        # epoch, loss.item()))
        torch.save(network.state_dict(), os.path.join(args.save_path, "model_" + str(epoch)))
        # logging.info("train performance at epoch %d"%(epoch))
        # test(train_loader,network, args)
        logging.info("valid performance at epoch %d"%(epoch))
        test(valid_loader,network, args, "valid")
        logging.info("test performance at epoch %d"%(epoch))
        test(test_loader,network, args, "test")
                # train_losses.append(loss.item())
                # train_counter.append(
                # (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
                # torch.save(network.state_dict(), '/results/model.pth')
                # torch.save(optimizer.state_dict(), '/results/optimizer.pth')

def sort_prob_gap_by_class(pred_labels, prob_gap_ls, label_ls, class_id, select_count):
    boolean_id_arrs = torch.logical_and((label_ls == class_id).view(-1), (pred_labels == label_ls).view(-1))

    # boolean_id_arrs = (label_ls == class_id).view(-1)

    sample_id_with_curr_class = torch.nonzero(boolean_id_arrs).view(-1)

    prob_gap_ls_curr_class = prob_gap_ls[boolean_id_arrs]

    sorted_probs, sorted_idx = torch.sort(prob_gap_ls_curr_class, dim = 0, descending = False)

    selected_sub_ids = (sorted_probs < 0.2).nonzero()
    # # # selected_sub_ids = (sorted_probs > 0.999).nonzero()
    select_count = min(select_count, len(selected_sub_ids))

    selected_sample_indx = sample_id_with_curr_class[sorted_idx[0:select_count]]

    selected_prob_gap_values = sorted_probs[0:select_count]

    return selected_sample_indx





# def find_representative_samples(net, train_loader, args, valid_ratio = 0.1):
#     prob_gap_ls = torch.zeros(len(train_loader.dataset))

#     label_ls = torch.zeros(len(train_loader.dataset), dtype =torch.long)

#     valid_count = int(len(train_loader.dataset)*valid_ratio)

#     pred_labels = torch.zeros(len(train_loader.dataset), dtype =torch.long)

#     sample_representation_vec_ls_by_class = dict()

#     for batch_id, (sample_ids, data, labels) in enumerate(train_loader):

#         if args.cuda:
#             data = data.cuda()
#             # labels = labels.cuda()

#         sample_representation = net.feature_forward(data)

#         for idx in range(len(labels)):
#             curr_label = labels[idx].item()
#             if curr_label not in sample_representation_vec_ls_by_class:
#                 sample_representation_vec_ls_by_class[curr_label] = []
#             sample_representation_vec_ls_by_class[curr_label].append(sample_representation[idx])

#     for label in sample_representation_vec_ls_by_class:
#         sample_representation_vec_ls_by_class[label] = torch.stack(sample_representation_vec_ls_by_class[label])

    
        # out_probs = torch.exp(net.(data))
        # sorted_probs, sorted_indices = torch.sort(out_probs, dim = 1, descending = True)

        # prob_gap = sorted_probs[:,0] - sorted_probs[:,1]

        # prob_gap_ls[sample_ids] = prob_gap.detach().cpu()

        # label_ls[sample_ids] = labels

        # pred_labels[sample_ids] = sorted_indices[:,0].detach().cpu()


def find_boundary_samples(net, train_loader, args, valid_ratio = 0.1):
    prob_gap_ls = torch.zeros(len(train_loader.dataset))

    label_ls = torch.zeros(len(train_loader.dataset), dtype =torch.long)

    valid_count = int(len(train_loader.dataset)*valid_ratio)

    pred_labels = torch.zeros(len(train_loader.dataset), dtype =torch.long)

    pred_correct_count = 0

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

    for label_id in unique_label_ls:
        selected_valid_ids = sort_prob_gap_by_class(pred_labels, prob_gap_ls, label_ls, label_id, int(valid_count/len(unique_label_ls)))
        selected_valid_ids_ls.append(selected_valid_ids)

    valid_ids = torch.cat(selected_valid_ids_ls)

    # valid_set = Subset(train_loader.dataset, valid_ids)
    valid_set = new_mnist_dataset2(train_loader.dataset.data[valid_ids].clone(), train_loader.dataset.targets[valid_ids].clone())

    meta_set = new_mnist_dataset2(train_loader.dataset.data[valid_ids].clone(), train_loader.dataset.targets[valid_ids].clone())




    origin_train_labels = train_loader.dataset.targets.clone()

    test(train_loader, net, args, "train")

    if args.flip_labels:

        logging.info("add errors to train set")

        train_dataset, _ = random_flip_labels_on_training(train_loader.dataset, ratio = args.err_label_ratio)


    

    train_dataset, _, _ = partition_train_valid_dataset_by_ids(train_dataset, origin_train_labels, valid_ids)

    train_loader, valid_loader, meta_loader, _ = create_data_loader(train_dataset, valid_set, meta_set, None, args)

    test(valid_loader, net, args, "valid")
    test(train_loader, net, args, "train")

    return train_loader, valid_loader, meta_loader
    # torch.sort(prob_gap_ls, dim = 0, descending = False)


def main(args):

    set_logger(args)
    
    logging.info("start")

    net = DNN_two_layers()

    
    pre_processing_mnist_main(args)
    if args.select_valid_set:
        # train_loader, test_loader = get_mnist_dataset_without_valid_without_perturbations(args)

        train_loader, test_loader = get_mnist_dataset_without_valid_without_perturbations2(args)

        net_state_dict = torch.load(os.path.join(args.data_dir, "model_full"), map_location=torch.device("cpu"))

        net.load_state_dict(net_state_dict, strict=False)

        if args.cuda:
            net = net.cuda()

        test(test_loader, net, args, "test")
        # test(train_loader, net, args)

        # train_loader, valid_loader, meta_loader = find_boundary_samples(net, train_loader, args)
        train_loader, valid_loader, meta_loader = find_representative_samples(net, train_loader, args)
    else:
        train_loader, valid_loader, meta_loader, test_loader = get_mnist_data_loader(args)

    net = DNN_two_layers()

    if args.cuda:
        net = net.cuda()
    
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)

    
    if args.do_train:
        logging.info("start basic training")
        basic_train(train_loader, valid_loader, test_loader, args, net, optimizer)
    else:

        logging.info("start meta training")
        meta_learning_model(args, net, optimizer, torch.nn.NLLLoss(), train_loader, meta_loader, valid_loader, test_loader, mnist_to_device, cached_w_array = None, target_id = None)


if __name__ == "__main__":
    args = parse_args()
    main(args)