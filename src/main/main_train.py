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
from main.meta_reweighting_rl import *
from datasets.dataloader import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
from utils.logger import setup_logger
import models
from lib.NCECriterion import NCESoftmaxLoss
from lib.lr_scheduler import get_scheduler
from models.resnet import *
import collections

cached_model_name="cached_model"

def vary_learning_rate(current_learning_rate, eps, args, model=None):
    # current_learning_rate = current_learning_rate / 2
    if args.dataset == 'MNIST':#args.dataset == 'MNIST'
        current_learning_rate = args.lr* ((0.5 ** int(eps >= 500)) * (0.5 ** int(eps >= 800)))
    else:
        if args.dataset.startswith('cifar'):
            current_learning_rate = args.lr* ((0.2 ** int(eps >= 50)) * (0.2 ** int(eps >= 100)))
    logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, eps))

    optimizer = None

    if model is not None:
        # optimizer = torch.optim.Adam(
        #     filter(lambda p: p.requires_grad, model.parameters()), 
        #     lr=current_learning_rate


        # )

        optimizer = torch.optim.SGD(model.parameters(), lr=current_learning_rate)

    return optimizer, current_learning_rate

def report_best_test_performance_so_far(test_loss_ls, test_acc_ls, test_loss, test_acc):
    test_loss_ls.append(test_loss)
    test_acc_ls.append(test_acc)

    test_loss_array = numpy.array(test_loss_ls)

    test_acc_array = numpy.array(test_acc_ls)

    min_loss_epoch = numpy.argmin(test_loss_array)

    min_test_loss = test_loss_array[min_loss_epoch]

    min_test_acc = test_acc_array[min_loss_epoch]

    logging.info("best test performance so far is in epoch %d: %f, %f"%(min_loss_epoch, min_test_loss, min_test_acc))

def report_final_performance_by_early_stopping(valid_loss_ls, valid_acc_ls, test_loss_ls, test_acc_ls, args, tol = 5, is_meta=True):
    valid_acc_arr = numpy.array(valid_acc_ls)

    best_valid_acc = numpy.max(valid_acc_arr)

    # for k in range(len(valid_acc_ls)):
    #     if valid_acc_ls[k] == best_valid_acc:


    best_valid_acc_epochs = numpy.reshape(numpy.nonzero(valid_acc_arr == best_valid_acc), (-1))

    for epoch in best_valid_acc_epochs:
        all_best = True
        for k in range(1, tol+1):
            if epoch + k <= len(valid_acc_ls) - 1:
                if not valid_acc_ls[epoch + k] <= best_valid_acc:
                    all_best = False
                    break

        if all_best:
            break

    final_epoch = min(epoch, args.epochs-1)
    final_test_loss = test_loss_ls[final_epoch]

    final_test_acc = test_acc_ls[final_epoch]

    logging.info("final test performance is in epoch %d: %f, %f"%(final_epoch, final_test_loss, final_test_acc))
    if is_meta:
        cache_sample_weights_given_epoch(final_epoch)
    else:
        cache_sample_weights_given_epoch_basic_train(final_epoch)


def report_final_performance_by_early_stopping2(valid_loss_ls, valid_acc_ls, test_loss_ls, test_acc_ls, args, tol = 5, is_meta=True):
    # valid_acc_arr = numpy.array(valid_acc_ls)

    # best_valid_acc = numpy.max(valid_acc_arr)

    # # for k in range(len(valid_acc_ls)):
    # #     if valid_acc_ls[k] == best_valid_acc:


    # best_valid_acc_epochs = numpy.reshape(numpy.nonzero(valid_acc_arr == best_valid_acc), (-1))

    # for epoch in best_valid_acc_epochs:
    #     all_best = True
    #     for k in range(1, tol+1):
    #         if epoch + k <= len(valid_acc_ls) - 1:
    #             if not valid_acc_ls[epoch + k] == best_valid_acc:
    #                 all_best = False
    #                 break

    #     if all_best:
    #         break


    test_loss_arr = torch.tensor(test_loss_ls)

    final_epoch = torch.argmin(test_loss_arr).item()



    # final_epoch = min(epoch + tol, args.epochs-1)
    final_test_loss = test_loss_ls[final_epoch]

    final_test_acc = test_acc_ls[final_epoch]

    logging.info("final test performance is in epoch %d: %f, %f"%(final_epoch, final_test_loss, final_test_acc))
    if is_meta:
        cache_sample_weights_given_epoch(final_epoch)
    else:
        cache_sample_weights_given_epoch_basic_train(final_epoch)


def cache_sample_weights_for_min_loss_epoch(args, test_loss_ls):
    min_loss_epoch = numpy.argmin(test_loss_ls)

    best_w_array = torch.load(os.path.join(args.save_path, 'sample_weights_' + str(min_loss_epoch)))

    best_model = torch.load(os.path.join(args.save_path, 'refined_model_' + str(min_loss_epoch)))

    logging.info("caching sample weights at epoch %d"%(min_loss_epoch))


    torch.save(best_w_array, os.path.join(args.save_path, "cached_sample_weights"))

    torch.save(best_model, os.path.join(args.save_path, cached_model_name))


def cache_sample_weights_given_epoch(epoch):
    best_w_array = torch.load(os.path.join(args.save_path, 'sample_weights_' + str(epoch)))

    best_model = torch.load(os.path.join(args.save_path, 'refined_model_' + str(epoch)))

    logging.info("caching sample weights at epoch %d"%(epoch))


    torch.save(best_w_array, os.path.join(args.save_path, "cached_sample_weights"))

    torch.save(best_model, os.path.join(args.save_path, cached_model_name))


def cache_sample_weights_given_epoch_basic_train(epoch):
    # best_w_array = torch.load(os.path.join(args.save_path, 'sample_weights_' + str(epoch)))

    best_model = torch.load(os.path.join(args.save_path, 'model_' + str(epoch)))

    # logging.info("caching sample weights at epoch %d"%(epoch))


    # torch.save(best_w_array, os.path.join(args.save_path, "cached_sample_weights"))

    torch.save(best_model, os.path.join(args.save_path, cached_model_name))

def resume_meta_training_by_loading_cached_info(args, net):

    # if args.resume_meta_train:
    #     prev_net, prev_weights, start_epoch= resume_meta_training_by_loading_cached_info()
    #     net.load_state_dict(prev_net.state_dict(), strict=False)

    # if args.cuda:
    #     net = net.cuda()



    model = torch.load(os.path.join(args.save_path, 'curr_refined_model'), map_location=torch.device('cpu'))
    w_array = torch.load(os.path.join(args.save_path, 'curr_sample_weights'), map_location=torch.device('cpu'))
    start_ep = torch.load(os.path.join(args.save_path, "curr_epoch")).item()
    net.load_state_dict(model.state_dict(), strict=False)
    return net, w_array, start_ep

def meta_learning_model(
    args,
    model,
    opt,
    criterion,
    meta_criterion,
    train_loader,
    meta_loader,
    valid_loader,
    test_loader,
    to_device,
    cached_w_array=None,
    scheduler=None,
    target_id=None,
    start_ep=0,
    mile_stones_epochs=None,
):
    
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
        # w_array.data[:] = 1e-1
        # w_array =torch.ones(len(train_loader.dataset), requires_grad=True, device = device)*1e-4
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

    valid_loss_ls = []
    valid_acc_ls = []
    test_loss_ls = []
    test_acc_ls = []
    w_array_delta_ls = []

    model.train()
    for ep in tqdm(range(start_ep + 1, args.epochs+1)):
        
        train_loss, train_acc = 0, 0
        curr_w_array_delta = torch.zeros_like(w_array)

        avg_train_loss = 0

        train_pred_correct = 0

        for idx, inputs in enumerate(train_loader):
            # inputs, labels = inputs.to(device=args['device'], non_blocking=True),\
                                # labels.to(device=args['device'], non_blocking=True)
            # train_ids = (train_loader.dataset.indices.view(1,-1) == inputs[0].view(-1,1)).nonzero()[:,1]
            train_ids = inputs[0]
            if args.dataset == 'MNIST':
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

                labels = inputs[2]
                if isinstance(criterion, torch.nn.L1Loss):
                    labels = torch.nn.functional.one_hot(inputs[2],
                            num_classes=10)
                    meta_train_outputs = F.softmax(meta_train_outputs)
                meta_train_loss = torch.mean(criterion(meta_train_outputs, labels)*eps)
                
                
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
                
                meta_out = meta_inputs[2]
                model_out = meta_model(meta_inputs[1])
                if isinstance(meta_criterion, torch.nn.L1Loss):
                    meta_out = torch.nn.functional.one_hot(meta_inputs[2],
                            num_classes=10)
                    model_out = F.softmax(model_out)
                meta_val_loss = meta_criterion(model_out, meta_out)

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
            
            prev_w_array = w_array[train_ids].detach().clone()

            w_array[train_ids] =  w_array[train_ids]-curr_ilp_learning_rate*eps_grads
            
            w_array[train_ids] = torch.clamp(w_array[train_ids], max=1, min=1e-7) #torch.relu(w_array[train_ids])

            curr_w_array_delta[train_ids] = w_array[train_ids].detach() - prev_w_array


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
            
            model_out = model(inputs[1])
            labels = inputs[2]
            if isinstance(criterion, torch.nn.L1Loss):
                labels = torch.nn.functional.one_hot(inputs[2],
                        num_classes=10)
                model_out = F.softmax(meta_train_outputs)
            minibatch_loss = torch.mean(criterion(model_out, labels)*w_array[train_ids])

            minibatch_loss.backward()
            opt.step()
            

            avg_train_loss += minibatch_loss.detach().cpu().item()*inputs[1].shape[0]

            model_pred = torch.max(model_out, dim = 1)[1]

            train_pred_correct += torch.sum(model_pred.view(-1) == inputs[2].view(-1)).detach().cpu().item()


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

            avg_train_loss = avg_train_loss/len(train_loader.dataset)
            train_pred_acc_rate = train_pred_correct*1.0/len(train_loader.dataset)
            logging.info("average training loss at epoch %d:%f"%(ep, avg_train_loss))

            logging.info("training accuracy at epoch %d:%f"%(ep, train_pred_acc_rate))
            if criterion is not None:
                criterion.reduction = 'mean'
            logging.info("valid performance at epoch %d"%(ep))
            valid_loss, valid_acc = test(valid_loader, model, criterion, args, "valid")
            logging.info("test performance at epoch %d"%(ep))
            test_loss, test_acc = test(test_loader, model, criterion, args, "test")

            report_best_test_performance_so_far(valid_loss_ls, valid_acc_ls, valid_loss, valid_acc)

            report_best_test_performance_so_far(test_loss_ls, test_acc_ls, test_loss, test_acc)

        if scheduler is not None:
            logging.info("learning rate at iteration %d before using scheduler: %f" %(int(ep), float(opt.param_groups[0]['lr'])))

            scheduler.step()
            logging.info("learning rate at iteration %d after using scheduler: %f" %(int(ep), float(opt.param_groups[0]['lr'])))

        if args.lr_decay:
            if mile_stones_epochs is not None:
                ms_idx = 0
                for ms_ep_idx in range(len(mile_stones_epochs)-1):
                    if ep >= mile_stones_epochs[ms_ep_idx] and ep < mile_stones_epochs[ms_ep_idx+1]:
                        ms_idx = ms_ep_idx + 1
                        break
                if ep >= mile_stones_epochs[-1]:
                    ms_idx = len(mile_stones_epochs)

                curr_ilp_learning_rate = args.meta_lr*(0.1**(ms_idx))
                logging.info("meta learning rate at iteration %d: %f" %(int(ep), curr_ilp_learning_rate))
            else:


                if ep > 100:
                    
                    curr_ilp_learning_rate = args.meta_lr*0.1
                    logging.info("meta learning rate at iteration %d: %f" %(int(ep), curr_ilp_learning_rate))
            # min_valid_loss_epoch = numpy.argmin(numpy.array(valid_loss_ls))
            # if min_valid_loss_epoch < ep-1:
            # opt, curr_learning_rate = vary_learning_rate(curr_learning_rate, ep, args, model=model)

                
        w_array_delta_ls.append(curr_w_array_delta)
            

        torch.save(model, os.path.join(args.save_path, 'refined_model_' + str(ep)))
        torch.save(w_array, os.path.join(args.save_path, 'sample_weights_' + str(ep)))

        torch.save(model, os.path.join(args.save_path, 'curr_refined_model'))
        torch.save(w_array, os.path.join(args.save_path, 'curr_sample_weights'))
        torch.save(torch.tensor(ep), os.path.join(args.save_path, "curr_epoch"))
        torch.save(torch.stack(w_array_delta_ls, dim = 0), os.path.join(args.save_path, "curr_w_array_delta_ls"))
    # cache_sample_weights_for_min_loss_epoch(args, test_loss_ls)
    # if args.dataset == 'MNIST':
    report_final_performance_by_early_stopping(valid_loss_ls, valid_acc_ls, test_loss_ls, test_acc_ls, args, tol = 5)
    # else:
    #     report_final_performance_by_early_stopping2(valid_loss_ls, valid_acc_ls, test_loss_ls, test_acc_ls, args, tol = 5)
    w_array_delta_ls_tensor = torch.stack(w_array_delta_ls, dim = 0)
    torch.save(w_array_delta_ls_tensor, os.path.join(args.save_path, "cached_w_array_delta_ls"))
    
    torch.save(torch.sum(w_array_delta_ls_tensor, dim = 1), os.path.join(args.save_path, "cached_w_array_total_delta"))

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def basic_train(train_loader, valid_loader, test_loader, criterion, args, network, optimizer, scheduler = None):

    network.train()
    curr_lr = args.lr
    valid_loss_ls = []
    valid_acc_ls = []
    test_loss_ls = []
    test_acc_ls = []
    for epoch in range(args.epochs):

        for batch_idx, (_, data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if args.cuda:
                data = data.cuda()
                target = target.cuda()
            output = network(data)
            if isinstance(criterion, torch.nn.L1Loss):
                target = torch.nn.functional.one_hot(target, num_classes=10)
                output = F.softmax(output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
            # if batch_idx % log_interval == 0:
        # logging.info("Train Epoch: %d [{}/{} ({:.0f}%)]\tLoss: {:.6f}", (
        # epoch, batch_idx * len(data), len(train_loader.dataset),
        # 100. * batch_idx / len(train_loader), loss.item()))

        # logging.info("Train Epoch: %d \tLoss: %f", (
        # epoch, loss.item()))
        torch.save(network.state_dict(), os.path.join(args.save_path, "model_" + str(epoch)))
        # logging.info("train performance at epoch %d"%(epoch))
        # test(train_loader,network, args)
        logging.info("learning rate at epoch %d: %f"%(epoch, float(optimizer.param_groups[0]['lr'])))
        
        
        with torch.no_grad():
        
            logging.info("valid performance at epoch %d"%(epoch))
                
            if valid_loader is not None:
                valid_loss, valid_acc = test(valid_loader,network, criterion, args, "valid")
                report_best_test_performance_so_far(valid_loss_ls, valid_acc_ls, valid_loss, valid_acc)
            logging.info("test performance at epoch %d"%(epoch))
            test_loss, test_acc = test(test_loader,network, criterion,args, "test")

            report_best_test_performance_so_far(test_loss_ls, test_acc_ls, test_loss, test_acc)


    report_final_performance_by_early_stopping(valid_loss_ls, valid_acc_ls, test_loss_ls, test_acc_ls, args, tol = 5, is_meta=False)
        # if (epoch+1) % 40 == 0:
        #     curr_lr /= 10
        #     update_lr(optimizer, curr_lr)
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

    selected_sub_ids = (sorted_probs < 0.05).nonzero()
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

    flipped_labels = None

    if args.flip_labels:

        logging.info("add errors to train set")

        # train_dataset, _ = random_flip_labels_on_training(train_loader.dataset, ratio = args.err_label_ratio)
        flipped_labels = obtain_flipped_labels(train_loader.dataset, args)


    

    train_dataset, _, _ = partition_train_valid_dataset_by_ids(train_loader.dataset, origin_train_labels, flipped_labels, valid_ids)

    train_loader, valid_loader, meta_loader, _ = create_data_loader(train_dataset, valid_set, meta_set, None, args)

    test(valid_loader, net, args, "valid")
    test(train_loader, net, args, "train")

    return train_loader, valid_loader, meta_loader

def get_boundary_valid_ids(train_loader, net, args, valid_count):
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

    for label_id in unique_label_ls:
        selected_valid_ids = sort_prob_gap_by_class(pred_labels, prob_gap_ls, label_ls, label_id, int(valid_count/len(unique_label_ls)))
        selected_valid_ids_ls.append(selected_valid_ids)

    valid_ids = torch.cat(selected_valid_ids_ls)
    return valid_ids


def load_checkpoint(args, model):
    logger.info('==> Loading...')
    state = torch.load(os.path.join(args.data_dir, args.cached_model_name), map_location=torch.device("cpu"))

    # model_state = state['model']
    model_state = state

    model.load_state_dict(model_state, strict=False)

    return model


def load_checkpoint2(args, model):
    logger.info('==> Loading cached model...')
    if args.prev_save_path is not None:
        cached_model_file_name = os.path.join(args.prev_save_path, cached_model_name)
        if os.path.exists(cached_model_file_name):
            
            state = torch.load(cached_model_file_name, map_location=torch.device("cpu"))

            if type(state) is collections.OrderedDict:
                model.load_state_dict(state)
            else:
                model.load_state_dict(state.state_dict())
            logger.info('==> Loading cached model successfully')

    return model

    # state = {
    #     'opt': args,
    #     'model': model.state_dict(),
    #     'model_ema': model_ema.state_dict(),
    #     'contrast': contrast.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'scheduler': scheduler.state_dict(),
    #     'epoch': epoch,
    #     'best_acc': best_acc,
    # }
    # if args.amp_opt_level != "O0":
    #     state['amp'] = amp.state_dict()
    # torch.save(state, os.path.join(args.save_path, 'current.pth'))
    # if epoch % args.save_freq == 0:
    #     torch.save(state, os.path.join(args.save_path, f'ckpt_epoch_{epoch}.pth'))











def find_boundary_and_representative_samples(net, train_loader, args, valid_ratio = 0.1):


    
    

    valid_count = int(len(train_loader.dataset)*valid_ratio)


    valid_ids2 = get_boundary_valid_ids(train_loader, net, args, int(valid_count/10))
    valid_ids1 = get_representative_valid_ids(train_loader, args, net, valid_count - len(valid_ids2))

    
    
    valid_ids = torch.zeros(len(train_loader.dataset))

    valid_ids[valid_ids1] = 1
    valid_ids[valid_ids2] = 1

    valid_ids = torch.nonzero(valid_ids).view(-1)

    # valid_set = Subset(train_loader.dataset, valid_ids)
    valid_set = new_mnist_dataset2(train_loader.dataset.data[valid_ids].clone(), train_loader.dataset.targets[valid_ids].clone())

    meta_set = new_mnist_dataset2(train_loader.dataset.data[valid_ids].clone(), train_loader.dataset.targets[valid_ids].clone())




    origin_train_labels = train_loader.dataset.targets.clone()

    test(train_loader, net, args, "train")

    flipped_labels = None

    if args.flip_labels:

        logging.info("add errors to train set")

        # train_dataset, _ = random_flip_labels_on_training(train_loader.dataset, ratio = args.err_label_ratio)


        flipped_labels = obtain_flipped_labels(train_loader.dataset, args)
    

    train_dataset, _, _ = partition_train_valid_dataset_by_ids(train_loader.dataset, origin_train_labels, flipped_labels, valid_ids)

    train_loader, valid_loader, meta_loader, _ = create_data_loader(train_dataset, valid_set, meta_set, None, args)

    test(valid_loader, net, args, "valid")
    test(train_loader, net, args, "train")

    return train_loader, valid_loader, meta_loader
    # torch.sort(prob_gap_ls, dim = 0, descending = False)

def load_pretrained_model(args, net):
    pretrained_model_state = torch.load(os.path.join(args.data_dir, "pretrained_model"))

    net.load_state_dict(pretrained_model_state, strict=False)

    return net

def main2(args):

    set_logger(args)
    
    logging.info("start")
    print('==> Preparing data..')

    if args.dataset == 'MNIST':
        pretrained_rep_net = DNN_three_layers(args.nce_k, low_dim=args.low_dim).cuda()
    else:
        pretrained_rep_net = ResNet18().cuda()
    criterion = torch.nn.CrossEntropyLoss()

    meta_criterion = criterion
    if args.l1_meta_loss:
        meta_criterion = torch.nn.L1Loss()

    if args.unsup_rep:
        pretrained_rep_net = load_checkpoint(args, pretrained_rep_net)
    else:
        pretrained_rep_net = load_checkpoint2(args, pretrained_rep_net)

    cached_sample_weights = None
    if args.load_cached_weights:
        cached_sample_weights = torch.load(os.path.join(args.prev_save_path, args.cached_sample_weights_name))
        logging.info("sample weights loaded successfully")
    if args.cuda:
        pretrained_rep_net = pretrained_rep_net.cuda()
    
    if args.select_valid_set:
        trainloader, validloader, metaloader, testloader = get_dataloader_for_meta(
            args,
            split_method='cluster',
            pretrained_model=pretrained_rep_net,
            cached_sample_weights=cached_sample_weights,
        )
    else:
        trainloader, validloader, metaloader, testloader = get_dataloader_for_meta(
            args,
            split_method='random',
            pretrained_model=pretrained_rep_net,
        )

    if args.l1_loss:
        criterion = torch.nn.L1Loss()

    if args.bias_classes:
        num_train = len(trainloader.dataset.targets)
        num_val = len(metaloader.dataset.targets)
        num_test = len(testloader.dataset.targets)
        if type(trainloader.dataset.targets) is numpy.ndarray:
            vsum = np.sum
        else:
            vsum = torch.sum

        if type(testloader.dataset.targets) is list:
            if type(testloader.dataset.data) is numpy.ndarray:
                testloader.dataset.targets = np.array(testloader.dataset.targets)
            else:
                testloader.dataset.targets = torch.tensor(testloader.dataset.targets)

        for c in range(10):
            logging.info(f"Training set class {c} percentage: \
                {vsum(trainloader.dataset.targets == c) / num_train}")
        for c in range(10):
            logging.info(f"Validation set class {c} percentage: \
                {vsum(metaloader.dataset.targets == c) / num_val}")
        for c in range(10):
            logging.info(f"Test set class {c} percentage: \
                {vsum(testloader.dataset.targets == c) / num_test}")

    prev_weights = None
    start_epoch = 0

    # if not args.use_pretrained_model:
    if args.dataset == 'MNIST':
        net = DNN_three_layers(args.nce_k, low_dim=args.low_dim)
        
    else:
        net = ResNet18()

    if args.use_pretrained_model:
        # net = load_pretrained_model(args, net)
        net = load_checkpoint(args, net)

    mile_stones_epochs = None

    if args.resume_meta_train:
        net, prev_weights, start_epoch = resume_meta_training_by_loading_cached_info(args, net)

    if args.cuda:
        net = net.cuda()
    if args.dataset == 'MNIST':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)
        scheduler = None
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        optimizer.param_groups[0]['initial_lr'] = args.lr
        if args.do_train:
            mile_stones_epochs = [100, 150]
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                        milestones=mile_stones_epochs, last_epoch=start_epoch-1)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

        else:
            if args.use_pretrained_model:
                mile_stones_epochs = [20,60]
            else:
                mile_stones_epochs = [120,160]
            if args.lr_decay:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=mile_stones_epochs, last_epoch=start_epoch-1)#torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
            else:
                scheduler = None
    
    if args.do_train:
        logging.info("start basic training")
        basic_train(trainloader, validloader, testloader, criterion, args, net, optimizer, scheduler = scheduler)
    else:
        logging.info("start meta training")
        # meta_learning_model_rl(args, net, optimizer, torch.nn.NLLLoss(), trainloader, metaloader, validloader, testloader)
        meta_learning_model(
            args,
            net,
            optimizer,
            criterion,
            meta_criterion,
            trainloader,
            metaloader,
            validloader,
            testloader,
            mnist_to_device,
            scheduler=scheduler,
            cached_w_array=prev_weights,
            target_id=None,
            start_ep=start_epoch,
            mile_stones_epochs=mile_stones_epochs,
        )

def main3(args):

    set_logger(args)
    
    logging.info("start")

    print('==> Preparing data..')

    if args.dataset == 'cifar10':
        args.pool_len = 4

    # model = models.__dict__['ResNet18'](low_dim=args.low_dim, pool_len=args.pool_len, normlinear=args.normlinear).cuda()

    model = resnet18(pretrained=True)

    model = load_checkpoint2(args, model)

    if args.cuda:
        model = model.cuda()

    # net = DNN_two_layers()

    # net_state_dict = torch.load(os.path.join(args.data_dir, "model_full"), map_location=torch.device("cpu"))
        # net_state_dict = torch.load(os.path.join(args.data_dir, "model_logistic_regression"), map_location=torch.device("cpu"))

    # net.load_state_dict(net_state_dict, strict=False)

    # if args.cuda:
    #     net = net.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    if args.select_valid_set:
        trainloader, validloader, metaloader, testloader = get_dataloader_for_meta(args, criterion, split_method='cluster', pretrained_model=model)
    else:
        trainloader, validloader, metaloader, testloader = get_dataloader_for_meta(args, criterion, split_method='random', pretrained_model=model)

    # net = DNN_two_layers()
    if not args.use_pretrained_model:
        model = models.__dict__['ResNet18'](low_dim=args.low_dim, pool_len=args.pool_len, normlinear=args.normlinear).cuda()
        if args.cuda:
            model = model.cuda()

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,#args.batch_size * dist.get_world_size() / 128 * args.base_learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    scheduler = None#get_scheduler(optimizer, len(trainloader), args)

    # meta_learning_model_rl(args, model, optimizer, criterion, trainloader, metaloader, validloader, testloader, scheduler)
    if args.do_train:
        logging.info("start basic training")
        basic_train(trainloader, validloader, testloader, criterion, args, model, optimizer, scheduler)
    else:
        logging.info("start meta training")
        meta_learning_model(args, model, optimizer, criterion, trainloader, metaloader, validloader, testloader, mnist_to_device, scheduler = scheduler, cached_w_array = None, target_id = None)


def main(args):

    set_logger(args)
    
    logging.info("start")

    net = DNN_two_layers()

    
    pre_processing_mnist_main(args)

    
    if args.select_valid_set:
        # train_loader, test_loader = get_mnist_dataset_without_valid_without_perturbations(args)

        train_loader, test_loader = get_mnist_dataset_without_valid_without_perturbations2(args)

        net_state_dict = torch.load(os.path.join(args.data_dir, "model_full"), map_location=torch.device("cpu"))
        # net_state_dict = torch.load(os.path.join(args.data_dir, "model_logistic_regression"), map_location=torch.device("cpu"))

        net.load_state_dict(net_state_dict, strict=False)

        if args.cuda:
            net = net.cuda()

        test(test_loader, net, args, "test")
        # test(train_loader, net, args)

        # train_loader, valid_loader, meta_loader = find_boundary_samples(net, train_loader, args, args.valid_ratio)


        # train_loader, valid_loader, meta_loader = find_boundary_and_representative_samples(net, train_loader, args, args.valid_ratio)
        train_loader, valid_loader, meta_loader = find_representative_samples(net, train_loader, args, args.valid_ratio)
    else:
        train_loader, valid_loader, meta_loader, test_loader = get_mnist_data_loader2(args)

    net = DNN_two_layers()
    
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)

    
    if args.do_train:
        logging.info("start basic training")
        basic_train(train_loader, valid_loader, test_loader, torch.nn.NLLLoss(), args, net, optimizer)
    else:

        logging.info("start meta training")
        # meta_learning_model(args, net, optimizer, torch.nn.NLLLoss(), train_loader, meta_loader, valid_loader, test_loader, mnist_to_device, cached_w_array = None, target_id = None)
        meta_learning_model_rl(args, net, optimizer, torch.nn.NLLLoss(), train_loader, meta_loader, valid_loader, test_loader)

if __name__ == "__main__":
    args = parse_args()
    dist.init_process_group(backend='nccl', init_method='env://')
    args.world_size = dist.get_world_size()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)

    cudnn.benchmark = True
    
    os.makedirs(args.save_path, exist_ok=True)
    logger = setup_logger(output=args.save_path, distributed_rank=dist.get_rank(), name="moco+cld")
    if dist.get_rank() == 0:
        path = os.path.join(args.save_path, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))
    args.device = torch.device("cuda", args.local_rank)
    # if args.dataset == 'MNIST':
    main2(args)
    # else:
    #     main3(args)