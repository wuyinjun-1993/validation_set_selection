import torch

import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from common.parse_args import *
from exp_datasets.mnist import *
from models.DNN import *
from common.utils import *
from tqdm.notebook import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import itertools
import torch_higher as higher
from main.find_valid_set import *
from main.meta_reweighting_rl import *
from exp_datasets.dataloader import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import json
from utils.logger import setup_logger
import models
from lib.NCECriterion import NCESoftmaxLoss
from lib.lr_scheduler import get_scheduler
from lib.BootstrappingLoss import SoftBootstrappingLoss, HardBootstrappingLoss
from models.resnet import *
from models.resnet3 import *
from models.bert import *
import collections
from models.LeNet5 import *
import models.TAVAAL

import torchvision.models

cached_model_name="cached_model"
pretrained_model_name="pretrained_model"

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

def report_best_test_performance_so_far(logger, test_loss_ls, test_acc_ls,
        test_loss, test_acc, set_name):
    test_loss_ls.append(test_loss)
    test_acc_ls.append(test_acc)

    test_loss_array = numpy.array(test_loss_ls)
    test_acc_array = numpy.array(test_acc_ls)
    max_acc_epoch = numpy.argmax(test_acc_array)

    min_test_loss = test_loss_array[max_acc_epoch]
    min_test_acc = test_acc_array[max_acc_epoch]

    logger.info("best %s performance is in epoch %d: %f, %f"%(set_name, max_acc_epoch, min_test_loss, min_test_acc))

def report_final_performance_by_early_stopping(valid_loss_ls, valid_acc_ls,
        test_loss_ls, test_acc_ls, args, logger, is_meta=True):
    valid_acc_arr = numpy.array(valid_acc_ls)
    best_valid_acc_idx = numpy.argmax(valid_acc_arr)

    final_test_loss = test_loss_ls[best_valid_acc_idx]
    final_test_acc = test_acc_ls[best_valid_acc_idx]

    logger.info("final test performance is in epoch %d: %f, %f"%(best_valid_acc_idx, final_test_loss, final_test_acc))

    torch.save(best_valid_acc_idx, os.path.join(args.save_path, "early_stopping_epoch"))

    if is_meta:
        cache_sample_weights_given_epoch(best_valid_acc_idx, args)
    else:
        cache_sample_weights_given_epoch_basic_train(best_valid_acc_idx, args)

    return best_valid_acc_idx


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

    torch.save(final_epoch, os.path.join(args.save_path, "early_stopping_epoch"))
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


def cache_sample_weights_given_epoch(epoch, args):
    best_w_array = torch.load(os.path.join(args.save_path, 'sample_weights_' + str(epoch)))
    best_model = torch.load(os.path.join(args.save_path, 'refined_model_' + str(epoch)))

    logging.info("caching sample weights at epoch %d"%(epoch))

    torch.save(best_w_array, os.path.join(args.save_path, "cached_sample_weights"))
    torch.save(best_model, os.path.join(args.save_path, cached_model_name))
    torch.save(best_model, os.path.join(args.save_path, pretrained_model_name))


def cache_sample_weights_given_epoch_basic_train(epoch, args):
    best_model = torch.load(os.path.join(args.save_path, 'model_' + str(epoch)))
    torch.save(best_model, os.path.join(args.save_path, cached_model_name))
    torch.save(best_model, os.path.join(args.save_path, pretrained_model_name))


def remove_intermediate_models(epochs_to_remove):
    for ep in epochs_to_remove:
        model_path = os.path.join(args.save_path, 'model_' + str(ep))
        if os.path.exists(model_path):
            os.remove(model_path)


def resume_training_by_epoch(args, model):
    model_file_name = os.path.join(args.save_path, "model_" + str(args.resumed_training_epoch))
    
    state = torch.load(model_file_name)
    if type(state) is collections.OrderedDict:
        model.load_state_dict(state)
    else:
        model.load_state_dict(state.state_dict())
    return model

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
    logger,
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
    heuristic=None,
    gt_training_labels=None,
):
    logger.info("Meta set set: {}".format(len(meta_loader.dataset)))
    
    
    if args.cuda:
        device = torch.device("cuda:" + str(torch.cuda.current_device()))
    else:
        device = torch.device("cpu")

    if cached_w_array is None:
        if args.w_rectified_gaussian_init:
            w_array = torch.normal(
                mean=0.0,
                std=1.0,
                size=(len(train_loader.dataset),),
                device=device,
            )
            w_array = F.relu(w_array)
            w_array = (w_array / torch.sum(w_array)) * w_array.shape[0]
            w_array.requires_grad = True
        else:
            w_array = torch.rand(len(train_loader.dataset), requires_grad=True, device = device)
            # w_array = torch.ones(len(train_loader.dataset), requires_grad=True, device=device)
    else:
        cached_w_array.requires_grad = False
        w_array = cached_w_array.clone()
        w_array = w_array.to(device)
        w_array.requires_grad = True
    
    total_iter_count = 1

    curr_ilp_learning_rate = args.meta_lr

    valid_loss_ls = []
    valid_acc_ls = []
    test_loss_ls = []
    test_acc_ls = []
    w_array_delta_ls = []

    for ep in tqdm(range(start_ep, args.epochs)):
        
        train_loss, train_acc = 0, 0
        rand_epoch_seed = random.randint(0, args.epochs*10)
        train_loader.sampler.set_epoch(rand_epoch_seed)
        curr_w_array_delta = torch.zeros_like(w_array)

        avg_train_loss = 0

        train_pred_correct = 0

        if args.active_learning:
            with torch.no_grad():
                # Select 10 samples based on heuristic and assign correct label
                _, indices = torch.sort(
                    heuristic(model.module, train_loader), descending=True
                )
                metaset = meta_loader.dataset
                metaset.targets = torch.cat(
                    (meta_loader.dataset.targets, gt_training_labels[indices[:5]]),
                    dim=0,
                )
                metaset.data = torch.cat(
                    (meta_loader.dataset.data, train_loader.dataset.data[indices[:5]]),
                    dim=0,
                )
                torch.save(indices[:5], os.path.join(args.save_path,
                    'actively_chosen_samples_' + str(ep)))
                logger.info("meta dataset size::%d"%(len(metaset.targets)))
                meta_sampler = torch.utils.data.distributed.DistributedSampler(
                    metaset,
                    num_replicas=args.world_size,
                    rank=args.local_rank,
                )
                meta_loader = torch.utils.data.DataLoader(
                    metaset,
                    batch_size=args.test_batch_size,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    sampler=meta_sampler,
                )

        metaloader = itertools.cycle(meta_loader)

        model.train()
        for idx, inputs in enumerate(train_loader):
            # inputs, labels = inputs.to(device=args['device'], non_blocking=True),\
                                # labels.to(device=args['device'], non_blocking=True)
            # train_ids = (train_loader.dataset.indices.view(1,-1) == inputs[0].view(-1,1)).nonzero()[:,1]
            train_ids = inputs[0]
            if args.dataset == 'MNIST':
                assert len(torch.nonzero(train_loader.dataset.__getitem__(train_ids[0])[1] - inputs[1][0])) == 0

            
            # inputs = to_device(inputs, args)
            if args.cuda:
                inputs[1], inputs[2] = train_loader.dataset.to_cuda(inputs[1], inputs[2])
            
            w_array.requires_grad = True
            
            opt.zero_grad()
            with higher.innerloop_ctx(model.module, opt) as (meta_model, meta_opt):
                
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
                if type(meta_train_outputs) is tuple:
                    meta_train_outputs = meta_train_outputs[0]

                labels = inputs[2]
                if isinstance(criterion, torch.nn.L1Loss):
                    labels = F.one_hot(inputs[2],
                            num_classes=10)
                    meta_train_outputs = F.softmax(meta_train_outputs)
                meta_train_loss = torch.mean(criterion(meta_train_outputs, labels)*eps)
                
                
                meta_opt.step(meta_train_loss)
    
                # 2. Compute grads of eps on meta validation data
                # meta_inputs, meta_labels =  next(meta_loader)
                # meta_inputs, meta_labels = meta_inputs.to(device=args['device'], non_blocking=True),\
                #                  meta_labels.to(device=args['device'], non_blocking=True)
                meta_inputs =  next(metaloader)
                
                # print(meta_inputs)
                if args.cuda:
                    meta_inputs[1], meta_inputs[2] = train_loader.dataset.to_cuda(meta_inputs[1], meta_inputs[2])
                # meta_inputs = to_device(meta_inputs, args)
                
                if criterion is not None:
                    criterion.reduction = 'mean'
                
                # meta_val_loss,meta_valid_numbers = loss_func(meta_inputs, meta_model, criterion, no_agg = False)
                
                meta_out = meta_inputs[2]
                model_out = meta_model(meta_inputs[1])
                if type(model_out) is tuple:
                    model_out = model_out[0]
                if isinstance(meta_criterion, torch.nn.L1Loss):
                    meta_out = F.one_hot(meta_inputs[2],
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
            if type(model_out) is tuple:
                model_out = model_out[0]
            labels = inputs[2]
            if isinstance(criterion, torch.nn.L1Loss):
                labels = F.one_hot(inputs[2],
                        num_classes=10)
                model_out = F.softmax(meta_train_outputs)
            minibatch_loss = torch.mean(criterion(model_out, labels)*w_array[train_ids])

            minibatch_loss.backward()
            opt.step()
            

            avg_train_loss += minibatch_loss.detach().cpu().item()*inputs[2].shape[0]

            model_pred = torch.max(model_out, dim = 1)[1]

            train_pred_correct += torch.sum(model_pred.view(-1).detach().cpu() == inputs[2].detach().cpu().view(-1)).item()


            total_iter_count += 1

            if idx % 10 == 0:
                logger.info("Loss at iter %d: %f"%(idx, minibatch_loss.detach().cpu()))


            # del meta_inputs[2], inputs[2]

            # if type(meta_inputs[1]) is not torch.Tensor:
            #     meta_inputs[1] = list(meta_inputs[1])
            #     lenth = len(meta_inputs[1])
            #     for tensor_idx in range(lenth):
            #         del meta_inputs[1][lenth - 1 - tensor_idx]
            # else:
            #     del meta_inputs[1]

            # if type(inputs[1]) is not torch.Tensor:
            #     inputs[1] = list(inputs[1])
            #     lenth = len(inputs[1])
            #     for tensor_idx in range(lenth):
            #         del inputs[1][lenth - 1 - tensor_idx]

            # else:
            #     del inputs[1]


            # del prev_w_array, meta_out, meta_inputs[0], inputs[0], minibatch_loss, meta_val_loss, meta_train_loss, model_out, eps_grads, model_pred, meta_train_outputs, labels, eps
            # if args.cuda:
            #     torch.cuda.empty_cache()

        # inference after epoch
        with torch.no_grad():
            if args.local_rank == 0:
                avg_train_loss = avg_train_loss/len(train_loader.dataset)
                train_pred_acc_rate = train_pred_correct*1.0/len(train_loader.dataset)
                logger.info("average training loss at epoch %d:%f"%(ep, avg_train_loss))

                logger.info("training accuracy at epoch %d:%f"%(ep, train_pred_acc_rate))
                if criterion is not None:
                    criterion.reduction = 'mean'
                if len(valid_loader) > 0:
                    logger.info("valid performance at epoch %d"%(ep))
                    valid_loss, valid_acc = test(valid_loader, model, criterion, args, logger, "valid")
                    report_best_test_performance_so_far(logger, valid_loss_ls,
                        valid_acc_ls, valid_loss, valid_acc, "valid")

                logger.info("test performance at epoch %d"%(ep))
                test_loss, test_acc = test(test_loader, model, criterion, args, logger, "test")
                report_best_test_performance_so_far(logger, test_loss_ls,
                        test_acc_ls, test_loss, test_acc, "test")

        if scheduler is not None:
            logger.info("learning rate at iteration %d before using scheduler: %f" %(int(ep), float(opt.param_groups[0]['lr'])))

            scheduler.step()
            logger.info("learning rate at iteration %d after using scheduler: %f" %(int(ep), float(opt.param_groups[0]['lr'])))

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
                
            else:
                if ep > 120:
                    curr_ilp_learning_rate = args.meta_lr*0.1
                    # logger.info("meta learning rate at iteration %d: %f" %(int(ep), curr_ilp_learning_rate))

            logger.info("meta learning rate at iteration %d: %f" %(int(ep), curr_ilp_learning_rate))
            # min_valid_loss_epoch = numpy.argmin(numpy.array(valid_loss_ls))
            # if min_valid_loss_epoch < ep-1:
            # opt, curr_learning_rate = vary_learning_rate(curr_learning_rate, ep, args, model=model)

                
        w_array_delta_ls.append(curr_w_array_delta)
            

        if args.local_rank == 0:
            torch.save(model.module.state_dict(), os.path.join(args.save_path, 'refined_model_' + str(ep)))
            torch.save(w_array, os.path.join(args.save_path, 'sample_weights_' + str(ep)))
            torch.save(model.module.state_dict(), os.path.join(args.save_path, 'curr_refined_model'))
            torch.save(w_array, os.path.join(args.save_path, 'curr_sample_weights'))
            torch.save(torch.tensor(ep), os.path.join(args.save_path, "curr_epoch"))
            torch.save(torch.stack(w_array_delta_ls, dim = 0), os.path.join(args.save_path, "curr_w_array_delta_ls"))

    if args.local_rank == 0:
        report_final_performance_by_early_stopping(valid_loss_ls, valid_acc_ls,
                test_loss_ls, test_acc_ls, args, logger)
    # w_array_delta_ls_tensor = torch.stack(w_array_delta_ls, dim = 0)

    # if args.local_rank == 0:
    #     torch.save(w_array_delta_ls_tensor, os.path.join(args.save_path, "cached_w_array_delta_ls"))
    #     torch.save(torch.sum(w_array_delta_ls_tensor, dim = 1), os.path.join(args.save_path, "cached_w_array_total_delta"))

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def basic_train(train_loader, valid_loader, test_loader, criterion, args,
        network, optimizer, scheduler=None, heuristic=None,
        warmup_scheduler=None, gt_training_labels=None, start_epoch = 0):

    curr_lr = args.lr
    valid_loss_ls = []
    valid_acc_ls = []
    test_loss_ls = []
    test_acc_ls = []
    for epoch in tqdm(range(start_epoch, args.epochs+start_epoch)):
        rand_epoch_seed = random.randint(0, args.epochs*10)
        train_loader.sampler.set_epoch(rand_epoch_seed)
        if args.active_learning:
            with torch.no_grad():
                # Select 10 samples based on heuristic and assign correct label
                _, indices = torch.sort(
                    heuristic(network, train_loader), descending=True
                )
                train_loader.dataset.targets[indices[:10]] = gt_training_labels[indices[:10]]

        network.train()
        for batch_idx, (_, data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = train_loader.dataset.to_cuda(data, target)

            output = network(data)
            if isinstance(criterion, torch.nn.L1Loss):
                target = F.one_hot(target, num_classes=10)
                output = F.softmax(output)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                logger.info("Loss at batch %d: %f"%(batch_idx, loss))
            # if epoch < args.warm and warmup_scheduler is not None:
            #     warmup_scheduler.step()
        

        if scheduler is not None:
            scheduler.step(epoch)
        if args.local_rank == 0:
            model_path = os.path.join(args.save_path, "model_" + str(epoch))
            torch.save(network.module.state_dict(), model_path)

        logger.info("learning rate at epoch %d: %f"%(epoch, float(optimizer.param_groups[0]['lr'])))
        with torch.no_grad():
            if valid_loader is not None and args.local_rank == 0:
                valid_loss, valid_acc = test(valid_loader, network, criterion, args, logger, "valid")
                report_best_test_performance_so_far(logger, valid_loss_ls,
                        valid_acc_ls, valid_loss, valid_acc, "valid")
            
            if args.local_rank == 0:
                test_loss, test_acc = test(test_loader, network, criterion, args, logger, "test")
                report_best_test_performance_so_far(logger, test_loss_ls,
                        test_acc_ls, test_loss, test_acc, "test")

    if args.local_rank == 0:
        best_index = report_final_performance_by_early_stopping(valid_loss_ls,
                valid_acc_ls, test_loss_ls, test_acc_ls, args, logger, is_meta=False)


def get_confusion_for_glc(meta_loader, net, num_classes):
    meta_data = []
    meta_target = []
    for _, (_, data, target) in enumerate(meta_loader):
        meta_data.append(data)
        meta_target.append(target)
    meta_data = torch.cat(meta_data)
    meta_target = torch.cat(meta_target)

    C = torch.zeros((num_classes, num_classes)).cuda()
    for i in range(num_classes):
        num_examples = 0
        matches = torch.nonzero(meta_target == i)
        for match in matches:
            num_examples += 1
            with torch.no_grad():
                C[i, :] += net(meta_data[match]).flatten()
        C[i, :] /= num_examples
    return C


def glc_train(train_loader, valid_loader, test_loader, meta_set, criterion, args,
        network, optimizer, scheduler=None, heuristic=None,
        warmup_scheduler=None, gt_training_labels=None, start_epoch = 0):
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10

    curr_lr = args.lr
    valid_loss_ls = []
    valid_acc_ls = []
    test_loss_ls = []
    test_acc_ls = []

    network.eval()
    C = get_confusion_for_glc(meta_set, network, num_classes)
    for epoch in tqdm(range(start_epoch, args.epochs+start_epoch)):
        rand_epoch_seed = random.randint(0, args.epochs*10)
        train_loader.sampler.set_epoch(rand_epoch_seed)
        network.train()
        for batch_idx, (_, data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = train_loader.dataset.to_cuda(data, target)

            output = network(data)
            if isinstance(criterion, torch.nn.L1Loss):
                target = F.one_hot(target, num_classes=num_classes)
                output = F.softmax(output)
            loss = criterion(torch.matmul(output, C), target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if epoch < args.warm and warmup_scheduler is not None:
            #     warmup_scheduler.step()
        

        if scheduler is not None:
            scheduler.step(epoch)
        if args.local_rank == 0:
            model_path = os.path.join(args.save_path, "model_" + str(epoch))
            torch.save(network.module.state_dict(), model_path)

        logger.info("learning rate at epoch %d: %f"%(epoch, float(optimizer.param_groups[0]['lr'])))
        with torch.no_grad():
            if valid_loader is not None and args.local_rank == 0:
                valid_loss, valid_acc = test(valid_loader, network, criterion, args, logger, "valid")
                report_best_test_performance_so_far(logger, valid_loss_ls,
                        valid_acc_ls, valid_loss, valid_acc, "valid")
            
            if args.local_rank == 0:
                test_loss, test_acc = test(test_loader, network, criterion, args, logger, "test")
                report_best_test_performance_so_far(logger, test_loss_ls,
                        test_acc_ls, test_loss, test_acc, "test")

    if args.local_rank == 0:
        best_index = report_final_performance_by_early_stopping(valid_loss_ls,
                valid_acc_ls, test_loss_ls, test_acc_ls, args, logger, is_meta=False)


def uncertainty_heuristic(model, train_loader):
    vals = torch.zeros((train_loader.dataset.targets.shape[0],))
    for _, (indices, data, _) in enumerate(train_loader):
        if args.cuda:
            data = data.cuda()
        output = model(data)
        vals[indices] = F.cross_entropy(output, output).cpu()
    return vals


def active_learning(train_loader, valid_loader, test_loader, criterion,
        gt_training_labels, heuristic, args, network, optimizer, scheduler = None):

    valid_loss_ls = []
    valid_acc_ls = []
    test_loss_ls = []
    test_acc_ls = []
    for epoch in range(args.epochs):
        with torch.no_grad():
            # Select 10 samples based on heuristic and assign correct label
            _, indices = torch.sort(
                heuristic(network, train_loader), descending=True
            )
            train_loader.dataset.targets[indices[:10]] = gt_training_labels[indices[:10]]

        network.train()
        for _, (_, data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            if args.cuda:
                data = data.cuda()
                target = target.cuda()
            output = network(data)
            if isinstance(criterion, torch.nn.L1Loss):
                target = F.one_hot(target, num_classes=10)
                output = F.softmax(output)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if args.local_rank == 0:
            torch.save(network.module.state_dict(), os.path.join(args.save_path, "model_" + str(epoch)))
        logger.info("learning rate at epoch %d: %f"%(epoch, float(optimizer.param_groups[0]['lr'])))
        
        with torch.no_grad():
            if valid_loader is not None and args.local_rank == 0:
                valid_loss, valid_acc = test(valid_loader, network, criterion, args, logger, "valid")
                report_best_test_performance_so_far(logger, valid_loss_ls,
                        valid_acc_ls, valid_loss, valid_acc, "valid")

            if args.local_rank == 0:
                test_loss, test_acc = test(test_loader, network, criterion, args, logger, "test")
                report_best_test_performance_so_far(logger, test_loss_ls,
                        test_acc_ls, test_loss, test_acc, "test")

    if args.local_rank == 0:
        report_final_performance_by_early_stopping(valid_loss_ls, valid_acc_ls,
                test_loss_ls, test_acc_ls, args, logger, is_meta=False)


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
    args.logger.info('==> Loading cached model...')
    if args.prev_save_path is not None:
        cached_model_file_name = os.path.join(args.prev_save_path, cached_model_name)
        if os.path.exists(cached_model_file_name):
            
            state = torch.load(cached_model_file_name, map_location=torch.device("cpu"))

            if type(state) is collections.OrderedDict:
                model.load_state_dict(state)
            else:
                model.load_state_dict(state.state_dict())
            args.logger.info('==> Loading cached model successfully')
            del state
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
    if os.path.exists(os.path.join(args.prev_save_path, pretrained_model_name)):
        logging.info("load pretrained models")

        pretrained_model_state = torch.load(os.path.join(args.prev_save_path, pretrained_model_name))

        if hasattr(pretrained_model_state, "state_dict"):
            pretrained_model_state = pretrained_model_state.state_dict()

        net.load_state_dict(pretrained_model_state, strict=False)

    return net
    

def main2(args, logger):
    logger.info("start")
    logger.info('==> Preparing data..')

    if args.dataset == 'MNIST':
        # if args.model_type == 'lenet':
        pretrained_rep_net = LeNet5()
        # else:
        #     pretrained_rep_net = DNN_three_layers(args.nce_k, low_dim=args.low_dim).cuda()
        optimizer = torch.optim.SGD(pretrained_rep_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        optimizer.param_groups[0]['initial_lr'] = args.lr
    elif args.dataset.startswith('cifar'):
        if args.dataset == 'cifar10':
            if args.model_type == 'resnet18':
                pretrained_rep_net = ResNet18(num_classes=10).cuda()
                args.num_class=10
            else:    
                pretrained_rep_net = resnet34(num_classes=10).cuda()
                args.num_class=10
            optimizer = torch.optim.SGD(pretrained_rep_net.parameters(),
                    lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        else:
            if args.model_type == 'resnet18':
                pretrained_rep_net = ResNet18(num_classes=100).cuda()
                args.num_class=100
            else:
                pretrained_rep_net = resnet34(num_classes=100).cuda()
                args.num_class=100
            optimizer = torch.optim.SGD(pretrained_rep_net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
        pretrained_rep_net.eval()
        
        optimizer.param_groups[0]['initial_lr'] = args.lr
    elif args.dataset == 'retina':
        pretrained_rep_net = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1).cuda()
        pretrained_rep_net.fc = nn.Linear(512, 5)
        optimizer = torch.optim.Adam(pretrained_rep_net.parameters(), lr=args.lr, weight_decay=5e-4)
        pretrained_rep_net.eval()
        
        optimizer.param_groups[0]['initial_lr'] = args.lr
    elif args.dataset.startswith('sst2'):
        pretrained_rep_net = custom_Bert(2)
        args.num_class=2
        # pretrained_rep_net = init_model_with_pretrained_model_weights(pretrained_rep_net)
        optimizer = torch.optim.Adam(pretrained_rep_net.parameters(), lr=args.lr)# get_bert_optimizer(net, args.lr)
    elif args.dataset.startswith('sst5'):
        pretrained_rep_net = custom_Bert(5)
        args.num_class=5
        # pretrained_rep_net = init_model_with_pretrained_model_weights(pretrained_rep_net)
        optimizer = torch.optim.Adam(pretrained_rep_net.parameters(), lr=args.lr)# get_bert_optimizer(net, args.lr)
    elif args.dataset.startswith('imdb'):
        pretrained_rep_net = custom_Bert(2)
        args.num_class=2
        # pretrained_rep_net = init_model_with_pretrained_model_weights(pretrained_rep_net)
        optimizer = torch.optim.Adam(pretrained_rep_net.parameters(), lr=args.lr)# get_bert_optimizer(net, args.lr)
    elif args.dataset.startswith('trec'):
        pretrained_rep_net = custom_Bert(6)
        args.num_class=6
        # pretrained_rep_net = init_model_with_pretrained_model_weights(pretrained_rep_net)
        optimizer = torch.optim.Adam(pretrained_rep_net.parameters(), lr=args.lr)# get_bert_optimizer(net, args.lr)
    else:
        raise NotImplementedError
        # pretrained_rep_net = ResNet18().cuda()
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
        logger.info("sample weights loaded successfully")
    if args.cuda:
        pretrained_rep_net.cuda()
    
    if args.select_valid_set:
        trainloader, validloader, metaloader, testloader, origin_labels = get_dataloader_for_meta(
            criterion,
            optimizer,
            args,
            'cluster',
            logger,
            pretrained_model=pretrained_rep_net,
            cached_sample_weights=cached_sample_weights,
        )
    elif args.uncertain_select:
        trainloader, validloader, metaloader, testloader, origin_labels = get_dataloader_for_meta(
            criterion,
            optimizer,
            args,
            'uncertainty',
            logger,
            pretrained_model=pretrained_rep_net,
            cached_sample_weights=cached_sample_weights,
        )
    elif args.certain_select:
        trainloader, validloader, metaloader, testloader, origin_labels = get_dataloader_for_meta(
            criterion,
            optimizer,
            args,
            'certainty',
            logger,
            pretrained_model=pretrained_rep_net,
            cached_sample_weights=cached_sample_weights,
        )
    elif args.craige:
        trainloader, validloader, metaloader, testloader, origin_labels = get_dataloader_for_meta(
            criterion,
            optimizer,
            args,
            'craige',
            logger,
            pretrained_model=pretrained_rep_net,
            cached_sample_weights=cached_sample_weights,
        )
    else:
        trainloader, validloader, metaloader, testloader, origin_labels = get_dataloader_for_meta(
            criterion,
            optimizer,
            args,
            'random',
            logger,
            pretrained_model=pretrained_rep_net,
        )

    del pretrained_rep_net

    if args.cuda:
        torch.cuda.empty_cache()

    if args.l1_loss:
        criterion = torch.nn.L1Loss()
    elif args.soft_bootstrapping_loss:
        criterion = SoftBootstrappingLoss()
    elif args.hard_bootstrapping_loss:
        criterion = HardBootstrappingLoss()

    if args.bias_classes:
        num_train = len(trainloader.dataset.targets)
        num_val = len(validloader.dataset.targets)

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
            logger.info(f"Training set class {c} percentage: \
                {vsum(trainloader.dataset.targets == c) / num_train}")
        for c in range(10):
            logger.info(f"Validation set class {c} percentage: \
                {vsum(validloader.dataset.targets == c) / num_val}")
        if metaloader is not None:
            num_meta = len(metaloader.dataset.targets)
            for c in range(10):
                logger.info(f"Meta set class {c} percentage: \
                    {vsum(metaloader.dataset.targets == c) / num_meta}")
        for c in range(10):
            logger.info(f"Test set class {c} percentage: \
                {vsum(testloader.dataset.targets == c) / num_test}")

    prev_weights = None
    start_epoch = 0

    if args.dataset == 'MNIST':
        # if args.model_type == 'lenet':
        net = LeNet5()
        # else:
        #     net = DNN_three_layers(args.nce_k, low_dim=args.low_dim)
    elif args.dataset.startswith('cifar'):
        if args.dataset == 'cifar10':
            if args.model_type == 'resnet18':
                net = ResNet18(num_classes=10)
            else:
                net = resnet34(num_classes=10)
        elif args.dataset == 'cifar100':
            if args.model_type == 'resnet18':
                net = ResNet18(num_classes=100)
            else:
                net = resnet34(num_classes=100)
    elif args.dataset == 'retina':
        net = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1).cuda()
        for param in net.conv1.parameters():
            param.requires_grad = False
        for param in net.layer1.parameters():
            param.requires_grad = False
        for param in net.layer2.parameters():
            param.requires_grad = False
        net.fc = nn.Linear(512, 5)
    elif args.dataset.startswith('sst2'):
        net = custom_Bert(2)
    elif args.dataset.startswith('sst5'):
        net = custom_Bert(5)
    elif args.dataset.startswith('imdb'):
        net = custom_Bert(2)
    elif args.dataset.startswith('trec'):
        net = custom_Bert(6)
    else:
        raise NotImplementedError

    if args.use_pretrained_model:
        net = load_checkpoint2(args, net)

    mile_stones_epochs = None

    if not args.do_train and args.resume_meta_train:
        net, prev_weights, start_epoch = resume_meta_training_by_loading_cached_info(args, net)
    if args.do_train and args.resume_train:
        net = resume_training_by_epoch(args, net)
        start_epoch = args.resumed_training_epoch
        # net, prev_weights, start_epoch = resume_meta_training_by_loading_cached_info(args, net)

    if args.cuda:
        net = net.cuda()

    net = DDP(net, device_ids=[args.local_rank])
    optimizer, (scheduler, mile_stones_epochs) = obtain_optimizer_scheduler(args, net, start_epoch = start_epoch)
    
    warmup_scheduler = None

    # if args.dataset == 'cifar100':
    #     iter_per_epoch = len(trainloader)
    #     warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.do_train:
        logger.info("starting basic training")
        basic_train(
            trainloader,
            validloader,
            testloader,
            criterion,
            args,
            net,
            optimizer,
            scheduler=scheduler,
            heuristic=uncertainty_heuristic if args.active_learning else None,
            warmup_scheduler=warmup_scheduler,
            gt_training_labels=torch.tensor(origin_labels) if args.active_learning else None,
            start_epoch=start_epoch
        )
    elif args.finetune:
        logger.info("starting finetuning")
        basic_train(
            metaloader,
            validloader,
            testloader,
            criterion,
            args,
            net,
            optimizer,
            scheduler=scheduler,
            heuristic=uncertainty_heuristic if args.active_learning else None,
            warmup_scheduler=warmup_scheduler,
            gt_training_labels=torch.tensor(origin_labels) if args.active_learning else None,
            start_epoch=start_epoch
        )
    elif args.glc_train:
        logger.info("starting glc training")
        glc_train(
            trainloader,
            validloader,
            testloader,
            metaloader,
            criterion,
            args,
            net,
            optimizer,
            scheduler=scheduler,
            heuristic=uncertainty_heuristic if args.active_learning else None,
            warmup_scheduler=warmup_scheduler,
            gt_training_labels=torch.tensor(origin_labels) if args.active_learning else None,
            start_epoch=start_epoch
        )
    elif args.ta_vaal_train:
        logger.info("starting TA-VAAL training")
        models.TAVAAL.main_train_taaval(
            args,
        )
    else:
        logger.info("starting meta training")
        logger.info("meta dataset size::%d"%(len(metaloader.dataset)))
        meta_learning_model(
            args,
            logger,
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
            heuristic=uncertainty_heuristic if args.active_learning else None,
            gt_training_labels=torch.tensor(origin_labels) if args.active_learning else None,
        )

# def main3(args):

#     set_logger(args)
    
#     logging.info("start")

#     print('==> Preparing data..')

#     if args.dataset == 'cifar10':
#         args.pool_len = 4

#     # model = models.__dict__['ResNet18'](low_dim=args.low_dim, pool_len=args.pool_len, normlinear=args.normlinear).cuda()

#     model = resnet18(pretrained=True)

#     model = load_checkpoint2(args, model)

#     if args.cuda:
#         model = model.cuda()

#     # net = DNN_two_layers()

#     # net_state_dict = torch.load(os.path.join(args.data_dir, "model_full"), map_location=torch.device("cpu"))
#         # net_state_dict = torch.load(os.path.join(args.data_dir, "model_logistic_regression"), map_location=torch.device("cpu"))

#     # net.load_state_dict(net_state_dict, strict=False)

#     # if args.cuda:
#     #     net = net.cuda()
#     criterion = nn.CrossEntropyLoss().cuda()
#     if args.select_valid_set:
#         trainloader, validloader, metaloader, testloader = get_dataloader_for_meta(criterion, optimizer,args, criterion, split_method='cluster', pretrained_model=model)
#     else:
#         trainloader, validloader, metaloader, testloader = get_dataloader_for_meta(criterion, optimizer,args, criterion, split_method='random', pretrained_model=model)

#     # net = DNN_two_layers()
#     if not args.use_pretrained_model:
#         model = models.__dict__['ResNet18'](low_dim=args.low_dim, pool_len=args.pool_len, normlinear=args.normlinear).cuda()
#         if args.cuda:
#             model = model.cuda()

#     # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    
#     optimizer = torch.optim.SGD(model.parameters(),
#                                 lr=args.lr,#args.batch_size * dist.get_world_size() / 128 * args.base_learning_rate,
#                                 momentum=args.momentum,
#                                 weight_decay=args.weight_decay)
#     scheduler = None#get_scheduler(optimizer, len(trainloader), args)

#     # meta_learning_model_rl(args, model, optimizer, criterion, trainloader, metaloader, validloader, testloader, scheduler)
#     if args.do_train:
#         logging.info("start basic training")
#         basic_train(trainloader, validloader, testloader, criterion, args, model, optimizer, scheduler)
#     else:
#         logging.info("start meta training")
#         meta_learning_model(args, model, optimizer, criterion, trainloader, metaloader, validloader, testloader, mnist_to_device, scheduler = scheduler, cached_w_array = None, target_id = None)


# def main(args):

#     set_logger(args)
    
#     logging.info("start")

#     net = DNN_two_layers()

    
#     pre_processing_mnist_main(args)

    
#     if args.select_valid_set:
#         # train_loader, test_loader = get_mnist_dataset_without_valid_without_perturbations(args)

#         train_loader, test_loader = get_mnist_dataset_without_valid_without_perturbations2(args)

#         net_state_dict = torch.load(os.path.join(args.data_dir, "model_full"), map_location=torch.device("cpu"))
#         # net_state_dict = torch.load(os.path.join(args.data_dir, "model_logistic_regression"), map_location=torch.device("cpu"))

#         net.load_state_dict(net_state_dict, strict=False)

#         if args.cuda:
#             net = net.cuda()

#         test(test_loader, net, args, "test")
#         # test(train_loader, net, args)

#         # train_loader, valid_loader, meta_loader = find_boundary_samples(net, train_loader, args, args.valid_ratio)


#         # train_loader, valid_loader, meta_loader = find_boundary_and_representative_samples(net, train_loader, args, args.valid_ratio)
#         train_loader, valid_loader, meta_loader = find_representative_samples(net, train_loader, args, args.valid_ratio)
#     else:
#         train_loader, valid_loader, meta_loader, test_loader = get_mnist_data_loader2(args)

#     net = DNN_two_layers()
    
#     optimizer = torch.optim.SGD(net.parameters(), lr=args.lr)

    
#     if args.do_train:
#         logging.info("start basic training")
#         basic_train(train_loader, valid_loader, test_loader, torch.nn.NLLLoss(), args, net, optimizer)
#     else:

#         logging.info("start meta training")
#         # meta_learning_model(args, net, optimizer, torch.nn.NLLLoss(), train_loader, meta_loader, valid_loader, test_loader, mnist_to_device, cached_w_array = None, target_id = None)
#         meta_learning_model_rl(args, net, optimizer, torch.nn.NLLLoss(), train_loader, meta_loader, valid_loader, test_loader)

if __name__ == "__main__":
    args = parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    args.world_size = dist.get_world_size()

    cudnn.benchmark = True
    
    os.makedirs(args.save_path, exist_ok=True)
    logger = setup_logger(output=args.save_path, distributed_rank=dist.get_rank(), name="valid-selec")
    
    if dist.get_rank() == 0:
        path = os.path.join(args.save_path, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))
    args.device = torch.device("cuda", args.local_rank)
    args.logger = logger
    with logging_redirect_tqdm():
        main2(args, logger)
