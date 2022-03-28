import torch
import torch.nn as nn
import logging
import itertools

import torch_higher as higher

from tqdm.notebook import tqdm
import os,sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from main.helper_func import *
from models.magic_module import *
import torch.optim as optim
from torch.autograd import Variable

EPSILON = 1e-5


def init_weights(n_examples, w_init):
        # assert self._ren is False and self._baseline is False
    weights = torch.tensor(
            [w_init] * n_examples, requires_grad=True).to('cuda')
        # w_decay = w_decay
    return weights

def get_weights(args, model, optimizer, batch, valid_loader, w_array, w_decay):
    model.eval()

    ids, inputs, labels = tuple(t.to('cuda') for t in batch)
    batch_size = inputs.shape[0]

    # if self._baseline:
    #     return torch.ones(batch_size).to('cuda')
    # elif self._ren:
    #     weights = torch.zeros(batch_size, requires_grad=True).to('cuda')
    # else:
    weights = w_array[ids]

    magic_model = MagicModule(model)
    criterion = nn.CrossEntropyLoss()

    model_tmp = copy.deepcopy(model)
    if args.cuda:
        model_tmp = model_tmp.cuda()
    optimizer_hparams = optimizer.state_dict()['param_groups'][0]
    optimizer_tmp = optim.SGD(
        model_tmp.parameters(),
        lr=optimizer_hparams['lr'],
        momentum=optimizer_hparams['momentum'],
        weight_decay=optimizer_hparams['weight_decay'])

    inputs = Variable(inputs)
    labels = Variable(labels)

    for i in range(batch_size):
        model_tmp.load_state_dict(model.state_dict())
        optimizer_tmp.load_state_dict(optimizer.state_dict())

        model_tmp.zero_grad()

        if i > 0:
            l, r, t = i - 1, i + 1, 1
        else:
            l, r, t = i, i + 2, 0

        curr_input = inputs[l:r]
        curr_labels = labels[i:i+1]
        logits = model_tmp(curr_input)[t:t+1]
        loss = criterion(logits, curr_labels)
        loss.backward()
        optimizer_tmp.step()

        deltas = {}
        for (name, param), (name_tmp, param_tmp) in zip(
                model.named_parameters(),
                model_tmp.named_parameters()):
            assert name == name_tmp
            deltas[name] = weights[i] * (param_tmp.data - param.data)

            del param, param_tmp
        magic_model.update_params(deltas)
        del deltas, logits, loss, curr_input, curr_labels
        # torch.cuda.empty_cache()

    weights_grad_list = []
    for step, val_batch in enumerate(valid_loader):
        if args.cuda:
            val_batch = (t.cuda() for t in val_batch)
        _, val_inputs, val_labels = val_batch
        val_batch_size = val_labels.shape[0]

        if weights.grad is not None:
            weights.grad.zero_()
        val_logits = magic_model(val_inputs)
        val_loss = criterion(val_logits, val_labels)
        val_loss = val_loss * float(val_batch_size) / float(
            len(valid_loader.dataset))

        weights_grad = torch.autograd.grad(
            val_loss, weights, retain_graph=True)[0]
        weights_grad_list.append(weights_grad)

    weights_grad = sum(weights_grad_list)

    # if self._ren:
    #     return -weights_grad
    # else:
    w_array[ids] = weights.data / w_decay - weights_grad
    w_array[ids] = torch.max(w_array[ids], torch.ones_like(
        w_array[ids]).fill_(EPSILON))

    return w_array[ids].data






def linear_normalize(weights):
    weights = torch.max(weights, torch.zeros_like(weights))
    if torch.sum(weights) > 1e-8:
        return weights / torch.sum(weights)
    return torch.zeros_like(weights)


def linear_bound(weights):
    weights = torch.max(weights, torch.zeros_like(weights))
    weights = torch.min(weights, torch.ones_like(weights))
    return weights/len(weights.view(-1))
    # if torch.sum(weights) > 1e-8:
    #     return weights / torch.sum(weights)
    # return torch.zeros_like(weights)


def softmax_normalize(weights, temperature):
    return nn.functional.softmax(weights / temperature, dim=0)

def meta_learning_model_rl(args, model, opt, criterion, train_loader, meta_loader, valid_loader, test_loader, scheduler = None):
    
    w_array = init_weights(len(train_loader.dataset), args.w_init)
    for ep in range(args.epochs):
        for batch in tqdm(train_loader, desc='Training Epoch'):
            if args.cuda:
                ids, inputs, labels = tuple(t.cuda() for t in batch)

            # if is_pretrain:
            #     weights = linear_normalize(
            #         torch.ones(inputs.shape[0]).to('cuda'))
            # else:
            if args.norm_fn == 'softmax':
                weights = softmax_normalize(
                    # model, optimizer, batch, valid_loader, w_array, w_decay
                    get_weights(args, model, opt, batch, meta_loader, w_array, args.w_decay),
                    temperature=args.image_softmax_norm_temp)
            elif args.norm_fn == 'linear':

                weights = linear_normalize(get_weights(args, model, opt, batch, meta_loader, w_array, args.w_decay))
            else:
                weights = linear_bound(get_weights(args, model, opt, batch, meta_loader, w_array, args.w_decay))

            model.train()

            opt.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss = torch.sum(loss * weights.data)
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step()

        with torch.no_grad():
            logging.info("valid performance at epoch %d"%(ep))
            test(valid_loader, model, args, "valid")
            logging.info("test performance at epoch %d"%(ep))
            test(test_loader, model, args, "test")

        torch.save(model, os.path.join(args.save_path, 'refined_model_' + str(ep)))
        torch.save(w_array, os.path.join(args.save_path, 'sample_weights_' + str(ep)))