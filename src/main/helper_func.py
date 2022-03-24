import torch

import logging

import torch.nn.functional as F

def test(test_loader, network, args, prefix = "test"):
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for _, data, target in test_loader:
            if args.cuda:
                data = data.cuda()
                target = target.cuda()
            output = network(data)
            test_loss += F.nll_loss(output, target).item()*data.shape[0]
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    # test_losses.append(test_loss)
    logging.info(prefix + ' performance: Avg. loss: %f, Accuracy: %d/%d (%f)\n'%(
        test_loss, correct.item(), len(test_loader.dataset),
        100. * correct.item()*1.0 / len(test_loader.dataset)))