import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

def test(test_loader, network, criterion, args, logger, prefix = "test"):
    test_loss = 0
    correct = 0
    y_hat = []
    y = []
    with torch.no_grad():
        network.eval()
        for test_batch in test_loader:
            if len(test_batch) == 3:
                _, data, target = test_batch
            elif len(test_batch) == 2:
                data, target = test_batch

            if args.cuda:
                data, target = test_loader.dataset.to_cuda(data, target)
                # data = data.cuda()
                # target = target.cuda()
            output = network(data)
            orig_target = target.clone()
            if isinstance(criterion, torch.nn.L1Loss):
                target = F.one_hot(target, num_classes=10)
                output = F.softmax(output)
            if type(output) is tuple:
                output = output[0]
            test_loss += criterion(output, target).item()*target.shape[0]
            pred = output.data.max(1, keepdim=True)[1]
            y_hat.append(pred)
            y.append(orig_target.data.view_as(pred))
            correct += pred.eq(orig_target.data.view_as(pred)).sum()
    y_hat = torch.cat(y_hat, dim=0).cpu()
    y = torch.cat(y, dim=0).cpu()
    quadratic_kappa = torch.tensor(cohen_kappa_score(y_hat, y, weights='quadratic'),device='cuda:0')

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct.item()*1.0 / len(test_loader.dataset)
    # test_losses.append(test_loss)
    logger.info(prefix + ' performance: Avg. loss: %f, Accuracy: %d/%d (%f), Quadratic Kappa: %f'%(
        test_loss, correct.item(), len(test_loader.dataset),test_acc, quadratic_kappa))

    return test_loss, test_acc
