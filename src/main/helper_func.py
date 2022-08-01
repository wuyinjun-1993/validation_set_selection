import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score
import sklearn
def test(test_loader, network, criterion, args, logger, prefix = "test"):
    test_loss = 0
    correct = 0
    y_hat = []
    y = []
    pred_prob_ls = []
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
                if output.shape[1] > 1:
                    output = F.softmax(output)
                # else:
                #     output = F.sigmoid(output)
            if type(output) is tuple:
                output = output[0]
            test_loss += criterion(output, target).item()*target.shape[0]
            if len(output.shape) > 1:
                pred = output.data.max(1, keepdim=True)[1]
            else:
                pred = (output>0.5).long().view(-1)
            y_hat.append(pred)
            y.append(orig_target.data.view_as(pred))

            if len(output.shape) > 1:
                pred_prob_ls.append(F.softmax(output,dim=1))
            else:
                pred_prob_ls.append(output)
            correct += pred.eq(orig_target.data.view_as(pred)).sum()
    y_hat = torch.cat(y_hat, dim=0).cpu()
    y = torch.cat(y, dim=0).cpu()
    pred_prob_ls_tensor = torch.cat(pred_prob_ls)
    quadratic_kappa = torch.tensor(cohen_kappa_score(y_hat, y, weights='quadratic'),device='cuda:0')
    # if pred_prob_ls_tensor.shape[1] > 2:
    auc_score = sklearn.metrics.roc_auc_score(y.data.cpu().numpy(), pred_prob_ls_tensor.cpu().numpy(), multi_class='ovr')
    # else:
    #     auc_score = sklearn.metrics.roc_auc_score(y.data.cpu().numpy(), pred_prob_ls_tensor[:,1].cpu().numpy())
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct.item()*1.0 / len(test_loader.dataset)
    # test_losses.append(test_loss)
    logger.info(prefix + ' performance: Avg. loss: %f, Accuracy: %d/%d (%f), Quadratic Kappa: %f, AUC score: %f'%(
        test_loss, correct.item(), len(test_loader.dataset),test_acc, quadratic_kappa, auc_score))

    return test_loss, test_acc, quadratic_kappa, auc_score
