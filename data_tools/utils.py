""" Evaluate ROC
Returns:
    auc, eer: Area under the curve, Equal Error Rate
"""
from __future__ import print_function

import os
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


def evaluate(labels, scores, metric='roc'):
    if metric == 'roc':
        return roc(labels, scores)
    elif metric == 'auprc':
        return auprc(labels, scores)
    elif metric == 'f1_score':
        threshold = 0.20
        scores[scores >= threshold] = 1
        scores[scores <  threshold] = 0
        return f1_score(labels, scores)
    else:
        raise NotImplementedError("Check the evaluation metric.")


##
def roc(labels, scores, saveto=None):
    """Compute ROC curve and ROC area for each class"""
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    labels = labels.cpu()
    scores = scores.cpu()

    # True/False Positive Rates.
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)#this function return a folat value

    # Equal Error Rate
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    if saveto:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='(AUC = %0.2f, EER = %0.2f)' % (roc_auc, eer))
        plt.plot([eer], [1-eer], marker='o', markersize=5, color="navy")
        plt.plot([0, 1], [1, 0], color='navy', lw=1, linestyle=':')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(saveto, "ROC.pdf"))
        plt.close()

    return roc_auc


def auprc(labels, scores):
    ap = average_precision_score(labels, scores)
    return ap


import torch
def L1_measure(input, target):
    """ L1 error for ITAE evaluate.

    Args:
        input (FloatTensor): Input tensor, the restore images
        target (FloatTensor): Output tensor, the label images

    Returns:
        [FloatTensor]: L1 distance between input and target, the output shape is [batchsize]
    """
    assert input.size() == target.size(), 'the input size is not equal to target'
    b, c, h, w = input.size()
    input = input.view(b, -1)
    target = target.view(b, -1)
    return torch.mean(torch.abs(input - target), dim=1)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import mmcv
import datetime

class  log_txt(object):
    ### train logger
    def __init__(self, path, description):
        mmcv.mkdir_or_exist(path)
        self.file = os.path.join(path, 'train_log.txt')
        logger = open(self.file, 'a')
        logger.write('\n' + description + '\n')
        logger.close()

    def log_train(self, epoch, loss, optimizer):
        curr_time = datetime.datetime.now()
        curr_time = datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')
        logger = open(self.file, 'a')
        logger.write('\n' + '--'*40 +
            '\n {Time}\tEpoch: {epoch} \tLoss: {loss:0.4f}\tLR: {lr:0.6f}'.format(Time=curr_time, epoch=epoch, loss=loss, lr=optimizer.param_groups[0]['lr'])
                     )
        logger.close()

    def log_test(self, auroc, best_auroc):
        logger = open(self.file, 'a')
        logger.write('\n \t \t  Test Auroc : {acc}\t  Best Auroc : {best}'.format(acc=auroc, best=best_auroc))
        logger.close()