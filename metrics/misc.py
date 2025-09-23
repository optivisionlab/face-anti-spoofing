from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import math
import torch


def compute_eer(labels, scores):
    """Compute the Equal Error Rate (EER) from the predictions and scores.
    Args:
        labels (list[int]): values indicating whether the ground truth
            value is positive (1) or negative (0).
        scores (list[float]): the confidence of the prediction that the
            given sample is a positive.
    Return:
        (float, thresh): the Equal Error Rate and the corresponding threshold
    NOTES:
       The EER corresponds to the point on the ROC curve that intersects
       the line given by the equation 1 = FPR + TPR.
       The implementation of the function was taken from here:
       https://yangcha.github.io/EER-ROC/
    """
    scores_lst = [val[labels[idx]].int() for idx, val in enumerate(scores)]
    fpr, tpr, thresholds = roc_curve(labels.tolist(), scores_lst, pos_label=1)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)
    labels = labels.to("cuda")
    scores = scores.to("cuda")
    return eer, thresh


def adjust_learning_rate(optimizer, global_step, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if global_step < args.warmup_steps:
        minlr = lr * 0.01
        lr = minlr + (lr - minlr) * global_step / (args.warmup_steps - 1)
    else:
        if args.cos:
            lr *= 0.5 * (1. + math.cos(math.pi * (global_step - args.warmup_steps) / (args.total_steps - args.warmup_steps)))
        else:  # stepwise lr schedule
            milestones = args.schedule.split(',')
            milestones = [int(milestone) for milestone in milestones]
            for milestone in milestones:
                lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

def compute_accuracy(pred, target):
    pred = torch.argmax(pred, dim=1)
    equal = torch.mean(pred.eq(target).type(torch.FloatTensor))
    return equal.item()


def compute_acer(preds, target):
    """
    preds: [N, 2] hoặc [N], xác suất/logits dự đoán
    labels: [N], ground truth (0: bona fide, 1: attack)
    """
    # Nếu preds là 2D (softmax/logits), lấy class dự đoán
    if preds.ndim == 2:
        pred_classes = preds.argmax(dim=1)
    else:
        pred_classes = (preds > 0.5).long()

    TP = ((pred_classes == 0) & (target == 0)).sum().item()  # bona fide đúng
    TN = ((pred_classes == 1) & (target == 1)).sum().item()  # attack đúng
    FP = ((pred_classes == 0) & (target == 1)).sum().item()  # attack dự đoán thành bona fide
    FN = ((pred_classes == 1) & (target == 0)).sum().item()  # bona fide dự đoán thành attack

    # Tính APCER, BPCER
    APCER = FP / (FP + TN + 1e-8)  # tránh chia 0
    BPCER = FN / (TP + FN + 1e-8)

    ACER = (APCER + BPCER) / 2
    return ACER, APCER, BPCER