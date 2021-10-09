import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    '''
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
    '''

    smooth = 1e-5
    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    intersection = (output_ & target_).sum()
    return (2. * intersection + smooth) / (output_.sum() + target_.sum() + smooth)


def precision(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    TP = (output_ & target_).sum()
    P = output_.sum()

    return (TP + smooth) / (P + smooth)


def recall(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).data.cpu().numpy()
    target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    TP = (output_ & target_).sum()
    TP_FN = target_.sum()

    return (TP + smooth) / (TP_FN + smooth)
