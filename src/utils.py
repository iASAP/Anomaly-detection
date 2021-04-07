import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
import cv2
import math
from collections import OrderedDict
import copy
import time
from sklearn.metrics import roc_auc_score

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def psnr(mse):

    return 10 * math.log10(1 / mse)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def normalize_img(img):

    img_re = copy.copy(img)
    
    img_re = (img_re - np.min(img_re)) / (np.max(img_re) - np.min(img_re))
    
    return img_re

def point_score(outputs, imgs):
    
    loss_func_mse = nn.MSELoss(reduction='none')
    error = loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)
    normal = (1-torch.exp(-error))
    score = (torch.sum(normal*loss_func_mse((outputs[0]+1)/2,(imgs[0]+1)/2)) / torch.sum(normal)).item()
    return score
    
def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr-min_psnr))

def anomaly_score_inv(psnr, max_psnr, min_psnr):
    return (1.0 - ((psnr - min_psnr) / (max_psnr-min_psnr)))

def anomaly_score_array(psnr):
    return anomaly_score(psnr, np.max(psnr), np.min(psnr))

def anomaly_score_array_inv(psnr):
    return anomaly_score_inv(psnr, np.max(psnr), np.min(psnr))

def AUC(anomal_scores, labels):
    return roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))

def score_sum(arr1, arr2, alpha):
    return alpha*arr1 + (1-alpha)*arr2

def is_power_of_2(n):
    return (n & (n-1) == 0) and n != 0

def nearest_power_of_2(target):
    if target > 1:
        for i in range(1, int(target)):
            if (2 ** i >= target):
                return 2 ** i
    else:
        return 1

def sliding_window(image, step_size, win_size):
    """

    """
    # slide a window across the image
    for y in range(0, image.shape[0]-win_size[1], step_size[1]):
        for x in range(0, image.shape[1]-win_size[0], step_size[0]):
            # yield the current window
            yield (x, y, image[y:y + win_size[1], x:x + win_size[0]])
