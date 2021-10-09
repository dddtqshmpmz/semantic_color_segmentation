from math import nan
import math
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import pytorch_ssim
import cv2
import os
import sys

def psnr(reconst_img,target_img):
    mse = F.mse_loss(reconst_img, target_img.detach())
    pixel_max = 1.0
    psnr_result = 20.0 * torch.log10(pixel_max/torch.sqrt(mse))
    return psnr_result

def ssim(reconst_img, target_img):
    # target (bn,3,h,w)
    ssim_res = pytorch_ssim.ssim(reconst_img,target_img)
    # ssim_loss = pytorch_ssim.SSIM(window_size=11)
    # ssim_loss_res = ssim_loss(reconst_img,target_img)
    return ssim_res

def reconst_loss(reconst_img, target_img, type='mse'):
    if type == 'mse':
        loss = F.mse_loss(reconst_img, target_img.detach())
    elif type == 'l1':
        loss = F.l1_loss(reconst_img, target_img.detach())
    elif type == 'vgg':
        pass
    return loss


def temp_distance(primary_color_layers, alpha_layers, rgb_layers):
    """
    　共分散行列をeye(3)とみなして簡単にしたもの．
    　３次元空間でのprimary_colorへのユークリッド距離を
    　ピクセルごとに算出し，alpha_layersで重み付けし，
    　最後にsum

    　primary_color_layers, rgb_layers: (bn, ln, 3, h, w)

     协方差矩阵的简化版本，被视为眼睛 (3)。
     3D 空间中到primary_color 的欧几里德距离
     为每个像素计算，由 alpha_layers 加权，
     最后求和
    """
    diff = (primary_color_layers - rgb_layers)
    distance = (diff * diff).sum(dim=2, keepdim=True) # out: (bn, ln, 1, h, w)
    #loss = (distance * alpha_layers).sum(dim=1, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True).view(-1)
    loss = (distance * alpha_layers).sum(dim=1, keepdim=True).mean()
    #print('temp loss: ', loss)
    return loss # shape = (bn)

def squared_mahalanobis_distance_loss(primary_color_layers, alpha_layers, rgb_layers):
    """
     実装していない
     No implement of squared_mahalanobis_distance_loss
    """
    loss = temp_distance(primary_color_layers, alpha_layers, rgb_layers)
    return loss

def alpha_normalize(alpha_layers):
    # constraint (sum = 1)
    # layersの状態で受け取り，その形で返す. bn, ln, 1, h, w  以层的状态接收并以该形式返回
    return alpha_layers / (alpha_layers.sum(dim=1, keepdim=True) + 1e-8)

def mono_color_reconst_loss(mono_color_reconst_img, target_img):
    loss = F.l1_loss(mono_color_reconst_img, target_img.detach())

    return loss

