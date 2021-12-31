from __future__ import print_function
import argparse
from math import isnan, nan
import math
import numpy as np
import torch
import torch.utils.data
from torch import mode, nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid, save_image
from net import MaskGenerator, ResiduePredictor
from mydataset import MyDataset, MyDatasetIHC, MyDatasetMulti, MyDatasetIHC_365, MyDatasetKvasir
import pytorch_ssim
import cv2
import os
import sys
from collections import OrderedDict
from glob import glob
import pandas as pd
import torch.optim as optim
from torchsummary import summary
from dataParallel import BalancedDataParallel

import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import segmentation_models.segmentation_models_pytorch as smp

import archs
import losses
from dataset import Dataset, DatasetSemanColor, DatasetSemanColor2Decoder
from metrics import iou_score, dice_coef,precision,recall
from utils import AverageMeter, str2bool
from util import psnr, ssim, reconst_loss, temp_distance, squared_mahalanobis_distance_loss, alpha_normalize, mono_color_reconst_loss

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

gpu_num = len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) - 1
# gpu0_bsz = 5
# other_gpu_bsz = 8
# init_batch_size = gpu0_bsz + gpu_num * other_gpu_bsz

# gpu0_bsz_after = gpu0_bsz
# other_gpu_bsz_after = 12
# init_batch_size_after = gpu0_bsz_after + gpu_num * other_gpu_bsz_after

init_batch_size = 20
init_batch_size_after = 27

parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--run_name', type=str, default='train',
                    help='run-name. This name is used for output folder.')
parser.add_argument('--batch_size', type=int, default=init_batch_size, metavar='N',  # 32-> 4
                    help='input batch size for training (default: 32)')

parser.add_argument('--after_batch_size', type=int, default=init_batch_size_after, metavar='N',  # 32-> 4
                    help='input batch size for training after color model complete (default: 32)')
parser.add_argument('--epochs', type=int, default=60, metavar='N',  # 10
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--num_primary_color', type=int, default=7,  # 6->7
                    help='num of layers')
parser.add_argument('--rec_loss_lambda', type=float, default=1.0,
                    help='reconst_loss lambda')
parser.add_argument('--m_loss_lambda', type=float, default=1.0,   # 1.0
                    help='m_loss_lambda')
parser.add_argument('--sparse_loss_lambda', type=float, default=0.0,  # 1.0
                    help='sparse_loss lambda')
parser.add_argument('--distance_loss_lambda', type=float, default=0.5,  # 1.0
                    help='distance_loss_lambda')

parser.add_argument('--save_layer_train', type=int, default=1,
                    help='save_layer_train')

# 让颜色分割网络收敛的慢一些
parser.add_argument('--color_lr', default=5e-4, type=float,
                    help='color learning rate')  # 1e-3


parser.add_argument('--num_workers', type=int, default=8,
                    help='num_workers of dataloader')
parser.add_argument('--csv_path_ihc', type=str, default='ihc_30k.csv')
parser.add_argument('--csv_path_365', type=str, default='train.csv')
parser.add_argument('--csv_path_kva', type=str, default='kvasircvc.csv')

parser.add_argument('--log_interval', type=int, default=100, metavar='N',  # 200-> 20 ->30
                    help='how many batches to wait before logging training status')
parser.add_argument('--reconst_loss_type', type=str,
                    default='l1', help='[mse | l1 | vgg]')


parser.add_argument('--name', default='multitask',
                    help='model name: (default: arch+timestamp)')

# model
parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                    choices=ARCH_NAMES,
                    help='model architecture: ' +
                    ' | '.join(ARCH_NAMES) +
                    ' (default: NestedUNet)')
parser.add_argument('--deep_supervision', default=False, type=str2bool)
parser.add_argument('--input_channels', default=3, type=int,
                    help='input channels')
parser.add_argument('--num_classes', default=1, type=int,
                    help='number of classes')
parser.add_argument('--input_w', default=320, type=int,
                    help='image width')
parser.add_argument('--input_h', default=320, type=int,
                    help='image height')

# loss
parser.add_argument('--loss', default='BCEDiceLoss',
                    choices=LOSS_NAMES,
                    help='loss: ' +
                    ' | '.join(LOSS_NAMES) +
                    ' (default: BCEDiceLoss)')

# dataset
parser.add_argument('--dataset', default='patient_datasetRandom',
                    help='dataset name')
parser.add_argument('--img_ext', default='.png',
                    help='image file extension')
parser.add_argument('--mask_ext', default='.png',
                    help='mask file extension')

# optimizer
parser.add_argument('--optimizer', default='Adam',
                    choices=['Adam', 'SGD'],
                    help='loss: ' +
                    ' | '.join(['Adam', 'SGD']) +
                    ' (default: Adam)')
parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='weight decay')
parser.add_argument('--nesterov', default=False, type=str2bool,
                    help='nesterov')

# scheduler
parser.add_argument('--scheduler', default='CosineAnnealingLR',
                    choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
parser.add_argument('--min_lr', default=1e-5, type=float,
                    help='minimum learning rate')
parser.add_argument('--factor', default=0.1, type=float)
parser.add_argument('--patience', default=2, type=int)
parser.add_argument('--milestones', default='1,2', type=str)
parser.add_argument('--gamma', default=2/3, type=float)
parser.add_argument('--early_stopping', default=-1, type=int,
                    metavar='N', help='early stopping (default: -1)')


args = parser.parse_args()
config = vars(parser.parse_args())
with open('models/%s/config.yml' % config['name'], 'w') as f:
    yaml.dump(config, f)

args.cuda = not args.no_cuda and torch.cuda.is_available()

try:
    os.makedirs('results/%s' % args.run_name)
except OSError:
    pass

# 打印所有数据到日志
log = open("train_process.log", "a")
sys.stdout = log  # log sys.__stdout__
torch.manual_seed(args.seed)
cudnn.benchmark = True

device = torch.device("cuda" if args.cuda else "cpu")

train_dataset = MyDatasetIHC_365(
    args.csv_path_365, args.csv_path_ihc, args.num_primary_color, mode='train')
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    worker_init_fn=lambda x: np.random.seed(),
    drop_last=True,
    pin_memory=True
)


train_after_dataset = MyDatasetKvasir(
    args.csv_path_kva, args.num_primary_color, mode='train')
# train_after_dataset = MyDatasetIHC(
#     args.csv_path_ihc, args.num_primary_color, mode='train')
train_after_loader = torch.utils.data.DataLoader(
    train_after_dataset,
    batch_size=args.after_batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    worker_init_fn=lambda x: np.random.seed(),
    drop_last=True,
    pin_memory=True
)


val_after_dataset = MyDatasetKvasir(
    args.csv_path_kva, args.num_primary_color, mode='val')
# val_after_dataset = MyDatasetIHC(
#     args.csv_path_ihc, args.num_primary_color, mode='val')
val_after_loader = torch.utils.data.DataLoader(
    val_after_dataset,
    batch_size=args.after_batch_size,  # 1
    shuffle=False,
    num_workers=args.num_workers,  # 1
    drop_last=True  # 加的这句
)

# mask_generator = MaskGenerator(args.num_primary_color)
# mask_generator = BalancedDataParallel(gpu0_bsz, mask_generator, dim = 0).to(device)
# mask_generator = mask_generator.cuda()

# residue_predictor = ResiduePredictor(args.num_primary_color)
# residue_predictor = BalancedDataParallel( gpu0_bsz, residue_predictor, dim=0).to(device)
# residue_predictor = residue_predictor.cuda()


mask_generator = MaskGenerator(args.num_primary_color).to(device)
mask_generator = nn.DataParallel(mask_generator)
mask_generator = mask_generator.cuda()

residue_predictor = ResiduePredictor(args.num_primary_color).to(device)
residue_predictor = nn.DataParallel(residue_predictor)
residue_predictor = residue_predictor.cuda()

params = list(mask_generator.parameters())
params += list(residue_predictor.parameters())


optimizer = optim.Adam(
    params, lr=config['color_lr'], betas=(0.0, 0.99))  # 0926

# 加载新参数
# model = smp.UnetWithColor('efficientnet-b3', in_channels= 3+7 ,
#                      classes= 1, encoder_weights="imagenet")
# model = BalancedDataParallel(gpu0_bsz, model, dim=0).to(device)
# model = model.cuda()

# model = smp.UnetWithColor('efficientnet-b3', in_channels=3+7,
#                           classes=1, encoder_weights="imagenet").cuda()
model = smp.Unet('efficientnet-b3', in_channels=3,
                           classes=1, encoder_weights="imagenet").cuda()
model = model.cuda()
model = nn.DataParallel(model, device_ids=[0,1,2])


paramsSeg = filter(lambda p: p.requires_grad, model.parameters())

optimizerSeg = optim.Adam(
    paramsSeg, lr=config['lr'], weight_decay=config['weight_decay'])

if config['scheduler'] == 'CosineAnnealingLR':
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizerSeg, T_max=config['epochs'], eta_min=config['min_lr'])

if config['loss'] == 'BCEWithLogitsLoss':
    criterion = nn.BCEWithLogitsLoss().cuda()
else:
    criterion = losses.__dict__[config['loss']]().cuda()


def sparse_loss(alpha_layers):
    # alpha_layers: bn, ln, 1, h, w
    #print('alpha_layers.mean().item(): ', alpha_layers.mean().item())
    alpha_layers = alpha_layers.sum(
        dim=1, keepdim=True) / (alpha_layers * alpha_layers).sum(dim=1, keepdim=True)
    loss = F.l1_loss(alpha_layers, torch.ones_like(alpha_layers).to(device))
    return loss


def train(epoch, min_train_loss):
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter()}  # iou

    mask_generator.train()
    residue_predictor.train()
    model.train()

    train_loss = 0
    r_loss_mean = 0
    m_loss_mean = 0
    s_loss_mean = 0
    d_loss_mean = 0
    batch_num = 0

    pbar = tqdm(total=len(train_loader))
    for batch_idx, (target_img, primary_color_layers, mask, img_365, primary_color_layers_365) in enumerate(train_loader):

        target_img = target_img.to(device)  # bn, 3ch, h, w
        primary_color_layers = primary_color_layers.to(device)
        mask = mask.to(device)
        img_365 = img_365.to(device)
        primary_color_layers_365 = primary_color_layers_365.to(device)

        # primary_color_layers = primary_color_layers.to(device) # bn, num_primary_color, 3ch, h, w

        optimizer.zero_grad()

        # networkにforwardにする
        primary_color_pack_365 = primary_color_layers_365.view(
            target_img.size(0), -1, target_img.size(2), target_img.size(3))
        pred_alpha_layers_pack_365 = mask_generator(
            img_365, primary_color_pack_365)

        pred_alpha_layers_365 = pred_alpha_layers_pack_365.view(
            target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))

        # 正規化などのprocessingを行う
        processed_alpha_layers_365 = alpha_normalize(pred_alpha_layers_365)

        # mono_color_layers_packの作成．ひとつのtensorにしておく．
        #mono_color_layers = primary_color_layers * processed_alpha_layers
        mono_color_layers_365 = torch.cat(
            (primary_color_layers_365, processed_alpha_layers_365), 2)  # shape: bn, ln, 4, h, w
        mono_color_layers_pack_365 = mono_color_layers_365.view(
            target_img.size(0), -1, target_img.size(2), target_img.size(3))

        # ResiduePredictorの出力をレイヤーごとにviewする 逐层查看Residue Predictor的输出
        residue_pack_365 = residue_predictor(
            img_365, mono_color_layers_pack_365)
        residue_365 = residue_pack_365.view(target_img.size(
            0), -1, 3, target_img.size(2), target_img.size(3))
        #pred_unmixed_rgb_layers = mono_color_layers + residue * processed_alpha_layers
        pred_unmixed_rgb_layers_365 = torch.clamp(
            (primary_color_layers_365 + residue_365), min=0., max=1.0)  # * processed_alpha_layers

        # alpha addしてreconst_imgを作成する
        #reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)
        reconst_img_365 = (pred_unmixed_rgb_layers_365 *
                           processed_alpha_layers_365).sum(dim=1)
        mono_color_reconst_img_365 = (
            primary_color_layers_365 * processed_alpha_layers_365).sum(dim=1)

        # Culculate loss.
        r_loss_365 = reconst_loss(
            reconst_img_365, img_365, type=args.reconst_loss_type) * args.rec_loss_lambda
        m_loss_365 = mono_color_reconst_loss(
            mono_color_reconst_img_365, img_365) * args.m_loss_lambda
        s_loss_365 = sparse_loss(
            processed_alpha_layers_365) * args.sparse_loss_lambda
        #print('total_loss: ', total_loss)
        d_loss_365 = squared_mahalanobis_distance_loss(primary_color_layers_365.detach(
        ), processed_alpha_layers_365, pred_unmixed_rgb_layers_365) * args.distance_loss_lambda

        total_loss_365 = r_loss_365 + m_loss_365 + s_loss_365 + d_loss_365
        if (math.isnan(total_loss_365.item())):
            min_train_loss = -1
            break

        batch_num += 1

        # networkにforwardにする
        primary_color_pack = primary_color_layers.view(
            target_img.size(0), -1, target_img.size(2), target_img.size(3))
        pred_alpha_layers_pack = mask_generator(target_img, primary_color_pack)
        #print('pred_alpha_layers_pack.size():', pred_alpha_layers_pack.size())

        # MaskGの出力をレイヤーごとにviewする
        pred_alpha_layers = pred_alpha_layers_pack.view(
            target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))

        # 正規化などのprocessingを行う
        processed_alpha_layers = alpha_normalize(pred_alpha_layers)

        # mono_color_layers_packの作成．ひとつのtensorにしておく．
        #mono_color_layers = primary_color_layers * processed_alpha_layers
        mono_color_layers = torch.cat(
            (primary_color_layers, processed_alpha_layers), 2)  # shape: bn, ln, 4, h, w
        mono_color_layers_pack = mono_color_layers.view(
            target_img.size(0), -1, target_img.size(2), target_img.size(3))

        # ResiduePredictorの出力をレイヤーごとにviewする 逐层查看Residue Predictor的输出
        residue_pack = residue_predictor(target_img, mono_color_layers_pack)
        residue = residue_pack.view(target_img.size(
            0), -1, 3, target_img.size(2), target_img.size(3))
        #pred_unmixed_rgb_layers = mono_color_layers + residue * processed_alpha_layers
        pred_unmixed_rgb_layers = torch.clamp(
            (primary_color_layers + residue), min=0., max=1.0)  # * processed_alpha_layers

        # alpha addしてreconst_imgを作成する
        #reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)
        reconst_img = (pred_unmixed_rgb_layers *
                       processed_alpha_layers).sum(dim=1)
        mono_color_reconst_img = (
            primary_color_layers * processed_alpha_layers).sum(dim=1)

        # Culculate loss.
        r_loss = reconst_loss(reconst_img, target_img,
                              type=args.reconst_loss_type) * args.rec_loss_lambda
        m_loss = mono_color_reconst_loss(
            mono_color_reconst_img, target_img) * args.m_loss_lambda
        # s_loss = sparse_loss(processed_alpha_layers) * args.sparse_loss_lambda
        #print('total_loss: ', total_loss)
        d_loss = squared_mahalanobis_distance_loss(primary_color_layers.detach(
        ), processed_alpha_layers, pred_unmixed_rgb_layers) * args.distance_loss_lambda

        total_loss = r_loss + m_loss + d_loss
        train_loss += total_loss.item()
        r_loss_mean += r_loss.item()
        m_loss_mean += m_loss.item()
        d_loss_mean += d_loss.item()

        processed_alpha_layers_ = torch.squeeze(
            processed_alpha_layers.detach(), dim=2)
        output = model(torch.cat((target_img.detach(), processed_alpha_layers_.detach()[
                       :, :, :, :]), 1), processed_alpha_layers_.detach())
        # output = model(target_img.detach())
        loss = criterion(output, mask)
        dice = dice_coef(output, mask)

        total_loss_365.backward(retain_graph=False)

        optimizerSeg.zero_grad()
        loss.backward()
        optimizer.step()
        optimizerSeg.step()

        avg_meters['loss'].update(loss.item(), target_img.size(0))
        avg_meters['dice'].update(dice, target_img.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('dice', avg_meters['dice'].avg),
            #('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)

        if batch_idx % args.log_interval == 0:
            print('')
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(target_img), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                total_loss.item() / len(target_img)))
            print('reconst_loss *lambda: ', r_loss.item() / len(target_img))
            # print('sparse_loss *lambda: ', s_loss.item() / len(target_img))
            print('squared_mahalanobis_distance_loss *lambda: ',
                  d_loss.item() / len(target_img))

            for save_layer_number in range(args.save_layer_train):
                save_image(primary_color_layers[save_layer_number, :, :, :, :],
                           'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_primary_color_layers.png' % save_layer_number)
                save_image(reconst_img[save_layer_number, :, :, :].unsqueeze(0),
                           'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_reconst_img.png' % save_layer_number)
                save_image(target_img[save_layer_number, :, :, :].unsqueeze(0),
                           'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_target_img.png' % save_layer_number)

    pbar.close()

    train_loss = train_loss / batch_num
    r_loss_mean = r_loss_mean / batch_num
    m_loss_mean = m_loss_mean / batch_num
    d_loss_mean = d_loss_mean / batch_num

    print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, train_loss))
    print('====> Epoch: {} Average reconst_loss *lambda: {:.6f}'.format(epoch, r_loss_mean))
    print('====> Epoch: {} Average mono_loss *lambda: {:.6f}'.format(epoch, m_loss_mean))
    # print('====> Epoch: {} Average sparse_loss *lambda: {:.6f}'.format(epoch, s_loss_mean ))
    print('====> Epoch: {} Average squared_mahalanobis_distance_loss *lambda: {:.6f}'.format(epoch, d_loss_mean))

    if (math.isnan(train_loss)):
        min_train_loss = -1

    # save best model
    if (train_loss < min_train_loss):
        min_train_loss = train_loss
        torch.save(mask_generator.state_dict(),
                   'results/%s/mask_generator.pth' % (args.run_name))
        torch.save(residue_predictor.state_dict(),
                   'results/%s/residue_predictor.pth' % args.run_name)

    return OrderedDict([('loss', avg_meters['loss'].avg), ('dice', avg_meters['dice'].avg), ('min_train_loss', min_train_loss)])


def train_after_finish_color_seg(epoch, min_train_loss):
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter()}  # iou

    # eval mode
    mask_generator.eval()
    residue_predictor.eval()
    model.train()

    pbar = tqdm(total=len(train_after_loader))
    for batch_idx, (target_img, primary_color_layers, mask) in enumerate(train_after_loader):

        target_img = target_img.to(device)  # bn, 3ch, h, w
        primary_color_layers = primary_color_layers.to(device)
        mask = mask.to(device)

        # primary_color_layers = primary_color_layers.to(device) # bn, num_primary_color, 3ch, h, w

        # networkにforwardにする
        primary_color_pack = primary_color_layers.view(
            target_img.size(0), -1, target_img.size(2), target_img.size(3))
        pred_alpha_layers_pack = mask_generator(target_img, primary_color_pack)
        #print('pred_alpha_layers_pack.size():', pred_alpha_layers_pack.size())

        # MaskGの出力をレイヤーごとにviewする
        pred_alpha_layers = pred_alpha_layers_pack.view(
            target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))

        # 正規化などのprocessingを行う
        processed_alpha_layers = alpha_normalize(pred_alpha_layers)

        # mono_color_layers_packの作成．ひとつのtensorにしておく．
        #mono_color_layers = primary_color_layers * processed_alpha_layers
        mono_color_layers = torch.cat(
            (primary_color_layers, processed_alpha_layers), 2)  # shape: bn, ln, 4, h, w
        mono_color_layers_pack = mono_color_layers.view(
            target_img.size(0), -1, target_img.size(2), target_img.size(3))

        # ResiduePredictorの出力をレイヤーごとにviewする 逐层查看Residue Predictor的输出
        residue_pack = residue_predictor(target_img, mono_color_layers_pack)
        residue = residue_pack.view(target_img.size(
            0), -1, 3, target_img.size(2), target_img.size(3))
        #pred_unmixed_rgb_layers = mono_color_layers + residue * processed_alpha_layers
        pred_unmixed_rgb_layers = torch.clamp(
            (primary_color_layers + residue), min=0., max=1.0)  # * processed_alpha_layers

        # alpha addしてreconst_imgを作成する
        #reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)
        reconst_img = (pred_unmixed_rgb_layers *
                       processed_alpha_layers).sum(dim=1)
        mono_color_reconst_img = (
            primary_color_layers * processed_alpha_layers).sum(dim=1)

        processed_alpha_layers_ = torch.squeeze(
            processed_alpha_layers.detach(), dim=2)
        output = model(torch.cat((target_img.detach(), processed_alpha_layers_.detach()[
                       :, :7, :, :]), 1), processed_alpha_layers_.detach())
        # output = model(target_img.detach())
        loss = criterion(output, mask)
        dice = dice_coef(output, mask)

        optimizerSeg.zero_grad()
        loss.backward()
        optimizerSeg.step()

        avg_meters['loss'].update(loss.item(), target_img.size(0))
        avg_meters['dice'].update(dice, target_img.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('dice', avg_meters['dice'].avg),
            #('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)

        if batch_idx % args.log_interval == 0:
            for save_layer_number in range(args.save_layer_train):
                save_image(primary_color_layers[save_layer_number, :, :, :, :],
                           'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_primary_color_layers.png' % save_layer_number)
                save_image(reconst_img[save_layer_number, :, :, :].unsqueeze(0),
                           'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_reconst_img.png' % save_layer_number)
                save_image(target_img[save_layer_number, :, :, :].unsqueeze(0),
                           'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_target_img.png' % save_layer_number)

    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg), ('dice', avg_meters['dice'].avg), ('min_train_loss', min_train_loss)])


def val(epoch, min_val_loss):
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter()}  # iou

    mask_generator.eval()
    residue_predictor.eval()
    model.eval()

    with torch.no_grad():
        val_loss = 0
        r_loss_mean = 0
        m_loss_mean = 0
        s_loss_mean = 0
        d_loss_mean = 0
        batch_num = 0

        pbar = tqdm(total=len(val_after_loader))

        for batch_idx, (target_img, primary_color_layers, mask) in enumerate(val_after_loader):
            target_img = target_img.to(device)  # bn, 3ch, h, w
            primary_color_layers = primary_color_layers.to(device)
            mask = mask.to(device)

            primary_color_pack = primary_color_layers.view(
                target_img.size(0), -1, target_img.size(2), target_img.size(3))
            pred_alpha_layers_pack = mask_generator(
                target_img, primary_color_pack)
            pred_alpha_layers = pred_alpha_layers_pack.view(
                target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))
            processed_alpha_layers = alpha_normalize(pred_alpha_layers)
            mono_color_layers = torch.cat(
                (primary_color_layers, processed_alpha_layers), 2)  # shape: bn, ln, 4, h, w
            mono_color_layers_pack = mono_color_layers.view(
                target_img.size(0), -1, target_img.size(2), target_img.size(3))
            residue_pack = residue_predictor(
                target_img, mono_color_layers_pack)
            residue = residue_pack.view(target_img.size(
                0), -1, 3, target_img.size(2), target_img.size(3))
            pred_unmixed_rgb_layers = torch.clamp(
                (primary_color_layers + residue), min=0., max=1.0)
            reconst_img = (pred_unmixed_rgb_layers *
                           processed_alpha_layers).sum(dim=1)
            mono_color_reconst_img = (
                primary_color_layers * processed_alpha_layers).sum(dim=1)

            # 计算loss 打印出来
            r_loss = reconst_loss(
                reconst_img, target_img, type=args.reconst_loss_type) * args.rec_loss_lambda
            m_loss = mono_color_reconst_loss(
                mono_color_reconst_img, target_img) * args.m_loss_lambda
            s_loss = sparse_loss(processed_alpha_layers) * \
                args.sparse_loss_lambda
            #print('total_loss: ', total_loss)
            d_loss = squared_mahalanobis_distance_loss(primary_color_layers.detach(
            ), processed_alpha_layers, pred_unmixed_rgb_layers) * args.distance_loss_lambda

            batch_num += 1
            total_loss = r_loss + m_loss + s_loss + d_loss
            val_loss += total_loss.item()
            r_loss_mean += r_loss.item()
            m_loss_mean += m_loss.item()
            s_loss_mean += s_loss.item()
            d_loss_mean += d_loss.item()

            save_layer_number = 0
            if batch_idx <= 1:
                # batchsizeは１で計算されているはず．それぞれ保存する．
                save_image(primary_color_layers[save_layer_number, :, :, :, :],
                           'results/%s/val_ep_' % args.run_name + str(epoch) + '_idx_%02d_primary_color_layers.png' % batch_idx)
                save_image(reconst_img[save_layer_number, :, :, :].unsqueeze(0),
                           'results/%s/val_ep_' % args.run_name + str(epoch) + '_idx_%02d_reconst_img.png' % batch_idx)
                save_image(target_img[save_layer_number, :, :, :].unsqueeze(0),
                           'results/%s/val_ep_' % args.run_name + str(epoch) + '_idx_%02d_target_img.png' % batch_idx)

            processed_alpha_layers_ = torch.squeeze(
                processed_alpha_layers.detach(), dim=2)
            output = model(torch.cat((target_img.detach(), processed_alpha_layers_.detach()[
                           :, :7, :, :]), 1), processed_alpha_layers_.detach())
            # output = model(target_img.detach())

            loss = criterion(output, mask)
            dice = dice_coef(output, mask)
            avg_meters['loss'].update(loss.item(), target_img.size(0))
            avg_meters['dice'].update(dice, target_img.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('dice', avg_meters['dice'].avg),
                #('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)

        pbar.close()

        val_loss = val_loss / batch_num
        r_loss_mean = r_loss_mean / batch_num
        m_loss_mean = m_loss_mean / batch_num
        s_loss_mean = s_loss_mean / batch_num
        d_loss_mean = d_loss_mean / batch_num

        print('====> Epoch: {} Average val loss: {:.6f}'.format(epoch, val_loss))
        print('====> Epoch: {} Average val reconst_loss *lambda: {:.6f}'.format(epoch, r_loss_mean))
        print('====> Epoch: {} Average val mono_loss *lambda: {:.6f}'.format(epoch, m_loss_mean))
        print('====> Epoch: {} Average val sparse_loss *lambda: {:.6f}'.format(epoch, s_loss_mean))
        print('====> Epoch: {} Average val squared_mahalanobis_distance_loss *lambda: {:.6f}'.format(epoch, d_loss_mean))

        if (math.isnan(val_loss)):
            min_val_loss = -1

        if (val_loss < min_val_loss):
            min_val_loss = val_loss

        return OrderedDict([('loss', avg_meters['loss'].avg), ('dice', avg_meters['dice'].avg), ('min_val_loss', min_val_loss)])


def val_after_finish_color_seg(epoch, min_val_loss):
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter()}  # iou

    mask_generator.eval()
    residue_predictor.eval()
    model.eval()

    with torch.no_grad():

        pbar = tqdm(total=len(val_after_loader))

        for batch_idx, (target_img, primary_color_layers, mask) in enumerate(val_after_loader):
            target_img = target_img.to(device)  # bn, 3ch, h, w
            primary_color_layers = primary_color_layers.to(device)
            mask = mask.to(device)

            primary_color_pack = primary_color_layers.view(
                target_img.size(0), -1, target_img.size(2), target_img.size(3))
            pred_alpha_layers_pack = mask_generator(
                target_img, primary_color_pack)
            pred_alpha_layers = pred_alpha_layers_pack.view(
                target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))
            processed_alpha_layers = alpha_normalize(pred_alpha_layers)
            mono_color_layers = torch.cat(
                (primary_color_layers, processed_alpha_layers), 2)  # shape: bn, ln, 4, h, w
            mono_color_layers_pack = mono_color_layers.view(
                target_img.size(0), -1, target_img.size(2), target_img.size(3))
            residue_pack = residue_predictor(
                target_img, mono_color_layers_pack)
            residue = residue_pack.view(target_img.size(
                0), -1, 3, target_img.size(2), target_img.size(3))
            pred_unmixed_rgb_layers = torch.clamp(
                (primary_color_layers + residue), min=0., max=1.0)
            reconst_img = (pred_unmixed_rgb_layers *
                           processed_alpha_layers).sum(dim=1)
            mono_color_reconst_img = (
                primary_color_layers * processed_alpha_layers).sum(dim=1)

            # output = model(torch.cat((target_img.detach(),mono_color_layers_pack.detach()[:,:28,:,:]),1),mono_color_layers_pack.detach())
            processed_alpha_layers_ = torch.squeeze(
                processed_alpha_layers.detach(), dim=2)
            output = model(torch.cat((target_img.detach(), processed_alpha_layers_.detach()[
                           :, :, :, :]), 1), processed_alpha_layers_.detach())
            # output = model(target_img.detach())

            loss = criterion(output, mask)
            dice = dice_coef(output, mask)
            avg_meters['loss'].update(loss.item(), target_img.size(0))
            avg_meters['dice'].update(dice, target_img.size(0))

            if (batch_idx <= 1):
                '''
                try:
                    os.makedirs('results/val')
                except OSError:
                    pass'''

                save_layer_number = 0
                save_image(primary_color_layers[save_layer_number, :, :, :, :],
                           'results/train/' + 'val_ep_%d_img-%02d_primary_color_layers.png' % (epoch, batch_idx))
                save_image(reconst_img[save_layer_number, :, :, :].unsqueeze(0),
                           'results/train/' + 'val_ep_%d_img-%02d_reconst_img.png' % (epoch, batch_idx))
                save_image(target_img[save_layer_number, :, :, :].unsqueeze(0),
                           'results/train/' + 'val_ep_%d_img-%02d_target_img.png' % (epoch,batch_idx))
                
                '''
                # 下段代码作图用的
                # RGBAの４chのpngとして保存する 另存为RGBA 4ch png
                # out: bn, ln = 7, 4, h, w
                RGBA_layers = torch.cat(
                    (pred_unmixed_rgb_layers, processed_alpha_layers), dim=2)
                # test ではバッチサイズが１なので，bn部分をなくす 在测试中，批量大小为1，因此消除了bn部分。
                RGBA_layers = RGBA_layers[0]  # ln, 4. h, w
                # ln ごとに結果を保存する
                for i in range(len(RGBA_layers)):
                    save_image(
                        RGBA_layers[i, :, :, :], 'results/val/img-%02d_layer-%02d.png' % (batch_idx, i))

                # 処理後のアルファの保存 processed_alpha_layers 保存已处理的alpha已处理的alpha_layers
                for i in range(len(processed_alpha_layers[0])):
                    save_image(
                        processed_alpha_layers[0, i, :, :, :], 'results/val/proc-alpha-%02d-00_layer-%02d.png' % (batch_idx, i))

                # 処理後のRGBの保存 处理后保存RGB
                for i in range(len(pred_unmixed_rgb_layers[0])):
                    save_image(
                        pred_unmixed_rgb_layers[0, i, :, :, :], 'results/val/rgb-00_layer-%02d-%02d.png' % (batch_idx, i))'''

                # output the pred mask pics
                seg_output = torch.sigmoid(output)
                zero = torch.zeros_like(seg_output)
                one = torch.ones_like(seg_output)
                seg_output = torch.where(seg_output > 0.5, one, zero)
                seg_output = seg_output.data.cpu().numpy()

                cv2.imwrite(os.path.join('results', 'train', 'semantic-output-%02d.png' % (batch_idx)),
                            (seg_output[0, 0] * 255).astype('uint8'))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('dice', avg_meters['dice'].avg),
                #('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)

        pbar.close()

        return OrderedDict([('loss', avg_meters['loss'].avg), ('dice', avg_meters['dice'].avg), ('min_val_loss', min_val_loss)])


def test_for_paper():
    avg_meters = {'loss': AverageMeter(),
                  'dice': AverageMeter(),
                  'iou': AverageMeter(),
                  'precision': AverageMeter(),
                  'recall': AverageMeter()}  # iou

    mask_generator.eval()
    residue_predictor.eval()
    model.eval()

    with torch.no_grad():

        pbar = tqdm(total=len(val_after_loader))

        for batch_idx, (target_img, primary_color_layers, mask) in enumerate(val_after_loader):
            target_img = target_img.to(device)  # bn, 3ch, h, w
            primary_color_layers = primary_color_layers.to(device)
            mask = mask.to(device)

            primary_color_pack = primary_color_layers.view(
                target_img.size(0), -1, target_img.size(2), target_img.size(3))
            pred_alpha_layers_pack = mask_generator(
                target_img, primary_color_pack)
            pred_alpha_layers = pred_alpha_layers_pack.view(
                target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))
            processed_alpha_layers = alpha_normalize(pred_alpha_layers)
            mono_color_layers = torch.cat(
                (primary_color_layers, processed_alpha_layers), 2)  # shape: bn, ln, 4, h, w
            mono_color_layers_pack = mono_color_layers.view(
                target_img.size(0), -1, target_img.size(2), target_img.size(3))
            residue_pack = residue_predictor(
                target_img, mono_color_layers_pack)
            residue = residue_pack.view(target_img.size(
                0), -1, 3, target_img.size(2), target_img.size(3))
            pred_unmixed_rgb_layers = torch.clamp(
                (primary_color_layers + residue), min=0., max=1.0)
            reconst_img = (pred_unmixed_rgb_layers *
                           processed_alpha_layers).sum(dim=1)
            mono_color_reconst_img = (
                primary_color_layers * processed_alpha_layers).sum(dim=1)

            # output = model(torch.cat((target_img.detach(),mono_color_layers_pack.detach()[:,:28,:,:]),1),mono_color_layers_pack.detach())
            processed_alpha_layers_ = torch.squeeze(
                processed_alpha_layers.detach(), dim=2)
            # output = model(torch.cat((target_img.detach(), processed_alpha_layers_.detach()[
            #                :, :, :, :]), 1), processed_alpha_layers_.detach())
            output = model(target_img.detach())

            loss = criterion(output, mask)
            dice = dice_coef(output, mask)
            iou_ = iou_score(output,mask)
            precision_ = precision(output,mask)
            recall_ = recall(output,mask)

            avg_meters['loss'].update(loss.item(), target_img.size(0))
            avg_meters['dice'].update(dice, target_img.size(0))
            avg_meters['iou'].update(iou_, target_img.size(0))
            avg_meters['precision'].update(precision_, target_img.size(0))
            avg_meters['recall'].update(recall_, target_img.size(0))

            if (True):
                save_dir = 'results/val/train_1207_3'
                
                try:
                    os.makedirs(save_dir)
                except OSError:
                    pass

                save_layer_numbers = [0,1]
                mask = mask.data.cpu().numpy()

                for save_layer_number in save_layer_numbers:
                    save_image(primary_color_layers[save_layer_number, :, :, :, :],
                            save_dir + '/img-%02d_%02d_primary_color_layers.png' % (batch_idx, save_layer_number))
                    save_image(reconst_img[save_layer_number, :, :, :].unsqueeze(0),
                            save_dir + '/img-%02d_%02d_reconst_img.png' %  (batch_idx, save_layer_number))
                    save_image(target_img[save_layer_number, :, :, :].unsqueeze(0),
                            save_dir + '/img-%02d_%02d_target_img.png' % (batch_idx, save_layer_number))
                
                
                    # RGBAの４chのpngとして保存する 另存为RGBA 4ch png
                    # out: bn, ln = 7, 4, h, w
                    RGBA_layers = torch.cat(
                        (pred_unmixed_rgb_layers, processed_alpha_layers), dim=2)
                    # test ではバッチサイズが１なので，bn部分をなくす 在测试中，批量大小为1，因此消除了bn部分。
                    RGBA_layers = RGBA_layers[save_layer_number]  # ln, 4. h, w
                    # ln ごとに結果を保存する
                    for i in range(len(RGBA_layers)):
                        save_image(
                            RGBA_layers[i, :, :, :], save_dir+'/img-%02d_%02d_layer-%02d.png' % (batch_idx,save_layer_number, i))
                    '''
                    # 処理後のアルファの保存 processed_alpha_layers 保存已处理的alpha已处理的alpha_layers
                    for i in range(len(processed_alpha_layers[0])):
                        save_image(
                            processed_alpha_layers[0, i, :, :, :], 'results/val/proc-alpha-%02d-00_layer-%02d.png' % (batch_idx, i))

                    # 処理後のRGBの保存 处理后保存RGB
                    for i in range(len(pred_unmixed_rgb_layers[0])):
                        save_image(
                            pred_unmixed_rgb_layers[0, i, :, :, :], 'results/val/rgb-00_layer-%02d-%02d.png' % (batch_idx, i))'''

                    # output the pred mask pics
                    seg_output = torch.sigmoid(output)
                    zero = torch.zeros_like(seg_output)
                    one = torch.ones_like(seg_output)
                    seg_output = torch.where(seg_output > 0.5, one, zero)
                    seg_output = seg_output.data.cpu().numpy()

                    cv2.imwrite(os.path.join(save_dir, 'semantic-output-%02d-%02d.png' % (batch_idx,save_layer_number)),
                                (seg_output[save_layer_number, 0] * 255).astype('uint8'))
                    cv2.imwrite(os.path.join(save_dir, 'semantic-mask-%02d-%02d.png' % (batch_idx,save_layer_number)),
                                (mask[save_layer_number, 0] * 255).astype('uint8'))
                    

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('dice', avg_meters['dice'].avg),
                ('iou', avg_meters['iou'].avg),
                ('precision', avg_meters['precision'].avg),
                ('recall', avg_meters['recall'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)

        print(pbar.postfix)
        pbar.close()

        return OrderedDict([('loss', avg_meters['loss'].avg), ('dice', avg_meters['dice'].avg), ('min_val_loss', min_val_loss)])


if __name__ == "__main__":

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),  # ('iou', []),
        ('dice', []),
        ('val_loss', []),
        ('val_dice', []),  # ('val_iou', [])
    ])

    min_train_loss = 100
    min_val_loss = 100
    best_dice = 0
    trigger = 0
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    load_flag = 0
    train_all_flag = 1
    path_mask_generator = 'results/train/dedicated_color_models/mask_generator.pth'
    path_residue_predictor = 'results/train/dedicated_color_models/residue_predictor.pth'

    mask_generator.load_state_dict(torch.load(path_mask_generator))
    residue_predictor.load_state_dict(torch.load(path_residue_predictor))

    path_model = 'models/multitask/20211207_3/model_epoch_70.pth'
    model.load_state_dict(torch.load(path_model))

    for epoch in range(1, args.epochs + 1):
        print('Start training')

        if train_all_flag == 1:
            test_for_paper()
            # train_log = train(epoch,min_train_loss)
            # val_log = val_after_finish_color_seg(epoch, min_val_loss)

            train_log = train_after_finish_color_seg(epoch, min_train_loss)
            if (train_log['min_train_loss'] == -1):
                train_all_flag = 0
        else:
            train_log = train_after_finish_color_seg(epoch, min_train_loss)

        # if (load_flag == 0 and train_all_flag == 0):
        #     load_flag = 1
        #     mask_generator.load_state_dict(torch.load(path_mask_generator))
        #     residue_predictor.load_state_dict(torch.load(path_residue_predictor))

        if train_all_flag == 1:
            # val_log = val(epoch,min_val_loss)
            val_log = val_after_finish_color_seg(epoch, min_val_loss)
        else:
            val_log = val_after_finish_color_seg(epoch, min_val_loss)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()

        print('loss %.4f - dice %.4f - val_loss %.4f - val_dice %.4f'
              % (train_log['loss'], train_log['dice'], val_log['loss'], val_log['dice']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['dice'].append(train_log['dice'])  # iou
        log['val_loss'].append(val_log['loss'])
        log['val_dice'].append(val_log['dice'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if epoch % 10 == 0:
            torch.save(model.state_dict(), 'models/%s/model_epoch_%d.pth' %
                       (config['name'], epoch))
            print("=> saved model every 10 epochs")

        if val_log['dice'] > best_dice:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_dice = val_log['dice']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()
