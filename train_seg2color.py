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
from net import MaskGenerator, ResiduePredictor, MaskGeneratorSeg2Color, ResiduePredictorSeg2Color
from mydataset import MyDataset, MyDatasetIHC,MyDatasetMulti, MyDatasetIHC_365
import pytorch_ssim
import cv2
import os
import sys
from collections import OrderedDict
from glob import glob
import pandas as pd
import torch.optim as optim
from torchsummary import summary

import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import segmentation_models.segmentation_models_pytorch as smp

import archs
import losses
from dataset import Dataset,DatasetSemanColor,DatasetSemanColor2Decoder
from metrics import iou_score, dice_coef
from utils import AverageMeter, str2bool
from util import psnr,ssim,reconst_loss,temp_distance,squared_mahalanobis_distance_loss,alpha_normalize,mono_color_reconst_loss

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'


parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--run_name', type=str, default='train', help='run-name. This name is used for output folder.')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',  ## 32-> 4
                    help='input batch size for training (default: 32)')

parser.add_argument('--after_batch_size', type=int, default= 18, metavar='N',  ## 32-> 4
                    help='input batch size for training after color model complete (default: 32)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', ## 10
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
parser.add_argument('--sparse_loss_lambda', type=float, default=0.0, # 1.0
                    help='sparse_loss lambda')
parser.add_argument('--distance_loss_lambda', type=float, default=0.5, # 1.0 
                    help='distance_loss_lambda')

parser.add_argument('--save_layer_train', type=int, default=1,
                    help='save_layer_train')

# ???????????????????????????????????????
parser.add_argument('--color_lr',default=5e-4, type=float,
                    help='color learning rate')  # 1e-3


parser.add_argument('--num_workers', type=int, default=8,
                    help='num_workers of dataloader')
parser.add_argument('--csv_path_ihc', type=str, default='ihc_30k.csv')
parser.add_argument('--csv_path_365',type=str, default='train.csv')

parser.add_argument('--log_interval', type=int, default=100, metavar='N', ## 200-> 20 ->30 
                    help='how many batches to wait before logging training status')
parser.add_argument('--reconst_loss_type', type=str, default='l1', help='[mse | l1 | vgg]')





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

# ???????????????????????????
log = open("train_process.log", "a")
sys.stdout = log  # log sys.__stdout__
torch.manual_seed(args.seed)
cudnn.benchmark = True

device = torch.device("cuda" if args.cuda else "cpu")

train_dataset = MyDatasetIHC_365( args.csv_path_365 ,args.csv_path_ihc ,args.num_primary_color, mode='train')
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    worker_init_fn=lambda x: np.random.seed(),
    drop_last=True,
    pin_memory=True
    )


train_after_dataset =  MyDatasetIHC( args.csv_path_ihc ,args.num_primary_color, mode='train')
train_after_loader = torch.utils.data.DataLoader(
    train_after_dataset,
    batch_size=args.after_batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    worker_init_fn=lambda x: np.random.seed(),
    drop_last=True,
    pin_memory=True
)


val_after_dataset =  MyDatasetIHC( args.csv_path_ihc ,args.num_primary_color, mode='val')
val_after_loader = torch.utils.data.DataLoader(
    val_after_dataset,
    batch_size= args.after_batch_size , # 1
    shuffle=False,
    num_workers= args.num_workers , # 1
    drop_last=True  # ????????????
)

mask_generator = MaskGenerator(args.num_primary_color).to(device)
mask_generator = nn.DataParallel(mask_generator)
mask_generator = mask_generator.cuda()

residue_predictor = ResiduePredictor(args.num_primary_color).to(device)
residue_predictor = nn.DataParallel(residue_predictor)
residue_predictor = residue_predictor.cuda()

# model C
mask_generator_seg2color = MaskGeneratorSeg2Color(args.num_primary_color).to(device)
mask_generator_seg2color = nn.DataParallel(mask_generator_seg2color)
mask_generator_seg2color = mask_generator_seg2color.cuda()

residue_predictor_seg2color = ResiduePredictorSeg2Color(args.num_primary_color).to(device)
residue_predictor_seg2color = nn.DataParallel(residue_predictor_seg2color)
residue_predictor_seg2color = residue_predictor_seg2color.cuda()

params = list(mask_generator_seg2color.parameters())
params += list(residue_predictor_seg2color.parameters())


optimizer = optim.Adam(params, lr=config['color_lr'], betas=(0.0, 0.99)) # 0926


# ???????????????
model = smp.UnetWithColor('efficientnet-b3', in_channels= 3+7 ,
                     classes= 1, encoder_weights='imagenet').cuda()
model = model.cuda()
model= nn.DataParallel(model,device_ids=[0,1,2])



if config['loss'] == 'BCEWithLogitsLoss':
    criterion = nn.BCEWithLogitsLoss().cuda()
else:
    criterion = losses.__dict__[config['loss']]().cuda()



def sparse_loss(alpha_layers):
    # alpha_layers: bn, ln, 1, h, w
    #print('alpha_layers.mean().item(): ', alpha_layers.mean().item())
    alpha_layers = alpha_layers.sum(dim=1, keepdim=True) / (alpha_layers * alpha_layers).sum(dim=1, keepdim=True)
    loss = F.l1_loss(alpha_layers, torch.ones_like(alpha_layers).to(device))
    return loss

def train(epoch,min_train_loss):

    mask_generator_seg2color.train()
    residue_predictor_seg2color.train()

    mask_generator.eval()
    # residue_predictor.eval()
    model.eval()

    train_loss = 0
    r_loss_mean = 0
    m_loss_mean = 0
    # s_loss_mean = 0
    d_loss_mean = 0
    batch_num = 0

    pbar = tqdm(total=len(train_after_loader))
    for batch_idx, (target_img, primary_color_layers) in enumerate(train_after_loader):

        target_img = target_img.to(device) # bn, 3ch, h, w
        primary_color_layers = primary_color_layers.to(device)
        # mask = mask.to(device)

        optimizer.zero_grad()


        primary_color_pack_old = primary_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3))
        pred_alpha_layers_pack_old = mask_generator(target_img, primary_color_pack_old)
        pred_alpha_layers_old = pred_alpha_layers_pack_old.view(target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))
        processed_alpha_layers_old = alpha_normalize(pred_alpha_layers_old)

        processed_alpha_layers_old = torch.squeeze(processed_alpha_layers_old,dim=2)
        output = model(torch.cat((target_img.detach(),processed_alpha_layers_old.detach()),1),processed_alpha_layers_old.detach())


        # network???forward?????????
        primary_color_pack = primary_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3))
        pred_alpha_layers_pack = mask_generator_seg2color(target_img.detach(), primary_color_pack.detach(),output.detach())  # ??????target_img??????detach??????
        pred_alpha_layers = pred_alpha_layers_pack.view(target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))

        # ??????????????????processing?????????
        processed_alpha_layers = alpha_normalize(pred_alpha_layers)

        # mono_color_layers_pack????????????????????????tensor??????????????????
        #mono_color_layers = primary_color_layers * processed_alpha_layers
        mono_color_layers = torch.cat((primary_color_layers, processed_alpha_layers), 2) #shape: bn, ln, 4, h, w
        mono_color_layers_pack = mono_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3))

        # ResiduePredictor?????????????????????????????????view?????? ????????????Residue Predictor?????????
        residue_pack  = residue_predictor_seg2color(target_img.detach(), mono_color_layers_pack, output.detach())
        residue = residue_pack.view(target_img.size(0), -1, 3, target_img.size(2), target_img.size(3))
        #pred_unmixed_rgb_layers = mono_color_layers + residue * processed_alpha_layers
        pred_unmixed_rgb_layers = torch.clamp((primary_color_layers + residue), min=0., max=1.0)# * processed_alpha_layers

        # alpha add??????reconst_img???????????????
        reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)
        mono_color_reconst_img = (primary_color_layers * processed_alpha_layers).sum(dim=1)

        # Culculate loss.
        r_loss = reconst_loss(reconst_img, target_img, type=args.reconst_loss_type) * args.rec_loss_lambda
        m_loss = mono_color_reconst_loss(mono_color_reconst_img, target_img) * args.m_loss_lambda
        # s_loss = sparse_loss(processed_alpha_layers) * args.sparse_loss_lambda
        #print('total_loss: ', total_loss)
        d_loss = squared_mahalanobis_distance_loss(primary_color_layers.detach(), processed_alpha_layers, pred_unmixed_rgb_layers) * args.distance_loss_lambda

        total_loss = r_loss + m_loss + d_loss
        if ( math.isnan(total_loss.item()) ):
            min_train_loss = -1
            break
        
        batch_num += 1

        train_loss += total_loss.item()
        r_loss_mean += r_loss.item()
        m_loss_mean += m_loss.item()
        # s_loss_mean += s_loss.item()
        d_loss_mean += d_loss.item()


        total_loss.backward()
        optimizer.step()

        pbar.update(1)


        if batch_idx % args.log_interval == 0:
            print('')
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(target_img), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                total_loss.item() / len(target_img)))
            print('reconst_loss *lambda: ', r_loss.item() / len(target_img))
            # print('sparse_loss *lambda: ', s_loss.item() / len(target_img))
            print('squared_mahalanobis_distance_loss *lambda: ', d_loss.item() / len(target_img))


            for save_layer_number in range(args.save_layer_train):
                save_image(primary_color_layers[save_layer_number,:,:,:,:],
                       'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_primary_color_layers.png' % save_layer_number)
                save_image(reconst_img[save_layer_number,:,:,:].unsqueeze(0),
                       'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_reconst_img.png' % save_layer_number)
                save_image(target_img[save_layer_number,:,:,:].unsqueeze(0),
                       'results/%s/train_ep_' % args.run_name + str(epoch) + '_ln_%02d_target_img.png' % save_layer_number)
                
    pbar.close()

    train_loss = train_loss / batch_num
    r_loss_mean = r_loss_mean / batch_num
    m_loss_mean = m_loss_mean / batch_num
    # s_loss_mean = s_loss_mean / batch_num
    d_loss_mean = d_loss_mean / batch_num

    print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, train_loss ))
    print('====> Epoch: {} Average reconst_loss *lambda: {:.6f}'.format(epoch, r_loss_mean ))
    print('====> Epoch: {} Average mono_loss *lambda: {:.6f}'.format(epoch, m_loss_mean ))
    # print('====> Epoch: {} Average sparse_loss *lambda: {:.6f}'.format(epoch, s_loss_mean ))
    print('====> Epoch: {} Average squared_mahalanobis_distance_loss *lambda: {:.6f}'.format(epoch, d_loss_mean ))

    if (math.isnan(train_loss)):
        min_train_loss = -1

    # save best model
    if (train_loss < min_train_loss):
        min_train_loss = train_loss
        torch.save(mask_generator_seg2color.state_dict(), 'results/%s/mask_generator_seg2color.pth' % (args.run_name))
        torch.save(residue_predictor_seg2color.state_dict(), 'results/%s/residue_predictor_seg2color.pth' % args.run_name)

    return min_train_loss



def val(epoch,min_val_loss):

    mask_generator.eval()
    residue_predictor.eval()
    model.eval()

    mask_generator_seg2color.eval()
    residue_predictor_seg2color.eval()

    with torch.no_grad():
        val_loss = 0
        r_loss_mean = 0
        m_loss_mean = 0
        # s_loss_mean = 0
        d_loss_mean = 0
        batch_num = 0

        pbar = tqdm(total=len(val_after_loader))

        for batch_idx, (target_img, primary_color_layers  ) in enumerate(val_after_loader):
            target_img = target_img.to(device) # bn, 3ch, h, w
            primary_color_layers = primary_color_layers.to(device)
            # mask = mask.to(device)


            primary_color_pack_old = primary_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3)) 
            pred_alpha_layers_pack_old = mask_generator(target_img, primary_color_pack_old)
            pred_alpha_layers_old = pred_alpha_layers_pack_old.view(target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))
            processed_alpha_layers_old = alpha_normalize(pred_alpha_layers_old)

            processed_alpha_layers_old = torch.squeeze(processed_alpha_layers_old,dim=2)
            output = model(torch.cat((target_img.detach(),processed_alpha_layers_old.detach()),1),processed_alpha_layers_old.detach())


            # network???forward?????????
            primary_color_pack = primary_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3))
            pred_alpha_layers_pack = mask_generator_seg2color(target_img.detach(), primary_color_pack.detach(),output.detach())
            pred_alpha_layers = pred_alpha_layers_pack.view(target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))

            # ??????????????????processing?????????
            processed_alpha_layers = alpha_normalize(pred_alpha_layers)

            mono_color_layers = torch.cat((primary_color_layers, processed_alpha_layers), 2) #shape: bn, ln, 4, h, w
            mono_color_layers_pack = mono_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3))

            # ResiduePredictor?????????????????????????????????view?????? ????????????Residue Predictor?????????
            residue_pack  = residue_predictor_seg2color(target_img.detach(), mono_color_layers_pack.detach(), output.detach())
            residue = residue_pack.view(target_img.size(0), -1, 3, target_img.size(2), target_img.size(3))
            pred_unmixed_rgb_layers = torch.clamp((primary_color_layers + residue), min=0., max=1.0)# * processed_alpha_layers

            # alpha add??????reconst_img???????????????
            reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)
            mono_color_reconst_img = (primary_color_layers * processed_alpha_layers).sum(dim=1)


            # ??????loss ????????????
            r_loss = reconst_loss(reconst_img, target_img, type=args.reconst_loss_type) * args.rec_loss_lambda
            m_loss = mono_color_reconst_loss(mono_color_reconst_img, target_img) * args.m_loss_lambda
            # s_loss = sparse_loss(processed_alpha_layers) * args.sparse_loss_lambda
            #print('total_loss: ', total_loss)
            d_loss = squared_mahalanobis_distance_loss(primary_color_layers.detach(), processed_alpha_layers, pred_unmixed_rgb_layers) * args.distance_loss_lambda

            batch_num += 1
            total_loss = r_loss + m_loss + d_loss
            val_loss += total_loss.item()
            r_loss_mean += r_loss.item()
            m_loss_mean += m_loss.item()
            # s_loss_mean += s_loss.item()
            d_loss_mean += d_loss.item()

            pbar.update(1)

            save_layer_number = 0
            if batch_idx <= 1:
                # batchsize??????????????????????????????????????????????????????????????????
                save_image(primary_color_layers[save_layer_number,:,:,:,:],
                    'results/%s/val_ep_' % args.run_name + str(epoch) + '_idx_%02d_primary_color_layers.png' % batch_idx)
                save_image(reconst_img[save_layer_number,:,:,:].unsqueeze(0),
                    'results/%s/val_ep_' % args.run_name + str(epoch) + '_idx_%02d_reconst_img.png' % batch_idx)
                save_image(target_img[save_layer_number,:,:,:].unsqueeze(0),
                    'results/%s/val_ep_' % args.run_name + str(epoch) + '_idx_%02d_target_img.png' % batch_idx)

        pbar.close()
            
        val_loss = val_loss / batch_num
        r_loss_mean = r_loss_mean / batch_num
        m_loss_mean = m_loss_mean / batch_num
        # s_loss_mean = s_loss_mean / batch_num
        d_loss_mean = d_loss_mean / batch_num

        print('====> Epoch: {} Average val loss: {:.6f}'.format(epoch, val_loss ))
        print('====> Epoch: {} Average val reconst_loss *lambda: {:.6f}'.format(epoch, r_loss_mean ))
        print('====> Epoch: {} Average val mono_loss *lambda: {:.6f}'.format(epoch, m_loss_mean ))
        # print('====> Epoch: {} Average val sparse_loss *lambda: {:.6f}'.format(epoch, s_loss_mean ))
        print('====> Epoch: {} Average val squared_mahalanobis_distance_loss *lambda: {:.6f}'.format(epoch, d_loss_mean ))

        if (math.isnan(val_loss)):
            min_val_loss = -1
    
        # save best val model
        if (val_loss < min_val_loss):
            min_val_loss = val_loss
            torch.save(mask_generator_seg2color.state_dict(), 'results/%s/mask_generator_seg2color_val.pth' % (args.run_name))
            torch.save(residue_predictor_seg2color.state_dict(), 'results/%s/residue_predictor_seg2color_val.pth' % args.run_name)

        return min_val_loss



if __name__ == "__main__":

    min_train_loss = 100
    min_val_loss = 100
    best_dice = 0


    path_mask_generator = 'results/train/dedicated_color_models/mask_generator.pth'
    path_residue_predictor = 'results/train/dedicated_color_models/residue_predictor.pth'
    path_seg_model = 'models/multitask/20211120/model.pth'


    mask_generator.load_state_dict(torch.load(path_mask_generator))
    residue_predictor.load_state_dict(torch.load(path_residue_predictor))
    model.load_state_dict(torch.load(path_seg_model))


    for epoch in range(1, args.epochs + 1):
        print('Start training')

        min_train_loss = train(epoch,min_train_loss)
        if (min_train_loss == 1):
            break

        min_val_loss = val(epoch,min_val_loss)

        torch.cuda.empty_cache()
    
