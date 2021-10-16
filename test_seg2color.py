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
from guided_filter_pytorch.guided_filter import GuidedFilter
import time

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

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


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

# 让颜色分割网络收敛的慢一些
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

# 打印所有数据到日志
log = open("train_process.log", "a")
sys.stdout = log  # log sys.__stdout__
torch.manual_seed(args.seed)
cudnn.benchmark = True

device = torch.device("cuda" if args.cuda else "cpu")


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



# 加载新参数
model = smp.UnetWithColor('efficientnet-b3', in_channels= 3+7 ,
                     classes= 1, encoder_weights='imagenet').cuda()
model = model.cuda()
model= nn.DataParallel(model,device_ids=[0,1,2,3])



if config['loss'] == 'BCEWithLogitsLoss':
    criterion = nn.BCEWithLogitsLoss().cuda()
else:
    criterion = losses.__dict__[config['loss']]().cuda()



def sparse_loss(alpha_layers,device):
    # alpha_layers: bn, ln, 1, h, w
    #print('alpha_layers.mean().item(): ', alpha_layers.mean().item())
    alpha_layers = alpha_layers.sum(dim=1, keepdim=True) / (alpha_layers * alpha_layers).sum(dim=1, keepdim=True)
    loss = F.l1_loss(alpha_layers, torch.ones_like(alpha_layers).to(device))
    return loss

# 必要な関数を定義する
# 用人工挑选的颜色代替之前选择的主要颜色
def replace_color(primary_color_layers, manual_colors):
    temp_primary_color_layers = primary_color_layers.clone()
    for layer in range(len(manual_colors)):
        for color in range(3):
                temp_primary_color_layers[:,layer,color,:,:].fill_(manual_colors[layer][color])
    return temp_primary_color_layers


def cut_edge(target_img):
    #print(target_img.size())
    target_img = F.interpolate(target_img, scale_factor=resize_scale_factor, mode='area')
    #print(target_img.size())
    h = target_img.size(2)
    w = target_img.size(3)
    h = h - (h % 32)
    w = w - (w % 32)
    target_img = target_img[:,:,:h,:w]
    #print(target_img.size())
    return target_img

def alpha_normalize(alpha_layers):
    # constraint (sum = 1)
    # layersの状態で受け取り，その形で返す. bn, ln, 1, h, w 以层的状态接收并以该形式返回 Bn，ln，1，h，w
    return alpha_layers / (alpha_layers.sum(dim=1, keepdim=True) + 1e-8)

def read_backimage():
    img = cv2.imread('../FSCS/dataset/backimage.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2,0,1)) # c,h,w
    img = img/255
    img = torch.from_numpy(img.astype(np.float32))
    return img.view(1,3,256,256).to(device)

def proc_guidedfilter(alpha_layers, guide_img):
    # guide_imgは， 1chのモノクロに変換 guide_img转换为1通道单色
    # target_imgを使う． bn, 3, h, w
    guide_img = (guide_img[:, 0, :, :]*0.299 + guide_img[:, 1, :, :]*0.587 + guide_img[:, 2, :, :]*0.114).unsqueeze(1)
        
    # lnのそれぞれに対してguideg filterを実行 对ln的每个运行引导过滤器
    for i in range(alpha_layers.size(1)):
        # layerは，bn, 1, h, w
        layer = alpha_layers[:, i, :, :, :]
        
        processed_layer = GuidedFilter(3, 1*1e-6)(guide_img, layer) #可以去了解一下什么是GuidedFilter smooth the image
        # レイヤーごとの結果をまとめてlayersの形に戻す (bn, ln, 1, h, w)
        # 将每个图层的结果放回图层（bn，ln，1，h，w）
        if i == 0: 
            processed_alpha_layers = processed_layer.unsqueeze(1)
        else:
            processed_alpha_layers = torch.cat((processed_alpha_layers, processed_layer.unsqueeze(1)), dim=1)
    
    return processed_alpha_layers

## Define functions for mask operation.
# マスクを受け取る関数
# target_layer_numberが冗長なレイヤーの番号（２つ）のリスト．これらのレイヤーに操作を加える
# 接收蒙版的功能
# 层编号（2）的列表，其中target_layer_number是冗余的。 向这些层添加操作

def load_mask(mask_path):
    mask = cv2.imread(mask_path, 0) #白黒で読み込み ＃黑白阅读
    mask[mask<128] = 0.
    mask[mask >= 128] = 1.
    # tensorに変換する #转换为张量
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().cuda()
    
    return mask

def mask_operate(alpha_layers, target_layer_number, mask_path):
    layer_A = alpha_layers[:, target_layer_number[0], :, :, :]
    layer_B = alpha_layers[:, target_layer_number[1], :, :, :]
    processed_alpha_layers
    layer_AB = layer_A + layer_B
    mask = load_mask(mask_path)
    
    mask = cut_edge(mask)
    
    layer_A = layer_AB * mask
    layer_B = layer_AB * (1. - mask)
    
    return_alpha_layers = alpha_layers.clone()
    return_alpha_layers[:, target_layer_number[0], :, :, :] = layer_A
    return_alpha_layers[:, target_layer_number[1], :, :, :] = layer_B
    
    return return_alpha_layers



#### User inputs
run_name = 'train_1015_1'
num_primary_color = 7
csv_path = 'train.csv' # なんでも良い．後方でパスを置き換えるから
csv_path_ihc = 'train_IHC_256_2w.csv'
csv_path_test = 'train_IHC.csv'

# 设置loss权重
rec_loss_lambda = 1.0
m_loss_lambda = 1.0
sparse_loss_lambda = 0.0
distance_loss_lambda = 0.5

# 设置哪台机器上跑test
device_id = 0
device = 'cuda'

# sys.stdout = sys.__stdout__
# 打印所有数据到日志
log = open("test_%s.log" % (run_name), "a")
sys.stdout = log

resize_scale_factor = 1  



test_dataset = MyDataset(csv_path, csv_path_ihc, csv_path_test,num_primary_color, mode='test')
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_workers,
    )

# define model
mask_generator = MaskGenerator(num_primary_color).to(device)
residue_predictor = ResiduePredictor(num_primary_color).to(device)
mask_generator_seg2color = MaskGeneratorSeg2Color(num_primary_color).to(device)
residue_predictor_seg2color = ResiduePredictorSeg2Color(num_primary_color).to(device)


# 注意使用多gpu并行训练 需要添加如下代码
mask_generator = nn.DataParallel(mask_generator,device_ids=[0,1,2,3])
mask_generator = mask_generator.cuda(device)
residue_predictor = nn.DataParallel(residue_predictor,device_ids=[0,1,2,3])
residue_predictor = residue_predictor.cuda(device)

mask_generator_seg2color = nn.DataParallel(mask_generator_seg2color,device_ids=[0,1,2])
mask_generator_seg2color = mask_generator_seg2color.cuda(device)
residue_predictor_seg2color = nn.DataParallel(residue_predictor_seg2color,device_ids=[0,1,2])
residue_predictor_seg2color = residue_predictor_seg2color.cuda(device)


backimage = read_backimage()

target_layer_number = [0, 1] # マスクで操作するレイヤーの番号 ＃层号与遮罩一起使用
mask_path = 'path/to/mask.image'



print('Start!')


path_mask_generator = 'results/train/20211013_1/mask_generator.pth'
path_residue_predictor = 'results/train/20211013_1/residue_predictor.pth'
path_seg_model = 'models/multitask/20211013_1_9004/model.pth'

path_mask_generator_seg2color = 'results/train/20211015_1/mask_generator.pth'
path_residue_predictor_seg2color = 'results/train/20211015_1/residue_predictor.pth'


mask_generator.load_state_dict(torch.load(path_mask_generator))
residue_predictor.load_state_dict(torch.load(path_residue_predictor))
model.load_state_dict(torch.load(path_seg_model))
mask_generator_seg2color.load_state_dict(torch.load(path_mask_generator_seg2color))
residue_predictor_seg2color.load_state_dict(torch.load(path_residue_predictor_seg2color))

mask_generator_seg2color.eval()
residue_predictor_seg2color.eval()

mask_generator.eval()
residue_predictor.eval()
model.eval()
   

mean_estimation_time = 0
img_index = 0

with torch.no_grad():
    test_loss = 0   
    r_loss_mean = 0
    m_loss_mean = 0
    s_loss_mean = 0
    d_loss_mean = 0
    psnr_mean = 0
    ssim_mean = 0
    batch_num = 0
    
    pbar = tqdm(total=len(test_loader))

    for batch_idx, (target_img, primary_color_layers) in enumerate(test_loader):
        pbar.update(1)
        if batch_idx < 378: 
            continue
        print('img #', batch_idx)

        target_img = cut_edge(target_img)
        target_img = target_img.to(device) # bn, 3ch, h, w
        primary_color_layers = primary_color_layers.to(device)

        primary_color_layers = primary_color_layers.view(primary_color_layers.size(0), -1 , primary_color_layers.size(3), primary_color_layers.size(4))
        primary_color_layers = cut_edge(primary_color_layers)
        primary_color_layers = primary_color_layers.view(primary_color_layers.size(0),7,3,primary_color_layers.size(2),primary_color_layers.size(3))

        
        start_time = time.time()
        target_img = target_img.to(device) # bn, 3ch, h, w

        primary_color_pack_old = primary_color_layers.view(primary_color_layers.size(0), -1 , primary_color_layers.size(3), primary_color_layers.size(4))

        pred_alpha_layers_pack_old = mask_generator(target_img, primary_color_pack_old)
        pred_alpha_layers_old = pred_alpha_layers_pack_old.view(target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))
        processed_alpha_layers_old = alpha_normalize(pred_alpha_layers_old)

        processed_alpha_layers_old = torch.squeeze(processed_alpha_layers_old,dim=2)
        output = model(torch.cat((target_img.detach(),processed_alpha_layers_old.detach()),1),processed_alpha_layers_old.detach())


        # networkにforwardにする
        primary_color_pack = primary_color_layers.view(primary_color_layers.size(0), -1 , primary_color_layers.size(3), primary_color_layers.size(4))

        pred_alpha_layers_pack = mask_generator_seg2color(target_img.detach(), primary_color_pack.detach(),output.detach())
        pred_alpha_layers = pred_alpha_layers_pack.view(target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))

        # 正規化などのprocessingを行う
        processed_alpha_layers = alpha_normalize(pred_alpha_layers)
        processed_alpha_layers = proc_guidedfilter(processed_alpha_layers, target_img) # Option
        processed_alpha_layers = alpha_normalize(processed_alpha_layers)  # Option


        mono_color_layers = torch.cat((primary_color_layers, processed_alpha_layers), 2) #shape: bn, ln, 4, h, w
        mono_color_layers_pack = mono_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3))

        # ResiduePredictorの出力をレイヤーごとにviewする 逐层查看Residue Predictor的输出
        residue_pack  = residue_predictor_seg2color(target_img.detach(), mono_color_layers_pack.detach(), output.detach())
        residue = residue_pack.view(target_img.size(0), -1, 3, target_img.size(2), target_img.size(3))
        pred_unmixed_rgb_layers = torch.clamp((primary_color_layers + residue), min=0., max=1.0)# * processed_alpha_layers

        # alpha addしてreconst_imgを作成する
        reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)
        mono_color_reconst_img = (primary_color_layers * processed_alpha_layers).sum(dim=1)

        end_time = time.time()
        estimation_time = end_time - start_time
        print('estimation_time: ',estimation_time)
        mean_estimation_time += estimation_time

        batch_num += 1

        # 计算loss 打印出来
        r_loss = reconst_loss(reconst_img, target_img, 'l1') * rec_loss_lambda
        m_loss = mono_color_reconst_loss(mono_color_reconst_img, target_img) * m_loss_lambda
        # s_loss = sparse_loss(processed_alpha_layers,device) # 注意这个没有乘以 权重
        #print('total_loss: ', total_loss)
        d_loss = squared_mahalanobis_distance_loss(primary_color_layers.detach(), processed_alpha_layers, pred_unmixed_rgb_layers) * distance_loss_lambda
        psnr_res = psnr(reconst_img, target_img)
        ssim_res = ssim(reconst_img, target_img)

        total_loss = r_loss + m_loss + d_loss
        test_loss += total_loss.item()
        r_loss_mean += r_loss.item()
        m_loss_mean += m_loss.item()
        # s_loss_mean += s_loss.item()
        d_loss_mean += d_loss.item()

        psnr_mean += psnr_res.item()
        ssim_mean += ssim_res.item()

        print('r_loss:', r_loss)
        print('psnr:', psnr_res)
        print('ssim:', ssim_res)
        # print('sparsity:',s_loss)

        

        if (batch_idx %10 == 0 and True): # 
            img_index = 1
            try:
                os.makedirs('results/%s/test' % (run_name))
            except OSError:
                pass
            save_layer_number = 0
            save_image(primary_color_layers[save_layer_number,:,:,:,:],
                   'results/%s/test/test' % (run_name) + '_img-%02d_primary_color_layers.png' % batch_idx)
            save_image(reconst_img[save_layer_number,:,:,:].unsqueeze(0),
                   'results/%s/test/test' % (run_name)  + '_img-%02d_reconst_img.png' % batch_idx)
            save_image(target_img[save_layer_number,:,:,:].unsqueeze(0),
                   'results/%s/test/test' % (run_name)  + '_img-%02d_target_img.png' % batch_idx)

            print('Saved to results/%s/test/...' % (run_name))
           
    pbar.close()

    test_loss = test_loss / batch_num
    r_loss_mean = r_loss_mean / batch_num
    m_loss_mean = m_loss_mean / batch_num
    # s_loss_mean = s_loss_mean / batch_num
    d_loss_mean = d_loss_mean / batch_num
    
    psnr_mean = psnr_mean / batch_num
    ssim_mean = ssim_mean / batch_num

    print('====> Average test loss: {:.6f}'.format(test_loss ))
    print('====> Average test reconst_loss *lambda: {:.6f}'.format( r_loss_mean ))
    print('====> Average test mono_loss *lambda: {:.6f}'.format( m_loss_mean ))
    print('====> Average test squared_mahalanobis_distance_loss *lambda: {:.6f}'.format( d_loss_mean ))

    print('====> Average test psnr: {:.6f}'.format( psnr_mean ))
    print('====> Average test ssim: {:.6f}'.format( ssim_mean ))
    # print('====> Average test sparse_loss: {:.6f}'.format( s_loss_mean ))

    print('mean_estimation_time: ', mean_estimation_time / batch_num )
    


