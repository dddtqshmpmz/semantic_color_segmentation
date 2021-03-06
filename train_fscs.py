from __future__ import print_function
import argparse
from math import nan
import math
import numpy as np
import torch
import torch.utils.data
from torch import le, mode, nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from net import MaskGenerator, ResiduePredictor
from mydataset import MyDataset, MyDatasetIHC, MyDataset365_with_IHC
from dataParallel import BalancedDataParallel
import pytorch_ssim
import cv2
import os
import sys
from tqdm import tqdm
import lpips
from util import psnr,ssim,reconst_loss,temp_distance,squared_mahalanobis_distance_loss,alpha_normalize,mono_color_reconst_loss,delta_E
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# gpu_num = len(os.environ['CUDA_VISIBLE_DEVICES'].split(',') ) - 1
# gpu0_bsz = 3
# other_gpu_bsz = 22
# batch_size = gpu0_bsz + gpu_num * other_gpu_bsz

parser = argparse.ArgumentParser(description='baseline')
parser.add_argument('--run_name', type=str, default='train', help='run-name. This name is used for output folder.')
parser.add_argument('--batch_size', type=int, default= 1, metavar='N',  ## 32-> 4
                    help='input batch size for training (default: 32)')
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
parser.add_argument('--lpips_loss_lambda',type=float,default=0.1, 
                    help='lpips_loss_lambda') 

parser.add_argument('--save_layer_train', type=int, default=1,
                    help='save_layer_train')


parser.add_argument('--num_workers', type=int, default=8,
                    help='num_workers of dataloader')
parser.add_argument('--csv_path', type=str, default='train.csv', help='path to csv of images path') # sample / places
parser.add_argument('--csv_path_ihc', type=str, default='ihc_30k.csv', help='path to ihc_256 dataset csv of images path')
parser.add_argument('--csv_path_test',type=str, default='train_IHC.csv', help='path to test ihc csv of images path')

parser.add_argument('--log_interval', type=int, default=100, metavar='N', ## 200-> 20 ->30 
                    help='how many batches to wait before logging training status')
parser.add_argument('--reconst_loss_type', type=str, default='l1', help='[mse | l1 | vgg]')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

try:
    os.makedirs('results/%s' % args.run_name)
except OSError:
    pass

# ???????????????????????????
log = open("train_process.log", "a")
sys.stdout = log # log  sys.__stdout__
torch.manual_seed(args.seed)
cudnn.benchmark = True

device = torch.device("cuda" if args.cuda else "cpu")

train_dataset = MyDatasetIHC(args.csv_path_ihc, args.num_primary_color, mode='train')
# train_dataset = MyDataset365_with_IHC(args.csv_path, args.csv_path_ihc, args.num_primary_color, mode = 'train')
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    worker_init_fn=lambda x: np.random.seed(),
    drop_last=True,
    pin_memory=True
    )


val_dataset = MyDatasetIHC(args.csv_path_ihc, args.num_primary_color, mode='val')
# val_dataset = MyDataset365_with_IHC(args.csv_path, args.csv_path_ihc, args.num_primary_color, mode = 'val')
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    drop_last=True
    )


mask_generator = MaskGenerator(args.num_primary_color).to(device)
mask_generator = nn.DataParallel(mask_generator)
mask_generator = mask_generator.cuda()

residue_predictor = ResiduePredictor(args.num_primary_color).to(device)
residue_predictor = nn.DataParallel(residue_predictor)
residue_predictor = residue_predictor.cuda()

# mask_generator = MaskGenerator(args.num_primary_color)
# mask_generator = BalancedDataParallel(gpu0_bsz, mask_generator, dim = 0).to(device)
# mask_generator = mask_generator.cuda()

# residue_predictor = ResiduePredictor(args.num_primary_color)
# residue_predictor = BalancedDataParallel( gpu0_bsz, residue_predictor, dim=0).to(device)
# residue_predictor = residue_predictor.cuda()

loss_fn_alex  = lpips.LPIPS(net='alex',eval_mode=False)
loss_fn_alex = loss_fn_alex.cuda()

loss_fn_alex_eval  = lpips.LPIPS(net='alex',eval_mode=True)
loss_fn_alex_eval = loss_fn_alex_eval.cuda()

params = list(mask_generator.parameters())
params += list(residue_predictor.parameters())


optimizer = optim.Adam(params, lr=5e-4, betas=(0.0, 0.99)) # 1e-3 -> 0.2

def sparse_loss(alpha_layers):
    # alpha_layers: bn, ln, 1, h, w
    #print('alpha_layers.mean().item(): ', alpha_layers.mean().item())
    alpha_layers = alpha_layers.sum(dim=1, keepdim=True) / (alpha_layers * alpha_layers).sum(dim=1, keepdim=True)
    loss = F.l1_loss(alpha_layers, torch.ones_like(alpha_layers).to(device))
    return loss

def im2tensor(image, imtype=np.uint8, cent=1., factor=2.):
    return (image * factor - cent)

def im2tensor2(image, imtype=np.uint8, cent=1., factor=2.):
    return ( (image+cent)/factor )


def train(epoch,min_train_loss):
    mask_generator.train()
    residue_predictor.train()

    train_loss = 0
    r_loss_mean = 0
    m_loss_mean = 0
    # s_loss_mean = 0
    d_loss_mean = 0
    l_loss_mean = 0
    batch_num = 0

    pbar = tqdm(total=len(train_loader))
    for batch_idx, (target_img, primary_color_layers) in enumerate(train_loader):
        target_img = target_img.to(device) # bn, 3ch, h, w
        primary_color_layers = primary_color_layers.to(device)
        #primary_color_layers = primary_color_layers.to(device) # bn, num_primary_color, 3ch, h, w

        optimizer.zero_grad()


        # network???forward?????????
        primary_color_pack = primary_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3))
        pred_alpha_layers_pack = mask_generator(target_img, primary_color_pack)
        #print('pred_alpha_layers_pack.size():', pred_alpha_layers_pack.size())

        # MaskG?????????????????????????????????view??????
        pred_alpha_layers = pred_alpha_layers_pack.view(target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))

        # ??????????????????processing?????????
        processed_alpha_layers = alpha_normalize(pred_alpha_layers)

        # mono_color_layers_pack????????????????????????tensor??????????????????
        #mono_color_layers = primary_color_layers * processed_alpha_layers
        mono_color_layers = torch.cat((primary_color_layers, processed_alpha_layers), 2) #shape: bn, ln, 4, h, w
        mono_color_layers_pack = mono_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3))

        # ResiduePredictor?????????????????????????????????view?????? ????????????Residue Predictor?????????
        residue_pack  = residue_predictor(target_img, mono_color_layers_pack)
        residue = residue_pack.view(target_img.size(0), -1, 3, target_img.size(2), target_img.size(3))
        #pred_unmixed_rgb_layers = mono_color_layers + residue * processed_alpha_layers
        pred_unmixed_rgb_layers = torch.clamp((primary_color_layers + residue), min=0., max=1.0)# * processed_alpha_layers

        # alpha add??????reconst_img???????????????
        #reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)
        reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)
        mono_color_reconst_img = (primary_color_layers * processed_alpha_layers).sum(dim=1)

        # Culculate loss.
        r_loss = reconst_loss(reconst_img, target_img, type=args.reconst_loss_type) * args.rec_loss_lambda
        m_loss = mono_color_reconst_loss(mono_color_reconst_img, target_img) * args.m_loss_lambda
        # s_loss = sparse_loss(processed_alpha_layers) * args.sparse_loss_lambda
        #print('total_loss: ', total_loss)
        d_loss = squared_mahalanobis_distance_loss(primary_color_layers.detach(), processed_alpha_layers, pred_unmixed_rgb_layers) * args.distance_loss_lambda
        
        reconst_img = im2tensor(reconst_img)
        target_img_ = im2tensor(target_img.detach())
        l_loss = loss_fn_alex.forward(reconst_img,target_img_).mean() * args.lpips_loss_lambda
        reconst_img = im2tensor2(reconst_img)

        total_loss = r_loss + m_loss + d_loss  + l_loss

        if (math.isnan(total_loss.item())):
            print('----------------------------------------- total_loss is nan, continue... -----------------------------------------')
            continue
        
        batch_num += 1

        pbar.update(1)

        total_loss.backward()
        train_loss += total_loss.item()
        r_loss_mean += r_loss.item()
        m_loss_mean += m_loss.item()
        # s_loss_mean += s_loss.item()
        d_loss_mean += d_loss.item()
        l_loss_mean += l_loss.item()

        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('')
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(target_img), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                total_loss.item() / len(target_img)))
            print('reconst_loss *lambda: ', r_loss.item() / len(target_img))
            # print('sparse_loss *lambda: ', s_loss.item() / len(target_img))
            print('squared_mahalanobis_distance_loss *lambda: ', d_loss.item() / len(target_img))
            print('lpips_loss *lambda: ', l_loss.item() / len(target_img))


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
    l_loss_mean = l_loss_mean / batch_num

    print('====> Epoch: {} Average loss: {:.6f}'.format(epoch, train_loss ))
    print('====> Epoch: {} Average reconst_loss *lambda: {:.6f}'.format(epoch, r_loss_mean ))
    print('====> Epoch: {} Average mono_loss *lambda: {:.6f}'.format(epoch, m_loss_mean ))
    # print('====> Epoch: {} Average sparse_loss *lambda: {:.6f}'.format(epoch, s_loss_mean ))
    print('====> Epoch: {} Average squared_mahalanobis_distance_loss *lambda: {:.6f}'.format(epoch, d_loss_mean ))
    print('====> Epoch: {} Average lpips_loss *lambda: {:.6f}'.format(epoch, l_loss_mean ))

    if (math.isnan(train_loss)):
        return -1

    # save best model
    if (train_loss < min_train_loss):
        min_train_loss = train_loss
        torch.save(mask_generator.state_dict(), 'results/%s/mask_generator.pth' % (args.run_name))
        torch.save(residue_predictor.state_dict(), 'results/%s/residue_predictor.pth' % args.run_name)

    return min_train_loss



def val(epoch, min_val_loss):
    mask_generator.eval()
    residue_predictor.eval()

    with torch.no_grad():
        val_loss = 0
        r_loss_mean = 0
        m_loss_mean = 0
        # s_loss_mean = 0
        d_loss_mean = 0
        l_loss_mean = 0
        batch_num = 0

        pbar = tqdm(total=len(val_loader))

        for batch_idx, (target_img, primary_color_layers) in enumerate(val_loader):
            target_img = target_img.to(device) # bn, 3ch, h, w
            primary_color_layers = primary_color_layers.to(device)

            primary_color_pack = primary_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3))
            pred_alpha_layers_pack = mask_generator(target_img, primary_color_pack)
            pred_alpha_layers = pred_alpha_layers_pack.view(target_img.size(0), -1, 1, target_img.size(2), target_img.size(3))
            processed_alpha_layers = alpha_normalize(pred_alpha_layers)
            mono_color_layers = torch.cat((primary_color_layers, processed_alpha_layers), 2) #shape: bn, ln, 4, h, w
            mono_color_layers_pack = mono_color_layers.view(target_img.size(0), -1 , target_img.size(2), target_img.size(3))
            residue_pack  = residue_predictor(target_img, mono_color_layers_pack)
            residue = residue_pack.view(target_img.size(0), -1, 3, target_img.size(2), target_img.size(3))
            pred_unmixed_rgb_layers = torch.clamp((primary_color_layers + residue), min=0., max=1.0)
            reconst_img = (pred_unmixed_rgb_layers * processed_alpha_layers).sum(dim=1)
            mono_color_reconst_img = (primary_color_layers * processed_alpha_layers).sum(dim=1)

            # ??????loss ????????????
            r_loss = reconst_loss(reconst_img, target_img, type=args.reconst_loss_type) * args.rec_loss_lambda
            m_loss = mono_color_reconst_loss(mono_color_reconst_img, target_img) * args.m_loss_lambda
            # s_loss = sparse_loss(processed_alpha_layers) * args.sparse_loss_lambda
            #print('total_loss: ', total_loss)
            d_loss = squared_mahalanobis_distance_loss(primary_color_layers.detach(), processed_alpha_layers, pred_unmixed_rgb_layers) * args.distance_loss_lambda

            reconst_img = im2tensor(reconst_img)
            target_img_ = im2tensor(target_img.detach())
            l_loss = loss_fn_alex_eval.forward(reconst_img,target_img_).mean() * args.lpips_loss_lambda
            reconst_img = im2tensor2(reconst_img)

            total_loss = r_loss + m_loss + d_loss + l_loss

            val_loss += total_loss.item()
            r_loss_mean += r_loss.item()
            m_loss_mean += m_loss.item()
            # s_loss_mean += s_loss.item()
            d_loss_mean += d_loss.item()
            l_loss_mean += l_loss.item()

            pbar.update(1)
            batch_num += 1

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
        d_loss_mean = d_loss_mean / batch_num
        l_loss_mean = l_loss_mean / batch_num

        print('====> Epoch: {} Average val loss: {:.6f}'.format(epoch, val_loss ))
        print('====> Epoch: {} Average val reconst_loss *lambda: {:.6f}'.format(epoch, r_loss_mean ))
        print('====> Epoch: {} Average val mono_loss *lambda: {:.6f}'.format(epoch, m_loss_mean ))
        # print('====> Epoch: {} Average val sparse_loss *lambda: {:.6f}'.format(epoch, s_loss_mean ))
        print('====> Epoch: {} Average val squared_mahalanobis_distance_loss *lambda: {:.6f}'.format(epoch, d_loss_mean ))
        print('====> Epoch: {} Average val lpips_loss *lambda: {:.6f}'.format(epoch, l_loss_mean ))

        # save best val model
        if (val_loss < min_val_loss):
            min_val_loss = val_loss
            torch.save(mask_generator.state_dict(), 'results/%s/mask_generator_val.pth' % (args.run_name))
            torch.save(residue_predictor.state_dict(), 'results/%s/residue_predictor_val.pth' % args.run_name)

        return min_val_loss

if __name__ == "__main__":
    
    min_train_loss = 100
    min_val_loss = 100
    for epoch in range(1, args.epochs + 1):
        print('Start training')
        min_train_loss = train(epoch,min_train_loss)
        if (min_train_loss==-1):
            break
        min_val_loss = val(epoch,min_val_loss)

