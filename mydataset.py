from random import seed
import torch
from torch.utils.data.dataset import Dataset
import cv2
import pandas as pd
import numpy as np
import os
from torchvision.utils import make_grid, save_image

class MyDataset(Dataset):
    def __init__(self, csv_path, csv_path_ihc, csv_path_test, num_primary_color, mode=None):
        self.csv_path = csv_path
        ihc_num = 10000 # ihc 数据集设置数量
        val_num_train = 6500
        val_num_ihc = 200

        if mode == 'train':
            
            self.imgs_path = np.array(pd.read_csv(csv_path, header=None)).reshape(-1)[:-val_num_train] #csvリストの後ろをvaldataに
            self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path), header=None)).reshape(-1, num_primary_color*3)[:-val_num_train]
            
            self.imgs_path_ihc = np.array(pd.read_csv(csv_path_ihc, header=None)).reshape(-1)[:ihc_num] #csvリストの後ろをvaldataに
            self.palette_list_ihc = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path_ihc), header=None)).reshape(-1, num_primary_color*3)[:ihc_num]
            '''
            # -----以下两行拷贝
            self.imgs_path = self.imgs_path_ihc
            self.palette_list = self.palette_list_ihc
            '''
            
            self.imgs_path = np.concatenate((self.imgs_path,self.imgs_path_ihc),axis=0)  
            self.palette_list = np.concatenate((self.palette_list,self.palette_list_ihc),axis=0)

            # 图片路径换了
            self.imgs_path = np.array( [x.replace('../','../FSCS/')  for x in self.imgs_path] )
        
        if mode == 'val':
            self.imgs_path = np.array(pd.read_csv(csv_path, header=None)).reshape(-1)[-val_num_train:] #csvリストの後ろをvaldataに
            self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path), header=None)).reshape(-1, num_primary_color*3)[-val_num_train:]
            
            self.imgs_path_ihc = np.array(pd.read_csv(csv_path_ihc, header=None)).reshape(-1)[-val_num_ihc:] #csvリストの後ろをvaldataに
            self.palette_list_ihc = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path_ihc), header=None)).reshape(-1, num_primary_color*3)[-val_num_ihc:]
            
            self.imgs_path = np.concatenate((self.imgs_path,self.imgs_path_ihc),axis=0)  
            self.palette_list = np.concatenate((self.palette_list,self.palette_list_ihc),axis=0)

            # 图片路径换了
            self.imgs_path = np.array( [x.replace('../','../FSCS/')  for x in self.imgs_path] )

        if mode == 'test':
            self.imgs_path = np.array(pd.read_csv(csv_path_test, header=None)).reshape(-1)
            self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path_test), header=None)).reshape(-1, num_primary_color*3)
            self.imgs_path = np.array( [x.replace('train','train_IHC')  for x in self.imgs_path] )

            self.imgs_path = np.array( [x.replace('../','../FSCS/')  for x in self.imgs_path] )

        self.num_primary_color = num_primary_color

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        
        #target_size = 256
        # img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2,0,1)) # h,w,c -> c,h,w
        target_img = img/255 # 0~1

        # select primary_color
        primary_color_layers = self.make_primary_color_layers(self.palette_list[index], target_img)

        # to Tensor
        target_img = torch.from_numpy(target_img.astype(np.float32))
        primary_color_layers = torch.from_numpy(primary_color_layers.astype(np.float32))

        return target_img, primary_color_layers # return torch.Tensor

    def __len__(self):
        return len(self.imgs_path)


    def make_primary_color_layers(self, palette_values, target_img):
        '''
         入力：パレットの色の値 调色板颜色值
         出力：primary_color_layers #out: (bn, ln, 3, h, w)
        '''
        primary_color = palette_values.reshape(self.num_primary_color, 3) / 255 # (ln, 3)
        primary_color_layers = np.tile(np.ones_like(target_img), (self.num_primary_color,1,1,1)) * primary_color.reshape(self.num_primary_color,3,1,1)
        
        return primary_color_layers


class MyDatasetIHC(Dataset):
    def __init__(self, csv_path_ihc, num_primary_color, mode=None):
        ihc_num = 24000 # ihc 数据集设置数量

        if mode == 'train':
            self.imgs_path = np.array(pd.read_csv(csv_path_ihc, header=None)).reshape(-1)[:ihc_num] #csvリストの後ろをvaldataに
            self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path_ihc), header=None)).reshape(-1, num_primary_color*3)[:ihc_num]
            self.masks_path =  np.array(pd.read_csv(csv_path_ihc, header=None)).reshape(-1)[:ihc_num]

            self.masks_path = np.array( [x.replace('images_10pics_30k_patches','masks_10pics_30k_patches')  for x in self.imgs_path] )

        
        if mode == 'val':
            self.imgs_path = np.array(pd.read_csv(csv_path_ihc, header=None)).reshape(-1)[ihc_num:] #csvリストの後ろをvaldataに
            self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path_ihc), header=None)).reshape(-1, num_primary_color*3)[ihc_num:]
            self.masks_path =  np.array(pd.read_csv(csv_path_ihc, header=None)).reshape(-1)[ihc_num:]
            
            self.masks_path = np.array( [x.replace('images_10pics_30k_patches','masks_10pics_30k_patches')  for x in self.imgs_path] )


        self.num_primary_color = num_primary_color

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        mask = cv2.imread(self.masks_path[index],cv2.IMREAD_GRAYSCALE)
 
        target_size = 320
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2,0,1)) # h,w,c -> c,h,w
        target_img = img/255 # 0~1


        mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        mask = np.expand_dims(mask, 2)
        mask = mask.transpose((2,0,1)) # h,w,c -> c,h,w
        mask = mask/255 # 0~1

        # select primary_color
        primary_color_layers = self.make_primary_color_layers(self.palette_list[index], target_img)

        # to Tensor
        target_img = torch.from_numpy(target_img.astype(np.float32))
        primary_color_layers = torch.from_numpy(primary_color_layers.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))

        return target_img, primary_color_layers , mask   # return torch.Tensor

    def __len__(self):
        return len(self.imgs_path)


    def make_primary_color_layers(self, palette_values, target_img):
        '''
         入力：パレットの色の値 调色板颜色值
         出力：primary_color_layers #out: (bn, ln, 3, h, w)
        '''
        primary_color = palette_values.reshape(self.num_primary_color, 3) / 255 # (ln, 3)
        primary_color_layers = np.tile(np.ones_like(target_img), (self.num_primary_color,1,1,1)) * primary_color.reshape(self.num_primary_color,3,1,1)
        
        return primary_color_layers


class MyDatasetIHC2(Dataset):
    def __init__(self, target_img_ids,target_img_dir,target_palette_dir):
        self.target_img_ids = target_img_ids
        self.target_img_dir = target_img_dir
        self.target_palette_dir = target_palette_dir
        self.img_ext = '.png'
        self.num_primary_color = 7
    
    def __len__(self):
        return len(self.target_img_ids)

    def __getitem__(self, idx):
        target_img_id = self.target_img_ids[idx]

        target_img = cv2.imread(os.path.join(self.target_img_dir,
                                                 target_img_id + self.img_ext))

        target_size = 320
        target_img = cv2.resize(target_img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        target_img = target_img.transpose((2,0,1)) # h,w,c -> c,h,w
        target_img = target_img/255 # 0~1

        # select primary_color
        palette_values = pd.read_csv(os.path.join(self.target_palette_dir,target_img_id+'.csv'),header=None).values # 这边验证的时候注意一下有无问题
        primary_color_layers = self.make_primary_color_layers(palette_values, target_img)

        # to Tensor
        target_img = torch.from_numpy(target_img.astype(np.float32))
        primary_color_layers = torch.from_numpy(primary_color_layers.astype(np.float32))

        return target_img, primary_color_layers, target_img_id


    def make_primary_color_layers(self, palette_values, target_img):
        '''
         入力：パレットの色の値 调色板颜色值
         出力：primary_color_layers #out: (bn, ln, 3, h, w)
        '''
        primary_color = palette_values.reshape(self.num_primary_color, 3) / 255 # (ln, 3)
        primary_color_layers = np.tile(np.ones_like(target_img), (self.num_primary_color,1,1,1)) * primary_color.reshape(self.num_primary_color,3,1,1)
        
        return primary_color_layers


class MyDatasetKvasir(Dataset):
    def __init__(self, csv_path_kvasir, num_primary_color, mode=None):
        kva_num = 1300 # kvasir+cvc训练数量

        if mode == 'train':
            self.imgs_path = np.array(pd.read_csv(csv_path_kvasir, header=None)).reshape(-1)[:kva_num] 
            self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path_kvasir), header=None)).reshape(-1, num_primary_color*3)[:kva_num]
            self.masks_path =  np.array(pd.read_csv(csv_path_kvasir, header=None)).reshape(-1)[:kva_num]

            self.masks_path = np.array( [x.replace('images','masks')  for x in self.imgs_path] )

        
        if mode == 'val':
            self.imgs_path = np.array(pd.read_csv(csv_path_kvasir, header=None)).reshape(-1)[kva_num:]
            self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path_kvasir), header=None)).reshape(-1, num_primary_color*3)[kva_num:]
            self.masks_path =  np.array(pd.read_csv(csv_path_kvasir, header=None)).reshape(-1)[kva_num:]
            
            self.masks_path = np.array( [x.replace('images','masks')  for x in self.imgs_path] )

        self.num_primary_color = num_primary_color

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        mask = cv2.imread(self.masks_path[index],cv2.IMREAD_GRAYSCALE)
 
        target_size = 320
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2,0,1)) # h,w,c -> c,h,w
        target_img = img/255 # 0~1


        mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        mask = np.expand_dims(mask, 2)
        mask = mask.transpose((2,0,1)) # h,w,c -> c,h,w
        mask = mask/255 # 0~1

        # select primary_color
        primary_color_layers = self.make_primary_color_layers(self.palette_list[index], target_img)

        # to Tensor
        target_img = torch.from_numpy(target_img.astype(np.float32))
        primary_color_layers = torch.from_numpy(primary_color_layers.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))

        return target_img, primary_color_layers , mask   # return torch.Tensor

    def __len__(self):
        return len(self.imgs_path)


    def make_primary_color_layers(self, palette_values, target_img):
        '''
         入力：パレットの色の値 调色板颜色值
         出力：primary_color_layers #out: (bn, ln, 3, h, w)
        '''
        primary_color = palette_values.reshape(self.num_primary_color, 3) / 255 # (ln, 3)
        primary_color_layers = np.tile(np.ones_like(target_img), (self.num_primary_color,1,1,1)) * primary_color.reshape(self.num_primary_color,3,1,1)
        
        return primary_color_layers


class MyDatasetMulti(Dataset):
    def __init__(self, csv_path_ihc, num_primary_color, mode=None):

        ihc_num = 24000 # ihc 数据集设置数量

        if mode == 'train':
            
            self.imgs_path = np.array(pd.read_csv(csv_path_ihc, header=None)).reshape(-1)[:ihc_num] #csvリストの後ろをvaldataに
            self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path_ihc), header=None)).reshape(-1, num_primary_color*3)[:ihc_num]
            self.masks_path =  np.array(pd.read_csv(csv_path_ihc, header=None)).reshape(-1)[:ihc_num]

            self.masks_path = np.array( [x.replace('images_10pics_30k_patches','masks_10pics_30k_patches')  for x in self.imgs_path] )

        
        if mode == 'val':
            self.imgs_path = np.array(pd.read_csv(csv_path_ihc, header=None)).reshape(-1)[ihc_num:] #csvリストの後ろをvaldataに
            self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path_ihc), header=None)).reshape(-1, num_primary_color*3)[ihc_num:]
            self.masks_path =  np.array(pd.read_csv(csv_path_ihc, header=None)).reshape(-1)[ihc_num:]
            
            self.masks_path = np.array( [x.replace('images_10pics_30k_patches','masks_10pics_30k_patches')  for x in self.imgs_path] )


        self.num_primary_color = num_primary_color

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        mask = cv2.imread(self.masks_path[index],cv2.IMREAD_GRAYSCALE)
        
        target_size = 320
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2,0,1)) # h,w,c -> c,h,w
        target_img = img/255 # 0~1


        mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        mask = np.expand_dims(mask, 2)
        mask = mask.transpose((2,0,1)) # h,w,c -> c,h,w
        mask = mask/255 # 0~1

        # select primary_color
        primary_color_layers = self.make_primary_color_layers(self.palette_list[index], target_img)

        # to Tensor
        target_img = torch.from_numpy(target_img.astype(np.float32))
        primary_color_layers = torch.from_numpy(primary_color_layers.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))

        return target_img, primary_color_layers, mask # return torch.Tensor

    def __len__(self):
        return len(self.imgs_path)


    def make_primary_color_layers(self, palette_values, target_img):
        '''
         入力：パレットの色の値 调色板颜色值
         出力：primary_color_layers #out: (bn, ln, 3, h, w)
        '''
        primary_color = palette_values.reshape(self.num_primary_color, 3) / 255 # (ln, 3)
        primary_color_layers = np.tile(np.ones_like(target_img), (self.num_primary_color,1,1,1)) * primary_color.reshape(self.num_primary_color,3,1,1)
        
        return primary_color_layers


class MyDatasetIHC_365(Dataset):
    def __init__(self,csv_path_365 , csv_path_ihc, num_primary_color, mode=None):

        ihc_num = 24000 # ihc 数据集设置数量

        if mode == 'train':
            self.imgs_path_365 =  np.array(pd.read_csv(csv_path_365, header=None)).reshape(-1)[:ihc_num]
            self.palette_list_365 = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path_365), header=None)).reshape(-1, num_primary_color*3)[:ihc_num]
            self.imgs_path_365 = np.array( [x.replace('../','../FSCS/')  for x in self.imgs_path_365] )

            self.imgs_path = np.array(pd.read_csv(csv_path_ihc, header=None)).reshape(-1)[:ihc_num] #csvリストの後ろをvaldataに
            self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path_ihc), header=None)).reshape(-1, num_primary_color*3)[:ihc_num]
            self.masks_path =  np.array(pd.read_csv(csv_path_ihc, header=None)).reshape(-1)[:ihc_num]

            self.masks_path = np.array( [x.replace('images_10pics_30k_patches','masks_10pics_30k_patches')  for x in self.imgs_path] )

        
        if mode == 'val':
            self.imgs_path_365 =  np.array(pd.read_csv(csv_path_365, header=None)).reshape(-1)[ihc_num:]
            self.palette_list_365 = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path_365), header=None)).reshape(-1, num_primary_color*3)[ihc_num:]
            self.imgs_path_365 = np.array( [x.replace('../','../FSCS/')  for x in self.imgs_path_365] )

            self.imgs_path = np.array(pd.read_csv(csv_path_ihc, header=None)).reshape(-1)[ihc_num:] #csvリストの後ろをvaldataに
            self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path_ihc), header=None)).reshape(-1, num_primary_color*3)[ihc_num:]
            self.masks_path =  np.array(pd.read_csv(csv_path_ihc, header=None)).reshape(-1)[ihc_num:]
            
            self.masks_path = np.array( [x.replace('images_10pics_30k_patches','masks_10pics_30k_patches')  for x in self.imgs_path] )


        self.num_primary_color = num_primary_color

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        mask = cv2.imread(self.masks_path[index],cv2.IMREAD_GRAYSCALE)

        # cv2.imwrite('img.png',img)
        # cv2.imwrite('mask.png',mask)

        img_365 = cv2.imread(self.imgs_path_365[index])
        
        target_size = 320
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2,0,1)) # h,w,c -> c,h,w
        target_img = img/255 # 0~1


        mask = cv2.resize(mask, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        mask = np.expand_dims(mask, 2)
        mask = mask.transpose((2,0,1)) # h,w,c -> c,h,w
        mask = mask/255 # 0~1

        img_365 = cv2.resize(img_365, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        img_365 = cv2.cvtColor(img_365, cv2.COLOR_BGR2RGB)
        img_365 = img_365.transpose((2,0,1)) # h,w,c -> c,h,w
        img_365 = img_365/255 # 0~1

        # select primary_color
        primary_color_layers = self.make_primary_color_layers(self.palette_list[index], target_img)

        primary_color_layers_365 = self.make_primary_color_layers(self.palette_list_365[index],img_365)

        # to Tensor
        target_img = torch.from_numpy(target_img.astype(np.float32))
        primary_color_layers = torch.from_numpy(primary_color_layers.astype(np.float32))
        mask = torch.from_numpy(mask.astype(np.float32))
        img_365 = torch.from_numpy(img_365.astype(np.float32))
        primary_color_layers_365 = torch.from_numpy(primary_color_layers_365.astype(np.float32))

        # save_image(primary_color_layers,'primary_color_layers.png')

        return target_img, primary_color_layers, mask, img_365, primary_color_layers_365 # return torch.Tensor

    def __len__(self):
        return len(self.imgs_path)


    def make_primary_color_layers(self, palette_values, target_img):
        '''
         入力：パレットの色の値 调色板颜色值
         出力：primary_color_layers #out: (bn, ln, 3, h, w)
        '''
        primary_color = palette_values.reshape(self.num_primary_color, 3) / 255 # (ln, 3)
        primary_color_layers = np.tile(np.ones_like(target_img), (self.num_primary_color,1,1,1)) * primary_color.reshape(self.num_primary_color,3,1,1)
        
        return primary_color_layers


class MyDataset365_with_IHC(Dataset):
    def __init__(self,csv_path_365 , csv_path_ihc, num_primary_color, mode=None):

        ihc_num = 24000 # ihc 数据集设置数量

        if mode == 'train':
            self.imgs_path_365 =  np.array(pd.read_csv(csv_path_365, header=None)).reshape(-1)[:ihc_num]
            self.palette_list_365 = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path_365), header=None)).reshape(-1, num_primary_color*3)[:ihc_num]
            self.imgs_path_365 = np.array( [x.replace('../','../FSCS/')  for x in self.imgs_path_365] )

            # self.imgs_path = np.array(pd.read_csv(csv_path_ihc, header=None)).reshape(-1)[:ihc_num] #csvリストの後ろをvaldataに
            # self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path_ihc), header=None)).reshape(-1, num_primary_color*3)[:ihc_num]

            # self.imgs_path = np.concatenate((self.imgs_path,self.imgs_path_365),axis=0)  
            # self.palette_list = np.concatenate((self.palette_list,self.palette_list_365),axis=0)

            self.imgs_path = self.imgs_path_365
            self.palette_list = self.palette_list_365
        
        if mode == 'val':
            self.imgs_path = np.array(pd.read_csv(csv_path_ihc, header=None)).reshape(-1)[ihc_num :] #csvリストの後ろをvaldataに
            self.palette_list = np.array(pd.read_csv('palette_%d_%s' % (num_primary_color, csv_path_ihc), header=None)).reshape(-1, num_primary_color*3)[ihc_num:]


        self.num_primary_color = num_primary_color

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])

        target_size = 320
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2,0,1)) # h,w,c -> c,h,w
        target_img = img/255 # 0~1

        # select primary_color
        primary_color_layers = self.make_primary_color_layers(self.palette_list[index], target_img)

        # to Tensor
        target_img = torch.from_numpy(target_img.astype(np.float32))
        primary_color_layers = torch.from_numpy(primary_color_layers.astype(np.float32))

        return target_img, primary_color_layers  # return torch.Tensor

    def __len__(self):
        return len(self.imgs_path)


    def make_primary_color_layers(self, palette_values, target_img):
        '''
         入力：パレットの色の値 调色板颜色值
         出力：primary_color_layers #out: (bn, ln, 3, h, w)
        '''
        primary_color = palette_values.reshape(self.num_primary_color, 3) / 255 # (ln, 3)
        primary_color_layers = np.tile(np.ones_like(target_img), (self.num_primary_color,1,1,1)) * primary_color.reshape(self.num_primary_color,3,1,1)
        
        return primary_color_layers

