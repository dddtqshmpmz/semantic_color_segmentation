# from curses import COLOR_PAIRS
import os

import cv2
import numpy as np
import torch
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir,  # str(i),
                                                img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}


class DatasetSemanColor(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir,color_img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.color_img_dir = color_img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img =cv2.resize( cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext)) ,(320,320) )
        h,w = img.shape[:2]

        color_img = []
        for j in range (0,7):
            
            color_img.append( cv2.resize( cv2.imread(os.path.join(self.color_img_dir,img_id+'_layer-%02d.png' % (j)  )
                , -1 ) ,(h,w)) [...,None] )
        color_img = np.dstack(color_img)
        color_img = np.squeeze(color_img,axis=3)
        img = np.concatenate((img,color_img),axis=2)

        mask = []
        for i in range(self.num_classes):
            mask.append( cv2.resize( cv2.imread(os.path.join(self.mask_dir,  # str(i),
                                                img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE),(h,w))[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']


        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)

        return img, mask, {'img_id': img_id}


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, img_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            X mask_dir: Mask file directory. 
            img_ext (str): Image file extension.
            X mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.img_ext = img_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        if self.transform is not None:
            augmented = self.transform(image=img)
            img = augmented['image']

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)

        return img, {'img_id': img_id}


class DatasetSemanColor2Decoder(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir,color_img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.color_img_dir = color_img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img =cv2.resize( cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext)) ,(320,320) )
        h,w = img.shape[:2]

        color_img = []
        for j in range (0,7):
            
            color_img.append( cv2.resize( cv2.imread(os.path.join(self.color_img_dir,img_id+'_layer-%02d.png' % (j)  )
                , -1 ) ,(h,w)) [...,None] )
        color_img = np.dstack(color_img)
        color_img = np.squeeze(color_img,axis=3)
        img = np.concatenate((img,color_img),axis=2)

        mask = []
        for i in range(self.num_classes):
            mask.append( cv2.resize( cv2.imread(os.path.join(self.mask_dir,  # str(i),
                                                img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE),(h,w))[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']


        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        color_img = color_img.astype('float32') / 255
        color_img = color_img.transpose(2,0,1)

        return img, color_img, mask, {'img_id': img_id}