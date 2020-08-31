import torch
import numpy as np
import cv2
import os
import h5py
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from utils import *

class MegVSR_Dataset(Dataset):
    def __init__(self, root_dir, crop_size=32, crop_per_image=4, mode='train'):
        super().__init__()
        self.root_dir = root_dir
        self.crop_per_image = crop_per_image
        self.crop_size = 32
        self.mode = mode
        self.initialization()
    
    def initialization(self):
        self.sub_dir = 'train_png' if self.mode == 'train' else 'test_png'
        self.data_dir = os.path.join(self.root_dir, self.sub_dir)
        self.video_id = 0
        self.lr_dirs = []
        self.hr_dirs = []
        if self.mode == 'train':
            # Dir
            for dirname in os.listdir(self.data_dir):
                if 'down4x' in dirname:
                    self.lr_dirs.append(dirname)
                else:
                    self.hr_dirs.append(dirname)
            self.num_of_videos = len(self.lr_dirs)
            # file path
            self.frame_paths = []
            for lr_dirname in tqdm(self.lr_dirs):
                video_id = int(lr_dirname[:2])
                lr_frames = []
                for filename in os.listdir(os.path.join(self.data_dir, lr_dirname)):
                    lr_path = os.path.join(self.data_dir, lr_dirname, filename)
                    lr_frames.append(lr_path)
                hr_frames = []
                hr_dirname = lr_dirname[:6] + '_frames'
                for filename in os.listdir(os.path.join(self.data_dir, hr_dirname)):
                    hr_path = os.path.join(self.data_dir, hr_dirname, filename)
                    hr_frames.append(hr_path)
                video_frames = {
                    'id': video_id,
                    'len': len(lr_frames),
                    'lr_frames': lr_frames,
                    'hr_frames': hr_frames,
                }
                self.frame_paths.append(video_frames)
            # 初始化完毕
            return True
        else:
            # 验证集只需要返回lr
            pass

    def __len__(self):
        return self.frame_paths[self.video_id]['len']
    
    def __getitem__(self, idx):
        data = {}
        video_frame = self.frame_paths[self.video_id]
        video_id = video_frame['id']
        lr_img = cv2.imread(video_frame['lr_frames'][idx])[:,:,::-1]
        hr_img = cv2.imread(video_frame['hr_frames'][idx])[:,:,::-1]
        if self.mode == 'train':
            lr_crops, hr_crops = random_crop(lr_img, hr_img, aug='SISR',
                    crop_size=self.crop_size, crop_per_image=self.crop_per_image)
        else:
            lr_crops = np.expand_dims(lr_img, 0)
            hr_crops = np.expand_dims(hr_img, 0)

        lr_crops = lr_crops.transpose(0,3,1,2).astype(np.float32) / 255.
        hr_crops = hr_crops.transpose(0,3,1,2).astype(np.float32) / 255.

        data['frame_id'] = '%03d' % idx
        data['video_id'] = '%02d' % video_id
        data['lr'] = np.ascontiguousarray(lr_crops)
        data['hr'] = np.ascontiguousarray(hr_crops)

        return data


class TestDataset(Dataset):
    def __init__(self, root_dir, ratio=1, sub_dir='eval'):
        super().__init__()
        self.root_dir = root_dir
        self.ratio = ratio
        self.sub_dir = sub_dir
        self.initialization()
    
    def initialization(self):
        self.data_dir = os.path.join(self.root_dir, self.sub_dir)
        self.dataname = os.listdir(self.data_dir)
        self.datapath = [os.path.join(self.data_dir, name) for name in 
                        self.dataname if 'dng' in name.lower()]
        self.raw_data = []

    def __len__(self):
        return len(self.dataname)
    
    def __getitem__(self, idx):
        data = {}
        # gt_img = process.postprocess_bayer(data['rawpath'], data['data'])
        # plt.imshow(gt_img)
        # plt.show()

        return data


def MegVSR_DataLoader(DataLoader):
    pass


def random_crop(lr_img, hr_img, crop_size=32, crop_per_image=8, aug=None):
    # 本函数用于将numpy随机裁剪成以crop_size为边长的方形crop_per_image等份
    is_tensor = torch.is_tensor(lr_img)
    device = 'cpu'
    dtype = lr_img.dtype
    if is_tensor:
        device = lr_img.device
        if device != 'cpu':
            lr_img = lr_img.cpu()
            hr_img = hr_img.cpu()
        lr_img = lr_img.numpy()
        hr_img = hr_img.numpy()

    h, w, c = lr_img.shape
    # 创建空numpy做画布
    lr_crops = np.zeros((crop_per_image, crop_size, crop_size, c))
    hr_crops = np.zeros((crop_per_image, crop_size*4, crop_size*4, c))

    # 往空tensor的通道上贴patchs
    for i in range(crop_per_image):
        h_start = np.random.randint(0, h - crop_size)
        w_start = np.random.randint(0, w - crop_size)
        h_end = h_start + crop_size
        w_end = w_start + crop_size 

        lr_crop = lr_img[h_start:h_end, w_start:w_end, :]
        hr_crop = hr_img[h_start*4:h_end*4, w_start*4:w_end*4, :]

        if aug is not None:
            mode = np.random.randint(8) if aug == 'SISR' else aug
            lr_crop = data_aug(lr_crop, mode)
            hr_crop = data_aug(hr_crop, mode)

        lr_crops[i:i+1, ...] = lr_crop
        hr_crops[i:i+1, ...] = hr_crop

    if is_tensor:
        lr_crops = torch.from_numpy(lr_crops).to(device).type(dtype)
        hr_crops = torch.from_numpy(hr_crops).to(device).type(dtype)

    return lr_crops, hr_crops


def data_aug(image, mode):
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate counterwise 90 degree
        return np.rot90(image)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        # rotate 180 degree
        return np.rot90(image, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        # rotate 270 degree
        return np.rot90(image, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        image = np.rot90(image, k=3)
        return np.flipud(image)


if __name__=='__main__':
    crop_per_image = 4
    dst = MegVSR_Dataset('F:/datasets/MegVSR', crop_per_image=crop_per_image)
    dataloader_train = DataLoader(dst, batch_size=8, shuffle=True, num_workers=0)
    for i in range(10):
        dst.video_id = i
        for k, data in enumerate(dataloader_train):
            # 由于crops的存在，Dataloader会把数据变成5维，需要view回4维
            imgs_lr = tensor_dim5to4(data['lr'])
            imgs_hr = tensor_dim5to4(data['hr'])
            print(data['video_id'], data['frame_id'])
            fig = plt.figure(figsize=(16,10))
            ax = [None]*2*crop_per_image
            for i in range(crop_per_image):
                ax[i*2] = fig.add_subplot(2, crop_per_image, i*2+1)
                ax[i*2+1] = fig.add_subplot(2, crop_per_image, i*2+2)
                input_out = np.uint8(imgs_lr[i].numpy().transpose(1,2,0)*255)
                gt_out = np.uint8(imgs_hr[i].numpy().transpose(1,2,0)*255)
                ax[i*2].imshow(input_out)
                ax[i*2+1].imshow(gt_out)
            plt.show()
            plt.close()
            print(len(dataloader_train),len(dst))
            break