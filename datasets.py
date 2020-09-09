import numpy as np
import cv2
import os
import h5py
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils import *

class MegVSR_Dataset(Dataset):
    def __init__(self, root_dir, crop_size=32, crop_per_image=4, 
                nflames=3, cv2_INTER=True, mode='train', shuffle=True):
        super().__init__()
        self.buffer = []
        self.nflames = nflames
        self.root_dir = root_dir
        self.crop_per_image = crop_per_image
        self.crop_size = crop_size
        self.cv2_INTER = cv2_INTER
        self.mode = mode
        self.length = 0
        self.shuffle = shuffle
        self.initialization()
    
    def initialization(self):
        self.sub_dir = 'train_png' if self.mode == 'train' else 'eval_png'
        if self.sub_dir == 'eval_png' and self.nflames > 1:
            self.sub_dir = 'eval_video'
        self.data_dir = os.path.join(self.root_dir, self.sub_dir)
        self.video_id = 0
        self.lr_dirs = []
        self.hr_dirs = []
        # Read Dirs
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
            lr_frames.sort()
            hr_frames.sort()
            video_frames = {
                'id': video_id,
                'len': len(lr_frames),
                'lr_frames': lr_frames,
                'hr_frames': hr_frames,
            }
            self.frame_paths.append(video_frames)
        # 初始化完毕
        self.next_video(0)
        return True

    def __len__(self):
        self.length = self.frame_paths[self.video_id]['len']
        return self.length
    
    def next_video(self, idx):
        self.video_id = idx
        del self.buffer
        self.buffer = [None]*self.__len__() if self.shuffle else []
        # crop corner
        self.get_flame_shape()
        self.init_random_crop_point()
    
    def init_random_crop_point(self):
        self.h_start = []
        self.w_start = []
        self.h_end = []
        self.w_end = []
        self.aug = []
        for i in range(self.crop_per_image):
            h_start = np.random.randint(0, self.h - self.crop_size)
            w_start = np.random.randint(0, self.w - self.crop_size)
            self.h_start.append(h_start)
            self.w_start.append(w_start)
            self.h_end.append(h_start + self.crop_size)
            self.w_end.append(w_start + self.crop_size)
            self.aug.append(np.random.randint(8))

    def get_flame_shape(self):
        video_frame = self.frame_paths[self.video_id]
        lr_img = cv2.imread(video_frame['lr_frames'][0])[:,:,::-1]
        self.h, self.w, self.c = lr_img.shape
    
    # 按照现有固定的crop_point裁剪video中指定位置的数据
    def video_crop(self, lr_img, hr_img):
        # 本函数用于将numpy随机裁剪成以crop_size为边长的方形crop_per_image等份
        if use_mge:
            is_tensor = False
        else:
            is_tensor = torch.is_tensor(lr_img)

        h, w, c = lr_img.shape
        # 创建空numpy做画布
        lr_crops = np.zeros((self.crop_per_image, self.crop_size, self.crop_size, c))
        hr_crops = np.zeros((self.crop_per_image, self.crop_size*4, self.crop_size*4, c))

        # 往空tensor的通道上贴patchs
        for i in range(self.crop_per_image):
            lr_crop = lr_img[self.h_start[i]:self.h_end[i], self.w_start[i]:self.w_end[i], :]
            hr_crop = hr_img[self.h_start[i]*4:self.h_end[i]*4, self.w_start[i]*4:self.w_end[i]*4, :]

            lr_crop = data_aug(lr_crop, self.aug[i])
            hr_crop = data_aug(hr_crop, self.aug[i])

            lr_crops[i:i+1, ...] = lr_crop
            hr_crops[i:i+1, ...] = hr_crop

        return lr_crops, hr_crops
    
    # 使用单帧的方法+hash获取多帧的数据
    def multi_frame_crop(self, idx):
        self.init_random_crop_point()
        r = self.nflames // 2
        temp_buffer = []
        for i in range(idx-r, idx+r+1):
            id = min(max(i, 0), self.length-1)
            if self.buffer[id] is None:
                self.buffer[id] = self.getitem(id)
            temp_buffer.append(self.getitem(id))
        return self.buffer_stack_on_channels(temp_buffer)

    # 将nframe帧的buffer中的数据叠成一个tensor形状ndarray
    def buffer_stack_on_channels(self, buffer=None):
        data = {}
        r = self.nflames // 2
        data['frame_id'] = buffer[r]['frame_id']
        data['video_id'] = buffer[r]['video_id']
        b, c, h, w = buffer[r]['lr'].shape
        data['lr'] = np.zeros((b, c*self.nflames, h, w))
        data['hr'] = np.zeros((b, c*self.nflames, h*4, w*4))
        if self.cv2_INTER:
            data['bc'] = np.zeros((b, c*self.nflames, h*4, w*4))

        for i, frame in enumerate(buffer):
            data['lr'][:, c*i:c*(i+1), :, :] = frame['lr']
            data['hr'][:, c*i:c*(i+1), :, :] = frame['hr']
            if self.cv2_INTER:
                data['bc'][:, c*i:c*(i+1), :, :] = frame['bc']
        
        return data

    # 获取单帧的一组数据
    def getitem(self, idx):
        data = {}
        video_frame = self.frame_paths[self.video_id]
        video_id = video_frame['id']
        lr_img = cv2.imread(video_frame['lr_frames'][idx])[:,:,::-1]
        hr_img = cv2.imread(video_frame['hr_frames'][idx])[:,:,::-1]
        if self.mode == 'train':
            if self.nflames > 1:
                lr_crops, hr_crops = self.video_crop(lr_img, hr_img)
            else:
                aug = 'SISR' if self.nflames == 1 else None
                lr_crops, hr_crops = random_crop(lr_img, hr_img, aug=aug,
                        crop_size=self.crop_size, crop_per_image=self.crop_per_image)
        else:
            lr_crops = np.expand_dims(lr_img, 0)
            hr_crops = np.expand_dims(hr_img, 0)

        if self.cv2_INTER:
            bc_crops = np.zeros_like(hr_crops)
            for i in range(lr_crops.shape[0]):
                bc_crops[i, ...] = cv2.resize(lr_crops[i], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            bc_crops = np.clip(bc_crops, 0, 255)

        lr_crops = lr_crops.transpose(0,3,1,2).astype(np.float32) / 255.
        hr_crops = hr_crops.transpose(0,3,1,2).astype(np.float32) / 255.
        if self.cv2_INTER:
            bc_crops = bc_crops.transpose(0,3,1,2).astype(np.float32) / 255.

        data['frame_id'] = '%04d' % idx
        data['video_id'] = '%02d' % video_id
        data['lr'] = np.ascontiguousarray(lr_crops)
        data['hr'] = np.ascontiguousarray(hr_crops)
        if self.cv2_INTER:
            data['bc'] = np.ascontiguousarray(bc_crops)

        return data

    def liner_buffer(self, idx):
        r = self.nflames // 2
        data = None
        # 首位置，前相邻帧为本帧
        if idx == 0:
            data = self.getitem(idx)
            self.buffer = [data] * (r + 1)
            for i in range(1, r):
                data = self.getitem(idx+i)
                self.buffer.append(data)
        else:
            # 其余位置删除最前帧，向末尾添加下一帧
            del self.buffer[0]

        # 最后一帧前，末尾都是下一帧
        if idx != self.frame_paths[self.video_id]['len'] - r:
            data = self.getitem(idx+r)
        else:
            # 最后一帧时，末尾为本帧（上一帧的下一帧）
            data = self.buffer[-1]

        self.buffer.append(data)

        return self.buffer_stack_on_channels(self.buffer)

    def __getitem__(self, idx):
        if self.nflames > 1:
            if self.shuffle:
                return self.multi_frame_crop(idx)
            else:
                return self.liner_buffer(idx)
        else:
            return self.getitem(idx)


class MegVSR_Test_Dataset(Dataset):
    def __init__(self, root_dir, nflames=3, cv2_INTER=True, shuffle=True):
        super().__init__()
        self.root_dir = root_dir
        self.nflames = nflames
        self.shuffle = shuffle
        self.cv2_INTER = cv2_INTER
        self.initialization()
    
    def initialization(self):
        self.sub_dir = 'test_png'
        self.data_dir = os.path.join(self.root_dir, self.sub_dir)
        self.video_id = 0
        self.lr_dirs = []
        # Read Dirs
        for dirname in os.listdir(self.data_dir):
            self.lr_dirs.append(dirname)
        self.num_of_videos = len(self.lr_dirs)
        self.lr_dirs.sort()
        # file path
        self.frame_paths = []
        for lr_dirname in tqdm(self.lr_dirs):
            video_id = int(lr_dirname[:2])
            lr_frames = []
            for filename in os.listdir(os.path.join(self.data_dir, lr_dirname)):
                lr_path = os.path.join(self.data_dir, lr_dirname, filename)
                lr_frames.append(lr_path)
            lr_frames.sort()
            video_frames = {
                'id': video_id,
                'len': len(lr_frames),
                'lr_frames': lr_frames,
            }
            self.frame_paths.append(video_frames)
        # 初始化完毕
        return True

    def __len__(self):
        return self.frame_paths[self.video_id-90]['len']
    
    def next_video(self, idx):
        self.video_id = idx
        del self.buffer
        self.buffer = [None]*self.__len__() if self.shuffle else []
    
    # 使用单帧的方法+hash获取多帧的数据
    def multi_frame_crop(self, idx):
        self.init_random_crop_point()
        r = self.nflames // 2
        temp_buffer = []
        for i in range(idx-r, idx+r+1):
            id = min(max(i, 0), self.__len__()-1)
            if self.buffer[id] is None:
                self.buffer[id] = self.getitem(id)
            temp_buffer.append(self.getitem(id))
        return self.buffer_stack_on_channels(temp_buffer)

    # 将nframe帧的buffer中的数据叠成一个tensor形状ndarray
    def buffer_stack_on_channels(self, buffer=None):
        data = {}
        r = self.nflames // 2
        data['frame_id'] = buffer[r]['frame_id']
        data['video_id'] = buffer[r]['video_id']
        b, c, h, w = buffer[r]['lr'].shape
        data['lr'] = np.zeros((b, c*self.nflames, h, w))
        if self.cv2_INTER:
            data['bc'] = np.zeros((b, c*self.nflames, h*4, w*4))

        for i, frame in enumerate(buffer):
            data['lr'][:, c*i:c*(i+1), :, :] = frame['lr']
            if self.cv2_INTER:
                data['bc'][:, c*i:c*(i+1), :, :] = frame['bc']
        
        return data
    
    def getitem(self, idx):
        data = {}
        video_frame = self.frame_paths[self.video_id-90]
        video_id = video_frame['id']
        lr_img = cv2.imread(video_frame['lr_frames'][idx])[:,:,::-1]
        lr_crops = np.expand_dims(lr_img, 0)
        if self.cv2_INTER:
            b, h, w, c = lr_crops.shape
            bc_crops = np.zeros((b, h*4, w*4, c))
            for i in range(b):
                bc_crops[i, ...] = cv2.resize(lr_crops[i], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            bc_crops = np.clip(bc_crops, 0, 255)

        lr_crops = scale_down(lr_crops.transpose(0,3,1,2))
        data['frame_id'] = '%04d' % idx
        data['video_id'] = '%02d' % video_id
        data['lr'] = np.ascontiguousarray(lr_crops)

        if self.cv2_INTER:
            bc_crops = scale_down(bc_crops.transpose(0,3,1,2))
            data['bc'] = np.ascontiguousarray(bc_crops)

        return data

    def liner_buffer(self, idx):
        r = self.nflames // 2
        data = None
        # 首位置，前相邻帧为本帧
        if idx == 0:
            data = self.getitem(idx)
            self.buffer = [data] * (r + 1)
            for i in range(1, r):
                data = self.getitem(idx+i)
                self.buffer.append(data)
        else:
            # 其余位置删除最前帧，向末尾添加下一帧
            del self.buffer[0]

        # 最后一帧前，末尾都是下一帧
        if idx != self.frame_paths[self.video_id]['len'] - r:
            data = self.getitem(idx+r)
        else:
            # 最后一帧时，末尾为本帧（上一帧的下一帧）
            data = self.buffer[-1]

        self.buffer.append(data)

        return self.buffer_stack_on_channels(self.buffer)

    def __getitem__(self, idx):
        if self.nflames > 1:
            if self.shuffle:
                return self.multi_frame_crop(idx)
            else:
                return self.liner_buffer(idx)
        else:
            return self.getitem(idx)


def random_crop(lr_img, hr_img, crop_size=32, crop_per_image=8, aug=None):
    # 本函数用于将numpy随机裁剪成以crop_size为边长的方形crop_per_image等份
    if use_mge:
        is_tensor = False
    else:
        is_tensor = torch.is_tensor(lr_img)
    if is_tensor:
        dtype = lr_img.dtype
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
    nflames = 3
    dst = MegVSR_Dataset('F:/datasets/MegVSR', crop_per_image=crop_per_image, 
                        mode='train',crop_size=100, nflames=nflames)
    # dst = MegVSR_Test_Dataset('F:/datasets/MegVSR/', crop_per_image=crop_per_image, 
                        # mode='train',crop_size=100, nflames=nflames)
    dataloader_train = DataLoader(dst, batch_size=1, shuffle=False, num_workers=2)
    for i in range(11,20):
        dst.next_video(i)
        for k, data in enumerate(dataloader_train):
            # 由于crops的存在，Dataloader会把数据变成5维，需要view回4维
            imgs_lr = tensor_dim5to4(data['bc'])
            imgs_hr = tensor_dim5to4(data['hr'])
            print(data['video_id'], data['frame_id'])
            fig = plt.figure(figsize=(20,10))
            ax = [None]*2*nflames
            input_out = np.uint8(imgs_lr[0].numpy().transpose(1,2,0)*255)
            gt_out = np.uint8(imgs_hr[0].numpy().transpose(1,2,0)*255)
            for i in range(nflames):
                ax[i*2] = fig.add_subplot(2, nflames, i+1)
                ax[i*2+1] = fig.add_subplot(2, nflames, i+nflames+1)
                ax[i*2].imshow(input_out[:,:,3*i:3*i+3])
                ax[i*2+1].imshow(gt_out[:,:,3*i:3*i+3])
            plt.show()
            plt.close()
            print(len(dataloader_train),len(dst))
            if k>5: break