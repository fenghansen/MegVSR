import numpy as np
import cv2
import os
import h5py
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils import *

class Global_Buffer(Dataset):
    def __init__(self, pool_size=15, index_range=1000, optflow=False):
        super().__init__()
        self.pool_size = pool_size
        self.index_range = index_range
        self.buffer = [None] * index_range
        self.video_frame_paths = None
        self.optflow = optflow
        self.start = 0
        self.end = 0
        self.check = [False] * index_range
    
    def __len__(self):
        return len(self.buffer)
    
    def video_init(self, video_frame_paths):
        self.video_frame_paths = video_frame_paths
        self.buffer = [None] * self.index_range
        self.check = [False] * self.index_range
        self.start = 0
        self.end = 0
    
    def update(self, idx):
        self.buffer[idx] = {'lr': cv2.imread(self.video_frame_paths['lr_frames'][idx])[:,:,::-1]}
        if 'hr_frames' in self.video_frame_paths:
            self.buffer[idx]['hr'] = cv2.imread(self.video_frame_paths['hr_frames'][idx])[:,:,::-1]
        if self.optflow:
            self.updateOptflow(idx)
        self.end = idx+1
        while self.end - self.start > self.pool_size:
            self.buffer[self.start] = None
            self.start += 1
    
    def updateOptflow(self, idx):
        if idx == 0:
            h, w, c = self.buffer[idx]['lr'].shape
            self.buffer[idx]['flow'] = np.zeros((h,w,2), dtype=np.float32)
            self.check[idx] = True
        else:
            # if self.buffer[idx-1]['lr'] is None:
            #     h, w, c = self.buffer[idx]['lr'].shape
            #     self.buffer[idx]['flow'] = np.zeros((h,w,2), dtype=np.float32)
            # else:
            try:
                self.buffer[idx]['flow'] = calOptflow(self.buffer[idx-1]['lr'], self.buffer[idx]['lr'])
                self.check[idx] = True
            except:
                h, w, c = self.buffer[idx]['lr'].shape
                self.buffer[idx]['flow'] = np.zeros((h,w,2), dtype=np.float32)
    
    def __getitem__(self, idx):
        if idx < self.end:
            if self.check[idx] is False:
                self.updateOptflow(idx)
            return self.buffer[idx]
        else:
            self.update(idx)
            return self.buffer[idx]

class MegVSR_Dataset(Dataset):
    def __init__(self, root_dir, crop_size=32, crop_per_image=4, global_buffer=None, 
                nframes=3, cv2_INTER=True, mode='train', shuffle=True, optflow=False):
        super().__init__()
        self.buffer = []
        self.global_buffer = global_buffer
        self.nframes = nframes
        self.root_dir = root_dir
        self.crop_per_image = crop_per_image
        self.crop_size = crop_size
        self.cv2_INTER = cv2_INTER
        self.optflow = optflow
        self.mode = mode
        self.length = 0
        self.shuffle = shuffle
        self.initialization()
    
    def initialization(self):
        self.sub_dir = 'train_png' if self.mode == 'train' else 'eval_png'
        if self.sub_dir == 'eval_png' :#and self.nframes > 1:
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
        self.global_buffer.video_init(self.frame_paths[idx])
        self.buffer = []
        # crop corner
        self.get_frame_shape()
        self.init_random_crop_point()
        gc.collect()
    
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

    def get_frame_shape(self):
        video_frame = self.frame_paths[self.video_id]
        lr_img = cv2.imread(video_frame['lr_frames'][0])[:,:,::-1]
        self.h, self.w, self.c = lr_img.shape
    
    # 按照现有固定的crop_point裁剪video中指定位置的数据
    def video_crop(self, lr_img, hr_img, flow=None):
        # 本函数用于将numpy随机裁剪成以crop_size为边长的方形crop_per_image等份
        data = {}
        h, w, c = lr_img.shape
        # 创建空numpy做画布
        lr_crops = np.zeros((self.crop_per_image, self.crop_size, self.crop_size, c), dtype=np.float32)
        hr_crops = np.zeros((self.crop_per_image, self.crop_size*4, self.crop_size*4, c), dtype=np.float32)
        if self.optflow:
            flow_crops = np.zeros((self.crop_per_image, self.crop_size, self.crop_size, 2), dtype=np.float32)

        # 往空tensor的通道上贴patchs
        for i in range(self.crop_per_image):
            lr_crop = lr_img[self.h_start[i]:self.h_end[i], self.w_start[i]:self.w_end[i], :]
            hr_crop = hr_img[self.h_start[i]*4:self.h_end[i]*4, self.w_start[i]*4:self.w_end[i]*4, :]

            lr_crop = data_aug(lr_crop, self.aug[i])
            hr_crop = data_aug(hr_crop, self.aug[i])

            lr_crops[i, ...] = lr_crop
            hr_crops[i, ...] = hr_crop
            if self.optflow:
                flow_crop = flow[self.h_start[i]:self.h_end[i], self.w_start[i]:self.w_end[i], :]
                flow_crop = data_aug(flow_crop, self.aug[i])
                flow_crops[i, ...] = flow_crop

        data['lr'] = lr_crops
        data['hr'] = hr_crops
        if self.optflow:
            data['flow'] = flow_crops
        
        return data
    
    # 使用单帧的方法+hash获取多帧的数据
    def multiworkers_buffer(self, idx):
        self.init_random_crop_point()
        r = self.nframes // 2
        temp_buffer = []
        for i in range(idx-r, idx+r+1):
            id = min(max(i, 0), self.length-1)
            temp_buffer.append(self.getitem(id))
        # if self.optflow:
        #     b,c,h,w = temp_buffer[0]['lr'].shape
        #     for i in range(self.nframes):
        #         temp_buffer[i]['flow'] = np.zeros((b,2,h,w), dtype=np.float32)
        #     for k in range(self.crop_per_image):
        #         prvs_crop = temp_buffer[0]['lr'][k].transpose(1,2,0)
        #         for i in range(1, self.nframes):
        #             next_crop = temp_buffer[i]['lr'][k].transpose(1,2,0)
        #             temp_buffer[i]['flow'][k] = calOptflow(prvs_crop, next_crop).transpose(2,0,1)
        #             prvs_crop = next_crop

        return self.buffer_stack_on_channels(temp_buffer)

    # 将nframe帧的buffer中的数据叠成一个tensor形状ndarray
    def buffer_stack_on_channels(self, buffer=None):
        data = {}
        cf = self.nframes // 2
        data['frame_id'] = buffer[cf]['frame_id']
        data['video_id'] = buffer[cf]['video_id']
        b, h, w, c = buffer[cf]['lr'].shape
        data['lr'] = np.zeros((b, h, w, c*self.nframes), dtype=np.float32)
        data['hr'] = np.zeros((b, h*4, w*4, c*self.nframes), dtype=np.float32)
        if self.cv2_INTER:
            data['bc'] = np.zeros((b, h*4, w*4, c*self.nframes), dtype=np.float32)

        for i, frame in enumerate(buffer):
            data['lr'][:, :, :, c*i:c*(i+1)] = frame['lr']
            data['hr'][:, :, :, c*i:c*(i+1)] = frame['hr']
            if self.cv2_INTER:
                data['bc'][:, :, :, c*i:c*(i+1)] = frame['bc']
        
        if self.optflow:
            data['flow'] = np.zeros((b, h, w, 2*self.nframes), dtype=np.float32)
            data['flow'][:,:,:,2*(cf-1):2*(cf-0)] = buffer[cf]['flow']
            data['flow'][:,:,:,2*(cf+1):2*(cf+2)] = -buffer[cf+1]['flow']

            if self.nframes == 5:
                data['flow'][:,:,:,2*(cf-2):2*(cf-1)] = buffer[cf-1]['flow'] + buffer[cf]['flow']
                data['flow'][:,:,:,2*(cf+2):2*(cf+3)] = -buffer[cf+1]['flow'] - buffer[cf+2]['flow']
        
            for crop_id in range(b):
                for i in range(self.nframes):
                    if i == cf: continue
                    image = data['lr'][crop_id,:,:,c*i:c*(i+1)]
                    flow = data['flow'][crop_id,:,:,2*i:2*(i+1)]
                    shift = FlowShift(image, flow)
                    data['lr'][crop_id,:,:,c*i:c*(i+1)] = shift
                    # show test
                #     bgr, ratio = visualize(flow, name=f"Frame{i-2}", show=False)
                #     final = bgr*0.5 + shift*127.5
                #     cv2.imshow(f"Frame{i-2}", final[:,:,::-1].astype(np.uint8))
                #     # cv2.imshow(f"ori{i-2}", shift[:,:,::-1]-image[:,:,::-1])
                # cv2.waitKey(30)
        
        data['lr'] = np.ascontiguousarray(data['lr'].transpose(0,3,1,2))
        data['hr'] = np.ascontiguousarray(data['hr'].transpose(0,3,1,2))
        if self.cv2_INTER:
            data['bc'] = np.ascontiguousarray(data['bc'].transpose(0,3,1,2))
        if self.optflow:
            data['flow'] = np.ascontiguousarray(data['flow'].transpose(0,3,1,2))

        return data

    # 获取单帧的一组数据
    def getitem(self, idx):
        data = {}
        video_frame = self.frame_paths[self.video_id]
        video_id = video_frame['id']
        if self.global_buffer is None:
            lr_img = cv2.imread(video_frame['lr_frames'][idx])[:,:,::-1]
            hr_img = cv2.imread(video_frame['hr_frames'][idx])[:,:,::-1]
        else:
            lr_img = self.global_buffer[idx]['lr']
            hr_img = self.global_buffer[idx]['hr']
            if self.optflow:
                flow = self.global_buffer[idx]['flow']
            
        if self.mode == 'train':
            if self.nframes > 1:
                crops = self.video_crop(lr_img, hr_img, flow=flow)
                lr_crops = crops['lr']
                hr_crops = crops['hr']
                flow_crops = crops['flow']
            else:
                aug = 'SISR' if self.nframes == 1 else np.random.randint(8)
                lr_crops, hr_crops = random_crop(lr_img, hr_img, aug=aug,
                        crop_size=self.crop_size, crop_per_image=self.crop_per_image)
        else:
            lr_crops = np.expand_dims(lr_img, 0)
            hr_crops = np.expand_dims(hr_img, 0)
            if self.optflow:
                flow_crops = np.expand_dims(flow, 0)

        if self.cv2_INTER:
            bc_crops = np.zeros_like(hr_crops)
            for i in range(lr_crops.shape[0]):
                bc_crops[i, ...] = cv2.resize(lr_crops[i], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            bc_crops = np.clip(bc_crops, 0, 255)

        lr_crops = lr_crops.astype(np.float32) / 255.
        hr_crops = hr_crops.astype(np.float32) / 255.
        if self.cv2_INTER:
            bc_crops = bc_crops.astype(np.float32) / 255.
            data['bc'] = bc_crops
        if self.optflow:
            flow_crops = flow_crops.astype(np.float32)
            data['flow'] = flow_crops

        data['frame_id'] = '%04d' % idx
        data['video_id'] = '%02d' % video_id
        data['lr'] = lr_crops
        data['hr'] = hr_crops

        return data

    def liner_buffer(self, idx):
        r = self.nframes // 2
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
        if idx < self.frame_paths[self.video_id]['len'] - r:
            data = self.getitem(idx+r)
        else:
            # 最后一帧时，末尾为本帧（上一帧的下一帧）
            data = self.buffer[-1]

        self.buffer.append(data)

        return self.buffer_stack_on_channels(self.buffer)
    
    def bulid_h5datasets(self):
        compute_optflow(self.root_dir)

    def __getitem__(self, idx):
        if self.nframes > 1:
            if self.shuffle:
                return self.multiworkers_buffer(idx)
            else:
                return self.liner_buffer(idx)
        else:
            return self.getitem(idx)


class MegVSR_Test_Dataset(Dataset):
    def __init__(self, root_dir, nframes=3, cv2_INTER=True, shuffle=False,
                global_buffer=None, optflow=False):
        super().__init__()
        self.root_dir = root_dir
        self.buffer = []
        self.global_buffer = global_buffer
        self.optflow = optflow
        self.nframes = nframes
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
        return self.frame_paths[self.video_id]['len']
    
    def next_video(self, idx):
        self.video_id = idx
        del self.buffer
        self.buffer = []
        self.global_buffer.video_init(self.frame_paths[idx])
    
    # 使用单帧的方法+hash获取多帧的数据
    def multiworkers_buffer(self, idx):
        r = self.nframes // 2
        temp_buffer = []
        for i in range(idx-r, idx+r+1):
            id = min(max(i, 0), self.__len__()-1)
            temp_buffer.append(self.getitem(id))
        return self.buffer_stack_on_channels(temp_buffer)

    # 将nframe帧的buffer中的数据叠成一个tensor形状ndarray
    def buffer_stack_on_channels(self, buffer=None):
        data = {}
        cf = self.nframes // 2
        data['frame_id'] = buffer[cf]['frame_id']
        data['video_id'] = buffer[cf]['video_id']
        b, h, w, c = buffer[cf]['lr'].shape
        data['lr'] = np.zeros((b, h, w, c*self.nframes), dtype=np.float32)
        if self.cv2_INTER:
            data['bc'] = np.zeros((b, h*4, w*4, c*self.nframes), dtype=np.float32)

        for i, frame in enumerate(buffer):
            data['lr'][:, :, :, c*i:c*(i+1)] = frame['lr']
            if self.cv2_INTER:
                data['bc'][:, :, :, c*i:c*(i+1)] = frame['bc']
        
        if self.optflow:
            data['flow'] = np.zeros((b, h, w, 2*self.nframes), dtype=np.float32)
            data['flow'][:,:,:,2*(cf-1):2*(cf-0)] = buffer[cf]['flow']
            data['flow'][:,:,:,2*(cf+1):2*(cf+2)] = -buffer[cf+1]['flow']

            if self.nframes == 5:
                data['flow'][:,:,:,2*(cf-2):2*(cf-1)] = buffer[cf-1]['flow'] + buffer[cf]['flow']
                data['flow'][:,:,:,2*(cf+2):2*(cf+3)] = -buffer[cf+1]['flow'] - buffer[cf+2]['flow']
        
            for crop_id in range(b):
                for i in range(self.nframes):
                    if i == cf: continue
                    image = data['lr'][crop_id,:,:,c*i:c*(i+1)]
                    flow = data['flow'][crop_id,:,:,2*i:2*(i+1)]
                    shift = FlowShift(image, flow)
                    data['lr'][crop_id,:,:,c*i:c*(i+1)] = shift
        
        data['lr'] = np.ascontiguousarray(data['lr'].transpose(0,3,1,2))
        if self.cv2_INTER:
            data['bc'] = np.ascontiguousarray(data['bc'].transpose(0,3,1,2))
        if self.optflow:
            data['flow'] = np.ascontiguousarray(data['flow'].transpose(0,3,1,2))

        return data
    
    def getitem(self, idx):
        data = {}
        video_frame = self.frame_paths[self.video_id]
        video_id = video_frame['id']
        lr_img = self.global_buffer[idx]['lr']
        # lr_img = cv2.imread(video_frame['lr_frames'][idx])[:,:,::-1]
        lr_crops = np.expand_dims(lr_img, 0)
        if self.optflow:
            flow = self.global_buffer[idx]['flow']
            flow_crops = np.expand_dims(flow, 0)
            flow_crops = flow_crops.astype(np.float32)
            data['flow'] = flow_crops

        if self.cv2_INTER:
            b, h, w, c = lr_crops.shape
            bc_crops = np.zeros((b, h*4, w*4, c))
            for i in range(b):
                bc_crops[i, ...] = cv2.resize(lr_crops[i], None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            bc_crops = np.clip(bc_crops, 0, 255)

        lr_crops = scale_down(lr_crops)
        data['frame_id'] = '%04d' % idx
        data['video_id'] = '%02d' % video_id
        data['lr'] = lr_crops

        if self.cv2_INTER:
            bc_crops = scale_down(bc_crops)
            data['bc'] = bc_crops

        return data

    def liner_buffer(self, idx):
        r = self.nframes // 2
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
        if idx < self.__len__() - r:
            data = self.getitem(idx+r)
        else:
            # 最后一帧时，末尾为本帧（上一帧的下一帧）
            data = self.buffer[-1]

        self.buffer.append(data)

        return self.buffer_stack_on_channels(self.buffer)

    def __getitem__(self, idx):
        if self.nframes > 1:
            if self.shuffle:
                return self.multiworkers_buffer(idx)
            else:
                return self.liner_buffer(idx)
        else:
            return self.getitem(idx)

def random_crop(lr_img, hr_img, crop_size=32, crop_per_image=8, aug=None):
    # 本函数用于将numpy随机裁剪成以crop_size为边长的方形crop_per_image等份
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
    crop_per_image = 2
    nframes = 5
    gbuffer_train = Global_Buffer(pool_size=15, optflow=True)
    dst = MegVSR_Dataset('F:/datasets/MegVSR', crop_per_image=crop_per_image, optflow=True,
                        mode='train',crop_size=200, nframes=nframes, global_buffer=gbuffer_train)
    # dst = MegVSR_Test_Dataset('F:/datasets/MegVSR/', crop_per_image=crop_per_image, 
                        # mode='train',crop_size=100, nframes=nframes)
    dataloader_train = DataLoader(dst, batch_size=1, shuffle=False, num_workers=0)
    for i in range(84,86):
        dst.next_video(i)
        for k, data in enumerate(dataloader_train):
            # 由于crops的存在，Dataloader会把数据变成5维，需要view回4维
            imgs_lr = tensor_dim5to4(data['bc'])
            imgs_hr = tensor_dim5to4(data['hr'])
            print(data['video_id'], data['frame_id'])
            # fig = plt.figure(figsize=(20,10))
            # ax = [None]*2*nframes
            # input_out = np.uint8(imgs_lr[0].numpy().transpose(1,2,0)*255)
            # gt_out = np.uint8(imgs_hr[0].numpy().transpose(1,2,0)*255)
            # for i in range(nframes):
            #     ax[i*2] = fig.add_subplot(2, nframes, i+1)
            #     ax[i*2+1] = fig.add_subplot(2, nframes, i+nframes+1)
            #     ax[i*2].imshow(input_out[:,:,3*i:3*i+3])
            #     ax[i*2+1].imshow(gt_out[:,:,3*i:3*i+3])
            # plt.show()
            # plt.close()
            print(len(dataloader_train),len(dst))
            # if k>5: break