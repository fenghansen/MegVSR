import numpy as np
import random
import h5py
import os
import cv2
import tarfile
import io
import av
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

TRAIN_RAW_DATA = "F:/datasets/MegVSR/train.tar"
TEST_RAW_DATA = "F:/datasets/MegVSR/test.tar"
TRAIN_DATA_STORAGE = "F:/datasets/MegVSR"

# 计算视频尺寸
def frame_shape(generator):
    for frame in generator:
        return frame.to_rgb().to_ndarray().shape

# 统计视频帧数
def frame_count(container, video_stream=0):
    def count(generator):
        res = 0
        for _ in generator:
            res += 1
        return res

    frames = container.streams.video[video_stream].frames
    if frames != 0:
        return frames
    frame_series = container.decode(video=video_stream)
    frames = count(frame_series)
    container.seek(0)
    return frames

random.seed(100)
tar = tarfile.open(TRAIN_RAW_DATA)
os.makedirs(TRAIN_DATA_STORAGE, exist_ok=True)

name_info = {}
todo_list = []
while True:
    tinfo = tar.next()
    if tinfo is None:
        break
    if not tinfo.isfile():
        continue
    tname = tinfo.name
    name_info[tname] = tinfo
    if tname.endswith("_down4x.mp4"):
        todo_list.append(tname)


def official_pairdata_maker():
    count = 0
    for tname in tqdm(todo_list):
        tinfo = name_info[tname]
        srcinfo = name_info[tname.replace('_down4x.mp4', '')]

        f_down4x = tar.extractfile(tinfo)    # 下采样版本的视频
        f_origin = tar.extractfile(srcinfo)  # 原始视频

        container_down4x = av.open(f_down4x)
        container_origin = av.open(f_origin)

        frames_down4x = container_down4x.decode(video=0)
        frames_origin = container_origin.decode(video=0)

        fc_down4x = frame_count(container_down4x)
        fc_origin = frame_count(container_origin)
        extra = fc_down4x - fc_origin
        print(tname, extra)
        
        # 由于视频编码和 FFmpeg 实现的问题，压缩前后的帧数可能会不等，下采样版本的视频可能数量少几帧。
        # 这时，您需要注意跳过下采样版本视频缺少的帧数。
        if extra > 0:
            for _ in range(extra):
                next(frames_down4x)

        for k, (frame_down4x,
                frame_origin) in enumerate(zip(frames_down4x, frames_origin)):
            if random.random() < 0.1:
                img_origin = frame_origin.to_rgb().to_ndarray()
                if img_origin.shape[0] < 256 or img_origin.shape[1] < 256:
                    continue
                    
                img_down4x = frame_down4x.to_rgb().to_ndarray()
                img_down4x = cv2.resize(
                    img_down4x, (img_origin.shape[1], img_origin.shape[0]))

                x0 = random.randrange(img_origin.shape[0] - 256 + 1)
                y0 = random.randrange(img_origin.shape[1] - 256 + 1)

                img_show = np.float32(
                    np.stack((img_down4x[x0:x0 + 256, y0:y0 + 256].transpose((2, 0, 1)),
                            img_origin[x0:x0 + 256, y0:y0 + 256].transpose((2, 0, 1))))) / 256
                np.save(os.path.join(TRAIN_DATA_STORAGE, '%04d.npy' % count), img_show)
                count += 1

        container_down4x.close()
        container_origin.close()
        f_down4x.close()
        f_origin.close()


def custom_datapair_maker():
    count = 0

    for frame_id, tname in enumerate(todo_list):
        tinfo = name_info[tname]
        srcinfo = name_info[tname.replace('_down4x.mp4', '')]

        f_down4x = tar.extractfile(tinfo)    # 下采样版本的视频
        f_origin = tar.extractfile(srcinfo)  # 原始视频

        container_down4x = av.open(f_down4x)
        container_origin = av.open(f_origin)

        frames_down4x = container_down4x.decode(video=0)
        frames_origin = container_origin.decode(video=0)

        fc_down4x = frame_count(container_down4x)
        fc_origin = frame_count(container_origin)
        extra = fc_down4x - fc_origin
        print(tname, extra)
        
        # 由于视频编码和 FFmpeg 实现的问题，压缩前后的帧数可能会不等，下采样版本的视频可能数量少几帧。
        # 这时，您需要注意跳过下采样版本视频缺少的帧数。
        if extra > 0:
            for _ in range(extra):
                next(frames_down4x)

        
        h1, w1, c1 = frame_shape(frames_origin)
        h2, w2, c2 = frame_shape(frames_down4x)
        f = h5py.File(os.path.join(TRAIN_DATA_STORAGE,'MegVSR_train_%02d.h5' % frame_id), 'w')
        dst_hr = f.create_dataset('hr', (fc_origin, c1, h1, w1), chunks=(1, c1, h1, w1))
        dst_lr = f.create_dataset('lr', (fc_origin, c2, h2, w2), chunks=(1, c2, h2, w2))

        with tqdm(total=fc_origin) as t:
            for k, (frame_down4x, frame_origin) in enumerate(zip(frames_down4x, frames_origin)):
                    
                img_origin = frame_origin.to_rgb().to_ndarray()
                img_down4x = frame_down4x.to_rgb().to_ndarray()
                
                if extra > 0:
                    img_bicubic = cv2.resize(img_origin, (img_down4x.shape[1], img_down4x.shape[0]), 
                                        interpolation=cv2.INTER_CUBIC)
                    psnr = compare_psnr(img_down4x, img_bicubic)
                    print(f'Frame {k}, PSNR = {psnr:.3f} dB')

                    cv2.imshow('down', img_down4x[:,:,::-1])
                    cv2.imshow('bicubic', img_bicubic[:,:,::-1])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                # img_show = np.float32(
                #     np.stack(img_down4x.transpose((2, 0, 1)),
                #             img_origin.transpose((2, 0, 1)))) / 255.0
                dst_hr[k] = img_origin.transpose(2,0,1).astype(np.uint8)
                dst_lr[k] = img_down4x.transpose(2,0,1).astype(np.uint8)
                # np.save(os.path.join(TRAIN_DATA_STORAGE, '%04d.npy' % count), img_show)
                count += 1
                t.set_description('Frame %02d:' % frame_id)
                t.update(1)

        container_down4x.close()
        container_origin.close()
        f_down4x.close()
        f_origin.close()
        f.close()

def get_frames_from_video(id=84):
    count = 0
    todo = [todo_list[id]]
    for tname in tqdm(todo):
        tinfo = name_info[tname]
        srcinfo = name_info[tname.replace('_down4x.mp4', '')]

        f_down4x = tar.extractfile(tinfo)    # 下采样版本的视频
        f_origin = tar.extractfile(srcinfo)  # 原始视频

        container_down4x = av.open(f_down4x)
        container_origin = av.open(f_origin)

        frames_down4x = container_down4x.decode(video=0)
        frames_origin = container_origin.decode(video=0)

        fc_down4x = frame_count(container_down4x)
        fc_origin = frame_count(container_origin)
        extra = fc_down4x - fc_origin
        print(tname, extra)
        
        # 由于视频编码和 FFmpeg 实现的问题，压缩前后的帧数可能会不等，下采样版本的视频可能数量少几帧。
        # 这时，您需要注意跳过下采样版本视频缺少的帧数。
        if extra > 0:
            for _ in range(extra):
                next(frames_down4x)

        imgs = []
        for k, (frame_down4x,
                frame_origin) in enumerate(zip(frames_down4x, frames_origin)):
            img_origin = frame_origin.to_rgb().to_ndarray()
            if img_origin.shape[0] < 256 or img_origin.shape[1] < 256:
                continue
                
            img_down4x = frame_down4x.to_rgb().to_ndarray()
            imgs.append(img_down4x)
            count += 1

        container_down4x.close()
        container_origin.close()
        f_down4x.close()
        f_origin.close()
        return imgs


if __name__ == '__main__':
    imgs = get_frames_from_video(84)
    print(len(imgs))