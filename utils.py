try:
    import megengine as torch
    import megengine.module as nn
    import megengine.functional as F
    from megengine.data import RandomSampler, SequentialSampler, DataLoader
    from megengine.data.dataset import Dataset
    from megengine.jit import trace
    from megengine.optimizer import Adam
    use_mge = True
    print('You are Using Megengine as utils backend...')
except:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    use_mge = False
    print('You are Using Pytorch as utils backend...')

import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import gc
from PIL import Image
import cv2
import time
import socket
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def log(string, log=None):
    log_string = f'{time.strftime("%H:%M:%S")} >>  {string}'
    print(log_string)
    if log is not None:
        with open(log,'a+') as f:
            f.write(log_string+'\n')


def load_weights(model, path, multi_gpu=False, by_name=False):
    pretrained_dict=torch.load(path)
    model_dict = model.module.state_dict() if multi_gpu else model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() 
                    if k in model_dict} if by_name else pretrained_dict
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    if multi_gpu:
        model.module.load_state_dict(model_dict)
    else:
        model.load_state_dict(model_dict)
    return model


def tensor_dim5to4(tensor):
    batchsize, crops, c, h, w = tensor.shape
    tensor = tensor.reshape(batchsize*crops, c, h, w)
    return tensor


def get_host_with_dir(dataset_name=''):
    multi_gpu = False
    hostname = socket.gethostname()
    log(f"User's hostname is '{hostname}'")
    if hostname == '1':
        host = '/data'
    elif hostname == 'DESKTOP-FCAMIOQ':
        host = 'F:/datasets'
    elif len(hostname) > 30:
        host = '/home/megstudio/workspace/datasets'
    else:
        multi_gpu = True
        host = '/mnt/lustre/fenghansen/datasets'
    return hostname, host + dataset_name, multi_gpu

def scale_down(img):
    return np.float32(img) / 255.

def scale_up(img):
    return np.uint8(img * 255.)

def plot_sample(img_lr, img_sr, img_hr, save_path, frame_id, plot_path='./images/samples', 
                model_name='RRDB', subplots=3, epoch=-1):
    # 变回uint8
    img_lr = scale_up(img_lr.transpose(1,2,0))
    img_sr = scale_up(img_sr.transpose(1,2,0))
    img_hr = scale_up(img_hr.transpose(1,2,0))
    # 获得bicubic缩放的图像
    h, w, c = img_hr.shape
    img_bc = cv2.resize(img_lr, (w,h), interpolation=cv2.INTER_CUBIC)
    # 计算PSNR和SSIM
    filename = frame_id
    psnr = []
    ssim = []
    psnr.append(compare_psnr(img_hr, img_bc))
    psnr.append(compare_psnr(img_hr, img_sr))
    ssim.append(compare_ssim(img_hr, img_bc, multichannel=True))
    ssim.append(compare_ssim(img_hr, img_sr, multichannel=True))
    psnr.append(-1)
    ssim.append(-1)
    # Images and titles
    images = {
        'Bicubic Interpolation': img_bc,
        model_name: img_sr,
        'Original': img_hr
    }
    # if os._exists(save_path) is not True:
    #     os.makedirs(save_path)
    plt.imsave(os.path.join(save_path, "{}_out.png".format(filename)), img_sr)
    # Plot the images. Note: rescaling and using squeeze since we are getting batches of size 1
    fig, axes = plt.subplots(1, 3, figsize=(30, 8))
    for i, (title, img) in enumerate(images.items()):
        axes[i].imshow(img)
        axes[i].set_title("{} - {} - psnr:{:.2f} - ssim{:.4f}".format(title, img.shape, psnr[i], ssim[i]))
        axes[i].axis('off')
    plt.suptitle('{} - Epoch: {}'.format(filename, epoch))
    print('PSNR:', psnr)
    print('SSIM:', ssim)
    # Save directory
    savefile = os.path.join(save_path, "{}-Epoch{}.png".format(filename, epoch))
    fig.savefig(savefile, bbox_inches='tight')
    plt.close()
    gc.collect()


def save_picture(img_sr, save_path='./images/test',frame_id='0000'):
    # 变回uint8
    img_sr = scale_up(img_sr.transpose(1,2,0))
    if os._exists(save_path) is not True:
        os.makedirs(save_path, exist_ok=True)
    plt.imsave(os.path.join(save_path, frame_id+'.png'), img_sr)
    gc.collect()

def test_output_rename(root_dir):
    for dirs in os.listdir(root_dir):
        dirpath = root_dir + '/' + dirs
        f = os.listdir(dirpath)
        end = len(f)
        for i in range(len(f)):
            frame_id = int(f[end-i-1][:4])
            old_file = os.path.join(dirpath, "%04d.png" % frame_id)
            new_file = os.path.join(dirpath, "%04d.png" % (frame_id + 1))
            os.rename(old_file, new_file)
        log(f"path |{dirpath}|'s rename has finished...")

def transform_pth2mge(path):
    if use_mge is False:
        log('You need to load Megengine first...')
        return
    
    net = SRResnet(nb=16)
    model_dict = net.state_dict()
    
    optimizer = Adam(net.parameters(), lr=1e-4)

    pretrained_dict = pytorch.load('last_model.pth')['net']
    # 1. filter out unnecessary keys
    for k, v in pretrained_dict.items():
        if k in model_dict:
            pretrained_dict[k] = v.cpu().numpy()
            if ('bias' in k and 'conv' in k or 'mean' in k or 'var' in k
                or '0.bias' in k or k=='out.bias'):
                pretrained_dict[k] = pretrained_dict[k].reshape(1,-1,1,1)
            print(f"'{k}':{model_dict[k].shape} -> {pretrained_dict[k].shape}")
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    net.load_state_dict(model_dict)

if __name__ == '__main__':
    import torch as pytorch
    from models import *
    transform_pth2mge('last_model.pth')
    # test_output_rename(r'F:/DeepLearning/MegVSR/images/test')