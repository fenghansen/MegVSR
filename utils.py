import os
import torch
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
    tensor = tensor.view(-1, c, h, w)
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
        multi_gpu = True
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
    if torch.is_tensor(img_lr):
        img_lr = img_lr.detach().cpu().numpy()
        img_sr = img_sr.detach().cpu().numpy()
        img_hr = img_hr.detach().cpu().numpy()
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
    if torch.is_tensor(img_sr):
        img_sr = img_sr.detach().cpu().numpy()
    # 变回uint8
    img_sr = scale_up(img_sr.transpose(1,2,0))
    if os._exists(save_path) is not True:
        os.makedirs(save_path, exist_ok=True)
    plt.imsave(os.path.join(save_path, frame_id+'.png'), img_sr)
    gc.collect()