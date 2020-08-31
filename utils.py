import torch
import glob
import matplotlib.pyplot as plt
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
    else:
        multi_gpu = True
        host = '/mnt/lustre/fenghansen/datasets'
    return hostname, host + dataset_name, multi_gpu