import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

Sobel = np.array([[-1,-2,-1],
                  [ 0, 0, 0],
                  [ 1, 2, 1]])
Robert = np.array([[0, 0],
                  [-1, 1]])
Sobel = torch.Tensor(Sobel)
Robert = torch.Tensor(Robert)


def norm(gradient_orig):
    grad_min = torch.min(gradient_orig)
    grad_max = torch.max(gradient_orig)
    grad_norm = torch.div((gradient_orig - grad_min), (grad_max - grad_min + 0.0001))
    return grad_norm


# 已测试本模块没有问题，作用为提取一阶导数算子滤波图（边缘图）
def gradient(maps, direction, device='cuda', kernel='sobel'):
    channels = maps.size()[1]
    if kernel == 'robert':
        smooth_kernel_x = Robert.expand(channels, channels, 2, 2)
        maps = F.pad(maps, (0, 0, 1, 1))
    elif kernel == 'sobel':
        smooth_kernel_x = Sobel.expand(channels, channels, 3, 3)
        maps = F.pad(maps, (1, 1, 1, 1))
    smooth_kernel_y = smooth_kernel_x.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    kernel = kernel.to(device=device)
    # kernel size is (2, 2) so need pad bottom and right side
    gradient_orig = torch.abs(F.conv2d(maps, weight=kernel, padding=0))
    return gradient_orig


def Pyramid_Sample(img, max_scale=8):
    imgs = []
    sample = img
    power = 1
    while 2**power <= max_scale:
        sample = nn.AvgPool2d(2,2)(sample)
        imgs.append(sample)
        power += 1
    return imgs


def Pyramid_Loss(lows, highs, loss_fn=F.l1_loss, rate=1., norm=True):
    losses = []
    for low, high in zip(lows, highs):
        losses.append( loss_fn(low, high) )
    pyramid_loss = 0
    scale = 0
    lam = 1
    for i, loss in enumerate(losses):
        pyramid_loss += loss * lam
        scale += lam
        lam = lam * rate
    if norm:
        pyramid_loss = pyramid_loss / scale
    return pyramid_loss


class Unet_dpsv_Loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def grad_loss(self, low, high):
        grad_x = torch.abs(gradient(low, 'x') - gradient(high, 'x'))
        grad_y = torch.abs(gradient(low, 'y') - gradient(high, 'y'))
        grad_norm = torch.mean(grad_x + grad_y)
        return grad_norm

    def loss(self, low, high):
        loss_grad = self.grad_loss(low, high)
        loss_recon = F.l1_loss(low, high)
        return loss_recon + loss_grad

    def forward(self, output, target, o2, o4, o8):
        t2, t4, t8 = Pyramid_Sample(target, max_scale=8)
        # loss_restore = self.loss(output, target)
        loss_restore = Pyramid_Loss([output, o2, o4, o8], [target, t2, t4, t8],
                                    loss_fn=self.loss, rate=0.5, norm=False)
        return loss_restore


class Unet_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def grad_loss(self, low, high):
        grad_x = torch.abs(gradient(low, 'x') - gradient(high, 'x'))
        grad_y = torch.abs(gradient(low, 'y') - gradient(high, 'y'))
        grad_norm = torch.mean(grad_x + grad_y)
        return grad_norm
    
    def mutual_consistency(self, low, high, hook=-1):
        low_gradient_x = norm(gradient(low, "x"))
        high_gradient_x = norm(gradient(high, "x"))
        M_gradient_x = low_gradient_x + high_gradient_x
        x_loss = M_gradient_x * torch.exp(-10 * M_gradient_x)
        low_gradient_y = norm(gradient(low, "y"))
        high_gradient_y = norm(gradient(high, "y"))
        M_gradient_y = low_gradient_y + high_gradient_y
        y_loss = M_gradient_y * torch.exp(-10 * M_gradient_y)
        mutual_loss = torch.mean(x_loss + y_loss) 
        if hook > -1:
            feature_map_hook(low, high, low_gradient_x+low_gradient_y, high_gradient_x+high_gradient_y, 
                    M_gradient_x + M_gradient_y, x_loss+ y_loss, path=f'./images/samples-features/mutual_consist_epoch{hook}.png')
        return mutual_loss
    
    def loss(self, low, high):
        # loss_grad = self.grad_loss(low, high)
        loss_recon = F.l1_loss(low, high)
        return loss_recon# + loss_grad
    
    def pyramid_loss(self, low, high, loss_fn=F.l1_loss):
        h2, h4, h8 = Pyramid_Sample(high, max_scale=8)
        l2, l4, l8 = Pyramid_Sample(low, max_scale=8)
        loss = Pyramid_Loss([low, l2, l4, l8], [high, h2, h4, h8], loss_fn=F.l1_loss, rate=0.5, norm=True)
        return loss

    def forward(self, low, high, pyramid=False):
        if pyramid:
            loss = self.pyramid_loss(low, high, loss_fn=self.loss)
        else:
            loss = self.loss(low, high)
        # loss_recon = F.l1_loss(low, high)
        # loss_grad = self.grad_loss(low, high)
        # loss_enhance = self.mutual_consistency(low, high, hook)
        return loss


def PSNR_Loss(low, high):
    return -10.0 * torch.log(torch.mean(torch.pow(high-low, 2))) / torch.log(torch.as_tensor(10.0))


class GAN_Loss(nn.Module):
    def __init__(self, mode='RaSGAN'):
        super().__init__()
        self.gan_mode = mode
    
    def forward(self, D_real, D_fake, D_fake_for_G):
        y_ones = torch.ones_like(D_real)
        y_zeros = torch.zeros_like(D_fake)

        if self.gan_mode == 'RSGAN':
            ### Relativistic Standard GAN
            BCE_stable = torch.nn.BCEWithLogitsLoss()
            # Discriminator loss
            errD = BCE_stable(D_real - D_fake, y_ones)
            loss_D = torch.mean(errD)
            # Generator loss
            errG = BCE_stable(D_fake_for_G - D_real, y_ones)
            loss_G = torch.mean(errG)
        elif self.gan_mode == 'SGAN':
            criterion = torch.nn.BCEWithLogitsLoss()
            # Real data Discriminator loss
            errD_real = criterion(D_real, y_ones)
            # Fake data Discriminator loss
            errD_fake = criterion(D_fake, y_zeros)
            loss_D = torch.mean(errD_real + errD_fake) / 2
            # Generator loss
            errG = criterion(D_fake_for_G, y_ones)
            loss_G = torch.mean(errG)
        elif self.gan_mode == 'RaSGAN':
            BCE_stable = torch.nn.BCEWithLogitsLoss()
            # Discriminator loss
            errD = (BCE_stable(D_real - torch.mean(D_fake), y_ones) + 
                    BCE_stable(D_fake - torch.mean(D_real), y_zeros))/2
            loss_D = torch.mean(errD)
            # Generator loss
            errG = (BCE_stable(D_real - torch.mean(D_fake_for_G), y_zeros) + 
                    BCE_stable(D_fake_for_G - torch.mean(D_real), y_ones))/2
            loss_G = torch.mean(errG)
        elif self.gan_mode == 'RaLSGAN':
            # Discriminator loss
            errD = (torch.mean((D_real - torch.mean(D_fake) - y_ones) ** 2) + 
                    torch.mean((D_fake - torch.mean(D_real) + y_ones) ** 2))/2
            loss_D = errD
            # Generator loss (You may want to resample again from real and fake data)
            errG = (torch.mean((D_real - torch.mean(D_fake_for_G) + y_ones) ** 2) + 
                    torch.mean((D_fake_for_G - torch.mean(D_real) - y_ones) ** 2))/2
            loss_G = errG
        
        return loss_D, loss_G