# -*- coding: utf-8 -*-
"""
An implementation of the 3D U-Net paper:
     Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger:
     3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. 
     MICCAI (2) 2016: 424-432
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
The implementation is borrowed from: https://github.com/ozan-oktay/Attention-Gated-Networks and
                                   : https://github.com/HiLab-git/SSL4MIS

This implementation has been further modified to include MI confidence gating

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.nn import init

class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1), init_stride=(1,1,1)):
        super(UnetConv3, self).__init__()
        
        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                nn.InstanceNorm3d(out_size),
                nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(
                nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                nn.InstanceNorm3d(out_size),
                nn.ReLU(inplace=True))
        else:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(
                nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                nn.ReLU(inplace=True))
        
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs

class UnetUp3_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, 
                             kernel_size=(3,3,3), padding_size=(1,1,1))
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')
        
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))
    
    
class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear') )

    def forward(self, input):
        return self.dsv(input)



class UNet3D(nn.Module):
    def __init__(self, feature_scale=4, n_classes=4, is_deconv=True, in_channels=2, is_batchnorm=True):
        super(UNet3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Downsampling path
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        # Center
        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # Upsampling path
        self.up4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # deep supervision
        self.dsv4 = UnetDsv3(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv3(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv3(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv3d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)
        
        # Dropout layers
        self.dropout1 = nn.Dropout3d(p=0.5)
        self.dropout2 = nn.Dropout3d(p=0.3)
        self.dropout3 = nn.Dropout3d(p=0.2)
        self.dropout4 = nn.Dropout3d(p=0.1)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs, mi_map=None):
        # Downsample
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        # Upsample with MI feedback
        up4 = self.up4(conv4, center)
        up4 = self.dropout1(up4)
        if mi_map is not None:
            up4 = up4 * (1 - F.interpolate(mi_map[3], size=up4.shape[2:], mode='trilinear', align_corners=False)) 

        up3 = self.up3(conv3, up4)
        up3 = self.dropout2(up3)
        if mi_map is not None:
            up3 = up3 * (1 - F.interpolate(mi_map[2], size=up3.shape[2:], mode='trilinear', align_corners=False)) 

        up2 = self.up2(conv2, up3)
        up2 = self.dropout3(up2)
        if mi_map is not None:
            up2 = up2 * (1 - F.interpolate(mi_map[1], size=up2.shape[2:], mode='trilinear', align_corners=False)) 

        up1 = self.up1(conv1, up2)
        up1 = self.dropout4(up1)
        if mi_map is not None:
            up1 = up1 * (1 - F.interpolate(mi_map[0], size=up1.shape[2:], mode='trilinear', align_corners=False)) 

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)

        deep_supervision = [dsv1, dsv2, dsv3, dsv4]

        return {
            "final_output": dsv1,
            "deep_supervision": deep_supervision,
        }

    @staticmethod
    def apply_argmax_softmax(pred):
        return F.softmax(pred, dim=1)
    



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def gaussian_blur3d(x, kernel_size=3, sigma=1.0):
    """Manual 3D Gaussian blur with proper channel handling."""
    # Create 1D Gaussian kernel
    kernel_1d = torch.arange(-kernel_size//2 + 1, kernel_size//2 + 1, 
                            dtype=torch.float32, device=x.device)
    kernel_1d = torch.exp(-kernel_1d**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()  # Normalize

    # Reshape to 3D kernels (with channel dimension)
    kernel_d = kernel_1d.view(1, 1, -1, 1, 1).repeat(x.shape[1], 1, 1, 1, 1)  # [C_in, 1, k, 1, 1]
    kernel_h = kernel_1d.view(1, 1, 1, -1, 1).repeat(x.shape[1], 1, 1, 1, 1)  # [C_in, 1, 1, k, 1]
    kernel_w = kernel_1d.view(1, 1, 1, 1, -1).repeat(x.shape[1], 1, 1, 1, 1)  # [C_in, 1, 1, 1, k]

    # Apply separable convolutions (depth -> height -> width)
    padding = kernel_size // 2
    x = F.conv3d(x, kernel_d, padding=(padding, 0, 0), groups=x.shape[1])
    x = F.conv3d(x, kernel_h, padding=(0, padding, 0), groups=x.shape[1])
    x = F.conv3d(x, kernel_w, padding=(0, 0, padding), groups=x.shape[1])
    return x
