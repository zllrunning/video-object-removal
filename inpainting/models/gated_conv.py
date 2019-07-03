import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GatedConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation=1, padding=0, bias=False, type='3d', status='train'):
        super(GatedConvolution, self).__init__()
        assert type in ['2d', '3d']
        assert status in ['train', 'test']

        self.status = status
        self.type = type

        if type == '3d':
            self.conv = nn.Conv3d(in_channels, out_channels*2, kernel_size,  stride=stride, dilation=dilation, padding=padding, bias=bias)
        elif type == '2d':
            self.conv = nn.Conv2d(in_channels, out_channels*2, kernel_size,  stride=stride, dilation=dilation, padding=padding, bias=bias)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        c = x.size(1)
        phi, gate = torch.split(x,c//2,1)

        if self.status == 'train':
            return torch.sigmoid(gate)*self.relu(phi)
        else:
            return torch.sigmoid(gate)*self.relu(phi), torch.sigmoid(gate)
        

class GatedUpConvolution(nn.Module):
    def __init__(self, size, in_channels, out_channels, kernel_size, stride, padding, bias, mode='trilinear', type='3d', status='train'):
        super(GatedUpConvolution, self).__init__()
        assert type in ['2d', '3d']
        assert status in ['train', 'test']
        self.status = status
        self.type = type
        self.leaky_relu = nn.LeakyReLU(0.2)

        if type == '3d':
            self.conv = nn.Sequential(
                        nn.Upsample(size=size, mode=mode),
                        nn.Conv3d(in_channels, out_channels*2, kernel_size, stride=stride, padding=padding, bias=bias))
                            
        elif type == '2d':
            self.conv = nn.Sequential(
                        nn.Upsample(size=size, mode=mode),
                        nn.Conv2d(in_channels, out_channels*2, kernel_size, stride=stride, padding=padding, bias=bias))

    def forward(self, x):
        x = self.conv(x)
        c = x.size(1)
        phi, gate = torch.split(x,c//2,1)

        if self.status == 'train':
            return torch.sigmoid(gate)*self.leaky_relu(phi)
        else:
            return torch.sigmoid(gate)*self.leaky_relu(phi), torch.sigmoid(gate)

