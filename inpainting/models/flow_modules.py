import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from time import time
from inpainting.models.correlation_package.modules.correlation import Correlation
from inpainting.models.gated_conv import GatedConvolution, GatedUpConvolution
import pdb


def get_grid(x):
    torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([torchHorizontal, torchVertical], 1)
    return grid


def conv(batch_norm, in_planes, out_planes, kernel_size = 3, stride = 1, dilation = 1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )

def conv_(batch_norm, in_planes, out_planes, kernel_size = 3, stride = 1, dilation = 1):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = stride, dilation = dilation, padding = ((kernel_size - 1) * dilation) // 2, bias=True),
            nn.LeakyReLU(0.1,inplace=True)
        )


class MaskEstimator_(nn.Module):
    def __init__(self, args, ch_in):
        super(MaskEstimator_, self).__init__()
        self.args = args
        self.convs = nn.Sequential(
            conv_(False, ch_in, ch_in//2),
            conv_(False, ch_in//2, ch_in//2),
            nn.Conv2d(in_channels = ch_in//2, out_channels = 1, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.convs(x)

class WarpingLayer(nn.Module):
    
    def __init__(self):
        super(WarpingLayer, self).__init__()
    
    def forward(self, x, flow):
        # WarpingLayer uses F.grid_sample, which expects normalized grid
        # we still output unnormalized flow for the convenience of comparing EPEs with FlowNet2 and original code
        # so here we need to denormalize the flow
        flow_for_grip = torch.zeros_like(flow).cuda()
        flow_for_grip[:,0,:,:] = flow[:,0,:,:] / ((flow.size(3) - 1.0) / 2.0)
        flow_for_grip[:,1,:,:] = flow[:,1,:,:] / ((flow.size(2) - 1.0) / 2.0)

        grid = (get_grid(x).cuda() + flow_for_grip).permute(0, 2, 3, 1)
        x_warp = F.grid_sample(x, grid)
        return x_warp


class ContextNetwork(nn.Module):
    def __init__(self, args, ch_in):
        super(ContextNetwork, self).__init__()
        self.args = args
        self.convs = nn.Sequential(
            conv(args.batch_norm, ch_in, 128, 3, 1, 1),
            conv(args.batch_norm, 128, 128, 3, 1, 2),
            conv(args.batch_norm, 128, 128, 3, 1, 4),
            conv(args.batch_norm, 128, 96, 3, 1, 8),
            conv(args.batch_norm, 96, 64, 3, 1, 16),
            conv(args.batch_norm, 64, 32, 3, 1, 1),
            conv(args.batch_norm, 32, 2, 3, 1, 1)
        )
    def forward(self, x):
        return self.convs(x)


class LongFlowEstimatorCorr(nn.Module):
    def __init__(self, args, ch_in):
        super(LongFlowEstimatorCorr, self).__init__()
        self.args = args
        self.convs = nn.Sequential(
            conv(args.batch_norm, ch_in, 128),
            conv(args.batch_norm, 128, 128),
            conv(args.batch_norm, 128, 96),
            conv(args.batch_norm, 96, 64),
            conv(args.batch_norm, 64, 32)
        )
        self.conv1 = nn.Conv2d(in_channels = 32, out_channels = 2, kernel_size = 3, stride = 1, padding = 1, dilation = 1, groups = 1, bias = True)
        self.convs_fine = ContextNetwork(args, 32+2)
        
    def forward(self, x):
        x = self.convs(x)
        flo_coarse = self.conv1(x)
        flo_fine = self.convs_fine(torch.cat([x, flo_coarse],1))
        flo = flo_coarse + flo_fine
        return flo


class LongFlowNetCorr(nn.Module):
    def __init__(self, args, in_ch):
        super(LongFlowNetCorr, self).__init__()
        self.args = args
        self.corr = Correlation(pad_size = args.search_range, kernel_size = 1, max_displacement = args.search_range, stride1 = 1, stride2 = 1, corr_multiply = 1).cuda()
        self.flow_estimator = LongFlowEstimatorCorr(args, in_ch+(args.search_range*2+1)**2)
        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.ConvTranspose3d):
                if m.bias is not None: nn.init.uniform_(m.bias)
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x1, x2, upflow=None):
        corr = self.corr(x1.contiguous(), x2.contiguous())
        if upflow is not None:
            flow = self.flow_estimator(torch.cat([x1, corr, upflow], dim = 1))
        else:
            flow = self.flow_estimator(torch.cat([x1, corr], dim = 1))
        return flow

