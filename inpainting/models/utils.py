import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
from random import *
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def down_sample(x, size=None, scale_factor=None, mode='nearest'):
    # define size if user has specified scale_factor
    if size is None: size = (int(scale_factor*x.size(2)), int(scale_factor*x.size(3)))
    # create coordinates
    h = torch.arange(0,size[0]) / (size[0]-1) * 2 - 1
    w = torch.arange(0,size[1]) / (size[1]-1) * 2 - 1
    # create grid
    grid =torch.zeros(size[0],size[1],2)
    grid[:,:,0] = w.unsqueeze(0).repeat(size[0],1)
    grid[:,:,1] = h.unsqueeze(0).repeat(size[1],1).transpose(0,1)
    # expand to match batch size
    grid = grid.unsqueeze(0).repeat(x.size(0),1,1,1)
    if x.is_cuda: grid = Variable(grid).cuda()
    # do sampling
    return F.grid_sample(x, grid, mode=mode)


def reduce_mean(x):
    for i in range(4):
        if i==1: continue
        x = torch.mean(x, dim=i, keepdim=True)
    return x


def l2_norm(x):
    def reduce_sum(x):
        for i in range(4):
            if i==1: continue
            x = torch.sum(x, dim=i, keepdim=True)
        return x

    x = x**2
    x = reduce_sum(x)
    return torch.sqrt(x)


def show_image(real, masked, stage_1, stage_2, fake, offset_flow):
    batch_size = real.shape[0]

    (real, masked, stage_1, stage_2, fake, offset_flow) = (
                                var_to_numpy(real), 
                                var_to_numpy(masked), 
                                var_to_numpy(stage_1),
                                var_to_numpy(stage_2),
                                var_to_numpy(fake),
                                var_to_numpy(offset_flow)
                              )
    # offset_flow = (offset_flow*2).astype(int) -1
    for x in range(batch_size):
        if x > 5 :
            break
        fig, axs = plt.subplots(ncols=5, figsize=(15,3))
        axs[0].set_title('real image')
        axs[0].imshow(real[x])
        axs[0].axis('off')

        axs[1].set_title('masked image')
        axs[1].imshow(masked[x])
        axs[1].axis('off')

        axs[2].set_title('stage_1 image')
        axs[2].imshow(stage_1[x])
        axs[2].axis('off')

        axs[3].set_title('stage_2 image')
        axs[3].imshow(stage_2[x])
        axs[3].axis('off')

        axs[4].set_title('fake_image')
        axs[4].imshow(fake[x])
        axs[4].axis('off')

        # axs[5].set_title('C_Attn')
        # axs[5].imshow(offset_flow[x])
        # axs[5].axis('off')

        plt.show()


def var_to_numpy(obj, for_vis=True):
    if for_vis:
        obj = obj.permute(0,2,3,1)
        obj = (obj+1) / 2
    return obj.data.cpu().numpy()


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)