import csv
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.backends import cudnn
from random import *
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pdb

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)

    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    n_correct_elems = correct.float().sum().data[0]

    return n_correct_elems / batch_size


def tensor2img(x,opt):
    if opt.no_mean_norm:
        x = x.copy() * 255.
        if len(np.shape(x)) == 3:
            return x.transpose((1,2,0)).astype(np.uint8)
        else:
            return x.astype(np.uint8)

    x[0] = x[0] + opt.mean[0]
    x[1] = x[1] + opt.mean[1]
    x[2] = x[2] + opt.mean[2]
    x = x.transpose((1,2,0)).astype(np.uint8)
    return x

def cvimg2tensor(src):
    out = src.copy()
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    out = out.transpose((2,0,1)).astype(np.float64)
    # out = out / 255
    return out

def DxDy(x):
    # shift one pixel and get difference (for both x and y direction)
    #return x[:,:,:,:,:-1] - x[:,:,:,:,1:], x[:,:,:,:-1,:]-x[:,:,:,1:,:]
    return x[:,:,:,:-1] - x[:,:,:,1:], x[:,:,:-1,:]-x[:,:,1:,:]


######################################################################################
##  Image utility
######################################################################################


def rotate_image(img, degree, interp=cv2.INTER_LINEAR):

    height, width = img.shape[:2]
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, degree, 1.)

    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    img_out = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h), flags=interp+cv2.WARP_FILL_OUTLIERS)
  
    return img_out


def numpy_to_PIL(img_np):

    ## input image is numpy array in [0, 1]
    ## convert to PIL image in [0, 255]

    img_PIL = np.uint8(img_np * 255)
    img_PIL = Image.fromarray(img_PIL)

    return img_PIL

def PIL_to_numpy(img_PIL):

    img_np = np.asarray(img_PIL)
    img_np = np.float32(img_np) / 255.0

    return img_np


def read_img(filename):
    ## read image and convert to RGB in [0, 1]
    img = cv2.imread(filename)
    if img is None:
        raise Exception("Image %s does not exist" %filename)
    img = img[:, :, ::-1] ## BGR to RGB    
    img = np.float32(img) / 255.0
    return img

def save_img(img, filename):

    print("Save %s" %filename)
    ## clip to [0, 1]
    img = np.clip(img, 0, 1)
    ## quantize to [0, 255]
    img = np.uint8(img * 255.0)
    cv2.imwrite(filename, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def var_to_numpy(obj, for_vis=True):
    if for_vis:
        obj = obj.permute(0,2,3,1)
        obj = (obj+1) / 2
    return obj.data.cpu().numpy()


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)





### python lib
import os, sys, random, math, cv2, pickle, subprocess
import numpy as np
from PIL import Image

### torch lib
import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader

### custom lib
from inpainting.lib.resample2d_package.modules.resample2d import Resample2d

FLO_TAG = 202021.25
EPS = 1e-12

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


######################################################################################
##  Training utility
######################################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def normalize_ImageNet_stats(batch):

    mean = torch.zeros_like(batch)
    std = torch.zeros_like(batch)
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225

    batch_out = (batch - mean) / std

    return batch_out


def img2tensor(img):

    img_t = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img_t = torch.from_numpy(img_t.astype(np.float32))

    return img_t

def tensor2img(img_t):

    img = img_t[0].detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))

    return img


def save_model(model, optimizer, opts):

    # save opts
    opts_filename = os.path.join(opts.model_dir, "opts.pth")
    print("Save %s" %opts_filename)
    with open(opts_filename, 'wb') as f:
        pickle.dump(opts, f)

    # serialize model and optimizer to dict
    state_dict = {
        'model': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }

    model_filename = os.path.join(opts.model_dir, "model_epoch_%d.pth" %model.epoch)
    print("Save %s" %model_filename)
    torch.save(state_dict, model_filename)


def load_model(model, optimizer, opts, epoch):

    # load model
    model_filename = os.path.join(opts.model_dir, "model_epoch_%d.pth" %epoch)
    print("Load %s" %model_filename)
    state_dict = torch.load(model_filename)
    
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])

    ### move optimizer state to GPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()

    model.epoch = epoch ## reset model epoch

    return model, optimizer



class SubsetSequentialSampler(Sampler):

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

def create_data_loader(data_set, opts, mode):

    ### generate random index
    if mode == 'train':
        total_samples = opts.train_epoch_size * opts.batch_size
    else:
        total_samples = opts.valid_epoch_size * opts.batch_size

    num_epochs = int(math.ceil(float(total_samples) / len(data_set)))

    indices = np.random.permutation(len(data_set))
    indices = np.tile(indices, num_epochs)
    indices = indices[:total_samples]

    ### generate data sampler and loader
    sampler = SubsetSequentialSampler(indices)
    data_loader = DataLoader(dataset=data_set, num_workers=opts.threads, batch_size=opts.batch_size, sampler=sampler, pin_memory=True)

    return data_loader


def learning_rate_decay(opts, epoch):
    
    ###             1 ~ offset              : lr_init
    ###        offset ~ offset + step       : lr_init * drop^1
    ### offset + step ~ offset + step * 2   : lr_init * drop^2
    ###              ...
    
    if opts.lr_drop == 0: # constant learning rate
        decay = 0
    else:
        assert(opts.lr_step > 0)
        decay = math.floor( float(epoch) / opts.lr_step )
        decay = max(decay, 0) ## decay = 1 for the first lr_offset iterations

    lr = opts.lr_init * math.pow(opts.lr_drop, decay)
    lr = max(lr, opts.lr_init * opts.lr_min)

    return lr


def count_network_parameters(model):

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    N = sum([np.prod(p.size()) for p in parameters])

    return N



######################################################################################
##  Image utility
######################################################################################


def rotate_image(img, degree, interp=cv2.INTER_LINEAR):

    height, width = img.shape[:2]
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, degree, 1.)

    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    img_out = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h), flags=interp+cv2.WARP_FILL_OUTLIERS)
  
    return img_out


def numpy_to_PIL(img_np):

    ## input image is numpy array in [0, 1]
    ## convert to PIL image in [0, 255]

    img_PIL = np.uint8(img_np * 255)
    img_PIL = Image.fromarray(img_PIL)

    return img_PIL

def PIL_to_numpy(img_PIL):

    img_np = np.asarray(img_PIL)
    img_np = np.float32(img_np) / 255.0

    return img_np


def read_img(filename, grayscale=0):

    ## read image and convert to RGB in [0, 1]

    if grayscale:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise Exception("Image %s does not exist" %filename)

        img = np.expand_dims(img, axis=2)
    else:
        img = cv2.imread(filename)

        if img is None:
            raise Exception("Image %s does not exist" %filename)

        img = img[:, :, ::-1] ## BGR to RGB
    
    img = np.float32(img) / 255.0

    return img

def save_img(img, filename):

    print("Save %s" %filename)

    if img.ndim == 3:
        img = img[:, :, ::-1] ### RGB to BGR
    
    ## clip to [0, 1]
    img = np.clip(img, 0, 1)

    ## quantize to [0, 255]
    img = np.uint8(img * 255.0)

    cv2.imwrite(filename, img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


######################################################################################
##  Flow utility
######################################################################################

def read_flo(filename):

    with open(filename, 'rb') as f:
        tag = np.fromfile(f, np.float32, count=1)
        
        if tag != FLO_TAG:
            sys.exit('Wrong tag. Invalid .flo file %s' %filename)
        else:
            w = int(np.fromfile(f, np.int32, count=1))
            h = int(np.fromfile(f, np.int32, count=1))
            #print 'Reading %d x %d flo file' % (w, h)
                
            data = np.fromfile(f, np.float32, count=2*w*h)

            # Reshape data into 3D array (columns, rows, bands)
            flow = np.resize(data, (h, w, 2))

    return flow

def save_flo(flow, filename):

    with open(filename, 'wb') as f:

        tag = np.array([FLO_TAG], dtype=np.float32)

        (height, width) = flow.shape[0:2]
        w = np.array([width], dtype=np.int32)
        h = np.array([height], dtype=np.int32)
        tag.tofile(f)
        w.tofile(f)
        h.tofile(f)
        flow.tofile(f)
    
def resize_flow(flow, W_out=0, H_out=0, scale=0):

    if W_out == 0 and H_out == 0 and scale == 0:
        raise Exception("(W_out, H_out) or scale should be non-zero")

    H_in = flow.shape[0]
    W_in = flow.shape[1]

    if scale == 0:
        y_scale = float(H_out) / H_in
        x_scale = float(W_out) / W_in
    else:
        y_scale = scale
        x_scale = scale

    flow_out = cv2.resize(flow, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_LINEAR)

    flow_out[:, :, 0] = flow_out[:, :, 0] * x_scale
    flow_out[:, :, 1] = flow_out[:, :, 1] * y_scale

    return flow_out


def rotate_flow(flow, degree, interp=cv2.INTER_LINEAR):
    
    ## angle in radian
    angle = math.radians(degree)

    H = flow.shape[0]
    W = flow.shape[1]

    #rotation_matrix = cv2.getRotationMatrix2D((W/2, H/2), math.degrees(angle), 1)
    #flow_out = cv2.warpAffine(flow, rotation_matrix, (W, H))
    flow_out = rotate_image(flow, degree, interp)
    
    fu = flow_out[:, :, 0] * math.cos(-angle) - flow_out[:, :, 1] * math.sin(-angle)
    fv = flow_out[:, :, 0] * math.sin(-angle) + flow_out[:, :, 1] * math.cos(-angle)

    flow_out[:, :, 0] = fu
    flow_out[:, :, 1] = fv

    return flow_out

def hflip_flow(flow):

    flow_out = cv2.flip(flow, flipCode=0)
    flow_out[:, :, 0] = flow_out[:, :, 0] * (-1)

    return flow_out

def vflip_flow(flow):

    flow_out = cv2.flip(flow, flipCode=1)
    flow_out[:, :, 1] = flow_out[:, :, 1] * (-1)

    return flow_out

def flow_to_rgb(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    #print "max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv)

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.float32(img) / 255.0


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def compute_flow_magnitude(flow):

    flow_mag = flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2

    return flow_mag

def compute_flow_gradients(flow):

    H = flow.shape[0]
    W = flow.shape[1]

    flow_x_du = np.zeros((H, W))
    flow_x_dv = np.zeros((H, W))
    flow_y_du = np.zeros((H, W))
    flow_y_dv = np.zeros((H, W))
    
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]

    flow_x_du[:, :-1] = flow_x[:, :-1] - flow_x[:, 1:]
    flow_x_dv[:-1, :] = flow_x[:-1, :] - flow_x[1:, :]
    flow_y_du[:, :-1] = flow_y[:, :-1] - flow_y[:, 1:]
    flow_y_dv[:-1, :] = flow_y[:-1, :] - flow_y[1:, :]

    return flow_x_du, flow_x_dv, flow_y_du, flow_y_dv


def detect_occlusion(fw_flow, bw_flow):
    
    ## fw-flow: img1 => img2
    ## bw-flow: img2 => img1

    
    with torch.no_grad():

        ## convert to tensor
        fw_flow_t = img2tensor(fw_flow).cuda()
        bw_flow_t = img2tensor(bw_flow).cuda()

        ## warp fw-flow to img2
        flow_warping = Resample2d().cuda()
        fw_flow_w = flow_warping(fw_flow_t, bw_flow_t)
    
        ## convert to numpy array
        fw_flow_w = tensor2img(fw_flow_w)


    ## occlusion
    fb_flow_sum = fw_flow_w + bw_flow
    fb_flow_mag = compute_flow_magnitude(fb_flow_sum)
    fw_flow_w_mag = compute_flow_magnitude(fw_flow_w)
    bw_flow_mag = compute_flow_magnitude(bw_flow)

    mask1 = fb_flow_mag > 0.01 * (fw_flow_w_mag + bw_flow_mag) + 0.5
    
    ## motion boundary
    fx_du, fx_dv, fy_du, fy_dv = compute_flow_gradients(bw_flow)
    fx_mag = fx_du ** 2 + fx_dv ** 2
    fy_mag = fy_du ** 2 + fy_dv ** 2
    
    mask2 = (fx_mag + fy_mag) > 0.01 * bw_flow_mag + 0.002

    ## combine mask
    mask = np.logical_or(mask1, mask2)
    occlusion = np.zeros((fw_flow.shape[0], fw_flow.shape[1]))
    occlusion[mask == 1] = 1

    return occlusion

######################################################################################
##  Other utility
######################################################################################

def save_vector_to_txt(matrix, filename):

    with open(filename, 'w') as f:

        print("Save %s" %filename)
        
        for i in range(matrix.size):
            line = "%f" %matrix[i]
            f.write("%s\n"%line)

def run_cmd(cmd):
    print(cmd)
    subprocess.call(cmd, shell=True)

def make_video(input_dir, img_fmt, video_filename, fps=24):

    cmd = "ffmpeg -y -loglevel error -framerate %s -i %s/%s -vcodec libx264 -pix_fmt yuv420p -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" %s" \
            %(fps, input_dir, img_fmt, video_filename)

    run_cmd(cmd)
