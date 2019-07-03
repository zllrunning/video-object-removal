import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from inpainting.models.flow_modules import (WarpingLayer, LongFlowNetCorr, MaskEstimator_ )
from inpainting.models.gated_conv import GatedConvolution, GatedUpConvolution
from inpainting.models.utils import *
from inpainting.models.ConvLSTM import ConvLSTM
import pdb

class VI_2D_Encoder_3(nn.Module):
    def __init__(self, opt):
        super(VI_2D_Encoder_3, self).__init__()
        self.opt = opt
        ### ENCODER
        st = 2 if self.opt.double_size else 1
        self.ec0 = GatedConvolution(5, 32, kernel_size=(3,3), stride=(st,st), padding=(1,1), bias=False, type='2d')
        self.ec1 = GatedConvolution(32, 64, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False, type='2d')
        self.ec2 = GatedConvolution(64, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False, type='2d')
        self.ec3_1 = GatedConvolution(64, 96, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False, type='2d')
        self.ec3_2 = GatedConvolution(96, 96, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False, type='2d')
        self.ec4_1 = GatedConvolution(96, 128, kernel_size=(3,3), stride=(2,2), padding=(1,1), bias=False, type='2d')
        self.ec4 = GatedConvolution(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False, type='2d')
        self.ec5 = GatedConvolution(128, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False, type='2d')
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        out_1 = self.ec0(x)
        out_2 = self.ec2(self.ec1(out_1))
        out_4 = self.ec3_2(self.ec3_1(out_2))
        out = self.ec5(self.ec4(self.ec4_1(out_4)))
        
        return out, out_4, out_2, out_1

class VI_2D_Decoder_3(nn.Module):
    def __init__(self, opt):
        super(VI_2D_Decoder_3, self).__init__()
        self.opt=opt
        dv = 2 if self.opt.double_size else 1
        ### decoder
        self.dc0 = GatedConvolution(128, 128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        self.dc1 = GatedConvolution(128, 128, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        #### UPCONV
        self.dc1_1 = GatedUpConvolution((1, opt.crop_size//4//dv, opt.crop_size//4//dv), 128 , 96, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        
        self.dc2_1 = GatedConvolution(96+96, 96, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        self.dc2_bt1 =  GatedConvolution(96, 96, kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,2,2),padding=(0,2,2), bias=False)
        self.dc2_bt2 =  GatedConvolution(96, 96, kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,4,4),padding=(0,4,4), bias=False)
        self.dc2_bt3 =  GatedConvolution(96, 96, kernel_size=(1,3,3), stride=(1,1,1), dilation=(1,8,8),padding=(0,8,8), bias=False)
        #### UPCONV
        self.dc2_2 = GatedUpConvolution((1, opt.crop_size//2//dv, opt.crop_size//2//dv), 96, 64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        self.dc3_1 = GatedConvolution(64+64, 64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        self.dc3_2 =  GatedConvolution(64, 64, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        #### UPCONV
        self.dc4 = GatedUpConvolution((1, opt.crop_size//dv, opt.crop_size//dv), 64 , 32, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        if self.opt.double_size:
            self.upsample = nn.Upsample(size=(1,opt.crop_size, opt.crop_size), mode='trilinear')
        self.dc5 = GatedConvolution(32, 16, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        self.dc6 = nn.Conv3d(16, 3, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, x2_64_warp=None, x2_128_warp=None):
        x1_64 = self.dc1_1(self.dc1(self.dc0(x)))
        if  x2_64_warp is not None and x2_128_warp is not None:
            x1_64 = self.dc2_bt3(self.dc2_bt2(self.dc2_bt1(self.dc2_1(torch.cat([x1_64, x2_64_warp],1)))))
            x1_128 = self.dc2_2(x1_64)
            if self.opt.double_size:
                d6 = self.dc6(self.dc5(self.upsample(self.dc4(self.dc3_2(self.dc3_1(torch.cat([x1_128, x2_128_warp],1)))))))
            else:
                d6 = self.dc6(self.dc5(self.dc4(self.dc3_2(self.dc3_1(torch.cat([x1_128, x2_128_warp],1))))))
        return d6, None

class VI_2D_BottleNeck(nn.Module):
    def __init__(self, opt, in_ch):
        super(VI_2D_BottleNeck, self).__init__()
        self.opt = opt
        ### bottleneck
        self.bt0 =  GatedConvolution(in_ch, 128, kernel_size=(3,3), stride=(1,1), dilation=(1,1),padding=(1,1), bias=False, type='2d')
        self.bt1 =  GatedConvolution(128, 128, kernel_size=(3,3), stride=(1,1), dilation=(2,2),padding=(2,2), bias=False, type='2d')
        self.bt2 =  GatedConvolution(128, 128, kernel_size=(3,3), stride=(1,1), dilation=(4,4),padding=(4,4), bias=False, type='2d')
        self.bt3 =  GatedConvolution(128, 128, kernel_size=(3,3), stride=(1,1), dilation=(8,8),padding=(8,8), bias=False, type='2d')
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        res = self.bt3(self.bt2(self.bt1(self.bt0(x))))
        return res

class VI_Aggregator(nn.Module):
    def __init__(self, opt, in_ch, T):
        super(VI_Aggregator, self).__init__()
        self.opt = opt
        ### spatio-temporal aggregation
        self.stAgg =  GatedConvolution(in_ch, in_ch, kernel_size=(T,3,3), stride=(1,1,1), padding=(0,1,1), bias=False, type='3d')
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, x):
        return self.stAgg(x)


class VINet_final(nn.Module):
    def __init__(self, opt):
        super(VINet_final, self).__init__()
        self.opt = opt
        self.encoder1 = VI_2D_Encoder_3(self.opt)
        self.encoder2 = VI_2D_Encoder_3(self.opt)
        self.bottleneck = VI_2D_BottleNeck(self.opt, in_ch=256)

        self.convlstm = ConvLSTM(input_size=128, hidden_size=128, kernel_size=3)

        self.decoder = VI_2D_Decoder_3(self.opt)
        self.flownet = LongFlowNetCorr(self.opt, 128)
        self.flownet_64 = LongFlowNetCorr(self.opt, 96+2)
        self.flownet_128 = LongFlowNetCorr(self.opt, 64+2)

        if self.opt.prev_warp:
            self.flownet_256 = LongFlowNetCorr(self.opt, 32+2)
            self.masknet_256 = MaskEstimator_(self.opt, 32)
            from inpainting.lib.resample2d_package.modules.resample2d import Resample2d
            self.warping_256 = Resample2d().cuda()

        self.masknet = MaskEstimator_(self.opt, 128)
        self.masknet_64 = MaskEstimator_(self.opt, 96)
        self.masknet_128 = MaskEstimator_(self.opt, 64)

        self.st_agg = VI_Aggregator(self.opt, in_ch=128, T=5)
        self.st_agg_64 = VI_Aggregator(self.opt, in_ch=96, T=5)
        self.st_agg_128 = VI_Aggregator(self.opt, in_ch=64, T=5)
        
        self.warping = WarpingLayer()
        self.warping_64 = WarpingLayer()        
        self.warping_128 = WarpingLayer()

    def forward(self, masked_img, mask, prev_state=None, prev_feed=None, idx=0): # masked img: b x C x TxHxW
        T = masked_img.size(2)
        ref_idx = (T-1)//2 # 2 --> 0
        ones = to_var(torch.ones(mask.size()))
        
        # encoder
        enc_output = []
        enc_input = torch.cat([masked_img, ones, ones*mask], dim=1)

        f1, f1_64, f1_128, f1_256 = self.encoder1(enc_input[:,:,ref_idx,:,:])

        f2, f2_64, f2_128, _ = self.encoder2(enc_input[:,:,ref_idx-2,:,:])
        f3, f3_64, f3_128, _ = self.encoder2(enc_input[:,:,ref_idx-1,:,:])
        f4, f4_64, f4_128, _ = self.encoder2(enc_input[:,:,ref_idx+1,:,:])
        f5, f5_64, f5_128, _ = self.encoder2(enc_input[:,:,ref_idx+2,:,:])
        f6, f6_64, f6_128, f6_256 = self.encoder2(prev_feed)

        flow2 = self.flownet(f1, f2)
        flow3 = self.flownet(f1, f3)
        flow4 = self.flownet(f1, f4)
        flow5 = self.flownet(f1, f5)
        flow6 = self.flownet(f1, f6)

        f2_warp = self.warping(f2, flow2)
        f3_warp = self.warping(f3, flow3)
        f4_warp = self.warping(f4, flow4)
        f5_warp = self.warping(f5, flow5)
        f6_warp = self.warping(f6, flow6)

        f_stack_oth = torch.stack([f2_warp, f3_warp, f4_warp, f5_warp, f6_warp],2)
        f_agg = self.st_agg(f_stack_oth).squeeze(2)
        occlusion_mask = self.masknet(torch.abs(f1 - f_agg))
        f_syn = (1-occlusion_mask) * f1 + occlusion_mask * f_agg

        bott_input = torch.cat([f1, f_syn],1)
        output = self.bottleneck(bott_input)

        # CONV LSTM
        state = self.convlstm(output, prev_state)

        # ============================ SCALE - 1/4 : 64 =============================
        flow2_64 = F.upsample(flow2, scale_factor = 2, mode = 'bilinear')*2
        flow3_64 = F.upsample(flow3, scale_factor = 2, mode = 'bilinear')*2
        flow4_64 = F.upsample(flow4, scale_factor = 2, mode = 'bilinear')*2
        flow5_64 = F.upsample(flow5, scale_factor = 2, mode = 'bilinear')*2
        flow6_64 = F.upsample(flow6, scale_factor = 2, mode = 'bilinear')*2

        f2_64_warp = self.warping_64(f2_64, flow2_64)
        f3_64_warp = self.warping_64(f3_64, flow3_64)
        f4_64_warp = self.warping_64(f4_64, flow4_64)
        f5_64_warp = self.warping_64(f5_64, flow5_64)
        f6_64_warp = self.warping_64(f6_64, flow6_64)

        flow2_64 = self.flownet_64(f1_64, f2_64_warp, flow2_64) + flow2_64
        flow3_64 = self.flownet_64(f1_64, f3_64_warp, flow3_64) + flow3_64
        flow4_64 = self.flownet_64(f1_64, f4_64_warp, flow4_64) + flow4_64
        flow5_64 = self.flownet_64(f1_64, f5_64_warp, flow5_64) + flow5_64
        flow6_64 = self.flownet_64(f1_64, f6_64_warp, flow6_64) + flow6_64

        f2_64_warp = self.warping_64(f2_64, flow2_64)
        f3_64_warp = self.warping_64(f3_64, flow3_64)
        f4_64_warp = self.warping_64(f4_64, flow4_64)
        f5_64_warp = self.warping_64(f5_64, flow5_64)
        f6_64_warp = self.warping_64(f6_64, flow6_64)
        
        f_stack_64_oth = torch.stack([f2_64_warp, f3_64_warp, f4_64_warp, f5_64_warp, f6_64_warp],2)
        f_agg_64 = self.st_agg_64(f_stack_64_oth).squeeze(2)
        occlusion_mask_64 = self.masknet_64(torch.abs(f1_64 - f_agg_64))
        f_syn_64 = (1-occlusion_mask_64) * f1_64 + occlusion_mask_64 * f_agg_64

        # ============================= SCALE - 1/2 : 128 ===============================
        flow2_128 = F.upsample(flow2_64, scale_factor = 2, mode = 'bilinear')*2
        flow3_128 = F.upsample(flow3_64, scale_factor = 2, mode = 'bilinear')*2
        flow4_128 = F.upsample(flow4_64, scale_factor = 2, mode = 'bilinear')*2
        flow5_128 = F.upsample(flow5_64, scale_factor = 2, mode = 'bilinear')*2
        flow6_128 = F.upsample(flow6_64, scale_factor = 2, mode = 'bilinear')*2

        f2_128_warp = self.warping_128(f2_128, flow2_128)
        f3_128_warp = self.warping_128(f3_128, flow3_128)
        f4_128_warp = self.warping_128(f4_128, flow4_128)
        f5_128_warp = self.warping_128(f5_128, flow5_128)
        f6_128_warp = self.warping_128(f6_128, flow6_128)

        flow2_128 = self.flownet_128(f1_128, f2_128_warp, flow2_128) + flow2_128
        flow3_128 = self.flownet_128(f1_128, f3_128_warp, flow3_128) + flow3_128
        flow4_128 = self.flownet_128(f1_128, f4_128_warp, flow4_128) + flow4_128
        flow5_128 = self.flownet_128(f1_128, f5_128_warp, flow5_128) + flow5_128
        flow6_128 = self.flownet_128(f1_128, f6_128_warp, flow6_128) + flow6_128

        f2_128_warp = self.warping_128(f2_128, flow2_128)
        f3_128_warp = self.warping_128(f3_128, flow3_128)
        f4_128_warp = self.warping_128(f4_128, flow4_128)
        f5_128_warp = self.warping_128(f5_128, flow5_128)
        f6_128_warp = self.warping_128(f6_128, flow6_128)

        f_stack_128_oth = torch.stack([f2_128_warp, f3_128_warp, f4_128_warp, f5_128_warp, f6_128_warp],2)
        f_agg_128 = self.st_agg_128(f_stack_128_oth).squeeze(2)
        occlusion_mask_128 = self.masknet_128(torch.abs(f1_128 - f_agg_128))
        f_syn_128 = (1-occlusion_mask_128) * f1_128 + occlusion_mask_128 * f_agg_128

        output, _ = self.decoder(state[0].unsqueeze(2), x2_64_warp=f_syn_64.unsqueeze(2), x2_128_warp=f_syn_128.unsqueeze(2))
        occ_mask = F.upsample(occlusion_mask, scale_factor=8, mode='bilinear')
        occ_mask_64 = F.upsample(occlusion_mask_64, scale_factor=4, mode='bilinear')
        occ_mask_128 = F.upsample(occlusion_mask_128, scale_factor=2, mode='bilinear')

        flow6_256, flow6_512 = None, None
        if self.opt.prev_warp:
            if prev_state is not None or idx != 0:
                flow6_256 = F.upsample(flow6_128, scale_factor = 2, mode = 'bilinear')*2
                flow6_512 = F.upsample(flow6_128, scale_factor = 4, mode = 'bilinear')*4
                f6_256_warp = self.warping_256(f6_256, flow6_256)
                flow6_256 = self.flownet_256(f1_256, f6_256_warp, flow6_256) + flow6_256
                occlusion_mask_256 = self.masknet_256(torch.abs(f1_256 - f6_256_warp))
                output_ = output
                if self.opt.double_size:
                    prev_feed_warp = self.warping_256(prev_feed[:,:3], flow6_512)
                    occlusion_mask_512 = F.upsample(occlusion_mask_256, scale_factor=2, mode='nearest')
                    output = (1-occlusion_mask_512.unsqueeze(2)) * output + occlusion_mask_512 * prev_feed_warp.unsqueeze(2)
                    flow6_256=flow6_512
                else:
                    prev_feed_warp = self.warping_256(prev_feed[:,:3], flow6_256)                
                    output = (1-occlusion_mask_256.unsqueeze(2)) * output + occlusion_mask_256 * prev_feed_warp.unsqueeze(2)
                if self.opt.loss_on_raw:
                    output = (output, output_)

        return output, torch.stack([flow2_128,flow3_128,flow4_128,flow5_128, flow6_128],2), state, torch.stack([occ_mask, 1-occ_mask, occ_mask_64, 1-occ_mask_64, occ_mask_128, 1-occ_mask_128], 2), flow6_256



        # if not self.opt.no_train:
        #     if self.opt.prev_warp and prev_state is not None:
        #         if self.opt.loss_on_raw:
        #             output = (output, output_)
        #         return output, torch.stack([flow2_128,flow3_128,flow4_128,flow5_128, flow6_128],2), state, None, torch.stack([occ_mask, 1-occ_mask, occ_mask_64, 1-occ_mask_64, occ_mask_128, occlusion_mask_256], 2), flow6_256
        #     else:
        #         return output, torch.stack([flow2_128,flow3_128,flow4_128,flow5_128, flow6_128],2), state, None, torch.stack([occ_mask, 1-occ_mask, occ_mask_64, 1-occ_mask_64, occ_mask_128, 1-occ_mask_128], 2)

        # elif self.opt.test:
        #     if self.opt.prev_warp and idx !=0:         
        #         return output, torch.stack([flow2_128,flow3_128,flow4_128,flow5_128, flow6_128],2), state, None, torch.stack([occ_mask, 1-occ_mask, occ_mask_64, 1-occ_mask_64, occ_mask_128, occlusion_mask_256], 2), flow6_256
        #     else:
        #         return output, torch.stack([flow2_128,flow3_128,flow4_128,flow5_128, flow6_128],2), state, None, torch.stack([occ_mask, 1-occ_mask, occ_mask_64, 1-occ_mask_64, occ_mask_128, 1-occ_mask_128], 2)

