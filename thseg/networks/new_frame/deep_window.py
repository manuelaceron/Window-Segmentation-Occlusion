from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from torch import Tensor
import pdb, math
import timm
from ..common_func.base_func import _ConvBNReLU
from ..unet.unet import UNet
from ..unet.resunet import Res_UNet_50
from ..unet.unet import new_DeepL
import torchvision.models as models
import torchvision
from transformers import SegformerForSemanticSegmentation, PretrainedConfig
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

pre_model = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024" # "nvidia/mit-b2"

configuration_b2 = PretrainedConfig( 
            architectures = ["SegformerForSemanticSegmentation"],
            attention_probs_dropout_prob = 0.0,
            classifier_dropout_prob = 0.1,
            decoder_hidden_size = 768,  
            depths = [3, 4, 6, 3], 
            downsampling_rates = [1, 4, 8, 16],
            drop_path_rate = 0.1,
            hidden_act = 'gelu',
            hidden_dropout_prob = 0.0,
            hidden_sizes = [64, 128, 320, 512], 
            id2label = { "0": "background", "1": "occlusion"},
            image_size = 224, # I think is not used...
            initializer_range = 0.02,
            label2id = { "background": 0, "occlusion": 1 },
            layer_norm_eps = 1e-06,
            mlp_ratios = [4, 4, 4, 4], 
            model_type = "segformer",
            num_attention_heads = [1, 2, 5, 8], 
            num_channels = 3, 
            num_encoder_blocks = 4, 
            patch_sizes = [7, 3, 3, 3], #[7, 3, 3, 3], # Patch size (same b0-b5)
            reshape_last_stage = True,
            semantic_loss_ignore_index = 255,
            sr_ratios = [8, 4, 2, 1], # Reduction ratio of the efficient Self-Att (same b0-b5)
            strides = [4, 2, 2, 2], #[4, 2, 2, 2], # Stride between 2 adjacent patches (same b0-b5)
            ignore_mismatched_sizes=True,
            torch_dtype = "float32",
            transformers_version = "4.18.0"
            ) 

class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_c, num_class):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, num_class, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_class),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_class, num_class, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_class)            
            )
         

    def forward(self, x):
        #pdb.set_trace()
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_c, num_class):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_c, num_class, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(num_class),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_c=3, num_class=4, pretrained_path=None, latent_space = None): 
        super(UNet, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = conv_block(in_c, filters[0]) #3,16
        self.Conv2 = conv_block(filters[0], filters[1]) #16,32
        self.Conv3 = conv_block(filters[1], filters[2]) #32,64
        self.Conv4 = conv_block(filters[2], filters[3]) #64,128
        self.Conv5 = conv_block(filters[3], filters[4]) #128,256

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], num_class, kernel_size=3, stride=1, padding=1)

        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # Add latent space

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
        prob = out

        

        out = self.activate(out)
        
        return out

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

#https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py

class AttU_Net(nn.Module):
    #def __init__(self,img_ch=3,output_ch=1, pretrained_path=None):
    def __init__(self,in_c=3,num_class=1, pretrained_path=None):
        super(AttU_Net,self).__init__()
        
        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        """ self.Conv1 = conv_block(img_ch,64)
        self.Conv2 = conv_block(64,128)
        self.Conv3 = conv_block(128,256)
        self.Conv4 = conv_block(256,512)
        self.Conv5 = conv_block(512,1024) 
       

        self.Up5 = up_conv(1024,512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(1024, 512)

        self.Up4 = up_conv(512,256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(512, 256)
        
        self.Up3 = up_conv(256,128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(256, 128)
        
        self.Up2 = up_conv(128,64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(128, 64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)

        self.activate = nn.Softmax() if output_ch > 1 else nn.Sigmoid() """ 

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = conv_block(in_c, filters[0]) #3,16
        self.Conv2 = conv_block(filters[0], filters[1]) #16,32
        self.Conv3 = conv_block(filters[1], filters[2]) #32,64
        self.Conv4 = conv_block(filters[2], filters[3]) #64,128
        self.Conv5 = conv_block(filters[3], filters[4]) #128,256

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3],F_l=filters[3],F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2],F_l=filters[2],F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1],F_l=filters[1],F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0],F_l=filters[0],F_int=8)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], num_class, kernel_size=3, stride=1, padding=1)

        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv(d2)

        out = self.activate(d1)

        return out


class AlkModule(nn.Module):
    def __init__(self,in_c,out_c,k,r):
        super(AlkModule,self).__init__()

        # Vertical -> horizontal convolution
        self.leftStripConv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size = [1, k],  dilation =r, padding='same'),
            nn.Conv2d(out_c, out_c, kernel_size = [k, 1],  dilation =r, padding='same')
        )

        self.rightStripConv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size = [k, 1],  dilation =r, padding='same'),
            nn.Conv2d(out_c, out_c, kernel_size = [1, k],  dilation =r, padding='same')
        )
        
    def forward(self,x):

        l = self.leftStripConv(x)
        r = self.rightStripConv(x)

        return l+r

class ALKU_Net(nn.Module):
    def __init__(self, in_c=3, num_class=4, pretrained_path=None): 
        super(ALKU_Net, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = conv_block(in_c, filters[0]) #3,16
        self.Conv2 = conv_block(filters[0], filters[1]) #16,32
        self.Conv3 = conv_block(filters[1], filters[2]) #32,64
        self.Conv4 = conv_block(filters[2], filters[3]) #64,128
        self.Conv5 = conv_block(filters[3], filters[4]) #128,256


        k = 10
        self.AlkModule1 = AlkModule(filters[0], filters[0], k, 1)
        self.AlkModule2 = AlkModule(filters[1], filters[1], k, 1)
        self.AlkModule3 = AlkModule(filters[2], filters[2], k, 1)
        self.AlkModule4 = AlkModule(filters[3], filters[3], k, 1)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], num_class, kernel_size=3, stride=1, padding=1)

        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()

    def forward(self, x):
        #encoding 
        e1 = self.Conv1(x) #16

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2) #32

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3) #64

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4) #128

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        #decoding
        d5 = self.Up5(e5)
        e4 = self.AlkModule4(e4) #128
        
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        e3 = self.AlkModule3(e3) #64
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        e2 = self.AlkModule2(e2) #32
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        e1 = self.AlkModule1(e1) #16
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)
               

        out = self.activate(out)
        
        return out


class BayesianUNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, in_c=3, num_class=4, pretrained_path=None): 
        super(BayesianUNet, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv1 = conv_block(in_c, filters[0]) #3,16
        self.Conv2 = conv_block(filters[0], filters[1]) #16,32
        self.Conv3 = conv_block(filters[1], filters[2]) #32,64
        self.Conv4 = conv_block(filters[2], filters[3]) #64,128
        self.Conv5 = conv_block(filters[3], filters[4]) #128,256

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], num_class, kernel_size=3, stride=1, padding=1)

        self.Drop = nn.Dropout(0.5)

        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)
        e1 = self.Drop(e1)
        e2 = self.Maxpool1(e1)

        e2 = self.Conv2(e2)
        e2 = self.Drop(e2)
        e3 = self.Maxpool2(e2)

        e3 = self.Conv3(e3)
        e3 = self.Drop(e3)
        e4 = self.Maxpool3(e3)

        e4 = self.Conv4(e4)
        e4 = self.Drop(e4)
        e5 = self.Maxpool4(e4)

        e5 = self.Conv5(e5)
        e5 = self.Drop(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
        d5 = self.Drop(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d4 = self.Drop(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d3 = self.Drop(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2 = self.Drop(d2)

        out = self.Conv(d2)
        out = self.Drop(out) #?
              

        out = self.activate(out)
        
        return out




class _ASPPConv(nn.Module):
    def __init__(self, in_cannels, num_classannels, atrous_rate, norm_layer):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_cannels, num_classannels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(num_classannels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_cannels, num_classannels, norm_layer):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_cannels, num_classannels, 1, bias=False),
            norm_layer(num_classannels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out

class _ASPP(nn.Module):
    def __init__(self, in_cannels, atrous_rates, norm_layer):
        super(_ASPP, self).__init__()
        num_classannels = 256
        self.b0 = nn.Sequential(
            nn.Conv2d(in_cannels, num_classannels, 1, bias=False), #first conv kernel 1, no dilated conv
            norm_layer(num_classannels),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates) 
        self.b1 = _ASPPConv(in_cannels, num_classannels, rate1, norm_layer)
        self.b2 = _ASPPConv(in_cannels, num_classannels, rate2, norm_layer)
        self.b3 = _ASPPConv(in_cannels, num_classannels, rate3, norm_layer)
        self.b4 = _AsppPooling(in_cannels, num_classannels, norm_layer=norm_layer)

        self.project = nn.Sequential( #TODO: why 5?
            nn.Conv2d(5 * num_classannels, num_classannels, 1, bias=False),
            norm_layer(num_classannels),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x

class resLargeKernel(nn.Module):
    def __init__(self,in_c,out_c,k=15,r=1):
        super(resLargeKernel,self).__init__()

        # Vertical -> horizontal convolution
        self.leftStripConv = nn.Conv2d(in_c, out_c, kernel_size = [1, k],  dilation =r, padding='same')
            
        self.rightStripConv = nn.Conv2d(in_c, out_c, kernel_size = [k, 1],  dilation =r, padding='same')
        
    def forward(self,x):

        l = self.leftStripConv(x)
        r = self.rightStripConv(x)

        return l+r

class DeepWindow(nn.Module):
    """
    https://koreascience.kr/article/JAKO202210261250365.page#ref-37
    """

    def __init__(self, in_c=3, num_class=4, pretrained_path=None): 
        super(DeepWindow, self).__init__()

        #self.deepLab =  DeepLabV3Plus(in_c, num_class, pretrained_path)
        #self.unet = UNet(in_c+1, num_class, pretrained_path)

        inplanes = 2048 #512
        inter_channels = 256

        rate1, rate2, rate3 = tuple([2, 4, 6]) 
        self.resnet = timm.create_model('resnet50', pretrained=True, features_only=True, in_chans=in_c)#out_indices=[0,1,2,3,4]) #pre-trained on ImageNet
        
        self.dconv1 = nn.Conv2d(inplanes, inter_channels, kernel_size=3, padding=rate1, dilation=rate1, bias=False) #fix channels 
        self.dconv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=rate2, dilation=rate2, bias=False)
        self.dconv3 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=rate3, dilation=rate3, bias=False)
        
        self.conv = nn.Conv2d(inter_channels, num_class, 1)
        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()

        self.aspp = _ASPP(inter_channels, [12, 24, 36], norm_layer=nn.BatchNorm2d) 
        self.resLargek = resLargeKernel(inter_channels,inter_channels)

        self.conv3 = nn.Conv2d(inplanes, inter_channels, 3)

        self.rpc_net = ALKU_Net(2, num_class)
         

    def forward(self, x):
        size = x.size()[2:]

        ft_ex= self.resnet(x)
        
        ft_ex = ft_ex[4] 

        #ft_ex = self.conv3(ft_ex)
        #pdb.set_trace()
        
        dc1 = self.dconv1(ft_ex)
        dc2 = self.dconv2(dc1)
        dc3 = self.dconv3(dc2)

        # Occlusion detection
        c4 = self.conv(dc3)
        act1 = self.activate(c4) #([4, 1, 16, 16])      
        out_occ = F.interpolate(act1, size, mode='bilinear', align_corners=True)
        

        # Coarse segmentation
        cs1 = self.aspp(dc3)
        cs2 = self.resLargek(cs1)
        cs3 = cs1 + cs2
        cs4 = self.conv(cs3)
        cs5 = F.interpolate(cs4, size, mode='bilinear', align_corners=True)
        out_seg = self.activate(cs5) # ([4, 1, 16, 16]) TODO: check if softmax before upsampling   
        
        # Visible patterns
        inv_out_occ = 1-out_occ
        ninv_out_occ = (inv_out_occ > 0.5).float()
        nout_seg = (out_seg > 0.5).float()
        
        vis_pattern = torch.mul(ninv_out_occ, nout_seg)

        # Repetitive pattern completion
        input_comp = torch.cat((vis_pattern, out_occ), dim = 1)
        
        
        num_iterations = 3
        for iteration in range(num_iterations):
            completed_patterns = self.rpc_net(input_comp)
            input_comp = torch.cat((completed_patterns, out_occ), dim = 1)
        
        return out_occ, out_seg, completed_patterns

class _DeepLabHead(nn.Module):
    def __init__(self, num_class, inplanes, c1_channels=256, norm_layer=nn.BatchNorm2d):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(inplanes, [12, 24, 36], norm_layer) # I think 2048 is because of the output of resnet50
        
        self.c1_block = _ConvBNReLU(c1_channels, 48, 1, padding=0, norm_layer=norm_layer) #Paper: reduce low level feat to 48. If 1x1 conv, as paper, padding =0, if 3x3conv, padding 1
        self.block = nn.Sequential(
            
            _ConvBNReLU(304, 256, 3, padding=1, norm_layer=norm_layer), #304 = x(256) + c1 (48): size of x and c1 concat #kernel 3
            #nn.Dropout(0.5),
            #_ConvBNReLU(256, 256, 3, padding=1, norm_layer=norm_layer),
            #nn.Dropout(0.1),
            nn.Conv2d(256, num_class, 1))

    def forward(self, x, c1): #c1 low level features, x is the output of resnet
        
        size = c1.size()[2:]
        c1 = self.c1_block(c1) #_ConvBNReLU
        x = self.aspp(x)
        
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        
        return self.block(torch.cat([x, c1], dim=1)) #this is concat in decoder

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_cannels, num_classannels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_cannels, num_classannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_classannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classannels, num_classannels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_classannels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_cannels, num_classannels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_cannels // 2, in_cannels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_cannels, num_classannels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_cannels, num_classannels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_cannels, num_classannels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class deco_Res_UNet_50(nn.Module): 
    def __init__(self,  in_c, num_class,  pretrained_path=None, bilinear=True):
        super(deco_Res_UNet_50, self).__init__()

        self.num_class = num_class
        self.bilinear = bilinear

        self.up1 = Up(2048+1024, 256, bilinear)
        self.up2 = Up(512+256, 128, bilinear)
        self.up3 = Up(256+128, 64, bilinear)
        self.up4 = Up(64+64, 64, bilinear)

        self.outc = OutConv(64, num_class)
        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()

    def forward(self, x, x1, x2, x3, x4, x5):
        size = x.size()    
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = F.interpolate(x, size=(size[2], size[3]), mode='bilinear')
        out = self.outc(out)
        #out = self.activate(out)

        return out

class DeepWindow_rob(nn.Module):
    """
    https://koreascience.kr/article/JAKO202210261250365.page#ref-37
    """

    def __init__(self, in_c=3, num_class=4, pretrained_path=None): 
        super(DeepWindow_rob, self).__init__()

        inplanes = 2048 
        low_level_planes = 256

        
        self.resnet = timm.create_model('resnet50', pretrained=True, features_only=True, in_chans=in_c)#out_indices=[0,1,2,3,4]) #pre-trained on ImageNet
        self.DLhead = _DeepLabHead(num_class, inplanes, c1_channels=low_level_planes)
        self.decoResUnet50 = deco_Res_UNet_50(in_c, num_class) 
        self.unet = UNet(2, num_class)
        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()
      

    def forward(self, x):
        size = x.size()[2:]

        layers = self.resnet(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4] #Encoder

        # Occlusion detection
        out_occ = self.DLhead(x5, x2) #initially I trained this way
        #out_occ = self.decoResUnet50(x, x1, x2, x3, x4, x5) 
        out_occ = F.interpolate(out_occ, size, mode='bilinear', align_corners=True) 
        out_occ = self.activate(out_occ)

        # Initial window segmentation
        out_seg = self.decoResUnet50(x, x1, x2, x3, x4, x5) #initially I trained this way
        #out_seg = self.DLhead(x5, x2)
        #out_seg = F.interpolate(out_seg, size, mode='bilinear', align_corners=True) 
        #out_seg = self.activate(out_seg)

        
        
        
        # Visible patterns
        inv_out_occ = 1-out_occ
        ninv_out_occ = (inv_out_occ > 0.5).float()
        nout_seg = (out_seg > 0.5).float()
        
        vis_pattern = torch.mul(ninv_out_occ, nout_seg)

        # Repetitive pattern completion
        input_comp = torch.cat((vis_pattern, out_occ), dim = 1)
        
        
        num_iterations = 1
        for iteration in range(num_iterations):
            completed_patterns = self.unet(input_comp)
            input_comp = torch.cat((completed_patterns, out_occ), dim = 1)
        
        return out_occ, out_seg, completed_patterns, vis_pattern

class BCNN(nn.Module):
    """
    
    """

    def __init__(self, in_c=3, num_class=4, pretrained_path=True): 
        super(BCNN, self).__init__()

        self.resnet = models.resnet50(pretrained=True)
        self.drop = nn.Dropout(p=0.5)
        
        self.resnet.layer1[0].conv2 = nn.Sequential(self.resnet.layer1[0].conv2, nn.Dropout(p=0.5) )
        self.resnet.layer1[1].conv2 = nn.Sequential(self.resnet.layer1[1].conv2, nn.Dropout(p=0.5) )
        self.resnet.layer1[2].conv2 = nn.Sequential(self.resnet.layer1[2].conv2, nn.Dropout(p=0.5) )

        self.resnet.layer2[0].conv2 = nn.Sequential(self.resnet.layer2[0].conv2, nn.Dropout(p=0.5) )
        self.resnet.layer2[1].conv2 = nn.Sequential(self.resnet.layer2[1].conv2, nn.Dropout(p=0.5) )
        self.resnet.layer2[2].conv2 = nn.Sequential(self.resnet.layer2[2].conv2, nn.Dropout(p=0.5) )
        self.resnet.layer2[3].conv2 = nn.Sequential(self.resnet.layer2[3].conv2, nn.Dropout(p=0.5) )

        self.resnet.layer3[0].conv2 = nn.Sequential(self.resnet.layer3[0].conv2, nn.Dropout(p=0.5) )
        self.resnet.layer3[1].conv2 = nn.Sequential(self.resnet.layer3[1].conv2, nn.Dropout(p=0.5) )
        self.resnet.layer3[2].conv2 = nn.Sequential(self.resnet.layer3[2].conv2, nn.Dropout(p=0.5) )
        self.resnet.layer3[3].conv2 = nn.Sequential(self.resnet.layer3[3].conv2, nn.Dropout(p=0.5) )
        self.resnet.layer3[4].conv2 = nn.Sequential(self.resnet.layer3[4].conv2, nn.Dropout(p=0.5) )
        self.resnet.layer3[5].conv2 = nn.Sequential(self.resnet.layer3[5].conv2, nn.Dropout(p=0.5) )

        self.resnet.layer4[0].conv2 = nn.Sequential(self.resnet.layer4[0].conv2, nn.Dropout(p=0.5) )
        self.resnet.layer4[1].conv2 = nn.Sequential(self.resnet.layer4[1].conv2, nn.Dropout(p=0.5) )
        self.resnet.layer4[2].conv2 = nn.Sequential(self.resnet.layer4[2].conv2, nn.Dropout(p=0.5) )

       

        self.resnet.layer3[0].conv2[0].stride = (1, 1)
        self.resnet.layer3[0].conv2[0].padding = (1, 1)
        self.resnet.layer3[0].downsample[0].stride = (1, 1)
        self.resnet.layer3[0].downsample[0].padding = (0, 0)

        self.resnet.layer4[0].conv2[0].stride = (1, 1)
        self.resnet.layer4[0].conv2[0].padding = (1, 1)
        self.resnet.layer4[0].downsample[0].stride = (1, 1)
        self.resnet.layer4[0].downsample[0].padding = (0, 0)
    


        
        #for name, param in self.resnet.named_parameters():
        #    if 'downsample' in name:
        #        print(name, param.shape)
            
            
        #    if 'classifier.classifier' not in name:
        #        param.requires_grad = False

        self.features = nn.Sequential(*list(self.resnet.children())[:-2])

        self.conv = nn.Conv2d(2048, num_class, 1)
        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()

    def forward(self, x):
        size = x.size()[2:]

        x1 = self.features(x)
        x2 = self.conv(x1)
        act = self.activate(x2)
        
        out = F.interpolate(act, size, mode='bilinear', align_corners=True)
        
        
        return out


class OccSeg_deepParsing(nn.Module):
    """
    
    """

    def __init__(self, in_c=3, num_class=4, pretrained_path=True): 
        super(OccSeg_deepParsing, self).__init__()

        self.resnet = models.resnet50(pretrained=True)
        
        rate1 = 12 #6
        rate2 = 16 #8
        self.resnet.layer3[0].conv2 = nn.Conv2d(256,256, kernel_size=3, padding=rate1, dilation=rate1, bias=False) 
        #self.resnet.layer3[0].conv2.stride = (1, 1)
        #self.resnet.layer3[0].conv2.padding = (1, 1)
        self.resnet.layer3[0].downsample[0].stride = (1, 1)
        self.resnet.layer3[0].downsample[0].padding = (0, 0)

        
        self.resnet.layer4[0].conv2 = nn.Conv2d(512,512, kernel_size=3, padding=rate2, dilation=rate2, bias=False) 
        #self.resnet.layer4[0].conv2.stride = (1, 1)
        #self.resnet.layer4[0].conv2.padding = (1, 1)
        self.resnet.layer4[0].downsample[0].stride = (1, 1)
        self.resnet.layer4[0].downsample[0].padding = (0, 0)

        self.features = nn.Sequential(*list(self.resnet.children())[:-2])

        self.conv = nn.Conv2d(2048, num_class, 1)
        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()

    def forward(self, x):
        size = x.size()[2:]

        x1 = self.features(x)
        x2 = self.conv(x1)
        act = self.activate(x2)
        out = F.interpolate(act, size, mode='bilinear', align_corners=True)
        
        
        return out


class segFormer_ResUnet50(nn.Module):
    """
    
    """

    def __init__(self, in_c=3, num_class=4, pretrained_path=True): 
        super(segFormer_ResUnet50, self).__init__()

        pre_model = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024" # "nvidia/mit-b2"
        id2label = {0: 'background', 1: 'occlusion'}
        label2id = {'background': 0, 'occlusion': 1}
        
        self.segFormer = SegformerForSemanticSegmentation.from_pretrained(pre_model, ignore_mismatched_sizes=True,
                                                            num_labels=len(id2label), id2label=id2label, label2id=label2id,
                                                            reshape_last_stage=True)
        self.resUnet = Res_UNet_50(in_c, num_class) # in_c depends on input + or concatenated


    def forward(self, sf_x, sf_l, x):
        size = x.size()[2:]

        ################SEGFORMER + RESUNET50############################
        occ = self.segFormer(pixel_values=sf_x, labels= sf_l)
        upsampled_logits = nn.functional.interpolate(occ.logits, size=size, mode="bilinear", align_corners=False) 
        occ = upsampled_logits.argmax(dim=1) 
        occ = torch.unsqueeze(occ, 1) 
        #x1 = torch.cat((x,occ), dim = 1) 
        x1 = x + occ
        #x1 = x
        visible = self.resUnet(x1)
        #################################################################
        
        return upsampled_logits, visible

class segFormer_Deeplab(nn.Module):
    """
    
    """

    def __init__(self, in_c=3, num_class=4, pretrained_path=True): 
        super(segFormer_Deeplab, self).__init__()

        pre_model = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024" # "nvidia/mit-b2"
        id2label = {0: 'background', 1: 'occlusion'}
        label2id = {'background': 0, 'occlusion': 1}
        
        self.segFormer = SegformerForSemanticSegmentation.from_pretrained(pre_model, ignore_mismatched_sizes=True,
                                                            num_labels=len(id2label), id2label=id2label, label2id=label2id,
                                                            reshape_last_stage=True)
        self.deepLab =  new_DeepL(in_c, num_class)

    def forward(self, sf_x, sf_l, x):
        size = x.size()[2:]

        ################SEGFORMER + DEEPLABV3+###########################
        occ = self.segFormer(pixel_values=sf_x, labels= sf_l)
        upsampled_logits = nn.functional.interpolate(occ.logits, size=size, mode="bilinear", align_corners=False) 
        occ = upsampled_logits.argmax(dim=1) 
        occ = torch.unsqueeze(occ, 1) 
        #x1 = torch.cat((x,occ), dim = 1) 
        x1 = x + occ
        visible = self.deepLab(x1)
        #################################################################
        
        return upsampled_logits, visible

class segFormer_b2(nn.Module):
    def __init__(self, in_c=3, num_class=4, pretrained_path=True): 
        super(segFormer_b2, self).__init__()

        pre_model = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024" # "nvidia/mit-b2"
        id2label = {0: 'background', 1: 'occlusion'}
        label2id = {'background': 0, 'occlusion': 1}
        
        self.segFormer = SegformerForSemanticSegmentation.from_pretrained(pre_model, ignore_mismatched_sizes=True,
                                                            num_labels=len(id2label), id2label=id2label, label2id=label2id,
                                                            reshape_last_stage=True)
    
    def forward(self, sf_x, sf_l):
        size = sf_l.shape[-2:]
        out = self.segFormer(pixel_values=sf_x, labels= sf_l)
        upsampled_logits = nn.functional.interpolate(out.logits, size=size, mode="bilinear", align_corners=False) 
        out = upsampled_logits.argmax(dim=1) 
        out = torch.unsqueeze(out, 1) 

        return out

class winCompletion(nn.Module):
    """
    
    """

    def __init__(self, in_c=3, num_class=4, pretrained_path=True): 
        super(winCompletion, self).__init__()

        
        
        self.segFormer = SegformerForSemanticSegmentation.from_pretrained(pretrained_model_name_or_path = pre_model, ignore_mismatched_sizes=True, config = configuration_b2)
                                                            
        
        self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True, in_chans=in_c) 
        print('using: resnet50', ' pretrained: ', True )

        self.decoResUnet50 = deco_Res_UNet_50(in_c, num_class) 

        #self.resUnet = Res_UNet_50(in_c, num_class) # in_c depends on input + or concatenated

        self.unet = UNet(5, num_class)


    def forward(self, x, occ):
        size = x.size()[2:]

        ################SEGFORMER + RESUNET50############################
        
        occ = occ.argmax(dim=1) 
        occ = torch.unsqueeze(occ, 1) 
        # Segformer used frozen from train.py


        layers = self.backbone(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4] #Encoder
        visible = self.decoResUnet50(x, x1, x2, x3, x4, x5)

        new_in = torch.cat((x,occ, visible), dim = 1) 
        complete = self.unet(new_in)


        #merged_mask = torch.logical_or(visible_mask, invisible_mask)
        #################################################################
        
        return visible, complete, x5



class selfAttention(nn.Module):
    def __init__(self, embed_dim, num_attention_heads, dropout):
        super(selfAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_attention_heads = num_attention_heads
        self.dropout =  dropout
        bias = True

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.self_attn = nn.MultiheadAttention(self.embed_dim, self.num_attention_heads, self.dropout)

    def forward(self, x):

        # Embedding?
        k = self.k_proj(x)
        v= self.v_proj(x)
        q = self.q_proj(x)

        out, self_attn_weights = self.self_attn(q,k,v)

        return out, self_attn_weights

class Mask2FormerAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Here, we add position embeddings to the queries and
    keys (as explained in the DETR paper).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        key_value_position_embeddings: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        hidden_states = hidden_states.permute(1, 0, 2) if hidden_states is not None else None
        position_embeddings = position_embeddings.permute(1, 0, 2) if position_embeddings is not None else None
        key_value_states = key_value_states.permute(1, 0, 2) if key_value_states is not None else None
        key_value_position_embeddings = (
            key_value_position_embeddings.permute(1, 0, 2) if key_value_position_embeddings is not None else None
        )

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        batch_size, target_len, embed_dim = hidden_states.size()

        # add position embeddings to the hidden states before projecting to queries and keys
        if position_embeddings is not None:
            hidden_states_original = hidden_states
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        # add key-value position embeddings to the key value states
        if key_value_position_embeddings is not None:
            key_value_states_original = key_value_states
            key_value_states = self.with_pos_embed(key_value_states, key_value_position_embeddings)

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.v_proj(key_value_states_original), -1, batch_size)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.v_proj(hidden_states_original), -1, batch_size)

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.size(1)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size * self.num_heads, target_len, source_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size * self.num_heads, target_len, source_len):
                raise ValueError(
                    f"Attention mask should be of size {(target_len, batch_size * self.num_heads, source_len)}, but is"
                    f" {attention_mask.size()}"
                )
            attn_weights += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_heads, target_len, source_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_heads, target_len, source_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (batch_size * self.num_heads, target_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, target_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(batch_size, self.num_heads, target_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)

        attn_output = self.out_proj(attn_output).permute(1, 0, 2)

        return attn_output, attn_weights_reshaped

class DetrMLPPredictionHead(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class attentionDecoder(nn.Module):
    def __init__(self, embed_dim, num_attention_heads, dropout, dim_feedforward):
        super(attentionDecoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_attention_heads = num_attention_heads
        self.dropout =  dropout
        self.dim_feedforward = dim_feedforward
        
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.num_attention_heads, self.dropout)
        
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        #self.self_attn = selfAttention(self.embed_dim, self.num_attention_heads, self.dropout)

        self.self_attn = Mask2FormerAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_attention_heads,
            dropout=self.dropout,
            is_decoder=True,
        )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.activation_fn = nn.ReLU()
        self.activation_dropout = dropout
        self.fc1 = nn.Linear(self.embed_dim, self.dim_feedforward)
        self.fc2 = nn.Linear(self.dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, hidden_states, query_pos_emb, features, feat_pos_emb, mask = None): #here hidden_states is the query
        
        output_attentions = True
        
        residual = hidden_states # [100, 3, 256]

        
        # Cross-Attention Block
        if mask is None:
            hidden_states, cross_attn_weights = self.cross_attn(
                query = self.with_pos_embed(hidden_states, query_pos_emb), #[100, 3, 256])
                key=self.with_pos_embed(features, feat_pos_emb), #([256, 3, 256])
                value=features, #([256, 3, 256])  Why pos emb is not applied here?
                key_padding_mask=None
                )
        else:
            hw, bs, emb = features.shape
            h = w = math.sqrt(hw)
            
            mask = mask.bool()
            mask = ~mask
            mask = mask.float()
            
            re_mask = nn.functional.interpolate(mask, size=int(h), mode="bilinear", align_corners=False)
            
            hidden_states, cross_attn_weights = self.cross_attn(
                query = self.with_pos_embed(hidden_states, query_pos_emb), #[100, 3, 256])
                key=self.with_pos_embed(features, feat_pos_emb), #([256, 3, 256])
                value=features, #([256, 3, 256])  Why pos emb is not applied here?
                key_padding_mask=re_mask.reshape(bs, hw)
                )


        #cross_attn_weights -> [100,3,256] query, batch, embedding
        
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        # Self Attention Block
        residual = hidden_states
        
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states, #[100, 3, 256]
            position_embeddings=query_pos_emb,
            attention_mask=None,
            output_attentions=True,
        )
        
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        
        
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


class modified_attentionDecoder(nn.Module):
    def __init__(self, embed_dim, num_attention_heads, dropout, dim_feedforward, hidden_dim):
        super(modified_attentionDecoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_attention_heads = num_attention_heads
        self.dropout =  dropout
        self.dim_feedforward = dim_feedforward
        self.hidden_dim = hidden_dim
        
        
        self.cross_attn = nn.MultiheadAttention(self.embed_dim, self.num_attention_heads, self.dropout)
        self.self_attn = nn.MultiheadAttention(self.embed_dim, self.num_attention_heads, self.dropout)
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        #self.self_attn = selfAttention(self.embed_dim, self.num_attention_heads, self.dropout)
        
        

        self.adapt_pos2d = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.activation_fn = nn.ReLU()
        self.activation_dropout = dropout
        self.fc1 = nn.Linear(self.embed_dim, self.dim_feedforward)
        self.fc2 = nn.Linear(self.dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def pos2posemb2d(self,pos, num_pos_feats=128, temperature=10000):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x), dim=-1)
        return posemb

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, reference_points, features, feat_pos_emb, mask, output_attentions = True): #here hidden_states is the query

        
       
        hw, bs, emb = features.shape
        h = w = math.sqrt(hw)
        h = int(h)
        w = int(w)

        
        if mask is not None : 
            mask = mask.bool()
            mask = ~mask
            mask = mask.float()
            
            re_mask = nn.functional.interpolate(mask, size=h, mode="bilinear", align_corners=False)
            #re_mask = re_mask.squeeze(1).bool()
            #pos_col, pos_row = mask2pos(re_mask)
            
            #pos_2d = torch.cat([pos_row.unsqueeze(1).repeat(1, h, 1).unsqueeze(-1), pos_col.unsqueeze(2).repeat(1, 1, w).unsqueeze(-1)],dim=-1)
            #posemb_2d = self.adapt_pos2d(self.pos2posemb2d(pos_2d))

        
        # Adapt reference points to pose embeddings 2D: ([6, 400, 2]) -> ([6, 400, 256])
        query_pos = self.adapt_pos2d(self.pos2posemb2d(reference_points))

        # Permute to correcto format
        tgt = tgt.transpose(0, 1)
        query_pos = query_pos.transpose(0, 1)
        
        # Create q and k: Sum Embedding and query pos embedding  ([400, 6, 256])
        q = k = self.with_pos_embed(tgt, query_pos)
        
        
        # Output ([400, 6, 256]) - weights: ([6, 400, 400])
        out1, self_attn_weights = self.self_attn( 
            query = q,      #([400, 6, 256])
            key=k,          #([400, 6, 256])
            value=tgt       #([400, 6, 256])         
        )

        out1 = nn.functional.dropout(out1, p=self.dropout, training=self.training)
        out1 = tgt + out1
        out1 = self.self_attn_layer_norm(out1)

        
        # Cross-Attention Block
        # Output ([400, 6, 256]) - weights: ([6, 400, 256]) ->feaure level1
        if mask is not None : 
            out2, cross_attn_weights = self.cross_attn(
                query = (out1 + query_pos), #([400, 6, 256])
                key= (features + feat_pos_emb), #), #([256, 6, 256]) ->feaure level1 
                value= features, #([256, 3, 256]) ->feaure level1 
                key_padding_mask=re_mask.reshape(bs, hw)
                )
        else:
            out2, cross_attn_weights = self.cross_attn(
                query = (out1 + query_pos), #([400, 6, 256])
                key= (features + feat_pos_emb), #), #([256, 6, 256]) ->feaure level1 
                value= features, #([256, 3, 256]) ->feaure level1 
                key_padding_mask=None
                )
        
        out2 = nn.functional.dropout(out2, p=self.dropout, training=self.training)
        out2 = out1 + out2
        out2 = self.cross_attn_layer_norm(out2)

        # Fully Connected
        
        hidden_states = self.activation_fn(self.fc1(out2))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = out2 + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,) #([400, 6, 256])
        
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)
        
        return outputs

class Mask2FormerSinePositionEmbedding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one used by the Attention is all you
    need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float] = None
    ):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        
        if mask is not None:
            mask = torch.nn.functional.interpolate(mask, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)
            mask = mask.squeeze(1).bool()
        else: 
            mask = torch.ones((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)

        
        #not_mask = ~mask
        not_mask = mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32) #([4, 128, 128])
        x_embed = not_mask.cumsum(2, dtype=torch.float32) #([4, 128, 128])
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device) #([128])
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats) #([128])
        
        pos_x = x_embed[:, :, :, None] / dim_t #([4, 128, 128, 128])
        pos_y = y_embed[:, :, :, None] / dim_t #([4, 128, 128, 128])
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3) #([4, 128, 128, 128])
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3) #([4, 128, 128, 128])
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

# Copied from transformers.models.maskformer.modeling_maskformer.PredictionBlock with MaskFormer->Mask2Former
class Mask2FormerPredictionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None:
        super().__init__()
        self.layers = [nn.Linear(in_dim, out_dim), activation]
        # Maintain submodule indexing as if part of a Sequential block
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state

class Mask2FormerMLPPredictionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3):
        """
        A classic Multi Layer Perceptron (MLP).

        Args:
            input_dim (`int`):
                The input dimensions.
            hidden_dim (`int`):
                The hidden dimensions.
            output_dim (`int`):
                The output dimensions.
            num_layers (int, *optional*, defaults to 3):
                The number of layers.
        """
        super().__init__()
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]

        self.layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            activation = nn.ReLU() if i < num_layers - 1 else nn.Identity()
            layer = Mask2FormerPredictionBlock(in_dim, out_dim, activation=activation)
            self.layers.append(layer)
            # Provide backwards compatibility from when the class inherited from nn.Sequential
            # In nn.Sequential subclasses, the name given to the layer is its index in the sequence.
            # In nn.Module subclasses they derived from the instance attribute they are assigned to e.g.
            # self.my_layer_name = Layer()
            # We can't give instance attributes integer names i.e. self.0 is not permitted and so need to register
            # explicitly
            self.add_module(str(i), layer)

    def forward(self, input: Tensor) -> Tensor:
        hidden_state = input
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class Mask2FormerMaskPredictor(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mask_feature_size: torch.Tensor):
        """
        This class is used to get the predicted mask for a given Mask2FormerMaskedAttentionDecoder layer. It also
        generates the binarized attention mask associated with the given predicted mask. The attention mask obtained
        using predicted mask of the (l-1)th decoder layer is fed to the cross(masked)-attention block of the next
        decoder layer as input.

        Args:
            hidden_size (`int`):
                The feature dimension of the Mask2FormerMaskedAttentionDecoder
            num_heads (`int`):
                The number of heads used in the Mask2FormerMaskedAttentionDecoder
            mask_feature_size: (`torch.Tensor`):
                one of the output dimensions of the predicted masks for each query
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.mask_embedder = Mask2FormerMLPPredictionHead(self.hidden_size, self.hidden_size, mask_feature_size)

    def forward(self, outputs: torch.Tensor, pixel_embeddings: torch.Tensor, attention_mask_target_size: int = None):

        mask_embeddings = self.mask_embedder(outputs.transpose(0, 1))
        
        # Sum up over the channels
        outputs_mask = torch.einsum("bqc,   bchw -> bqhw", mask_embeddings, pixel_embeddings)

        attention_mask = nn.functional.interpolate(outputs_mask, size=attention_mask_target_size, mode="bilinear", align_corners=False)
        
        
        

        attention_mask = attention_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        attention_mask = (attention_mask.flatten(0, 1) < 0.5).bool()
        attention_mask = attention_mask.detach()
        

        return outputs_mask, attention_mask

class winCompletionAtt(nn.Module):
    """
    
    """

    def __init__(self, in_c=3, num_class=4, pretrained_path=True): 
        super(winCompletionAtt, self).__init__()

        
        self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True, in_chans=in_c) 
        print('using: resnet50', ' pretrained: ', True )

        #self.fpn = torchvision.ops.FeaturePyramidNetwork(
        #    in_channels_list=[self.backbone[layer].shape[1] for layer in range(1, 5)],  
        #    out_channels=256
        #)
        num_queries = 256
        
        self.in_features= 256
        self.embed_dim = 256
        self.hidden_dim = 256
        self.num_attention_heads = 8
        self.dropout =  0.0
        self.dim_feedforward = 2048

        self.queries_embedder = nn.Embedding(num_queries, self.hidden_dim) 
        self.queries_features = nn.Embedding(num_queries, self.hidden_dim)

        self.pos_embedder = Mask2FormerSinePositionEmbedding(num_pos_feats=self.hidden_dim // 2, normalize=True)
        self.input_projection_5 = nn.Conv2d(2048, self.hidden_dim, kernel_size=1)
        self.output_projection_5 = nn.Conv2d(self.hidden_dim, 2048, kernel_size=1)

        self.attDecoder = attentionDecoder(self.embed_dim, self.num_attention_heads, self.dropout, self.dim_feedforward)
        

        self.decoResUnet50 = deco_Res_UNet_50(in_c, num_class) 

        #self.resUnet = Res_UNet_50(in_c, num_class) # in_c depends on input + or concatenated

       

    def forward(self, x):
        size = x.size()[2:]
        
        # Encoder
        layers = self.backbone(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4] #Encoder
        #x1 - [3, 64, 256, 256]
        #x2 - 3, 256, 128, 128]
        #x3- [3, 512, 64, 64])
        #x4- [3, 1024, 32, 32])
        #x5 - [3, 2048, 16, 16])

        
        batch_size = x1.shape[0]

        # Multi-stage positional embeddings
        #feat_2, feat_3, feat_4, feat_5 = self.fpn([x2, x3, x4, x5])
        #feat_pos_emb2 = self.pos_embedder(feat_2)
        #feat_pos_emb3 = self.pos_embedder(feat_3)
        #feat_pos_emb4 = self.pos_embedder(feat_4)
        feat_5 = x5
        
        feat_pos_emb5 = self.pos_embedder(feat_5, None).flatten(2)
        #Only for the last one? TODO: check if other hidden features are also like this...
        feat_pos_emb5 = feat_pos_emb5.permute(2, 0, 1) #[256, 3, 256])
        
        feat_5 = self.input_projection_5(feat_5).flatten(2)
        feat_5 = feat_5.flatten(2).permute(2, 0, 1) #([256, 3, 2048]
        
        
        # Query embedding
        query_emb = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1) #[100, 3, 256]
        query_feat = self.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1) #[100, 3, 256]
        
        # Output: hidden_state [100, 3, 256]) -  self_attn_weights [3, 8, 100, 100] - cross_attn_weights [3, 100, 256]
        #atD2 = self.attDecoder(query_feat, query_emb, feat_2, feat_pos_emb2)
        #atD3 = self.attDecoder(query_feat, query_emb, feat_3, feat_pos_emb3)
        #atD4 = self.attDecoder(query_feat, query_emb, feat_4, feat_pos_emb4)
        atD5 = self.attDecoder(query_feat, query_emb, feat_5, feat_pos_emb5)

        #################################

        atD5_2 = self.attDecoder(atD5[0], query_emb, feat_5, feat_pos_emb5)
        atD5_3 = self.attDecoder(atD5_2[0], query_emb, feat_5, feat_pos_emb5)
        atD5_4 = self.attDecoder(atD5_3[0], query_emb, feat_5, feat_pos_emb5)
        atD5_5 = self.attDecoder(atD5_4[0], query_emb, feat_5, feat_pos_emb5)

        #################################

        natD5 = atD5_5[0].permute(1,2,0)
        natD5 = self.output_projection_5(natD5.reshape(x5.shape[0],self.hidden_dim,x5.shape[2],x5.shape[3]))
        
        #atx5 = x5 + natD5
        atx5 = natD5

        out = self.decoResUnet50(x, x1, x2, x3, x4, atx5)

        return out        

class winMask2Former(nn.Module):
    """
    DONT MODIFY THIS, THIS IS THE ONE WITH THE BETTER RESUTLS SO FAR
    CROSS-ATT -> SELF-ATT 
    LEARNED QUERY
    """

    def __init__(self, in_c=3, num_class=4, pretrained_path=True): 
        super(winMask2Former, self).__init__()

        
        self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True, in_chans=in_c) 
        print('using: resnet50', ' pretrained: ', True )
        
        l1= 64
        l2= 256
        l3= 512
        l4= 1024
        l5 = 2048

        
        num_queries = 256
        self.mask_feature_size= 256
        self.in_features= 256
        self.embed_dim = 256
        self.hidden_dim = 256
        self.num_attention_heads = 8
        self.dropout =  0.0
        self.dim_feedforward = 2048
        self.target_sizes = (512, 512)

        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=[l2, l3, l4, l5],  
            out_channels=self.hidden_dim
        )

        self.queries_embedder = nn.Embedding(num_queries, self.hidden_dim) 
        self.queries_features = nn.Embedding(num_queries, self.hidden_dim)

        self.pos_embedder = Mask2FormerSinePositionEmbedding(num_pos_feats=self.hidden_dim // 2, normalize=True)
        
        self.input_projection_2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.input_projection_3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.input_projection_4 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.input_projection_5 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)


        self.output_projection_2 = nn.Conv2d(self.hidden_dim, l2, kernel_size=1)
        self.output_projection_3 = nn.Conv2d(self.hidden_dim, l3, kernel_size=1)
        self.output_projection_4 = nn.Conv2d(self.hidden_dim, l4, kernel_size=1)
        self.output_projection_5 = nn.Conv2d(self.hidden_dim, l5, kernel_size=1)

        self.attDecoder = attentionDecoder(self.embed_dim, self.num_attention_heads, self.dropout, self.dim_feedforward)
        

        self.mask_predictor = Mask2FormerMaskPredictor(
            hidden_size=self.hidden_dim,
            num_heads=self.num_attention_heads,
            mask_feature_size=self.mask_feature_size,
        )
        self.layernorm = nn.LayerNorm(self.hidden_dim)
        self.class_predictor = nn.Linear(self.hidden_dim, num_class)

        self.convos = DoubleConv(256,64) #new
        self.outc = OutConv(64, num_class) #new

        #self.adapt_pos2d = nn.Sequential(
        #    nn.Linear(self.hidden_dim, self.hidden_dim),
        #    nn.ReLU(),
        #    nn.Linear(self.hidden_dim, self.hidden_dim),
        #)


        self.decoResUnet50 = deco_Res_UNet_50(in_c, num_class) 

        #self.resUnet = Res_UNet_50(in_c, num_class) # in_c depends on input + or concatenated

       
    def grid_points(self, batch_size):
        num_position = 256
        num_pattern = 1

        nx=ny=round(math.sqrt(num_position))
        num_position=nx*ny
        x = (torch.arange(nx) + 0.5) / nx
        y = (torch.arange(ny) + 0.5) / ny
        xy=torch.meshgrid(x,y)
        reference_points=torch.cat([xy[0].reshape(-1)[...,None],xy[1].reshape(-1)[...,None]],-1).cuda()
        #reference_points = reference_points.unsqueeze(0).repeat(batch_size, num_pattern, 1)
        return reference_points
    
    def pos2posemb2d(self,pos, num_pos_feats=128, temperature=10000):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x), dim=-1)
        return posemb

    def forward(self, x, mask = None):
        size = x.size()[2:]
        
        # Encoder
        layers = self.backbone(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4] #Encoder
        #x1 - [3, 64, 256, 256]
        #x2 - 3, 256, 128, 128]
        #x3- [3, 512, 64, 64])
        #x4- [3, 1024, 32, 32])
        #x5 - [3, 2048, 16, 16])

        
        batch_size = x1.shape[0]

        # Multi-stage positional embeddings
        x_feat =OrderedDict()
        x_feat['l2'] = x2
        x_feat['l3'] = x3
        x_feat['l4'] = x4
        x_feat['l5'] = x5

        fpn_output = self.fpn(x_feat)

        feat_pos_emb3 = self.pos_embedder(fpn_output['l3'], None).flatten(2)
        feat_pos_emb4 = self.pos_embedder(fpn_output['l4'], None).flatten(2)        
        feat_pos_emb5 = self.pos_embedder(fpn_output['l5'], None).flatten(2)
        

         #Only for the last one? TODO: check if other hidden features are also like this...
        feat_pos_emb3 = feat_pos_emb3.permute(2, 0, 1) #[256, 3, 256])
        feat_pos_emb4 = feat_pos_emb4.permute(2, 0, 1) #[256, 3, 256])
        feat_pos_emb5 = feat_pos_emb5.permute(2, 0, 1) #[256, 3, 256])
        
        feat_3 = self.input_projection_3(fpn_output['l3']).flatten(2)
        feat_3 = feat_3.flatten(2).permute(2, 0, 1) #([4096, 6, 256])

        feat_4 = self.input_projection_4(fpn_output['l4']).flatten(2)
        feat_4 = feat_4.flatten(2).permute(2, 0, 1) #[1024, 6, 256])

        feat_5 = self.input_projection_5(fpn_output['l5']).flatten(2)
        feat_5 = feat_5.flatten(2).permute(2, 0, 1) #([256, 6, 2048]

        
        
        # Learned Query embedding
        query_emb = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1) #[100, 3, 256]
        query_feat = self.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1) #[100, 3, 256]

        # Anchor Query embedding
        #reference_points = self.grid_points(batch_size).unsqueeze(1).repeat(1, batch_size, 1)
        
        #query_pos = self.adapt_pos2d(self.pos2posemb2d(reference_points))
        
        # Output: hidden_state [100, 3, 256]) -  self_attn_weights [3, 8, 100, 100] - cross_attn_weights [3, 100, 256]

        atD5 = self.attDecoder(query_feat, query_emb, feat_5, feat_pos_emb5)
        atD4 = self.attDecoder(atD5[0], query_emb, feat_4, feat_pos_emb4)
        atD3 = self.attDecoder(atD4[0], query_emb, feat_3, feat_pos_emb3)

        #atD5 = self.attDecoder(query_feat, query_pos, feat_5, feat_pos_emb5)
        #atD4 = self.attDecoder(atD5[0], query_pos, feat_4, feat_pos_emb4)
        #atD3 = self.attDecoder(atD4[0], query_pos, feat_3, feat_pos_emb3)

        
        intermediate_hidden_states = self.layernorm(atD3[0])

        
        
        predicted_mask, attention_mask = self.mask_predictor( #There is a predicted mask for each image in the batch and for each query...
                    intermediate_hidden_states,
                    fpn_output['l2'], #([3, 256, 128, 128])
                    x3.shape[-2:], #[torch.Size([64, 64])]

                )

        out = self.convos(predicted_mask)  #new. 3,256,128,128 -> 3,1,128,128
        out = self.outc(out) #new
        out = torch.nn.functional.interpolate(out, size=self.target_sizes, mode="bilinear", align_corners=False)

        
        #class_prediction = self.class_predictor(intermediate_hidden_states.transpose(0, 1))
        # predicted_mask: ([3, 256, 128, 128]) -> batch, num_query, height, width
        # class_prediction: ([3, 256, 1]) -> batch, num_query, num_classes (I didnt't add +1)

        #masks_queries_logits = torch.nn.functional.interpolate(
        #    predicted_mask, size=self.target_sizes, mode="bilinear", align_corners=False
        #)
        #masks_probs = masks_queries_logits.sigmoid()

        #masks_classes = class_prediction.softmax(dim=-1)
        #segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs)
        

        return out, {}#, atD3    

class winMask2Former_newVersion(nn.Module):
    """
    
    """

    def __init__(self, in_c=3, num_class=4, pretrained_path=True): 
        super(winMask2Former_newVersion, self).__init__()
        
        self.mode = 'learned' #'centers-learned' # # #'learned' # 'anchor-fixed'
        self.noAtt = False
        self.singleLayer = False
        

        
        self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True, in_chans=in_c) 
        print('using: resnet50', ' pretrained: ', True )
        
        l1= 64
        l2= 256
        l3= 512
        l4= 1024
        l5 = 2048

        
        self.num_queries = 256#100#
        self.mask_feature_size= 256
        self.hidden_dim = 256
        self.num_attention_heads = 8
        self.dropout =  0.0
        self.dim_feedforward = 2048
        self.target_sizes = (512, 512)

        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=[l2, l3, l4, l5],  
            out_channels=self.hidden_dim
        )

        if self.mode in {'fixed-anchor', 'centers-learned'}:
            # adapt_pos2s used for anchor Q
            self.adapt_pos2d = nn.Sequential( nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim),)

        
        if self.mode == 'centers-learned':
            self.queries_center_embedder = nn.Embedding(self.num_queries, 2) 
            
            # For bbox detection
            self.bbox_predictor = DetrMLPPredictionHead(input_dim=self.hidden_dim, hidden_dim=self.hidden_dim, output_dim=4, num_layers=3)
            self.class_predictor = nn.Linear(self.hidden_dim, num_class) 

            self.project = nn.Linear(2, self.hidden_dim) 
        
        self.queries_embedder = nn.Embedding(self.num_queries, self.hidden_dim) 
        self.queries_features = nn.Embedding(self.num_queries, self.hidden_dim)


        self.pos_embedder = Mask2FormerSinePositionEmbedding(num_pos_feats=self.hidden_dim // 2, normalize=True)
        
        self.input_projection_2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.input_projection_3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.input_projection_4 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.input_projection_5 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)


        self.output_projection_2 = nn.Conv2d(self.hidden_dim, l2, kernel_size=1)
        self.output_projection_3 = nn.Conv2d(self.hidden_dim, l3, kernel_size=1)
        self.output_projection_4 = nn.Conv2d(self.hidden_dim, l4, kernel_size=1)
        self.output_projection_5 = nn.Conv2d(self.hidden_dim, l5, kernel_size=1)

        self.attDecoder = attentionDecoder(self.hidden_dim, self.num_attention_heads, self.dropout, self.dim_feedforward)
        

        self.mask_predictor = Mask2FormerMaskPredictor(
            hidden_size=self.hidden_dim,
            num_heads=self.num_attention_heads,
            mask_feature_size=self.mask_feature_size,
        )
        self.layernorm = nn.LayerNorm(self.hidden_dim)
        self.class_predictor = nn.Linear(self.hidden_dim, num_class)

        self.convos = DoubleConv(self.num_queries,64) #new
        self.outc = OutConv(64, num_class) #new

        

        

        self.decoResUnet50 = deco_Res_UNet_50(in_c, num_class) 

        

       
    def grid_points(self, batch_size, num_position, num_pattern=1):
        
        nx=ny=round(math.sqrt(num_position))
        num_position=nx*ny
        x = (torch.arange(nx) + 0.5) / nx
        y = (torch.arange(ny) + 0.5) / ny
        xy=torch.meshgrid(x,y)
        reference_points=torch.cat([xy[0].reshape(-1)[...,None],xy[1].reshape(-1)[...,None]],-1).cuda()
        #reference_points = reference_points.unsqueeze(0).repeat(batch_size, num_pattern, 1)
        return reference_points
    
    def pos2posemb2d(self,pos, num_pos_feats=128, temperature=10000):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat((pos_y, pos_x), dim=-1)
        return posemb

    def forward(self, x, mask = None):
        size = x.size()[2:]
        outputs={}
        
        # Encoder
        layers = self.backbone(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4] #Encoder
        #x1 - [3, 64, 256, 256]
        #x2 - 3, 256, 128, 128]
        #x3- [3, 512, 64, 64])
        #x4- [3, 1024, 32, 32])
        #x5 - [3, 2048, 16, 16])

        
        batch_size = x1.shape[0]

        # Multi-stage positional embeddings
        x_feat =OrderedDict()
        x_feat['l2'] = x2
        x_feat['l3'] = x3
        x_feat['l4'] = x4
        x_feat['l5'] = x5

        fpn_output = self.fpn(x_feat)
        if self.noAtt:
            
            predicted_mask = fpn_output['l2']
            out = self.convos(predicted_mask)  #new. 3,256,128,128 -> 3,1,128,128
            out = self.outc(out) #new
            out = torch.nn.functional.interpolate(out, size=self.target_sizes, mode="bilinear", align_corners=False)
            return out, {}


        feat_pos_emb3 = self.pos_embedder(fpn_output['l3'], mask).flatten(2)
        feat_pos_emb4 = self.pos_embedder(fpn_output['l4'], mask).flatten(2)        
        feat_pos_emb5 = self.pos_embedder(fpn_output['l5'], mask).flatten(2)
        

        feat_pos_emb3 = feat_pos_emb3.permute(2, 0, 1) #[256, 3, 256])
        feat_pos_emb4 = feat_pos_emb4.permute(2, 0, 1) #[256, 3, 256])
        feat_pos_emb5 = feat_pos_emb5.permute(2, 0, 1) #[256, 3, 256])
        
        feat_3 = self.input_projection_3(fpn_output['l3']).flatten(2)
        feat_3 = feat_3.flatten(2).permute(2, 0, 1) #([4096, 6, 256])

        feat_4 = self.input_projection_4(fpn_output['l4']).flatten(2)
        feat_4 = feat_4.flatten(2).permute(2, 0, 1) #[1024, 6, 256])

        feat_5 = self.input_projection_5(fpn_output['l5']).flatten(2)
        feat_5 = feat_5.flatten(2).permute(2, 0, 1) #([256, 6, 2048]

        if self.mode == 'learned':
            # Learned Query embedding
            query_emb = self.queries_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1) #[256, 3, 256]
            query_feat = self.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1) #[256, 3, 256]            
        
        elif self.mode == 'anchor-fixed':            
            # Anchor Query embedding
            reference_points = self.grid_points(batch_size, self.num_queries).unsqueeze(1).repeat(1, batch_size, 1) #([256, 4, 2])
            query_emb = self.adapt_pos2d(self.pos2posemb2d(reference_points)) #([256, 4, 256])            
            query_feat = self.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1) #([256, 4, 256]) 
        
        elif self.mode == 'centers-learned':            
            # Anchor Query embedding
            reference_points = self.queries_center_embedder.weight.unsqueeze(1).repeat(1, batch_size, 1) #([256, 4, 2])
            query_emb = self.adapt_pos2d(self.pos2posemb2d(reference_points)) #([256, 4, 256])            
            query_feat = self.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1) #([256, 4, 256])             
        
            
            atD5 = self.attDecoder(query_feat, query_emb, feat_5, feat_pos_emb5, mask)
            pred_boxes = self.bbox_predictor(atD5[0]).sigmoid() #400, 4, 4 -> in each image, each query predicts one BBox
            em_pred_boxes = self.project(pred_boxes[:,:,0:2]) #400,4,2 -> 400,4,256
            
            
            atD4 = self.attDecoder(em_pred_boxes, query_emb, feat_4, feat_pos_emb4, mask)
            pred_boxes = self.bbox_predictor(atD4[0]).sigmoid() #400, 4, 4 -> in each image, each query predicts one BBox
            em_pred_boxes = self.project(pred_boxes[:,:,0:2]) #400,4,2 -> 400,4,256

            atD3 = self.attDecoder(em_pred_boxes, query_emb, feat_3, feat_pos_emb3, mask)
            pred_boxes = self.bbox_predictor(atD3[0]).sigmoid() #400, 4, 4 -> in each image, each query predicts one BBox

            class_prediction = self.class_predictor(atD3[0].transpose(0, 1)) #4, 400, 1

                        
            outputs["logits"] = class_prediction
            outputs["pred_boxes"] = pred_boxes

            intermediate_hidden_states = self.layernorm(atD3[0])

            predicted_mask, attention_mask = self.mask_predictor( #There is a predicted mask for each image in the batch and for each query...
                        intermediate_hidden_states,
                        fpn_output['l2'], #([3, 256, 128, 128])
                        x3.shape[-2:], #[torch.Size([64, 64])]

                    )

            out = self.convos(predicted_mask)  #new. 3,256,128,128 -> 3,1,128,128
            out = self.outc(out) #new
            out = torch.nn.functional.interpolate(out, size=self.target_sizes, mode="bilinear", align_corners=False)
            return out, outputs

        
        if self.singleLayer:
            # Output: hidden_state [256, 3, 256]) -  self_attn_weights [3, 8, 256, 256] - cross_attn_weights [3, 256, 256]
            atD5 = self.attDecoder(query_feat, query_emb, feat_5, feat_pos_emb5, mask)
            atD4 = self.attDecoder(atD5[0], query_emb, feat_4, feat_pos_emb4, mask)
            atD3 = self.attDecoder(atD4[0], query_emb, feat_3, feat_pos_emb3, mask)
        else:
                atD5 = self.attDecoder(query_feat, query_emb, feat_5, feat_pos_emb5, mask)
                atD4 = self.attDecoder(atD5[0], query_emb, feat_4, feat_pos_emb4, mask)
                atD3 = self.attDecoder(atD4[0], query_emb, feat_3, feat_pos_emb3, mask)

                atD5 = self.attDecoder(atD3[0], query_emb, feat_5, feat_pos_emb5, mask)
                atD4 = self.attDecoder(atD5[0], query_emb, feat_4, feat_pos_emb4, mask)
                atD3 = self.attDecoder(atD4[0], query_emb, feat_3, feat_pos_emb3, mask)

                atD5 = self.attDecoder(atD3[0], query_emb, feat_5, feat_pos_emb5, mask)
                atD4 = self.attDecoder(atD5[0], query_emb, feat_4, feat_pos_emb4, mask)
                atD3 = self.attDecoder(atD4[0], query_emb, feat_3, feat_pos_emb3, mask)

                atD5 = self.attDecoder(atD3[0], query_emb, feat_5, feat_pos_emb5, mask)
                atD4 = self.attDecoder(atD5[0], query_emb, feat_4, feat_pos_emb4, mask)
                atD3 = self.attDecoder(atD4[0], query_emb, feat_3, feat_pos_emb3, mask)

                atD5 = self.attDecoder(atD3[0], query_emb, feat_5, feat_pos_emb5, mask)
                atD4 = self.attDecoder(atD5[0], query_emb, feat_4, feat_pos_emb4, mask)
                atD3 = self.attDecoder(atD4[0], query_emb, feat_3, feat_pos_emb3, mask)

                atD5 = self.attDecoder(atD3[0], query_emb, feat_5, feat_pos_emb5, mask)
                atD4 = self.attDecoder(atD5[0], query_emb, feat_4, feat_pos_emb4, mask)
                atD3 = self.attDecoder(atD4[0], query_emb, feat_3, feat_pos_emb3, mask)

                atD5 = self.attDecoder(atD3[0], query_emb, feat_5, feat_pos_emb5, mask)
                atD4 = self.attDecoder(atD5[0], query_emb, feat_4, feat_pos_emb4, mask)
                atD3 = self.attDecoder(atD4[0], query_emb, feat_3, feat_pos_emb3, mask)

                atD5 = self.attDecoder(atD3[0], query_emb, feat_5, feat_pos_emb5, mask)
                atD4 = self.attDecoder(atD5[0], query_emb, feat_4, feat_pos_emb4, mask)
                atD3 = self.attDecoder(atD4[0], query_emb, feat_3, feat_pos_emb3, mask)

                atD5 = self.attDecoder(atD3[0], query_emb, feat_5, feat_pos_emb5, mask)
                atD4 = self.attDecoder(atD5[0], query_emb, feat_4, feat_pos_emb4, mask)
                atD3 = self.attDecoder(atD4[0], query_emb, feat_3, feat_pos_emb3, mask)

        
        intermediate_hidden_states = self.layernorm(atD3[0])

        
        predicted_mask, attention_mask = self.mask_predictor( #There is a predicted mask for each image in the batch and for each query...
                    intermediate_hidden_states,
                    fpn_output['l2'], #([3, 256, 128, 128])
                    x3.shape[-2:], #[torch.Size([64, 64])]

                )

        out = self.convos(predicted_mask)  #new. 3,256,128,128 -> 3,1,128,128
        out = self.outc(out) #new
        out = torch.nn.functional.interpolate(out, size=self.target_sizes, mode="bilinear", align_corners=False)

        
        return out, outputs   

    
class winMask2Former_this_one(nn.Module):
    """
    
    """

    def __init__(self, in_c=3, num_class=4, pretrained_path=True): 
        super(winMask2Former_this_one, self).__init__()

        
        self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True, in_chans=in_c) 
        print('using: resnet50', ' pretrained: ', True )
        
        l1= 64
        l2= 256
        l3= 512
        l4= 1024
        l5 = 2048

        
        num_queries = 256
        self.mask_feature_size= 256
        self.in_features= 256
        self.embed_dim = 256
        self.hidden_dim = 256
        self.num_attention_heads = 8
        self.dropout =  0.0
        self.dim_feedforward = 2048
        self.target_sizes = (512, 512)

        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=[l2, l3, l4, l5],  
            out_channels=self.hidden_dim
        )

        self.queries_embedder = nn.Embedding(num_queries, self.hidden_dim) 
        self.queries_features = nn.Embedding(num_queries, self.hidden_dim)

        self.pos_embedder = Mask2FormerSinePositionEmbedding(num_pos_feats=self.hidden_dim // 2, normalize=True)
        
        self.input_projection_2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.input_projection_3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.input_projection_4 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.input_projection_5 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)


        self.output_projection_2 = nn.Conv2d(self.hidden_dim, l2, kernel_size=1)
        self.output_projection_3 = nn.Conv2d(self.hidden_dim, l3, kernel_size=1)
        self.output_projection_4 = nn.Conv2d(self.hidden_dim, l4, kernel_size=1)
        self.output_projection_5 = nn.Conv2d(self.hidden_dim, l5, kernel_size=1)

        self.attDecoder = attentionDecoder(self.embed_dim, self.num_attention_heads, self.dropout, self.dim_feedforward)
        self.mod_attDecoder = modified_attentionDecoder(self.embed_dim, self.num_attention_heads, self.dropout, self.dim_feedforward, self.hidden_dim)

        self.occ_att_l4_dim = 256
        self.occ_att_l3_dim = 1024

        self.num_position = 400
        self.num_pattern = 1
        # For anchor query: self.embed_dim
        # For occlusion att maps: self.occ_att_l4_dim
        self.pattern = nn.Embedding(self.num_pattern, self.embed_dim)

        

        self.mask_predictor = Mask2FormerMaskPredictor(
            hidden_size=self.hidden_dim,
            num_heads=self.num_attention_heads,
            mask_feature_size=self.mask_feature_size,
        )
        self.layernorm = nn.LayerNorm(self.hidden_dim)
        self.class_predictor = nn.Linear(self.hidden_dim, num_class) #TODO: add the null class?

        self.convos = DoubleConv(self.num_position,self.embed_dim) #new
        self.outc = OutConv(self.embed_dim, num_class) #new

        self.adapt_pos2d = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        


        self.decoResUnet50 = deco_Res_UNet_50(in_c, num_class) 

        #self.resUnet = Res_UNet_50(in_c, num_class) # in_c depends on input + or concatenated

       
    def grid_points(self, batch_size, num_position, num_pattern):

        nx=ny=round(math.sqrt(num_position))
        num_position=nx*ny
        x = (torch.arange(nx) + 0.5) / nx
        y = (torch.arange(ny) + 0.5) / ny
        xy=torch.meshgrid(x,y)
        reference_points=torch.cat([xy[0].reshape(-1)[...,None],xy[1].reshape(-1)[...,None]],-1).cuda()
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, num_pattern, 1)
        return reference_points
    
    
    

    def forward(self, x, occ_mask=None):
        size = x.size()[2:]

        
        
        # Encoder
        layers = self.backbone(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4] #Encoder
        #x1 - [3, 64, 256, 256]
        #x2 - 3, 256, 128, 128]
        #x3- [3, 512, 64, 64])
        #x4- [3, 1024, 32, 32])
        #x5 - [3, 2048, 16, 16])

        
        batch_size = x1.shape[0]

        # Multi-stage feature maps
        x_feat =OrderedDict()
        x_feat['l2'] = x2
        x_feat['l3'] = x3
        x_feat['l4'] = x4
        x_feat['l5'] = x5

        # Output of FPN already has D_emb
        fpn_output = self.fpn(x_feat)
        c = fpn_output['l3'].shape[1]

        # Embedding for feature maps from encoder -> batch, D_emb, H*W
        
        feat_pos_emb2 = self.pos_embedder(fpn_output['l2'], None).flatten(2) #([6, 256, 16384])
        feat_pos_emb3 = self.pos_embedder(fpn_output['l3'], None).flatten(2) #([6, 256, 4096])
        feat_pos_emb4 = self.pos_embedder(fpn_output['l4'], None).flatten(2) #([6, 256, 1024])      
        feat_pos_emb5 = self.pos_embedder(fpn_output['l5'], None).flatten(2) #([6, 256, 256])
        
        
        # Permute feature embeddings to correct format -> H*W, batch, D_emb
        feat_pos_emb2 = feat_pos_emb2.permute(2, 0, 1) #([16384, 6, 256])
        feat_pos_emb3 = feat_pos_emb3.permute(2, 0, 1) #([4096, 6, 256])
        feat_pos_emb4 = feat_pos_emb4.permute(2, 0, 1) #([1024, 6, 256]) 
        feat_pos_emb5 = feat_pos_emb5.permute(2, 0, 1) #([256, 6, 256])
        
        # Permute feature maps to correct format -> H*W, batch, D_emb 
        feat_2 = fpn_output['l2'].flatten(2).permute(2, 0, 1) #[?, 6, 256])     
        feat_3 = fpn_output['l3'].flatten(2).permute(2, 0, 1) #[4096, 6, 256])      
        feat_4 = fpn_output['l4'].flatten(2).permute(2, 0, 1) #[1024, 6, 256])
        feat_5 = fpn_output['l5'].flatten(2).permute(2, 0, 1) #([256, 6, 256]
         
        # Anchor Query -> batch, anchorPoints*numPatterns, 2 (coordinates) : ([6, 289, 2])
        reference_points = self.grid_points(batch_size, self.num_position, self.num_pattern)

        # Occlusion map query
        #occ_att_l4 = occ_attn[-3:] # batch, 5, 1024, 256
        #occ_att_l4 = occ_attn[7:12] # batch, 8, 256, 256
        #tmp1 = torch.cat(occ_att_l4, dim=1)
        #reference_points = tmp1.mean(1) # Heads mean -> batch, 256, 256
        

        # Embedding -> batch, numPositions, D_emb ([6, 300, 256])
        # For anchor query
        tgt = self.pattern.weight.reshape(1, self.num_pattern, 1, c).repeat(batch_size, 1, self.num_position, 1).reshape(
            batch_size, self.num_pattern * self.num_position, c)
        
        # For occ att map query
        #tgt = self.pattern.weight.reshape(1, self.num_pattern, 1, c).repeat(batch_size, 1, self.occ_att_l4_dim, 1).reshape(
        #    batch_size, self.num_pattern * self.occ_att_l4_dim, c)
        
        
        # Output: hidden_state [400, 6, 256]) 
        # Layer 1
        atD5 = self.mod_attDecoder(tgt, reference_points, feat_5, feat_pos_emb5, occ_mask)
        atD4 = self.mod_attDecoder(atD5[0].transpose(0,1), reference_points, feat_4, feat_pos_emb4, occ_mask)
        atD3 = self.mod_attDecoder(atD4[0].transpose(0,1), reference_points, feat_3, feat_pos_emb3, occ_mask)
        ###atD2 = self.mod_attDecoder(atD3[0].transpose(0,1), reference_points, feat_2, feat_pos_emb2)

        # Layer2
        #atD5 = self.mod_attDecoder(atD3[0].transpose(0,1), reference_points, feat_5, feat_pos_emb5)
        #atD4 = self.mod_attDecoder(atD5[0].transpose(0,1), reference_points, feat_4, feat_pos_emb4)
        #atD3 = self.mod_attDecoder(atD4[0].transpose(0,1), reference_points, feat_3, feat_pos_emb3)

        # Layer 3
        #atD5 = self.mod_attDecoder(atD3[0].transpose(0,1), reference_points, feat_5, feat_pos_emb5)
        #atD4 = self.mod_attDecoder(atD5[0].transpose(0,1), reference_points, feat_4, feat_pos_emb4)
        #atD3 = self.mod_attDecoder(atD4[0].transpose(0,1), reference_points, feat_3, feat_pos_emb3)

        # Layer 4
        #atD5 = self.mod_attDecoder(atD3[0].transpose(0,1), reference_points, feat_5, feat_pos_emb5)
        #atD4 = self.mod_attDecoder(atD5[0].transpose(0,1), reference_points, feat_4, feat_pos_emb4)
        #atD3 = self.mod_attDecoder(atD4[0].transpose(0,1), reference_points, feat_3, feat_pos_emb3)
        



        hidden_states = self.layernorm(atD3[0])
        
        # Create predicted mask -> ([6, 400, 128, 128])
        predicted_mask, attention_mask = self.mask_predictor( #There is a predicted mask for each image in the batch and for each query...
                    hidden_states,
                    fpn_output['l2'], #([3, 256, 256, 256])
                    x3.shape[-2:], #[torch.Size([128, 128])]
                )

                        
        out = self.convos(predicted_mask)  #new. 3,256,128,128 -> 3,1,128,128
        out = self.outc(out) #new
        out = torch.nn.functional.interpolate(out, size=self.target_sizes, mode="bilinear", align_corners=False)
        
        outputs={}
        
        return out, outputs   


class winMask2Former_DETR(nn.Module):
    """
    This is the version that computes bbox and classes as well....
    """

    def __init__(self, in_c=3, num_class=4, pretrained_path=True): 
        super(winMask2Former_DETR, self).__init__()

        
        self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True, in_chans=in_c) 
        print('using: resnet50', ' pretrained: ', True )
        
        l1= 64
        l2= 256
        l3= 512
        l4= 1024
        l5 = 2048

        
        num_queries = 256
        self.mask_feature_size= 256
        self.in_features= 256
        self.embed_dim = 256
        self.hidden_dim = 256
        self.num_attention_heads = 8
        self.dropout =  0.0
        self.dim_feedforward = 2048
        self.target_sizes = (512, 512)

        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=[l1, l2, l3, l4, l5],  
            out_channels=self.hidden_dim
        )

        self.queries_embedder = nn.Embedding(num_queries, self.hidden_dim) 
        self.queries_features = nn.Embedding(num_queries, self.hidden_dim)

        self.pos_embedder = Mask2FormerSinePositionEmbedding(num_pos_feats=self.hidden_dim // 2, normalize=True)
        
        self.input_projection_2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.input_projection_3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.input_projection_4 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.input_projection_5 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)


        self.output_projection_2 = nn.Conv2d(self.hidden_dim, l2, kernel_size=1)
        self.output_projection_3 = nn.Conv2d(self.hidden_dim, l3, kernel_size=1)
        self.output_projection_4 = nn.Conv2d(self.hidden_dim, l4, kernel_size=1)
        self.output_projection_5 = nn.Conv2d(self.hidden_dim, l5, kernel_size=1)

        self.attDecoder = attentionDecoder(self.embed_dim, self.num_attention_heads, self.dropout, self.dim_feedforward)
        self.mod_attDecoder = modified_attentionDecoder(self.embed_dim, self.num_attention_heads, self.dropout, self.dim_feedforward, self.hidden_dim)

        self.bbox_predictor = DetrMLPPredictionHead(input_dim=self.embed_dim, hidden_dim=self.embed_dim, output_dim=4, num_layers=3)

        self.occ_att_l4_dim = 256
        self.occ_att_l3_dim = 1024

        self.num_position = 400
        self.num_pattern = 1
        # For anchor query: self.embed_dim
        # For occlusion att maps: self.occ_att_l4_dim
        self.pattern = nn.Embedding(self.num_pattern, self.embed_dim)

        

        self.mask_predictor = Mask2FormerMaskPredictor(
            hidden_size=self.hidden_dim,
            num_heads=self.num_attention_heads,
            mask_feature_size=self.mask_feature_size,
        )
        self.layernorm = nn.LayerNorm(self.hidden_dim)
        self.class_predictor = nn.Linear(self.hidden_dim, num_class) #TODO: add the null class?

        self.convos = DoubleConv(self.num_position,self.embed_dim) #new
        self.outc = OutConv(self.embed_dim, num_class) #new

        self.adapt_pos2d = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        


        self.decoResUnet50 = deco_Res_UNet_50(in_c, num_class) 

        #self.resUnet = Res_UNet_50(in_c, num_class) # in_c depends on input + or concatenated

       
    def grid_points(self, batch_size, num_position, num_pattern):

        nx=ny=round(math.sqrt(num_position))
        num_position=nx*ny
        x = (torch.arange(nx) + 0.5) / nx
        y = (torch.arange(ny) + 0.5) / ny
        xy=torch.meshgrid(x,y)
        reference_points=torch.cat([xy[0].reshape(-1)[...,None],xy[1].reshape(-1)[...,None]],-1).cuda()
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, num_pattern, 1)
        return reference_points
    
    
    

    def forward(self, x, occ_mask=None):
        size = x.size()[2:]

        
        
        # Encoder
        layers = self.backbone(x)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4] #Encoder
        #x1 - [3, 64, 256, 256]
        #x2 - 3, 256, 128, 128]
        #x3- [3, 512, 64, 64])
        #x4- [3, 1024, 32, 32])
        #x5 - [3, 2048, 16, 16])

        
        batch_size = x1.shape[0]

        # Multi-stage feature maps
        x_feat =OrderedDict()
        x_feat['l1'] = x1
        x_feat['l2'] = x2
        x_feat['l3'] = x3
        x_feat['l4'] = x4
        x_feat['l5'] = x5

        # Output of FPN already has D_emb
        fpn_output = self.fpn(x_feat)
        c = fpn_output['l3'].shape[1]

        # Embedding for feature maps from encoder -> batch, D_emb, H*W
        
        feat_pos_emb2 = self.pos_embedder(fpn_output['l2'], occ_mask).flatten(2) #([6, 256, 16384])
        feat_pos_emb3 = self.pos_embedder(fpn_output['l3'], occ_mask).flatten(2) #([6, 256, 4096])
        feat_pos_emb4 = self.pos_embedder(fpn_output['l4'], occ_mask).flatten(2) #([6, 256, 1024])      
        feat_pos_emb5 = self.pos_embedder(fpn_output['l5'], occ_mask).flatten(2) #([6, 256, 256])
        
        
        # Permute feature embeddings to correct format -> H*W, batch, D_emb
        feat_pos_emb2 = feat_pos_emb2.permute(2, 0, 1) #([16384, 6, 256])
        feat_pos_emb3 = feat_pos_emb3.permute(2, 0, 1) #([4096, 6, 256])
        feat_pos_emb4 = feat_pos_emb4.permute(2, 0, 1) #([1024, 6, 256]) 
        feat_pos_emb5 = feat_pos_emb5.permute(2, 0, 1) #([256, 6, 256])
        
        # Permute feature maps to correct format -> H*W, batch, D_emb 
        feat_2 = fpn_output['l2'].flatten(2).permute(2, 0, 1) #[?, 6, 256])     
        feat_3 = fpn_output['l3'].flatten(2).permute(2, 0, 1) #[4096, 6, 256])      
        feat_4 = fpn_output['l4'].flatten(2).permute(2, 0, 1) #[1024, 6, 256])
        feat_5 = fpn_output['l5'].flatten(2).permute(2, 0, 1) #([256, 6, 256]
         
        # Anchor Query -> batch, anchorPoints*numPatterns, 2 (coordinates) : ([6, 289, 2])
        reference_points = self.grid_points(batch_size, self.num_position, self.num_pattern)

        # Occlusion map query
        #occ_att_l4 = occ_attn[-3:] # batch, 5, 1024, 256
        #occ_att_l4 = occ_attn[7:12] # batch, 8, 256, 256
        #tmp1 = torch.cat(occ_att_l4, dim=1)
        #reference_points = tmp1.mean(1) # Heads mean -> batch, 256, 256
        

        # Embedding -> batch, numPositions, D_emb ([6, 300, 256])
        # For anchor query
        tgt = self.pattern.weight.reshape(1, self.num_pattern, 1, c).repeat(batch_size, 1, self.num_position, 1).reshape(
            batch_size, self.num_pattern * self.num_position, c)
        
        # For occ att map query
        #tgt = self.pattern.weight.reshape(1, self.num_pattern, 1, c).repeat(batch_size, 1, self.occ_att_l4_dim, 1).reshape(
        #    batch_size, self.num_pattern * self.occ_att_l4_dim, c)
        
        
        # Output: hidden_state [400, 6, 256]) 
        # Layer 1
        atD5 = self.mod_attDecoder(tgt, reference_points, feat_5, feat_pos_emb5, occ_mask)
        atD4 = self.mod_attDecoder(atD5[0].transpose(0,1), reference_points, feat_4, feat_pos_emb4, occ_mask)
        atD3 = self.mod_attDecoder(atD4[0].transpose(0,1), reference_points, feat_3, feat_pos_emb3, occ_mask)
        #atD2 = self.mod_attDecoder(atD3[0].transpose(0,1), reference_points, feat_2, feat_pos_emb2)

        # Layer2
        #atD5 = self.mod_attDecoder(atD3[0].transpose(0,1), reference_points, feat_5, feat_pos_emb5)
        #atD4 = self.mod_attDecoder(atD5[0].transpose(0,1), reference_points, feat_4, feat_pos_emb4)
        #atD3 = self.mod_attDecoder(atD4[0].transpose(0,1), reference_points, feat_3, feat_pos_emb3)

        # Layer 3
        #atD5 = self.mod_attDecoder(atD3[0].transpose(0,1), reference_points, feat_5, feat_pos_emb5)
        #atD4 = self.mod_attDecoder(atD5[0].transpose(0,1), reference_points, feat_4, feat_pos_emb4)
        #atD3 = self.mod_attDecoder(atD4[0].transpose(0,1), reference_points, feat_3, feat_pos_emb3)

        # Layer 4
        #atD5 = self.mod_attDecoder(atD3[0].transpose(0,1), reference_points, feat_5, feat_pos_emb5)
        #atD4 = self.mod_attDecoder(atD5[0].transpose(0,1), reference_points, feat_4, feat_pos_emb4)
        #atD3 = self.mod_attDecoder(atD4[0].transpose(0,1), reference_points, feat_3, feat_pos_emb3)
        



        hidden_states = self.layernorm(atD3[0])
        
        # Create predicted mask -> ([6, 400, 128, 128])
        predicted_mask, attention_mask = self.mask_predictor( #There is a predicted mask for each image in the batch and for each query...
                    hidden_states,
                    fpn_output['l2'], #([3, 256, 256, 256])
                    x3.shape[-2:], #[torch.Size([128, 128])]
                )

                        
        out = self.convos(predicted_mask)  #new. 3,256,128,128 -> 3,1,128,128
        out = self.outc(out) #new
        out = torch.nn.functional.interpolate(out, size=self.target_sizes, mode="bilinear", align_corners=False)
        
        #(center_x, center_y, width, height)
        #pred_boxes = self.bbox_predictor(atD3[0]).sigmoid() #400, 4, 4 -> in each image, each query predicts one BBox

        #class_prediction = self.class_predictor(atD3[0].transpose(0, 1)) #4, 400, 1
        outputs={}
        #outputs["logits"] = class_prediction
        #outputs["pred_boxes"] = pred_boxes
        


        # NEW: didn't works
        #predicted_mask = torch.nn.functional.interpolate(predicted_mask, size=(384, 384), mode="bilinear", align_corners=False) #([4, 400, 384, 384])
        #masks_probs = masks_queries_logits.sigmoid()  

        #class_prediction = self.class_predictor(atD3[0].transpose(0, 1)) #4, 400, 1
        #masks_classes = class_prediction.sigmoid() #4,400,1

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        #segmentation = torch.einsum("bqc, bqhw -> bchw", masks_classes, masks_probs) # ([3, 1, 384, 384])
        #out = torch.nn.functional.interpolate(masks_probs, size=self.target_sizes, mode="bilinear", align_corners=False)
        #####  
        
        
        

        return out, outputs#, atD3 

def mask2pos(mask):
    not_mask = ~mask
    y_embed = not_mask[:, :, 0].cumsum(1, dtype=torch.float32)
    x_embed = not_mask[:, 0, :].cumsum(1, dtype=torch.float32)
    y_embed = (y_embed - 0.5) / y_embed[:, -1:]
    x_embed = (x_embed - 0.5) / x_embed[:, -1:]
    
    return y_embed, x_embed        