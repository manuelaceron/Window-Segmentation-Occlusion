from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import pdb
import timm
from ..common_func.base_func import _ConvBNReLU
from ..unet.unet import UNet
from ..unet.resunet import Res_UNet_50
from ..unet.unet import new_DeepL
import torchvision.models as models
from transformers import SegformerForSemanticSegmentation 

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

    def __init__(self, in_c=3, num_class=4, pretrained_path=None): 
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

class segFormer_ResUnet50_vis_hid(nn.Module):
    """
    
    """

    def __init__(self, in_c=3, num_class=4, pretrained_path=True): 
        super(segFormer_ResUnet50_vis_hid, self).__init__()

        pre_model = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024" # "nvidia/mit-b2"
        id2label = {0: 'background', 1: 'occlusion'}
        label2id = {'background': 0, 'occlusion': 1}
        
        self.segFormer = SegformerForSemanticSegmentation.from_pretrained(pre_model, ignore_mismatched_sizes=True,
                                                            num_labels=len(id2label), id2label=id2label, label2id=label2id,
                                                            reshape_last_stage=True)
        
        self.backbone = timm.create_model('resnet50', pretrained=True, features_only=True, in_chans=in_c) 
        print('using: resnet50', ' pretrained: ', True )

        self.decoResUnet50 = deco_Res_UNet_50(in_c, num_class) 

        #self.resUnet = Res_UNet_50(in_c, num_class) # in_c depends on input + or concatenated

        self.unet = UNet(2, num_class)


    def forward(self, sf_x, sf_l, x):
        size = x.size()[2:]

        ################SEGFORMER + RESUNET50############################
        occ = self.segFormer(pixel_values=sf_x, labels= sf_l)
        upsampled_logits = nn.functional.interpolate(occ.logits, size=size, mode="bilinear", align_corners=False) 
        occ = upsampled_logits.argmax(dim=1) 
        occ = torch.unsqueeze(occ, 1) 
        #x1 = torch.cat((x,occ), dim = 1) 
        #predicted = occ.argmax(dim=1)???
        
        #x1 = x


        layers = self.backbone(x+occ)
        x1, x2, x3, x4, x5 = layers[0], layers[1], layers[2], layers[3], layers[4] #Encoder

        visible = self.decoResUnet50(x, x1, x2, x3, x4, x5)
        new_in = visible + occ
        out = self.unet(new_in)
        #invisible = self.decoResUnet50(x, x1, x2, x3, x4, x5)


        #merged_mask = torch.logical_or(visible_mask, invisible_mask)
        #################################################################
        
        return upsampled_logits, visible, out