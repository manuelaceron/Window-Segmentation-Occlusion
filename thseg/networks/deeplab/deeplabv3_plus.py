import torch
import torch.nn as nn
import torch.nn.functional as F
from ..common_func.get_backbone import get_model
from .deeplabv3 import _ASPP
from ..common_func.base_func import _ConvBNReLU
import pdb

class _FCNHead(nn.Module):
    def __init__(self, in_cannels, channels, norm_layer=nn.BatchNorm2d):
        super(_FCNHead, self).__init__()
        inter_channels = in_cannels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_cannels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


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
    def __init__(self, in_cannels, atrous_rates, norm_layer, num_classannels = 256):
        super(_ASPP, self).__init__()
        
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


class DeepLabV3Plus(nn.Module):
    r"""DeepLabV3Plus
    Parameters
    ----------
    num_class : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'xception').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic
        Image Segmentation."
    """

    def __init__(self, in_c, num_class, pretrained_path=None, **kwargs):
        super(DeepLabV3Plus, self).__init__()
        aux = True
        self.aux = aux
        self.num_class = num_class
        self.in_c = in_c
        
        
        inplanes = 2048
        low_level_planes = 256

        self.pretrained = get_model('resnet101', in_c = in_c, checkpoint_path=pretrained_path)

        # deeplabv3 plus
        self.head = _DeepLabHead(num_class, inplanes, c1_channels=low_level_planes, **kwargs)
        if aux:
            self.auxlayer = _FCNHead(728, num_class)
        self.activate = nn.Softmax() if num_class > 1 else nn.Sigmoid()

    def base_forward(self, x):
        # Entry flow
        features = self.pretrained(x) #why 5?
        
        return features[1], features[4] #c1: torch.Size([2, 256, 128, 128]) - c4: torch.Size([2, 2048, 16, 16]) #why 1 and 4?

    # 1. code starts here?
    def forward(self, x):
        size = x.size()[2:]
        c1, c4 = self.base_forward(x) #c1 are low level features, and c4 is feature extractor output
       
        x = self.head(c4, c1) #torch.Size([2, 1, 128, 128]) c1 and x concat 

        x = F.interpolate(x, size, mode='bilinear', align_corners=True) # Upsampled by 4
        x = self.activate(x) #torch.Size([2, 1, 512, 512]) Decoder output
        return x 


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

if __name__ == '__main__':
    net = DeepLabV3Plus(num_class=6)
    # nrte = meca(64, 3)

    x = torch.randn(2, 3, 256, 256)
    y = net(x)
    print(y.shape)