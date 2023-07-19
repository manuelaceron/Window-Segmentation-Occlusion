from networks.MAT.networks.basic_module import Conv2dLayer, FullyConnectedLayer
from networks.unet.unet import UNet
import numpy as np
import math
import sys

import torch, pdb
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple

def token2feature(x, x_size):
    B, N, C = x.shape
    h, w = x_size
    x = x.permute(0, 2, 1).reshape(B, C, h, w)
    return x


def feature2token(x):
    B, C, H, W = x.shape
    x = x.view(B, C, -1).transpose(1, 2)
    return x

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, down=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       activation='lrelu',
                                       down=down,
                                       )
        self.down = down

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.down != 1:
            ratio = 1 / self.down
            x_size = (int(x_size[0] * ratio), int(x_size[1] * ratio))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask

class PatchUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, up=2):
        super().__init__()
        self.conv = Conv2dLayerPartial(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       activation='lrelu',
                                       up=up,
                                       )
        self.up = up

    def forward(self, x, x_size, mask=None):
        x = token2feature(x, x_size)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(x, mask)
        if self.up != 1:
            x_size = (int(x_size[0] * self.up), int(x_size[1] * self.up))
        x = feature2token(x)
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = FullyConnectedLayer(in_features=in_features, out_features=hidden_features, activation='lrelu')
        self.fc2 = FullyConnectedLayer(in_features=hidden_features, out_features=out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, down_ratio=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.k = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.v = FullyConnectedLayer(in_features=dim, out_features=dim)
        self.proj = FullyConnectedLayer(in_features=dim, out_features=dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask_windows=None, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        norm_x = F.normalize(x, p=2.0, dim=-1)
        q = self.q(norm_x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(norm_x).view(B_, -1, self.num_heads, C // self.num_heads).permute(0, 2, 3, 1)
        v = self.v(x).view(B_, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k) * self.scale

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        if mask_windows is not None:
            attn_mask_windows = mask_windows.squeeze(-1).unsqueeze(1).unsqueeze(1)
            attn = attn + attn_mask_windows.masked_fill(attn_mask_windows == 0, float(-100.0)).masked_fill(
                attn_mask_windows == 1, float(0.0))
            with torch.no_grad():
                mask_windows = torch.clamp(torch.sum(mask_windows, dim=1, keepdim=True), 0, 1).repeat(1, N, 1)

        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x, mask_windows

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, down_ratio=1, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        if self.shift_size > 0:
            down_ratio = 1
        self.attn = WindowAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                                    down_ratio=down_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
                                    proj_drop=drop)

        self.fuse = FullyConnectedLayer(in_features=dim * 2, out_features=dim, activation='lrelu')

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size, mask=None):
        # H, W = self.input_resolution
        H, W = x_size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)
        if mask is not None:
            mask = mask.view(B, H, W, 1)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            if mask is not None:
                shifted_mask = torch.roll(mask, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            if mask is not None:
                shifted_mask = mask

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        if mask is not None:
            mask_windows = window_partition(shifted_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size, 1)
        else:
            mask_windows = None

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows, mask_windows = self.attn(x_windows, mask_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        else:
            attn_windows, mask_windows = self.attn(x_windows, mask_windows, mask=self.calculate_mask(x_size).to(x.device))  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        if mask is not None:
            mask_windows = mask_windows.view(-1, self.window_size, self.window_size, 1)
            shifted_mask = window_reverse(mask_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            if mask is not None:
                mask = torch.roll(shifted_mask, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
            if mask is not None:
                mask = shifted_mask
        x = x.view(B, H * W, C)
        if mask is not None:
            mask = mask.view(B, H * W, 1)

        # FFN
        x = self.fuse(torch.cat([shortcut, x], dim=-1))
        x = self.mlp(x)

        return x, mask

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, down_ratio=1,
                 mlp_ratio=2., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        if downsample is not None:
            # self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            self.downsample = downsample
        else:
            self.downsample = None

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, down_ratio=down_ratio, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        self.conv = Conv2dLayerPartial(in_channels=dim, out_channels=dim, kernel_size=3, activation='lrelu')

    def forward(self, x, x_size, mask=None):
        if self.downsample is not None:
            x, x_size, mask = self.downsample(x, x_size, mask)
        identity = x
        for blk in self.blocks:
            if self.use_checkpoint:
                x, mask = checkpoint.checkpoint(blk, x, x_size, mask)
            else:
                x, mask = blk(x, x_size, mask)
        if mask is not None:
            mask = token2feature(mask, x_size)
        x, mask = self.conv(token2feature(x, x_size), mask)
        x = feature2token(x) + identity
        if mask is not None:
            mask = feature2token(mask)
        return x, x_size, mask

class Conv2dLayerPartial(nn.Module):
    def __init__(self,
                 in_channels,                    # Number of input channels.
                 out_channels,                   # Number of output channels.
                 kernel_size,                    # Width and height of the convolution kernel.
                 bias            = True,         # Apply additive bias before the activation function?
                 activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
                 up              = 1,            # Integer upsampling factor.
                 down            = 1,            # Integer downsampling factor.
                 resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
                 conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
                 trainable       = True,         # Update the weights of this layer during training?
                 ):
        super().__init__()
        self.conv = Conv2dLayer(in_channels, out_channels, kernel_size, bias, activation, up, down, resample_filter,
                                conv_clamp, trainable)

        self.weight_maskUpdater = torch.ones(1, 1, kernel_size, kernel_size)
        self.slide_winsize = kernel_size ** 2
        self.stride = down
        self.padding = kernel_size // 2 if kernel_size % 2 == 1 else 0

    def forward(self, x, mask=None):
        if mask is not None:
            with torch.no_grad():
                if self.weight_maskUpdater.type() != x.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(x)
                update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding)
                mask_ratio = self.slide_winsize / (update_mask + 1e-8)
                update_mask = torch.clamp(update_mask, 0, 1)  # 0 or 1
                mask_ratio = torch.mul(mask_ratio, update_mask)
            x = self.conv(x)
            x = torch.mul(x, mask_ratio)
            return x, update_mask
        else:
            x = self.conv(x)
            return x, None

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
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) # Only for Bayeasian! comment unless is Bayesian TODO:     
        )

    def forward(self, x):
        x = self.up(x)
        return x

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
            nn.BatchNorm2d(num_class), 
            nn.Dropout(0.5) # Only for Bayeasian! comment unless is Bayesian TODO:          
            )
         

    def forward(self, x):
        x = self.conv(x)
        return x

class FirstStage(nn.Module):
    def __init__(self, img_channels, img_resolution=256, dim=180, use_noise=False, demodulate=True, activation='lrelu', num_class = 1):
        super().__init__()
        res = 64

        self.conv_first = Conv2dLayerPartial(in_channels=img_channels+1, out_channels=dim, kernel_size=3, activation=activation)
        self.enc_conv = nn.ModuleList()
        down_time = int(np.log2(img_resolution // res))
        for i in range(down_time):  # from input size to 64
            self.enc_conv.append(
                Conv2dLayerPartial(in_channels=dim, out_channels=dim, kernel_size=3, down=2, activation=activation)
            )

        # from 64 -> 16 -> 64
        depths = [2, 3, 4, 3, 2]
        ratios = [1, 1/2, 1/2, 2, 2]
        num_heads = 6
        window_sizes = [8, 16, 16, 16, 8]
        drop_path_rate = 0.1
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.tran = nn.ModuleList()
        for i, depth in enumerate(depths):
            res = int(res * ratios[i])
            if ratios[i] < 1:
                merge = PatchMerging(dim, dim, down=int(1/ratios[i]))
            elif ratios[i] > 1:
                merge = PatchUpsampling(dim, dim, up=ratios[i])
            else:
                merge = None
            self.tran.append(
                BasicLayer(dim=dim, input_resolution=[res, res], depth=depth, num_heads=num_heads,
                           window_size=window_sizes[i], drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                           downsample=merge)
            )

    
        #style_dim = dim * 3
        #self.dec_conv = nn.ModuleList()
        #for i in range(down_time):  # from 64 to input size
        #    res = res * 2
        #    self.dec_conv.append(DecStyleBlock(res, dim, dim, activation, style_dim, use_noise, demodulate, img_channels))

        dim_ = 180
        dim_cat = dim_*2
        self.Up4 = up_conv(dim_,dim_) 
        self.Up_conv4 = conv_block(dim_cat,dim_)

        self.Up3 = up_conv(dim_,dim_)
        self.Up_conv3 = conv_block(dim_cat,dim_)

        self.Up2 = up_conv(dim_,dim_)
        self.Up_conv2 = conv_block(dim_cat,dim_)

        self.Conv = nn.Conv2d(dim_, num_class, kernel_size=3, stride=1, padding=1)

    def forward(self, images_in, masks_in):
        x = torch.cat([masks_in - 0.5, images_in * masks_in], dim=1)
        
        skips = []
        x, mask = self.conv_first(x, masks_in)  # input size Update mask...
        
        skips.append(x)

        # Input: b, 1, 512, 512 -> b, 180, 64, 64
        # Mask: b, 1, 512, 512 -> b, 1, 64, 64

        #Convolutional head
        for i, block in enumerate(self.enc_conv):  # input size to 64 -> x3 Conv2dLayerPartial: update mask?
            x, mask = block(x, mask)
            if i != len(self.enc_conv) - 1:
                skips.append(x)
        
        
        
        x_size = x.size()[-2:] #[64, 64])
        x = feature2token(x) #[1, 4096, 180])
        mask = feature2token(mask) #[1, 4096, 1])
        mid = len(self.tran) // 2

        # 5 Transformers blocks
        for i, block in enumerate(self.tran):  # 64 to 16
            
            if i < mid: #block 1,2
                x, x_size, mask = block(x, x_size, mask)
                skips.append(x)
                
            elif i > mid: #block 4, 5 
                x, x_size, mask = block(x, x_size, None) # mask=None. why? - i= 3 x: b,1024,180
                
                x = x + skips[mid - i]
            else: #block 3
                x, x_size, mask = block(x, x_size, None)
        
        x = token2feature(x, x_size).contiguous() #[1, 180, 64, 64])

        
        # Convolutional tail


        #img = None
        #for i, block in enumerate(self.dec_conv):
        #    x, img = block(x, img, style, skips[len(self.dec_conv)-i-1], noise_mode=noise_mode)


        d4 = self.Up4(x) #180, 64, 64 -> 180, 128, 128
        d4 = torch.cat((skips[2], d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4) #180, 128, 128 -> 180, 256, 256
        d3 = torch.cat((skips[1], d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3) #180, 256, 256 -> 180, 512, 512
        d2 = torch.cat((skips[0], d2), dim=1)
        d2 = self.Up_conv2(d2)

        img = self.Conv(d2)
        
        # ensemble
        img = img * (1 - masks_in) + images_in * masks_in

        return img


class SynthesisNet(nn.Module):
    def __init__(self,
                 img_resolution,            # Output image resolution.
                 img_channels   = 3,        # Number of color channels.
                 channel_decay  = 1.0,
                 channel_max    = 512,      # Maximum number of channels in any layer.
                 activation     = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
                 drop_rate      = 0.5,
                 use_noise      = True,
                 demodulate     = True,
                 ):
        super().__init__()
        resolution_log2 = int(np.log2(img_resolution))
        assert img_resolution == 2 ** resolution_log2 and img_resolution >= 4

        self.num_layers = resolution_log2 * 2 - 3 * 2
        self.img_resolution = img_resolution
        self.resolution_log2 = resolution_log2

        # first stage
        self.first_stage = FirstStage(img_channels, img_resolution=img_resolution, use_noise=False, demodulate=demodulate, num_class = 1)

    def forward(self, images_in, masks_in):

        out_stg1 = self.first_stage(images_in, masks_in)

        
        return out_stg1
            
class Generator(nn.Module):
    def __init__(self,
                 z_dim,                  # Input latent (Z) dimensionality, 0 = no latent.
                 c_dim,                  # Conditioning label (C) dimensionality, 0 = no label.
                 w_dim,                  # Intermediate latent (W) dimensionality.
                 img_resolution,         # resolution of generated image
                 img_channels,           # Number of input color channels.
                 synthesis_kwargs = {},  # Arguments for SynthesisNetwork.
                 mapping_kwargs   = {},  # Arguments for MappingNetwork.
                 ):
        super().__init__()
        
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.synthesis = SynthesisNet(img_resolution=img_resolution,
                                      img_channels=img_channels,
                                      **synthesis_kwargs)
        
        self.unet = UNet(in_c=img_channels, num_class=1)
        

    def forward(self, images_in, masks_in, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False,
                noise_mode='random', return_stg1=False):
        
        stg1 = self.synthesis(images_in, masks_in)

        stg2 = self.unet(stg1)
        
            
        return stg2, stg1