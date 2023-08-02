import numpy as np
import torch, pdb, math, cv2, os
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
# from torch.nn.utils.parametrizations import spectral_norm
from ..new_frame.deep_window import Mask2FormerSinePositionEmbedding, attentionDecoder, Mask2FormerMaskPredictor
import matplotlib.pyplot as plt


def plot_features(inputF, layer):
        
    nothing = True
    if nothing:
        return
    else:           

        path = '/home/cero_ma/MCV/code220419_windows/0401_files/DFV2_ecp_0.0001_addMask/'   
        if os.path.exists(os.path.join(path, 'features')) is False:
            os.mkdir(os.path.join(path, 'features'))
        
        if os.path.exists(os.path.join(path, 'means')) is False:
            os.mkdir(os.path.join(path, 'means'))
        
        gray_scale = torch.sum(inputF[0],0)
        gray_scale = gray_scale / inputF.shape[0]
        cv2.imwrite(os.path.join(path, 'means', layer+'.png'), (np.asarray(gray_scale.detach().cpu())*255).astype(float))

        x = inputF.detach().cpu().numpy()
        gr = int(math.ceil(math.sqrt(x.shape[1])))    

        fig = plt.figure(figsize=(30, 50))
        for i in range(x.shape[1]):
            a = fig.add_subplot(gr, gr, i+1)
            imgplot = plt.imshow(x[0][i], cmap='gray')
            a.axis("off")        
        
        plt.savefig(os.path.join(path, 'features', layer+'.png'), bbox_inches='tight')
# ----------------------------------------------------------------------------

def _init_conv_layer(conv, activation, mode='fan_out'):
    if isinstance(activation, nn.LeakyReLU):
        torch.nn.init.kaiming_uniform_(conv.weight,
                                       a=activation.negative_slope,
                                       nonlinearity='leaky_relu',
                                       mode=mode)
    elif isinstance(activation, (nn.ReLU, nn.ELU)):
        torch.nn.init.kaiming_uniform_(conv.weight,
                                       nonlinearity='relu',
                                       mode=mode)
    else:
        pass
    if conv.bias != None:
        torch.nn.init.zeros_(conv.bias)


def output_to_image(out):
    out = (out[0].cpu().permute(1, 2, 0) + 1.) * 127.5
    out = out.to(torch.uint8).numpy()
    return out

# ----------------------------------------------------------------------------

#################################
########### GENERATOR ###########
#################################

class GConv(nn.Module):
    """Implements the gated 2D convolution introduced in 
       `Free-Form Image Inpainting with Gated Convolution`(Yu et al., 2019)
    """

    def __init__(self, cnum_in, cnum_out,
                 ksize, stride=1, padding='auto', rate=1,
                 activation=nn.ELU(),
                 bias=True, gated=True, use_conv1D=False):
        super().__init__()

        padding = rate*(ksize-1)//2 if padding == 'auto' else padding
        self.activation = activation
        self.cnum_out = cnum_out
        # num_conv_out = cnum_out if self.cnum_out == 3 or self.activation is None else 2*cnum_out
        num_conv_out = 2*cnum_out if gated else cnum_out

        if use_conv1D:
            self.conv1D = strip_conv(cnum_in,
                                    num_conv_out,
                                    ksize=ksize,
                                    stride=stride,
                                    rate=rate)
        else:
            self.conv = nn.Conv2d(cnum_in,
                                num_conv_out,
                                kernel_size=ksize,
                                stride=stride,
                                padding=padding,
                                dilation=rate,
                                bias=bias)

            _init_conv_layer(self.conv, activation=self.activation)
            

        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.padding = padding
        self.gated = gated
        self.use_conv1D = use_conv1D

    def forward(self, x):
        """
        Args:

        """
        if not self.gated: return self.conv(x)

        if self.use_conv1D:
            x = self.conv1D(x)
        else:
            x = self.conv(x)        
        x, y = torch.split(x, self.cnum_out, dim=1)
        x = self.activation(x)
        y = torch.sigmoid(y)
        x = x * y

        return x

# ----------------------------------------------------------------------------

class GDeConv(nn.Module):
    """Upsampling (x2) followed by convolution"""

    def __init__(self, cnum_in, cnum_out, padding=1):        
        super().__init__()

        self.conv = GConv(cnum_in, cnum_out, ksize=3, stride=1,
                          padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest',
                          recompute_scale_factor=False)
        x = self.conv(x)
        return x

# ----------------------------------------------------------------------------

class GDownsamplingBlock(nn.Module):
    """Strided convolution (s=2) followed by convolution (s=1)"""

    def __init__(self, cnum_in, cnum_out, cnum_hidden=None):
        super().__init__()

        cnum_hidden = cnum_out if cnum_hidden == None else cnum_hidden
        self.conv1_downsample = GConv(cnum_in, cnum_hidden, ksize=3, stride=2)
        self.conv2 = GConv(cnum_hidden, cnum_out, ksize=3, stride=1)

    def forward(self, x):
        x = self.conv1_downsample(x)
        x = self.conv2(x)
        return x

# ----------------------------------------------------------------------------

class GUpsamplingBlock(nn.Module):
    """Upsampling (x2) followed by two convolutions"""

    def __init__(self, cnum_in, cnum_out, cnum_hidden=None):
        super().__init__()
        cnum_hidden = cnum_out if cnum_hidden == None else cnum_hidden
        self.conv1_upsample = GDeConv(cnum_in, cnum_hidden)
        self.conv2 = GConv(cnum_hidden, cnum_out, ksize=3, stride=1)

    def forward(self, x):
        x = self.conv1_upsample(x)
        x = self.conv2(x)        
        return x

# ----------------------------------------------------------------------------


class CoarseGenerator(nn.Module):
    """Coarse Network (Stage I)"""

    def __init__(self, cnum_in, cnum_out, cnum):
        super().__init__()
        
        self.conv1 = GConv(cnum_in, cnum//2, ksize=5, stride=1, padding=2)

        # downsampling
        self.down_block1 = GDownsamplingBlock(cnum//2, cnum)
        self.down_block2 = GDownsamplingBlock(cnum, 2*cnum)

        # bottleneck
        self.conv_bn1 = GConv(2*cnum, 2*cnum, ksize=3, stride=1)

        use_conv1D = True
        if use_conv1D:
            # Test using 1D convolutions
            k = 7
            self.conv_bn2 = GConv(2*cnum, 2*cnum, ksize=k, rate=2, use_conv1D=use_conv1D)
            self.conv_bn3 = GConv(2*cnum, 2*cnum, ksize=k, rate=4, use_conv1D=use_conv1D)
            self.conv_bn4 = GConv(2*cnum, 2*cnum, ksize=k, rate=8, use_conv1D=use_conv1D)
            self.conv_bn5 = GConv(2*cnum, 2*cnum, ksize=k, rate=16, use_conv1D=use_conv1D)
        else:
            self.conv_bn2 = GConv(2*cnum, 2*cnum, ksize=3, rate=2, padding=2)
            self.conv_bn3 = GConv(2*cnum, 2*cnum, ksize=3, rate=4, padding=4)
            self.conv_bn4 = GConv(2*cnum, 2*cnum, ksize=3, rate=8, padding=8)
            self.conv_bn5 = GConv(2*cnum, 2*cnum, ksize=3, rate=16, padding=16)


        self.conv_bn6 = GConv(2*cnum, 2*cnum, ksize=3, stride=1)
        self.conv_bn7 = GConv(2*cnum, 2*cnum, ksize=3, stride=1)

        # upsampling
        self.up_block1 = GUpsamplingBlock(2*cnum, cnum)
        self.up_block2 = GUpsamplingBlock(cnum, cnum//4, cnum_hidden=cnum//2)

        # to RGB
        self.conv_to_rgb = GConv(cnum//4, cnum_out, ksize=3, stride=1, activation=None, gated=False)
        self.tanh = nn.Tanh()


    def forward(self, inpu):
 
        x= inpu
        
        
        x = self.conv1(x) #[1, 24, 512, 512]
        plot_features(x, '1-conv1')
        
        # downsampling
        x = self.down_block1(x) #([1, 48, 256, 256]
        plot_features(x, '2-down_block1')

        x = self.down_block2(x) #[1, 96, 128, 128]
        plot_features(x, '3-down_block2')
         
        # bottleneck
        x = self.conv_bn1(x) #[1, 96, 128, 128]
        x = self.conv_bn2(x)
        x = self.conv_bn3(x)
        plot_features(x, '4-conv_bn3')
        x = self.conv_bn4(x)
        x = self.conv_bn5(x)
        x = self.conv_bn6(x)
        x = self.conv_bn7(x) #[1, 96, 128, 128]
        plot_features(x, '5-conv_bn7')
        
        
        # upsampling
        x = self.up_block1(x)
        plot_features(x, '5-up_block1')
        x = self.up_block2(x)
        plot_features(x, '6-up_block2')

        # to RGB
        x = self.conv_to_rgb(x) #[1, 1, 512, 512]
        plot_features(x, '7-conv_to_rgb')
        x = self.tanh(x)
        plot_features(x, '8-tanh')
        
        
        return x

# ----------------------------------------------------------------------------

class FineGenerator(nn.Module):
    """Two Branch Refinement Network with Contextual Attention (Stage II)"""

    def __init__(self, cnum_in, cnum_out, cnum, return_flow=False):
        super().__init__()

        ### CONV BRANCH (B1) ###
        self.conv_conv1 = GConv(cnum_in, cnum//2, ksize=5, stride=1, padding=2)

        # downsampling
        self.conv_down_block1 = GDownsamplingBlock(
            cnum//2, cnum, cnum_hidden=cnum//2)
        self.conv_down_block2 = GDownsamplingBlock(
            cnum, 2*cnum, cnum_hidden=cnum)

        # bottleneck
        self.conv_conv_bn1 = GConv(2*cnum, 2*cnum, ksize=3, stride=1)
        
        use_conv1D = True
        self.transformer = False

        if use_conv1D:
            k = 7
            self.conv_conv_bn2 = GConv(2*cnum, 2*cnum, ksize=k, rate=2, use_conv1D= use_conv1D)
            self.conv_conv_bn3 = GConv(2*cnum, 2*cnum, ksize=k, rate=4, use_conv1D= use_conv1D)
            self.conv_conv_bn4 = GConv(2*cnum, 2*cnum, ksize=k, rate=8, use_conv1D= use_conv1D)
            self.conv_conv_bn5 = GConv(2*cnum, 2*cnum, ksize=k, rate=16, use_conv1D= use_conv1D)
        else:
            self.conv_conv_bn2 = GConv(2*cnum, 2*cnum, ksize=3, rate=2, padding=2)
            self.conv_conv_bn3 = GConv(2*cnum, 2*cnum, ksize=3, rate=4, padding=4)
            self.conv_conv_bn4 = GConv(2*cnum, 2*cnum, ksize=3, rate=8, padding=8)
            self.conv_conv_bn5 = GConv(2*cnum, 2*cnum, ksize=3, rate=16, padding=16)

        ### ATTENTION BRANCH (B2) ###
        self.ca_conv1 = GConv(cnum_in, cnum//2, 5, 1, padding=2)

        # downsampling
        self.ca_down_block1 = GDownsamplingBlock(cnum//2, cnum, cnum_hidden=cnum//2)
        self.ca_down_block2 = GDownsamplingBlock(cnum, 2*cnum)

        # bottleneck
        self.ca_conv_bn1 = GConv(2*cnum, 2*cnum, 3, 1, activation=nn.ReLU())
        self.contextual_attention = ContextualAttention(ksize=3,
                                                        stride=1,
                                                        rate=2,
                                                        fuse_k=3,
                                                        softmax_scale=10,
                                                        fuse=True,
                                                        device_ids=None,
                                                        return_flow=return_flow,
                                                        n_down=2)
        self.ca_conv_bn4 = GConv(2*cnum, 2*cnum, ksize=3, stride=1)
        self.ca_conv_bn5 = GConv(2*cnum, 2*cnum, ksize=3, stride=1)

        ### UNITED BRANCHES ###
        self.conv_bn6 = GConv(4*cnum, 2*cnum, ksize=3, stride=1)
        self.conv_bn7 = GConv(2*cnum, 2*cnum, ksize=3, stride=1)

        # upsampling
        self.up_block1 = GUpsamplingBlock(2*cnum, cnum)
        self.up_block2 = GUpsamplingBlock(cnum, cnum//4, cnum_hidden=cnum//2)

        
        if not self.transformer:
            # to RGB
            self.conv_to_rgb = GConv(cnum//4, cnum_out, ksize=3, stride=1, 
                                    activation=None, gated=False)
            self.tanh = nn.Tanh()
            
        else:
            self.trans = TransformerModule(cnum=cnum)

       

    def forward(self, x, mask):
        xnow = x #[1, 1, 512, 512])

        ### CONV BRANCH ###
        x = self.conv_conv1(xnow)
        plot_features(x, '9-conv_conv1')
        # downsampling
        x = self.conv_down_block1(x) #Can be useful as skip connection?
        plot_features(x, '10-conv_down_block1')
                
        x = self.conv_down_block2(x) #Can be useful as skip connection?
        plot_features(x, '11-conv_down_block2')

        # bottleneck
        x = self.conv_conv_bn1(x)
        x = self.conv_conv_bn2(x)
        x = self.conv_conv_bn3(x)
        plot_features(x, '12-conv_conv_bn3')
        x = self.conv_conv_bn4(x)
        x = self.conv_conv_bn5(x)
        x_hallu = x #([1, 96, 128, 128])
        plot_features(x, '13-conv_conv_bn5')

        ### ATTENTION BRANCH ###
        x = self.ca_conv1(xnow)
        plot_features(x, '14-ca_conv1')

        # downsampling
        x = self.ca_down_block1(x)
        x = self.ca_down_block2(x)
        plot_features(x, '15-ca_down_block2')

        # bottleneck
        x = self.ca_conv_bn1(x)
        x, offset_flow = self.contextual_attention(x, x, mask)
        plot_features(x, '16-contextual_attention')
        x = self.ca_conv_bn4(x)
        x = self.ca_conv_bn5(x)
        pm = x #([1, 96, 128, 128])

        plot_features(x, '17-ca_conv_bn5')

        # concatenate outputs from both branches
        x = torch.cat([x_hallu, pm], dim=1) #([1, 192, 128, 128])

        plot_features(x, '18-concat')
        

        ### UNITED BRANCHES ###
        
        x = self.conv_bn6(x) #([1, 96, 128, 128])
        plot_features(x, '19-conv_bn6')

        x7 = self.conv_bn7(x) #([1, 96, 128, 128])
        plot_features(x7, '20-conv_bn7')

        # upsampling
        xu1 = self.up_block1(x7) #[1, 48, 256, 256])
        plot_features(xu1, '21-up_block1')

        xu2 = self.up_block2(xu1) #[1, 12, 512, 512]) #this one makes arc-shape -> rectangular-shape
        plot_features(xu2, '22-up_block2')

        if not self.transformer:
            # to RGB
            x = self.conv_to_rgb(xu2)
            plot_features(xu2, '23-conv_to_rgb')
            x = self.tanh(x)
            plot_features(xu2, '24-tanh')

        else:   
            # Test using TransformerQuery
            x_feat =OrderedDict()
            x_feat['l3'] = xu2
            x_feat['l4'] = xu1
            x_feat['l5'] = x7  
            x =self.trans(x_feat)
        
        
        
        return x, offset_flow

# ----------------------------------------------------------------------------

class Generator(nn.Module):
    """Inpainting network consisting of a coarse and a refinement network. 
    Described in the paper 
    `Free-Form Image Inpainting with Gated Convolution, Yu et. al`.
    """

    def __init__(self, cnum_in=5, cnum_mid = 3, cnum_out=3, cnum=48, 
                 return_flow=False, checkpoint=None):
        super().__init__()

        #self.stage1 = CoarseGenerator(cnum_in, cnum_out, cnum)
        #self.stage2 = FineGenerator(cnum_out, cnum_out, cnum, return_flow)
        #Test: RGB -> RGB -> Bin mask:
        self.stage1 = CoarseGenerator(cnum_in, cnum_mid, cnum)
        self.stage2 = FineGenerator(cnum_mid, cnum_out, cnum, return_flow)

        self.return_flow = return_flow
        self.cnum_in = cnum_in

        if checkpoint is not None:
            generator_state_dict = torch.load(checkpoint)['G']
            self.load_state_dict(generator_state_dict, strict=True)

        self.eval()

    def forward(self, x, mask):
        """
        Args:
            x (Tensor): input of shape [batch, cnum_in, H, W]
            mask (Tensor): mask of shape [batch, 1, H, W]
        """
        do_stg1 = True
        keep_orig_bg = True
        
        if do_stg1:
            # get coarse result
            x_stage1 = self.stage1(x)
        else:
            x_stage1 = x
        
    
        if keep_orig_bg:
            input_channel = 1
            add = x.shape[1] - input_channel
            x = x_stage1*mask + x[:, :self.cnum_in-add]*(1.-mask) 
        else:
            x = x_stage1            

        # get refined result
        x_stage2, offset_flow = self.stage2(x, mask)
        
        

        if self.return_flow:
            return x_stage1, x_stage2, offset_flow

        return x_stage1, x_stage2

    @torch.inference_mode()
    def infer(self, image, mask,
              return_vals=['inpainted', 'stage1']):
        """
        Args:
            image (Tensor): input image of shape [cnum_out, H, W]
            mask (Tensor): mask of shape [*, H, W]
            return_vals (str | List[str]): options: inpainted, stage1, stage2, flow
        """

        if isinstance(return_vals, str):
            return_vals = [return_vals]

        _, h, w = image.shape
        grid = 8

        image = image[None, :self.cnum_in, :h//grid*grid, :w//grid*grid]
        mask = mask[None, :3, :h//grid*grid, :w//grid*grid].sum(1, keepdim=True)

        image = (image*2 - 1.)  # map image values to [-1, 1] range
        # 1.: masked 0.: unmasked
        mask = (mask > 0.).to(dtype=torch.float32)

        image_masked = image * (1.-mask)  # mask image

        ones_x = torch.ones_like(image_masked)[:, :1]  # sketch channel
        x = torch.cat([image_masked, ones_x, ones_x*mask],
                      dim=1)  # concatenate channels

        if self.return_flow:
            x_stage1, x_stage2, offset_flow = self.forward(x, mask)
        else:
            x_stage1, x_stage2 = self.forward(x, mask)

        image_compl = image * (1.-mask) + x_stage2 * mask

        output = []
        for return_val in return_vals:
            if return_val.lower() == 'stage1':
                output.append(output_to_image(x_stage1))
            elif return_val.lower() == 'stage2':
                output.append(output_to_image(x_stage2))
            elif return_val.lower() == 'inpainted':
                output.append(output_to_image(image_compl))
            elif return_val.lower() == 'flow' and self.return_flow:
                output.append(offset_flow)
            else:
                print(f'Invalid return value: {return_val}')

        if len(output) == 1:
            return output[0]

        return output

# ----------------------------------------------------------------------------

####################################
####### CONTEXTUAL ATTENTION #######
####################################

"""
adapted from: https://github.com/daa233/generative-inpainting-pytorch/blob/master/model/networks.py
"""

class ContextualAttention(nn.Module):
    """ Contextual attention layer implementation. 
        Contextual attention is first introduced in publication: 
         `Generative Image Inpainting with Contextual Attention, Yu et al`
    Args:
        ksize: Kernel size for contextual attention
        stride: Stride for extracting patches from b
        rate: Dilation for matching
        softmax_scale: Scaled softmax for attention
    """

    def __init__(self,
                 ksize=3,
                 stride=1,
                 rate=1,
                 fuse_k=3,
                 softmax_scale=10.,
                 n_down=2,
                 fuse=False,
                 return_flow=False,
                 device_ids=None):        
        super().__init__()

        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.device_ids = device_ids
        self.n_down = n_down
        self.return_flow = return_flow
        self.register_buffer('fuse_weight', torch.eye(
            fuse_k).view(1, 1, fuse_k, fuse_k))

    def forward(self, f, b, mask=None):
        """
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
        """
        device = f.device
        # get shapes
        raw_int_fs, raw_int_bs = list(f.size()), list(b.size())   # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksize=kernel,
                                      stride=self.rate*self.stride,
                                      rate=1, padding='auto')  # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L]
        L = raw_w.size(2)
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)    # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1./self.rate,
                          mode='nearest', recompute_scale_factor=False)
        b = F.interpolate(b, scale_factor=1./self.rate,
                          mode='nearest', recompute_scale_factor=False)
        int_fs, int_bs = list(f.size()), list(b.size())   # b*c*h*w
        # split tensors along the batch dimension
        f_groups = torch.split(f, 1, dim=0)
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, ksize=self.ksize,
                                  stride=self.stride,
                                  rate=1, padding='auto')
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, L)
        w = w.permute(0, 4, 1, 2, 3)    # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        if mask is None:
            mask = torch.zeros(
                [int_bs[0], 1, int_bs[2], int_bs[3]], device=device)
        else:
            mask = F.interpolate(
                mask, scale_factor=1./((2**self.n_down)*self.rate), mode='nearest', recompute_scale_factor=False)
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksize=self.ksize,
                                  stride=self.stride,
                                  rate=1, padding='auto')
        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, L)
        m = m.permute(0, 4, 1, 2, 3)    # m shape: [N, L, C, k, k]
        m = m[0]    # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]

        mm = (torch.mean(m, dim=[1, 2, 3], keepdim=True) == 0.).to(
            torch.float32)
        mm = mm.permute(1, 0, 2, 3)  # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        scale = self.softmax_scale    # to fit the PyTorch tensor image value range

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.sqrt(torch.sum(torch.square(wi), dim=[
                                1, 2, 3], keepdim=True)).clamp_min(1e-4)
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            yi = F.conv2d(xi, wi_normed, stride=1, padding=(self.ksize-1)//2) # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                # (B=1, I=1, H=32*32, W=32*32)
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                # (B=1, C=1, H=32*32, W=32*32)
                yi = F.conv2d(yi, self.fuse_weight, stride=1,
                              padding=(self.fuse_k-1)//2)
                # (B=1, 32, 32, 32, 32)
                yi = yi.contiguous().view(
                    1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])
                yi = yi.permute(0, 2, 1, 4, 3)

                yi = yi.contiguous().view(
                    1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = F.conv2d(yi, self.fuse_weight, stride=1,
                              padding=(self.fuse_k-1)//2)
                yi = yi.contiguous().view(
                    1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()

            # (B=1, C=32*32, H=32, W=32)
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi*scale, dim=1)
            yi = yi * mm  # [1, L, H, W]

            if self.return_flow:
                offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W

                if int_bs != int_fs:
                    # Normalize the offset value to match foreground dimension
                    times = (int_fs[2]*int_fs[3])/(int_bs[2]*int_bs[3])
                    offset = ((offset + 1).float() * times - 1).to(torch.int64)
                offset = torch.cat([torch.div(offset, int_fs[3], rounding_mode='trunc'),
                                    offset % int_fs[3]], dim=1)  # 1*2*H*W
                offsets.append(offset)

            # deconv for patch pasting
            wi_center = raw_wi[0]
            yi = F.conv_transpose2d(
                yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y = y.contiguous().view(raw_int_fs)

        if not self.return_flow:
            return y, None

        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view(int_fs[0], 2, *int_fs[2:])

        # case1: visualize optical flow: minus current position
        h_add = torch.arange(int_fs[2], device=device).view(
            [1, 1, int_fs[2], 1]).expand(int_fs[0], -1, -1, int_fs[3])
        w_add = torch.arange(int_fs[3], device=device).view(
            [1, 1, 1, int_fs[3]]).expand(int_fs[0], -1, int_fs[2], -1)
        offsets = offsets - torch.cat([h_add, w_add], dim=1)
        # to flow image
        flow = torch.from_numpy(flow_to_image(
            offsets.permute(0, 2, 3, 1).cpu().data.numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2)
        # case2: visualize which pixels are attended
        # flow = torch.from_numpy(highlight_flow((offsets * mask.long()).cpu().data.numpy()))

        if self.rate != 1:
            flow = F.interpolate(flow, scale_factor=self.rate,
                                 mode='bilinear', align_corners=True)

        return y, flow

# ----------------------------------------------------------------------------

def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u / (maxrad + np.finfo(float).eps)
        v = v / (maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))

# ----------------------------------------------------------------------------

def compute_color(u, v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    # colorwheel = COLORWHEEL
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))
    return img

# ----------------------------------------------------------------------------

def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col + YG, 0] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC,
               2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col + CB, 1] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM,
               0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col + MR, 2] = 255 - \
        np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    return colorwheel

# ----------------------------------------------------------------------------


def extract_image_patches(images, ksize, stride, rate, padding='auto'):
    """
    Extracts sliding local blocks \\
    see also: https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html
    """

    padding = rate*(ksize-1)//2 if padding == 'auto' else padding

    unfold = torch.nn.Unfold(kernel_size=ksize,
                             dilation=rate,
                             padding=padding,
                             stride=stride)
    patches = unfold(images)
    return patches  # [N, C*k*k, L], L is the total number of such blocks

# ----------------------------------------------------------------------------

#################################
######### DISCRIMINATOR #########
#################################

class Conv2DSpectralNorm(nn.Conv2d):
    """Convolution layer that applies Spectral Normalization before every call."""

    def __init__(self, cnum_in, cnum_out, 
                 kernel_size, stride, padding=0, bias=True,
                 n_iter=1, eps=1e-12):
        super().__init__(cnum_in,
                         cnum_out, kernel_size=kernel_size,
                         stride=stride, padding=padding, bias=bias)
        
        self.register_buffer("weight_u", torch.empty(self.weight.size(0), 1))
        nn.init.trunc_normal_(self.weight_u)
        self.n_iter = n_iter
        self.eps = eps

    def l2_norm(self, x):
        return F.normalize(x, p=2, dim=0, eps=self.eps)

    def forward(self, x):

        weight_orig = self.weight.flatten(1).detach()

        for _ in range(self.n_iter):
            v = self.l2_norm(weight_orig.t() @ self.weight_u)
            self.weight_u = self.l2_norm(weight_orig @ v)

        sigma = self.weight_u.t() @ weight_orig @ v
        self.weight.data.div_(sigma)

        x = super().forward(x)

        return x

# ----------------------------------------------------------------------------

class DConv(nn.Module):
    """Spectral-Normalized convolution followed by a LeakyReLU activation"""

    def __init__(self, cnum_in, cnum_out, 
                 ksize=5, stride=2, padding='auto'):
        super().__init__()

        padding = (ksize-1)//2 if padding == 'auto' else padding
        self.conv_sn = Conv2DSpectralNorm(
            cnum_in, cnum_out, ksize, stride, padding)
        #self.conv_sn = spectral_norm(nn.Conv2d(cnum_in, cnum_out, ksize, stride, padding))
        self.leaky = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x = self.conv_sn(x)
        x = self.leaky(x)
        return x

# ----------------------------------------------------------------------------

class Discriminator(nn.Module):
    """Fully Convolutional Spectral-Normalized Markovian Discriminator
    from the paper `Free-Form Image Inpainting with Gated Convolution, Yu et. al`."""

    def __init__(self, cnum_in, cnum):
        super().__init__()
        
        self.conv1 = DConv(cnum_in, cnum)
        self.conv2 = DConv(cnum, 2*cnum)
        self.conv3 = DConv(2*cnum, 4*cnum)
        self.conv4 = DConv(4*cnum, 4*cnum)
        self.conv5 = DConv(4*cnum, 4*cnum)
        self.conv6 = DConv(4*cnum, 4*cnum)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = nn.Flatten()(x)

        return x

# ----------------------------------------------------------------------------

class strip_conv(nn.Module):
    def __init__(self,in_c,out_c, ksize = 15, stride=1, padding='same',
                 rate =1):
        super(strip_conv,self).__init__()

        
        # Vertical -> horizontal convolution
        self.leftStripConv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size = [1, ksize],  dilation = rate, padding=padding),
            nn.Conv2d(out_c, out_c, kernel_size = [ksize, 1],  dilation = rate, padding=padding)
        )

        self.rightStripConv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size = [ksize, 1],  dilation = rate, padding=padding),
            nn.Conv2d(out_c, out_c, kernel_size = [1, ksize],  dilation = rate, padding=padding)
        )
        
    def forward(self,x):

        l = self.leftStripConv(x)
        r = self.rightStripConv(x)
        

        return l+r

# ----------------------------------------------------------------------------

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

class TransformerModule(nn.Module):

    def __init__(self, cnum=48, num_class=4):
        super().__init__()
        
        self.num_queries = 100
        self.hidden_dim = 256
        self.num_attention_heads = 8
        self.dim_feedforward = 256 #TODO: check
        self.mask_feature_size= 256
        self.dropout =  0.0

        l3= cnum//4 #96, 128,128
        l4= cnum # 48, 256,256
        l5 = cnum*2 #12, 512,512

        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=[l3, l4, l5],  
            out_channels=self.hidden_dim
        )

        self.pos_embedder = Mask2FormerSinePositionEmbedding(num_pos_feats=self.hidden_dim // 2, normalize=True)
        
        self.input_projection_3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.input_projection_4 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)
        self.input_projection_5 = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1)

        self.adapt_pos2d = nn.Sequential( nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Linear(self.hidden_dim, self.hidden_dim),)
        self.queries_features = nn.Embedding(self.num_queries, self.hidden_dim)

        self.attDecoder = attentionDecoder(self.hidden_dim, self.num_attention_heads, self.dropout, self.dim_feedforward)

        self.layernorm = nn.LayerNorm(self.hidden_dim)

        self.mask_predictor = Mask2FormerMaskPredictor(
            hidden_size=self.hidden_dim,
            num_heads=self.num_attention_heads,
            mask_feature_size=self.mask_feature_size,
        )

        self.convos = DoubleConv(self.num_queries,64) #new
        self.outc = OutConv(64, num_class)
    
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

    def grid_points(self, batch_size, num_position, num_pattern=1):
        
        nx=ny=round(math.sqrt(num_position))
        num_position=nx*ny
        x = (torch.arange(nx) + 0.5) / nx
        y = (torch.arange(ny) + 0.5) / ny
        xy=torch.meshgrid(x,y)
        reference_points=torch.cat([xy[0].reshape(-1)[...,None],xy[1].reshape(-1)[...,None]],-1).cuda()
        #reference_points = reference_points.unsqueeze(0).repeat(batch_size, num_pattern, 1)
        return reference_points

    def forward(self, x_feat, mask = None):
        
        batch_size = x_feat['l3'].shape[0]
        fpn_output = self.fpn(x_feat)
        

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

        # Query
        # Anchor Query embedding
        reference_points = self.grid_points(batch_size, self.num_queries).unsqueeze(1).repeat(1, batch_size, 1) #([256, 4, 2])
        query_emb = self.adapt_pos2d(self.pos2posemb2d(reference_points)) #([256, 4, 256])            
        query_feat = self.queries_features.weight.unsqueeze(1).repeat(1, batch_size, 1) #([256, 4, 256]) 

        atD5 = self.attDecoder(query_feat, query_emb, feat_5, feat_pos_emb5, mask)
        atD4 = self.attDecoder(atD5[0], query_emb, feat_4, feat_pos_emb4, mask)
        atD3 = self.attDecoder(atD4[0], query_emb, feat_3, feat_pos_emb3, mask)

        intermediate_hidden_states = self.layernorm(atD3[0]) #([100, 1, 256])
        
        predicted_mask, attention_mask = self.mask_predictor( #There is a predicted mask for each image in the batch and for each query...
                    intermediate_hidden_states,
                    fpn_output['l3'],
                    (64,64), 
                )
        
        
        #out = self.convos(predicted_mask)  #new. 3,256,128,128 -> 3,1,128,128
        #out = self.outc(out) #new
        

        
        return predicted_mask 