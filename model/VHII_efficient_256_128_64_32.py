''' Variable Hyperparameter Image Inpainting
'''
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from core.spectral_norm import spectral_norm as _spectral_norm

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class InpaintGenerator(BaseNetwork):
    def __init__(self, init_weights=True):
        super(InpaintGenerator, self).__init__()
        embed_hidden=[256, 256, 256, 256]
        embed_dims = [256, 128, 64, 32] 
        num_heads=[8, 4, 2, 1]
        depths=[2, 2, 2, 2]
        sr_ratios=[1, 1, 1, 1]
        patchsize = [(4, 4), (8, 8), (16, 16), (32, 32)]
        
        blocks = []
        reduce_channels=True
        
        if reduce_channels:
           for _ in range(depths[0]):
               blocks.append(DepthWiseTransformerBlock(patchsize[3], hidden=embed_hidden[0], embed_dims=embed_dims[0], num_heads=num_heads[0], sr_ratio=sr_ratios[0]))

           for _ in range(depths[1]):
               blocks.append(DepthWiseTransformerBlock(patchsize[2], hidden=embed_hidden[1], embed_dims=embed_dims[1], num_heads=num_heads[1], sr_ratio=sr_ratios[1]))

           for _ in range(depths[2]):
               blocks.append(DepthWiseTransformerBlock(patchsize[1], hidden=embed_hidden[2], embed_dims=embed_dims[2], num_heads=num_heads[2], sr_ratio=sr_ratios[2]))

           for _ in range(depths[3]):
               blocks.append(DepthWiseTransformerBlock(patchsize[0], hidden=embed_hidden[3], embed_dims=embed_dims[3], num_heads=num_heads[3], sr_ratio=sr_ratios[3]))
        else:
           for _ in range(depths[0]):
               blocks.append(TransformerBlock(patchsize[3], embed_dims=embed_dims[0], num_heads=num_heads[0], sr_ratio=sr_ratios[0]))

           for _ in range(depths[1]):
               blocks.append(TransformerBlock(patchsize[2], embed_dims=embed_dims[1], num_heads=num_heads[1], sr_ratio=sr_ratios[1]))

           for _ in range(depths[2]):
               blocks.append(TransformerBlock(patchsize[1], embed_dims=embed_dims[2], num_heads=num_heads[2], sr_ratio=sr_ratios[2]))

           for _ in range(depths[3]):
               blocks.append(TransformerBlock(patchsize[0], embed_dims=embed_dims[3], num_heads=num_heads[3], sr_ratio=sr_ratios[3]))


        self.transformer = nn.Sequential(*blocks)
        
        self.add_pos_emb = AddPosEmb(64,64, embed_hidden[0])

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, embed_hidden[0], kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # decoder: decode images from features
        self.decoder = nn.Sequential(
            pixelup(embed_hidden[3], 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            pixelup(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        )

        if init_weights:
            self.init_weights()

    def forward(self, masked_images, masks):
        # extracting features
        b, c, h, w = masked_images.size()
        masks = masks.view(b, 1, h, w)
        
        enc_feat = self.encoder(masked_images.view(b, c, h, w))
        _, c, h, w = enc_feat.size()
        masks = F.interpolate(masks, scale_factor=1.0/4)
 
        enc_feat = self.add_pos_emb(enc_feat)
        enc_feat = self.transformer({'x': enc_feat, 'm': masks})['x']

        output = self.decoder(enc_feat)
        output = torch.tanh(output)
        return output
        

class pixelup(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel*4,
                              kernel_size=kernel_size, stride=1, padding=padding, padding_mode='reflect')
        
        self.pixel = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel(x)
        return x
        
class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding, padding_mode='reflect')

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)


# #############################################################################
# ############################# Transformer  ##################################
# #############################################################################
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0),
        nn.LeakyReLU(0.2, inplace=True)
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1),
        nn.LeakyReLU(0.2, inplace=True)
    )
    
    
class AddPosEmb(nn.Module):
    def __init__(self, h, w, c):
        super(AddPosEmb, self).__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, h*w, c).float().normal_(mean=0, std=0.02), requires_grad=True)
        self.num_vecs = h*w

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.permute(0,2,3,1).contiguous().view(b, -1, self.num_vecs, c)
        x = x + self.pos_emb
        x = x.permute(0,1,3,2).contiguous().view(b, c, h, w)
        return x


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask):
        att = (query @ key.transpose(-2, -1)) * (1.0 / math.sqrt(key.size(-1)))
        att.masked_fill(mask, -1e9)
        p_attn = F.softmax(att, dim=-1)
        p_val = p_attn @ value
        return p_val, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsize, embed_dims, num_heads, sr_ratio):
        super().__init__()
        self.patchsize = patchsize
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.query_embedding = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            embed_dims, embed_dims, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True))
        self.attention = Attention()

    def forward(self, x, mask=None):
        b, c, h, w = x.size()
        num_dims_per_head = self.embed_dims // self.num_heads

        query = self.query_embedding(x)
        key = self.key_embedding(x)
        value = self.value_embedding(x)
       
        height = self.patchsize[0]
        width = self.patchsize[1]
        out_w, out_h = w // height, h // width

        mask = mask.view(b, 1, 1, out_h, height, out_w, width)
        mask = mask.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(b, 1, out_h*out_w, height*width)
        mask = (mask.mean(-1) > 0.5).unsqueeze(1).repeat(1, 1, out_h*out_w, 1)

        # 1) embedding and reshape
        query = query.view(b, self.num_heads, num_dims_per_head, out_h, height, out_w, width)
        query = query.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(b,  self.num_heads, out_h*out_w, num_dims_per_head*height*width)

        key = key.view(b, self.num_heads, num_dims_per_head, out_h, height, out_w, width)
        key = key.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(b,  self.num_heads, out_h*out_w, num_dims_per_head*height*width)
            
        value = value.view(b, self.num_heads, num_dims_per_head, out_h, height, out_w, width)
        value = value.permute(0, 1, 3, 5, 2, 4, 6).contiguous().view(b,  self.num_heads, out_h*out_w, num_dims_per_head*height*width)

        y, _ = self.attention(query, key, value, mask)

        # 2) "Concat" using a view and apply a final linear.
        y = y.view(b, self.num_heads, out_h, out_w, num_dims_per_head, height, width)
        y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(b, self.num_heads * num_dims_per_head, h, w)

        x = self.output_linear(y)
        return x


# Standard 2 layerd FFN of transformer
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """
    
    def __init__(self, patchsize, embed_dims=128, num_heads=4, sr_ratio=1):
        super().__init__()
        
        self.attention = MultiHeadedAttention(patchsize, embed_dims=embed_dims, num_heads=num_heads, sr_ratio=sr_ratio)
        self.feed_forward = FeedForward(embed_dims)

    def forward(self, x):
        x, m = x['x'], x['m']

        # Self-attention
        x = x + self.attention(x, m)

        # FFN
        x = x + self.feed_forward(x)

        return {'x': x, 'm': m}
        
        
class DepthWiseTransformerBlock(nn.Module):
    """
    MobileVitTransformer = Depthwise Separable Conv + Transformer + Depthwise Separable Conv
    """
    def __init__(self, patchsize, hidden=128, embed_dims=128, num_heads=4, sr_ratio=1, kernel_size=3, reduce_channels=True):
        super().__init__()
        self.hidden = hidden
        self.embed_dims = embed_dims
        self.conv1 = conv_nxn_bn(hidden, hidden, kernel_size)
        self.conv2 = conv_1x1_bn(hidden, embed_dims)

        self.transformer = TransformerBlock(patchsize, embed_dims=embed_dims, num_heads=num_heads)

        self.conv3 = conv_1x1_bn(embed_dims, hidden)
        self.conv4 = conv_nxn_bn(hidden, hidden, kernel_size)
    
    def forward(self, x):
        x, m = x['x'], x['m']

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representations
        x = self.transformer({'x': x, 'm': m})['x']

        # Fusion
        x = self.conv3(x)
        x = self.conv4(x)
        return {'x': x, 'm': m}


# ######################################################################
# ######################################################################


class Discriminator(BaseNetwork):
    def __init__(self, in_channels=3, use_sigmoid=False, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 64

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=nf*1, kernel_size=5, stride=2,
                                    padding=1, bias=not use_spectral_norm, padding_mode='reflect'), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(nf*1, nf*2, kernel_size=5, stride=2,
                                    padding=2, bias=not use_spectral_norm, padding_mode='reflect'), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(nf * 2, nf * 4, kernel_size=5, stride=2,
                                    padding=2, bias=not use_spectral_norm, padding_mode='reflect'), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(nf * 4, nf * 4, kernel_size=5, stride=2,
                                    padding=2, bias=not use_spectral_norm, padding_mode='reflect'), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(nf * 4, nf * 4, kernel_size=5, stride=2,
                                    padding=2, bias=not use_spectral_norm, padding_mode='reflect'), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 4, nf * 4, kernel_size=5,
                      stride=2, padding=2, padding_mode='reflect')
        )

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        feat = self.conv(xs)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        return feat


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module


