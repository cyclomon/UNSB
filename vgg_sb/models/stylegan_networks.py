"""
The network architectures is based on PyTorch implemenation of StyleGAN2Encoder.
Original PyTorch repo: https://github.com/rosinality/style-based-gan-pytorch
Original PyTorch repo of CUT: https://github.com/taesungp/contrastive-unpaired-translation
Origianl StyelGAN2 paper: https://github.com/NVlabs/stylegan2
We　use the network architecture for our single-image traning setting.
"""
import math
import numpy as np
import random
import torch
from torch import nn
from torch.nn import functional as F


##################################################################################
# Discriminator
##################################################################################
class StyleGAN2Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, size=None, opt=None):
        super(StyleGAN2Discriminator, self).__init__()
        self.opt = opt
        self.stddev_group = 16
        if size is None:
            size = 2 ** int((np.rint(np.log2(min(opt.load_size, opt.crop_size)))))
            if "patch" in self.opt.netD and self.opt.D_patch_size is not None:
                size = 2 ** int(np.log2(self.opt.D_patch_size))

        channel_multiplier = ndf / 64
        channels = {
            4: min(384, int(round(4096 * channel_multiplier))),
            8: min(384, int(round(2048 * channel_multiplier))),
            16: min(384, int(round(1024 * channel_multiplier))),
            32: min(384, int(round(512 * channel_multiplier))),
            64: int(round(256 * channel_multiplier)),
            128: int(round(128 * channel_multiplier)),
            256: int(round(64 * channel_multiplier)),
            512: int(round(32 * channel_multiplier)),
            1024: int(round(16 * channel_multiplier)),
        }
        blur_kernel = [1, 3, 3, 1]

        convs = [ConvLayer(input_nc, channels[size], 1)]
        log_size = int(math.log(size, 2))
        in_channel = channels[size]
        if 'smallpatch' in self.opt.netD:
            final_res_log2 = 5
        elif 'patch' in self.opt.netD:
            final_res_log2 = 4
        else:
            final_res_log2 = 3

        for i in range(log_size, final_res_log2, -1):
            out_channel = channels[2 ** (i -1)]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        if 'tile' in self.opt.netD:
            in_channel += 1
        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        if 'patch' in self.opt.netD:
            self.final_linear = ConvLayer(channels[4], 1, 3, bias=False, activate=False)
        else:
            size = 2 ** int(final_res_log2)
            self.final_linear = nn.Sequential(
                EqualLinear(channels[4] * size * size, channels[4], activation='fused_lrelu'),
                EqualLinear(channels[4], 1)
            )

    def forward(self, input, get_minibatch_features=False):
        if "patch" in self.opt.netD and self.opt.D_patch_size is not None:
            _, _, h, w = input.size()
            y = torch.randint(h - self.opt.D_patch_size, ())
            x = torch.randint(w - self.opt.D_patch_size, ())
            input = input[:, :, y:y + self.opt.D_patch_size, x:x + self.opt.D_patch_size]
        out = input
        out = self.convs(out)
        b, c, h, w = out.size()

        if get_minibatch_features and 'tile' in self.opt.netD:
            group = min(b, self.stddev_group)
            stddev = out.view(group, -1, 1, c // 1, h, w)
            stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
            stddev = stddev.mean([2, 3, 4], keepdim=True).squeeze(2)
            stddev = stddev.repeat(group, 1, h, w)
            out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)
        if 'patch' not in self.opt.netD:
            out = out.view(b, -1)
        out = self.final_linear(out)

        return out


##################################################################################
# Generator
##################################################################################
class StyleGAN2Generator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, opt=None):
        super(StyleGAN2Generator, self).__init__()
        self.opt = opt
        self.encoder = StyleGAN2Encoder(input_nc, ngf, n_blocks, opt)
        self.decoder = StyleGAN2Decoder(output_nc, ngf, n_blocks, opt)

    def forward(self, input, layers=[], encode_only=False):
        feat, feats = self.encoder(input, layers, encode_only)
        if encode_only:
            return None, feats
        else:
            fake = self.decoder(feat)
            return fake, feats


##################################################################################
# Encoder and Decoders
##################################################################################
class StyleGAN2Encoder(nn.Module):
    def __init__(self, input_nc, ngf=64, n_blocks=6, opt=None):
        super(StyleGAN2Encoder, self).__init__()
        assert opt is not None
        self.opt = opt
        channel_multiplier = ngf / 32
        channels = {
            4: min(512, int(round(4096 * channel_multiplier))),
            8: min(512, int(round(2048 * channel_multiplier))),
            16: min(512, int(round(1024 * channel_multiplier))),
            32: min(512, int(round(512 * channel_multiplier))),
            64: int(round(256 * channel_multiplier)),
            128: int(round(128 * channel_multiplier)),
            256: int(round(64 * channel_multiplier)),
            512: int(round(32 * channel_multiplier)),
            1024: int(round(16 * channel_multiplier)),
        }

        blur_kernel = [1, 3, 3, 1]

        cur_res = 2 ** int((np.rint(np.log2(min(opt.load_size, opt.crop_size)))))
        convs = [nn.Identity(),
                 ConvLayer(input_nc, channels[cur_res], 1)]

        num_downsampling = self.opt.stylegan2_G_num_downsampling
        for i in range(num_downsampling):
            in_channel = channels[cur_res]
            out_channel = channels[cur_res // 2]
            convs.append(ResBlock(in_channel, out_channel, blur_kernel, downsample=True))
            cur_res = cur_res // 2

        for i in range(n_blocks // 2):
            n_channel = channels[cur_res]
            convs.append(ResBlock(n_channel, n_channel, downsample=False))

        self.convs = nn.Sequential(*convs)

    def forward(self, input, layers=[], get_features=False):
        if len(layers) > 0:
            feat = input
            feats = []
            if -1 in layers:
                layers.append(len(self.convs) - 1)
            for layer_id, layer in enumerate(self.convs):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                if layer_id == layers[-1] and get_features:
                    return None, feats
            return feat, feats
        else:
            return self.convs(input), None


class StyleGAN2Decoder(nn.Module):
    def __init__(self, output_nc, ngf=64, n_blocks=6, opt=None):
        super(StyleGAN2Decoder, self).__init__()
        assert opt is not None
        self.opt = opt
        channel_multiplier = ngf / 32
        channels = {
            4: min(512, int(round(4096 * channel_multiplier))),
            8: min(512, int(round(2048 * channel_multiplier))),
            16: min(512, int(round(1024 * channel_multiplier))),
            32: min(512, int(round(512 * channel_multiplier))),
            64: int(round(256 * channel_multiplier)),
            128: int(round(128 * channel_multiplier)),
            256: int(round(64 * channel_multiplier)),
            512: int(round(32 * channel_multiplier)),
            1024: int(round(16 * channel_multiplier)),
        }

        blur_kernel = [1, 3, 3, 1]

        num_downsampling = self.opt.stylegan2_G_num_downsampling
        cur_res = 2 ** int((np.rint(np.log2(min(opt.load_size, opt.crop_size))))) // (2 ** num_downsampling)
        convs = []

        for i in range(n_blocks // 2):
            n_channel = channels[cur_res]
            convs.append(ResBlock(n_channel, n_channel, downsample=False))

        for i in range(num_downsampling):
            in_channel = channels[cur_res]
            out_channel = channels[cur_res * 2]
            inject_noise = "small" not in self.opt.netG
            convs.append(StyledConv(in_channel, out_channel, 3, upsample=True, blur_kernel=blur_kernel,
                                    inject_noise=inject_noise))
            cur_res = cur_res * 2

        convs.append(ConvLayer(channels[cur_res], output_nc, 1))

        self.convs = nn.Sequential(*convs)

    def forward(self, input):
        return F.tanh(self.convs(input))


class StyleGAN2DecoderWS(nn.Module):
    def __init__(self, size, style_dim, n_mlp, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], lr_mlp=0.01):
        super(StyleGAN2DecoderWS, self).__init__()

        self.size = size
        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))

        self.style = nn.Sequential(*layers)

        channels = {
            4: min(512, int(round(4096 * channel_multiplier))),
            8: min(512, int(round(2048 * channel_multiplier))),
            16: min(512, int(round(1024 * channel_multiplier))),
            32: min(512, int(round(512 * channel_multiplier))),
            64: int(round(256 * channel_multiplier)),
            128: int(round(128 * channel_multiplier)),
            256: int(round(64 * channel_multiplier)),
            512: int(round(32 * channel_multiplier)),
            1024: int(round(16 * channel_multiplier)),
        }

        self.input = ConstantInput(channels[4])
        self.conv1 = StyledConv(channels[4], channels[4], 3, style_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        in_channel = channels[4]
        for i in range(3, self.log_size + 1):
            out_channel = channels[2 ** i]
            self.convs.append(StyledConv(in_channel, out_channel, 3, style_dim, upsample=True, blur_kernel=blur_kernel))
            self.convs.append(StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel))
            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device
        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]
        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))
        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(n_latent, self.style_dim, device=self.input.input.device)
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(self, styles, return_latents=False, inject_index=None, truncation=1, truncation_latent=None,
        input_is_latent=False, noise=None, randomize_noise=True,):

        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)]

        if truncation < 1:
            style_t = []
            for style in styles:
                style_t.append(truncation_latent + truncation*(style - truncation_latent))
            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent
            if len(styles[0].shape) < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            else:
                latent = styles[0]
        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)
            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip
        if return_latents:
            return image, latent
        else:
            return image, None


##################################################################################
# Basic Blocks
##################################################################################
class ModulatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True, upsample=False,
                 downsample=False, blur_kernel=[1, 3, 3, 1]):
        super(ModulatedConv2d, self).__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            self.blur = Blur(blur_kernel, pad=((p + 1) // 2 + factor - 1, p // 2 + 1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            self.blur = Blur(blur_kernel, pad=((p + 1) // 2, p // 2))

        fan_in = in_channel * kernel_size ** 2
        self.scale = math.sqrt(1) / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))

        if style_dim is not None and style_dim > 0:
            self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})')

    def forward(self, input, style):
        b, in_channel, h, w = input.shape

        if style is not None:
            style = self.modulation(style).view(b, 1, in_channel, 1, 1)
        else:
            style = torch.ones(b, 1, in_channel, 1, 1).to(input.device)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(b, self.out_channel, 1, 1, 1)

        weight = weight.view(b * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        if self.upsample:
            input = input.view(1, b * in_channel, h, w)
            weight = weight.view(b, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(b * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=b)
            _, _, height, width = out.shape
            out = out.view(b, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, b * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=b)
            _, _, height, width = out.shape
            out = out.view(b, self.out_channel, height, width)

        else:
            input = input.view(1, b * in_channel, h, w)
            out = F.conv2d(input, weight, padding=self.padding, groups=b)
            _, _, height, width = out.shape
            out = out.view(b, self.out_channel, height, width)

        return out


class StyledConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim=None, upsample=False, blur_kernel=[1, 3, 3, 1],
                 demodulate=True, inject_noise=None):
        super(StyledConv, self).__init__()

        self.inject_noise = inject_noise
        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, style_dim, upsample=upsample,
                                    blur_kernel=blur_kernel, demodulate=demodulate)

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style=None, noise=None):
        out = self.conv(input, style)
        if self.inject_noise:
            out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip

        return out


class ConvLayer(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size, downsample=False, blur_kernel=[1, 3, 3, 1], bias=True, activate=True,):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            layers.append(Blur(blur_kernel, pad=((p + 1) // 2, p // 2)))
            stride = 2
            self.padding = 0
        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(EqualConv2d(in_channel, out_channel, kernel_size, padding=self.padding, stride=stride, bias=bias and not activate,))

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))
            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], downsample=True, skip_gain=1.0):
        super().__init__()

        self.skip_gain = skip_gain
        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample, blur_kernel=blur_kernel)

        if in_channel != out_channel or downsample:
            self.skip = ConvLayer(in_channel, out_channel, 1, downsample=downsample, activate=False, bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out * self.skip_gain + skip) / math.sqrt(self.skip_gain ** 2 + 1.0)

        return out


##################################################################################
# Basic Functions
##################################################################################
def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    return F.leaky_relu(input + bias, negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1, channel, 1, 1))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        out = fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)
        return out


def upfirdn2d_native(input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1):
    _, minor, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, minor, in_h, 1, in_w, 1)
    out = F.pad(out, [0, up_x - 1, 0, 0, 0, up_y - 1, 0, 0])
    out = out.view(-1, minor, in_h * up_y, in_w * up_x)

    out = F.pad(out, [max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)])
    out = out[:, :, max(-pad_y0, 0): out.shape[2] - max(-pad_y1, 0), max(-pad_x0, 0): out.shape[3] - max(-pad_x1, 0),]

    out = out.reshape([-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1])
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(-1, minor, in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1, in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,)

    return out[:, :, ::down_y, ::down_x]


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    return upfirdn2d_native(input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1])


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if len(k.shape) == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = math.sqrt(1) / math.sqrt(in_channel * (kernel_size ** 2))

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})')


class EqualLinear(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True, bias_init=0, lr_mul=1.0, activation=None):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel).fill_(bias_init))
        else:
            self.bias = None

        self.activation = activation

        self.scale = (math.sqrt(1) / math.sqrt(in_channel)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)
        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})')