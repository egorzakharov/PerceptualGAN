import torch
from torch import nn
import torch.nn.functional as f
from models.perceptual_loss import PerceptualLoss
import os



def get_loss_layer(loss_type, encoder=None):
    return {
        'l1': nn.L1Loss(),
        'l2': nn.MSELoss(),
        'huber': nn.SmoothL1Loss(),
        'perceptual': PerceptualLoss(
            input_range='tanh',
            average_loss=False,
            extractor=encoder)
    }[loss_type]

def get_norm_layer(norm_type):
    return {
        'batch': nn.BatchNorm2d,
        'instance': nn.InstanceNorm2d,
        'none': Identity
    }[norm_type]

def get_upsampling_layer(upsampling_type):

    def conv_transpose_layer(in_channels, out_channels, kernel_size, 
                             stride, bias):

        padding = (kernel_size - 1) // stride
        output_padding = 1 if kernel_size % 2 else 0

        return [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride,
                                   padding, output_padding, bias=bias)]

    def pixel_shuffle_layer(in_channels, out_channels, kernel_size, 
                            upscale_factor, bias):

        kernel_size -= kernel_size % 2 == 0
        padding = kernel_size // 2

        num_channels = out_channels * upscale_factor**2

        return [nn.Conv2d(in_channels, out_channels, kernel_size, 1,
                          padding),
                nn.PixelShuffle(upscale_factor)]

    def upsampling_nearest_layer(in_channels, out_channels, kernel_size, 
                         scale_factor, bias, mode):

        kernel_size -= kernel_size % 2 == 0
        padding = kernel_size // 2

        return [nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                nn.Conv2d(in_channels, out_channels, kernel_size, 1,
                          padding, bias=bias)]

    return {
        'conv_transpose': conv_transpose_layer,
        'pixel_shuffle': pixel_shuffle_layer,
        'upsampling_nearest': upsampling_nearest_layer
    }[upsampling_type]

def weights_init(module):

    classname = module.__class__.__name__

    if classname.find('Conv') != -1:
        module.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)


class Identity(nn.Module):

    def __init__(self, num_channels=None):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def __repr__(self):
        return ('{name}()'.format(name=self.__class__.__name__))


class ResBlock(nn.Module):
    
    def __init__(self, in_channels, norm_layer):
        super(ResBlock, self).__init__()

        norm_layer = Identity if norm_layer is None else norm_layer
        bias = norm_layer == Identity

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias),
            norm_layer(in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias),
            norm_layer(in_channels))

    def forward(self, input):

        return input + self.block(input)


class ConcatBlock(nn.Module):

    def __init__(
        self,
        enc_channels,
        out_channels, 
        nonlinear_layer=nn.ReLU,
        norm_layer=None,
        norm_layer_cat=None,
        kernel_size=3):
        super(ConcatBlock, self).__init__()

        norm_layer = Identity if norm_layer is None else norm_layer
        norm_layer_cat = Identity if norm_layer_cat is None else norm_layer_cat

        # Get branch from encoder
        layers = get_conv_block(
                enc_channels,
                out_channels,
                nonlinear_layer,
                norm_layer,
                'same', False,
                kernel_size)
        
        layers += [norm_layer_cat(out_channels)]

        self.enc_block = nn.Sequential(*layers)

    def forward(self, input, vgg_input):

        output_enc = self.enc_block(vgg_input)
        output_dis = input

        output = torch.cat([output_enc, output_dis], 1)

        return output


class View(nn.Module):

    def __init__(self):
        super(View, self).__init__()

    def forward(self, x, size=None):

        if len(x.shape) == 2:

            if size is None:
                return x.view(x.shape[0], -1, 1, 1)
            else:
                b, c = x.shape
                _, _, h, w = size
                return x.view(b, c, 1, 1).expand(b, c, h, w)

        elif len(x.shape) == 4:

            return x.view(x.shape[0], -1)

    def __repr__(self):
        return '{name}()'.format(name=self.__class__.__name__)


def save_checkpoint(model, prefix):

    prefix = str(prefix)

    if hasattr(model, 'gen_A'):
        torch.save(
            model.gen_A.module.cpu(),
            os.path.join(model.weights_path, '%s_gen_A.pkl' % prefix))
        model.gen_A.module.cuda(model.gpu_id)

    if hasattr(model, 'gen_B'):
        torch.save(
            model.gen_B.module.cpu(),
            os.path.join(model.weights_path, '%s_gen_B.pkl' % prefix))
        model.gen_B.module.cuda(model.gpu_id)
    
    if hasattr(model, 'dis_A'):
        torch.save(
            model.dis_A.module.cpu(),
            os.path.join(model.weights_path, '%s_dis_A.pkl' % prefix))
        model.dis_A.module.cuda(model.gpu_id)
    
    if hasattr(model, 'dis_B'):
        torch.save(
            model.dis_B.module.cpu(),
            os.path.join(model.weights_path, '%s_dis_B.pkl' % prefix))
        model.dis_B.module.cuda(model.gpu_id)

def load_checkpoint(model, prefix, path=''):

    prefix = str(prefix)

    print('\nLoading checkpoint %s from path %s' % (prefix, path))

    if not path: path = model.weights_path

    path_gen_A = os.path.join(path, '%s_gen_A.pkl' % prefix)
    path_gen_B = os.path.join(path, '%s_gen_B.pkl' % prefix)

    if hasattr(model, 'gen_A'):
        if os.path.exists(path_gen_A):
            print('Loading gen_A to gen_A')
            model.gen_A = torch.load(path_gen_A)
        elif os.path.exists(path_gen_B):
            print('Loading gen_B to gen_A')
            model.gen_A = torch.load(path_gen_B)

    if hasattr(model, 'gen_B'):
        if os.path.exists(path_gen_B):
            print('Loading gen_B to gen_B')
            model.gen_B = torch.load(path_gen_B)
        elif os.path.exists(path_gen_A):
            print('Loading gen_A to gen_B')
            model.gen_B = torch.load(path_gen_A)

    path_dis_A = os.path.join(path, '%s_dis_A.pkl' % prefix)
    path_dis_B = os.path.join(path, '%s_dis_B.pkl' % prefix)

    if hasattr(model, 'dis_A'):
        if os.path.exists(path_dis_A):
            model.dis_A = torch.load(path_dis_A)
        elif os.path.exists(path_dis_B):
            model.dis_A = torch.load(path_dis_B)

    if hasattr(model, 'dis_B'):
        if os.path.exists(path_dis_B):
            model.dis_B = torch.load(path_dis_B)
        elif os.path.exists(path_dis_A):
            model.dis_B = torch.load(path_dis_A)
