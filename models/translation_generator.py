from torch import nn
from src import utils
from math import log



def get_args(parser):
    """Add generator-specific options to the parser"""

    parser.add_argument(
        '--gen_num_channels', default=32, type=int,
        help='initial number of channels in convolutions')

    parser.add_argument(
        '--gen_max_channels', default=256, type=int,
        help='maximum number of channels in convolutions')

    parser.add_argument(
        '--gen_kernel_size', default=4, type=int,
        help='kernel size in downsampling convolutions')

    parser.add_argument(
        '--gen_latent_size', default=32, type=int,
        help='spatial size of a tensor at which residual blocks are operating')

    parser.add_argument(
        '--gen_num_res_blocks', default=7, type=int,
        help='number of residual blocks')

    parser.add_argument(
        '--gen_norm_layer', default='none', type=str,
        help='type of the normalization layer,'
             'allowed values: instance|batch|none')

    parser.add_argument(
        '--gen_upsampling_layer', default='conv_transpose', type=str,
        help='upsampling module,'
             'allowed values: conv_transpose|pixel_shuffle|upsampling_nearest')


class Generator(nn.Module):
    """Translation generator architecture by Johnston et al"""

    def __init__(self, opt):
        super(Generator, self).__init__()

        # Get constructors of the required modules
        norm_layer = utils.get_norm_layer(opt.gen_norm_layer)
        upsampling_layer = utils.get_upsampling_layer(opt.gen_upsampling_layer)

        # Calculate the amount of downsampling and upsampling convolutional blocks
        num_down_blocks = int(log(opt.image_size // opt.gen_latent_size, 2))
        num_up_blocks = int(log(opt.image_size // opt.gen_latent_size, 2))

        # Read parameters for convolutional blocks
        in_channels = opt.gen_num_channels
        padding = (opt.gen_kernel_size - 1) // 2
        bias = norm_layer != nn.BatchNorm2d

        # First block is without normalization
        layers = [
            nn.Conv2d(3, in_channels, 7, 1, 3, bias=False),
            nn.ReLU(True)]

        # Downsampling blocks
        for i in range(num_down_blocks):

            # Increase the number of channels by 2x
            out_channels = min(in_channels * 2, opt.gen_max_channels)

            layers += [
                nn.Conv2d(in_channels, out_channels, opt.gen_kernel_size, 2,
                          padding, bias),
                norm_layer(out_channels),
                nn.ReLU(True)]

            in_channels = out_channels

        # Residual blocks
        for i in range(opt.gen_num_res_blocks):
            layers += [utils.ResBlock(in_channels, norm_layer)]

        # Upsampling blocks
        for i in range(num_up_blocks):

            # Decrease the number of channels by 2x
            out_channels = opt.gen_num_channels * 2**(num_up_blocks-i-1)
            out_channels = max(min(out_channels, 
                                   opt.gen_max_channels), 
                               opt.gen_num_channels)

            layers += upsampling_layer(in_channels, out_channels, opt.gen_kernel_size, 2, 
                                       bias)
            layers += [
                norm_layer(out_channels),
                nn.ReLU(True)]

            in_channels = out_channels

        # Last block outputs values in range [-1, 1]
        layers += [
            nn.Conv2d(out_channels, 3, 7, 1, 3, bias=False),
            nn.Tanh()]

        self.generator = nn.Sequential(*layers)

        # Initialize weights
        self.apply(utils.weights_init)

    def forward(self, image):

        return self.generator(image)
