import torch
from torch import nn
from math import log
from src import utils



def get_args(parser):
    """Add discriminator-specific options to the parser"""

    parser.add_argument(
        '--dis_input_sizes', default='256,128,64,32,16', type=str,
        help='comma separated list of spatial sizes of input tensors')

    parser.add_argument(
        '--dis_input_num_channels', default='64,128,256,512,512', type=str,
        help='comma separated list of channels of input tensors')

    parser.add_argument(
        '--dis_output_sizes', default='8,4,1', type=str,
        help='comma separated list of spatial sizes of output tensors')

    parser.add_argument(
        '--dis_num_channels', default=64, type=int,
        help='initial number of channels in convolutions')

    parser.add_argument(
        '--dis_max_channels', default=256, type=int,
        help='maximum number of channels in convolutions')

    parser.add_argument(
        '--dis_adv_loss_type', default='gan', type=str,
        help='loss type for classifier outputs'
             'allowed values: gan|lsgan')

    parser.add_argument(
        '--dis_use_encoder', action='store_true',
        help='use pretrained encoder features as discriminator input')

    parser.add_argument(
        '--dis_kernel_size', default=4, type=int,
        help='kernel size in downsampling convolutions')

    parser.add_argument(
        '--dis_kernel_size_io', default=3, type=int,
        help='kernel size in input and output convolutions, must be an odd number')


class Discriminator(nn.Module):
    """Discriminator with input as features or image"""

    def __init__(self, opt):
        super(Discriminator, self).__init__()

        # Store options for forward pass
        self.dis_input_sizes = opt.dis_input_sizes
        self.dis_output_sizes = opt.dis_output_sizes

        # Calculate the amount of downsampling convolutional blocks
        num_down_blocks = int(log(
            opt.dis_input_sizes[0] // max(opt.dis_output_sizes[-1], 4), 2))

        # Read initial parameters
        in_channels = opt.dis_input_num_channels[0]
        out_channels = opt.dis_num_channels
        padding = (opt.dis_kernel_size - 1) // 2
        padding_io = opt.dis_kernel_size_io // 2
        spatial_size = opt.dis_input_sizes[0]

        # Convolutional blocks
        self.blocks = nn.ModuleList()
        self.input_blocks = nn.ModuleList()
        self.output_blocks = nn.ModuleList()

        for i in range(num_down_blocks):

            # Downsampling block
            self.blocks += [nn.Sequential(
                nn.Conv2d(in_channels, out_channels, opt.dis_kernel_size, 2,
                          padding),
                nn.LeakyReLU(0.2, True))]

            in_channels = out_channels
            spatial_size //= 2

            # If size of downsampling block's output is equal to one of the inputs
            if spatial_size in opt.dis_input_sizes:

                # Get the number of channels in the next input
                in_channels = opt.dis_input_num_channels[len(self.input_blocks)+1]

                self.input_blocks += [nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, opt.dis_kernel_size_io, 1,
                              padding_io),
                    nn.LeakyReLU(0.2, True))]

                in_channels = out_channels * 2

            # If classifier is operating at the block's output size
            if spatial_size in opt.dis_output_sizes:

                self.output_blocks += [
                    nn.Conv2d(in_channels, 1, opt.dis_kernel_size_io, 1, 
                              padding_io)]

            out_channels = min(out_channels * 2, opt.dis_max_channels)

        # If 1x1 classifier is required at the end
        if opt.dis_output_sizes[-1] == 1:
            self.output_blocks += [nn.Conv2d(out_channels, 1, 4, 4)]

        # Initialize weights
        self.apply(utils.weights_init)

    def forward(self, img_dst, img_src=None, encoder=None):

        # Encode inputs
        input = img_dst

        # Source input is concatenate via batch dim for speed up
        if img_src is not None:
            input = torch.cat([input, img_src, 0])

        inputs = encoder(input)

        # Reshape source inputs from batch dim to channels
        if img_src is not None:

            for i in range(len(inputs)):
                b, c, h, w = inputs[i].shape
                inputs[i] = inputs[i].view(b//2, c*2, h, w)

        output = inputs[0]

        # Current spatial size and indices for inputs and outputs
        spatial_size = output.shape[2]
        input_idx = 0
        output_idx = 0

        # List of multiscale predictions
        preds = []

        for block in self.blocks:

            output = block(output)

            spatial_size //= 2

            if spatial_size in self.dis_input_sizes:

                # Concatenate next input to current output
                input = self.input_blocks[input_idx](inputs[input_idx+1])

                output = torch.cat([output, input], 1)

                input_idx += 1

            if spatial_size in self.dis_output_sizes:

                # Predict probabilities in PatchGAN style
                preds += [self.output_blocks[output_idx](output)]

                output_idx += 1

        if 1 in self.dis_output_sizes:

            # Final probability prediction
            preds += [self.output_blocks[output_idx](output)]

        return preds
