import argparse
import torch
from PIL import Image
from torchvision import transforms as T
from torch.autograd import Variable
import os
import numpy as np



# Options specification
parser = argparse.ArgumentParser(conflict_handler='resolve')

# Dataset options
parser.add_argument(
    '--input_path', default='', type=str,
    help='path to input images')

parser.add_argument(
    '--img_list', default='', type=str,
    help='*optional* path to txt list with image names subset')

parser.add_argument(
    '--img_size', default=-1, type=int,
    help='rescale input image to that size')

parser.add_argument(
    '--net_path', default='', type=str,
    help='path to the .pkl file with the network')

parser.add_argument(
    '--output_path', default='', type=str,
    help='path to the output directory')

opt, _ = parser.parse_known_args()

net = torch.load(opt.net_path).cuda()

transforms = []

if opt.img_size > 0:
	transforms += [T.Scale(opt.img_size)]

transforms += [T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)]

in_t = T.Compose(transforms)

out_t = T.ToPILImage()

if opt.img_list:
	names = np.loadtxt(opt.img_list, dtype=str).tolist()
else:
	names = os.listdir(opt.input_path)

for name in names:

	in_img = Image.open(os.path.join(opt.input_path, name))

	out_img = out_t(((net(Variable(in_t(in_img)[None].cuda()))[0] + 1) / 2).detach().cpu())

	output_path = os.path.join(opt.output_path, name)

	in_img.save(os.path.join(opt.output_path, name.split('.')[0] + '_original.jpg'))
	out_img.save(os.path.join(opt.output_path, name.split('.')[0] + '_result.jpg'))
