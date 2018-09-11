import argparse
import torch
from PIL import Image
from torchvision import transforms as T
from torch.autograd import Variable
import os



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
    '--net_path', default='', type=str,
    help='path to the .pkl file with the network')

parser.add_argument(
    '--output_path', default='', type=str,
    help='path to the output directory')

opt, _ = parser.parse_known_args()

net = torch.load(opt.net_path)

in_t = T.Compose([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
out_t = T.ToPILImage()

if opt.img_list:
	names = np.loadtxt(opt.img_list, dtype=str).tolist()
else:
	names = os.listdir(opt.input_path)

for name in names:

	input_path = os.path.join(opt.input_path, name)
	
	in_img = Image.open(input_path)

	out_img = out_t(((net(Variable(in_t(in_img)[None].cuda()))[0] + 1) / 2).detach().cpu())

	output_path = os.path.join(opt.output_path, name)

	out_img.save(output_path)
