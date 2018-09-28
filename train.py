import argparse
import importlib
from src.dataset import get_dataloaders
from src.logs import Logs
from src import utils
import os
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR



# Options specification
parser = argparse.ArgumentParser(conflict_handler='resolve')

# Dataset options
parser.add_argument(
    '--train_img_A_path', default='', type=str,
    help='path to train images for domain A,'
         'should be a folder or txt file image names')

parser.add_argument(
    '--train_img_B_path', default='', type=str,
    help='path to train images for domain B,'
         'should be a folder or txt file image names')

parser.add_argument(
    '--test_img_A_path', default='', type=str,
    help='path to test folder for domain A,'
         'should be a folder or txt file image names')

parser.add_argument(
    '--test_img_B_path', default='', type=str,
    help='path to test images for domain B,'
         'should be a folder or txt file image names')

parser.add_argument(
    '--img_path', default='', type=str,
    help='required when train/test path options are txt lists')

parser.add_argument(
    '--num_workers', default=4, type=int,
    help='number of data loading workers')

parser.add_argument(
    '--batch_size', default=16, type=int)

parser.add_argument(
    '--input_transforms', default='', type=str,
    help='comma separated transformations,'
         'allowed values: scale|flip|crop|random_crop')

parser.add_argument(
    '--image_size', default=0, type=int,
    help='resolution of images')

# Training options
parser.add_argument(
    '--gpu_ids', default='0', type=str,
    help='list of GPUs to train the model on')

parser.add_argument(
    '--epoch_len', default=1000, type=int,
    help='number of batches per epoch')

parser.add_argument(
    '--num_epoch', default=50, type=int)

parser.add_argument(
    '--lr', default=1e-4, type=float)

parser.add_argument(
    '--beta1', default=0.9, type=float,
    help='beta1 for Adam')

parser.add_argument(
    '--manual_seed', default=9107, type=int)

parser.add_argument(
    '--experiment_name', default='', type=str,
    help='name of the folder to store training data in')

parser.add_argument(
    '--checkpoint_freq', default=1, type=int,
    help='frequency (in epoch) at which checkpoints are made')

parser.add_argument(
    '--which_epoch', default='latest', type=str,
    help='epoch to continue training from')

# Model options
parser.add_argument(
    '--model_type', default='cyclegan', type=str,
    help='allowed values: cyclegan|pix2pix')

parser.add_argument(
    '--enc_type', default='vgg19_pytorch_modified', type=str,
    help='network with pretrained features,'
         'allowed values: vgg19_caffe|vgg19_pytorch|vgg19_pytorch_modified')

parser.add_argument(
    '--mse_loss_type', default='', type=str,
    help='criterion used for MSE losses,'
         'allowed values: l1|l2|huber|perceptual')

parser.add_argument(
    '--mse_loss_weight', default=1., type=float,
    help='weight of each MSE loss term')

parser.add_argument(
    '--adv_loss_weight', default=1., type=float,
    help='weight of each adversarial loss term')

parser.add_argument(
    '--pretrained_gen_path', default='', type=str,
    help='path of pretrained generator')

# Generator-specific options
m = importlib.import_module('models.translation_generator')
m.get_args(parser)

# Discriminator-specific options
m = importlib.import_module('models.discriminator')
m.get_args(parser)

# Read options
opt, _ = parser.parse_known_args()

# Set random seed
np.random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)
torch.cuda.manual_seed_all(opt.manual_seed)

print(opt)

experiment_path = os.path.join('runs', opt.experiment_name)

# Make directories
if not os.path.exists(experiment_path):
    os.makedirs(experiment_path)
    os.makedirs(experiment_path + '/checkpoints')

# Save opts
file_name = experiment_path + '/opt.txt'
with open(file_name, 'wt') as opt_file:
    for k, v in sorted(vars(opt).items()):
        opt_file.write('%s: %s\n' % (str(k), str(v)))

# Preprocess the options
opt.input_transforms = opt.input_transforms.split(',')
opt.gpu_ids = [int(i) for i in opt.gpu_ids.split(',')]
opt.dis_input_sizes = [int(i) for i in opt.dis_input_sizes.split(',')]
opt.dis_input_num_channels = [int(i) for i in opt.dis_input_num_channels.split(',')]
opt.dis_output_sizes = [int(i) for i in opt.dis_output_sizes.split(',')]

# Get dataloaders
train_dataloader, test_dataloader = get_dataloaders(opt)

# Initialize model
m = importlib.import_module('models.' + opt.model_type)
model = m.Model(opt)

# Initialize optimizers
if hasattr(model, 'gen_params'):

    opt_G = Adam(model.gen_params, lr=opt.lr, betas=(opt.beta1, 0.999))

    path_opt_G = os.path.join(
        model.weights_path, 
        '%s_opt_G.pkl' % opt.which_epoch)

    if os.path.exists(path_opt_G):
        opt_G.load_state_dict(torch.load(path_opt_G))

if hasattr(model, 'dis_params'):

    opt_D = Adam(model.dis_params, lr=opt.lr, betas=(opt.beta1, 0.999))

    path_opt_D = os.path.join(
        model.weights_path, 
        '%s_opt_D.pkl' % opt.which_epoch)

    if os.path.exists(path_opt_D):
        opt_D.load_state_dict(torch.load(path_opt_D))

logs = Logs(model, opt)

epoch_start = 0 if opt.which_epoch == 'latest' else int(opt.which_epoch)

for epoch in range(epoch_start + 1, opt.num_epoch + 1):

    model.train()

    for inputs in train_dataloader:

        model.forward(inputs)

        if hasattr(model, 'dis_params'):

            for p in model.dis_params:
                p.requires_grad = False

        opt_G.zero_grad()
        model.backward_G()
        opt_G.step()

        if hasattr(model, 'dis_params'):

            for p in model.dis_params:
                p.requires_grad = True

            opt_D.zero_grad()
            model.backward_D()
            opt_D.step()
            
    logs.update_losses('train')

    model.eval()

    for inputs in test_dataloader:

        with torch.no_grad():

            model.forward(inputs)
            
            model.backward_G() # needed to calculate losses

            if hasattr(model, 'dis_params'): 
                model.backward_D()
            
        logs.update_losses('test')

    logs.update_tboard(epoch)
    
    # Save weights
    if not epoch % opt.save_every_epoch:

        utils.save_checkpoint(model, epoch)

        if hasattr(model, 'gen_params'):
            torch.save(
                opt_G.state_dict(),
                os.path.join(model.weights_path, '%d_opt_G.pkl' % epoch))

        if hasattr(model, 'dis_params'):
            torch.save(
                opt_D.state_dict(),
                os.path.join(model.weights_path, '%d_opt_D.pkl' % epoch))

logs.close()
utils.save_checkpoint(model, 'latest')