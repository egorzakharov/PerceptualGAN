import torch
from torch import nn
from .translation_generator import Generator
from .discriminator import Discriminator
from .discriminator_loss import DiscriminatorLoss
from .perceptual_loss import FeatureExtractor
from src import utils
from torch.autograd import Variable
import os
import importlib



class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()

        self.gpu_id = opt.gpu_ids[0]
        self.weights_path = os.path.join('runs', opt.experiment_name, 'checkpoints')

        # Generator
        self.gen_B = Generator(opt)

        # Discriminator
        if opt.adv_loss_weight: 

            self.dis_B = Discriminator(opt)

        # Load weights
        utils.load_checkpoint(self, opt.which_epoch, opt.pretrained_gen_path)

        # Print architectures
        print('\nGen A to B\n')
        num_params = 0
        for p in self.gen_B.parameters():
            num_params += p.numel()
        print(self.gen_B)
        print('Number of parameters: %d' % num_params)

        self.gen_params = self.gen_B.parameters()

        if opt.adv_loss_weight:

            print('\nDis B\n')
            num_params = 0
            for p in self.dis_B.parameters():
                num_params += p.numel()
            print(self.dis_B)
            print('Number of parameters: %d' % num_params)

            self.dis_params = self.dis_B.parameters()

            # Losses
            self.crit_dis = DiscriminatorLoss(opt)

        self.adv_weight = opt.adv_loss_weight

        # If an encoder is required, load the weights
        if (opt.mse_loss_type == 'perceptual' or 
            opt.adv_loss_weight and opt.dis_use_encoder):

            # Load encoder
            if opt.enc_type[:5] == 'vgg19':
                self.layers = '1,6,11,20,29'

            self.enc = FeatureExtractor(
                input_range='tanh',
                net_type=opt.enc_type,
                layers=self.layers).eval()

            print('')
            print(self.enc)
            print('')

        else:

            self.enc = None

        self.crit_mse = utils.get_loss_layer(opt.mse_loss_type, self.enc)
        self.mse_weight = opt.mse_loss_weight

        # Load onto gpus
        self.gen_B = nn.DataParallel(self.gen_B.cuda(self.gpu_id), opt.gpu_ids)
        if opt.adv_loss_weight:
            self.dis_B = nn.DataParallel(self.dis_B.cuda(self.gpu_id), opt.gpu_ids)
        if self.enc is not None:
            self.enc = nn.DataParallel(self.enc.cuda(self.gpu_id), opt.gpu_ids)

    def forward(self, inputs):

        real_A, real_B = inputs
        
        # Input images
        self.real_B = Variable(real_B.cuda(self.gpu_id))
        self.real_A = Variable(real_A.cuda(self.gpu_id))
        
        # Fake images
        self.fake_B = self.gen_B(self.real_A)

    def backward_G(self):

        # Identity loss
        self.loss_ident_B = self.crit_mse(self.fake_B, self.real_B)

        loss_mse = self.loss_ident_B

        # GAN loss
        if self.adv_weight:

            loss_dis, _ = self.crit_dis(
                dis=self.dis_B,
                img_real_dst=self.fake_B, 
                img_real_src=self.real_A,
                enc=self.enc)

        else:

            loss_dis = 0

        loss_G = loss_mse * self.mse_weight + loss_dis * self.adv_weight

        if self.training:
            loss_G.backward()

        # Get values for visualization
        self.loss_ident_B = self.loss_ident_B.data.item()

    def backward_D(self):

        loss_dis, self.losses_adv_B = self.crit_dis(
            dis=self.dis_B,
            img_real_dst=self.real_B, 
            img_fake_dst=self.fake_B.detach(),
            img_real_src=self.real_A,
            enc=self.enc)

        loss_D = loss_dis

        if self.training:
            loss_D.backward()

    def train(self, mode=True):

        self.training = mode
        
        self.gen_B.train(mode)

        if self.adv_weight:
            
            self.dis_B.train(mode)

        return self