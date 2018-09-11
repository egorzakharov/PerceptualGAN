import torch
from torch import nn
from .translation_generator import Generator
from .discriminator import Discriminator
from .discriminator_loss import DiscriminatorLoss
from .perceptual_loss import FeatureExtractor
from itertools import chain
from src import utils
from torch.autograd import Variable
import os



class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()

        self.gpu_id = opt.gpu_ids[0]
        self.weights_path = os.path.join('runs', opt.experiment_name, 'checkpoints')

        # Generators
        self.gen_A = Generator(opt)
        self.gen_B = Generator(opt)

        # Discriminators
        self.dis_A = Discriminator(opt)
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

        print('\nGen B to A\n')
        num_params = 0
        for p in self.gen_A.parameters():
            num_params += p.numel()
        print(self.gen_A)
        print('Number of parameters: %d' % num_params)

        print('\nDis A\n')
        num_params = 0
        for p in self.dis_A.parameters():
            num_params += p.numel()
        print(self.dis_A)
        print('Number of parameters: %d' % num_params)

        print('\nDis B\n')
        num_params = 0
        for p in self.dis_B.parameters():
            num_params += p.numel()
        print(self.dis_B)
        print('Number of parameters: %d' % num_params)

        self.gen_params = chain(
            self.gen_A.parameters(),
            self.gen_B.parameters())

        self.dis_params = chain(
            self.dis_A.parameters(),
            self.dis_B.parameters())

        # Losses
        self.crit_dis = DiscriminatorLoss(opt)

        self.adv_weight = opt.adv_loss_weight

        # If an encoder is required, load the weights
        if opt.mse_loss_type == 'perceptual' or opt.dis_use_encoder:

            # Load encoder
            if opt.enc_type[:5] == 'vgg19':
                layers = '1,6,11,20,29'

            self.enc = FeatureExtractor(
                input_range='tanh',
                net_type=opt.enc_type,
                layers=layers).eval()

            print('')
            print(self.enc)
            print('')

        else:

            self.enc = None

        self.crit_mse = utils.get_loss_layer(opt.mse_loss_type, self.enc)
        self.mse_weight = opt.mse_loss_weight

        # Load onto gpus
        self.gen_A = nn.DataParallel(self.gen_A.cuda(self.gpu_id), opt.gpu_ids)
        self.gen_B = nn.DataParallel(self.gen_B.cuda(self.gpu_id), opt.gpu_ids)
        self.dis_A = nn.DataParallel(self.dis_A.cuda(self.gpu_id), opt.gpu_ids)
        self.dis_B = nn.DataParallel(self.dis_B.cuda(self.gpu_id), opt.gpu_ids)
        if self.enc is not None: 
            self.enc = nn.DataParallel(self.enc.cuda(self.gpu_id), opt.gpu_ids)

    def forward(self, inputs):

        real_A, real_B = inputs

        # Input images
        self.real_A = Variable(real_A.cuda(self.gpu_id))
        self.real_B = Variable(real_B.cuda(self.gpu_id))

        # Fake images
        self.fake_B = self.gen_B(self.real_A)
        self.fake_A = self.gen_A(self.real_B)

    def backward_G(self):

        # Cycle loss
        cycle_A = self.gen_A(self.fake_B)
        cycle_B = self.gen_B(self.fake_A)

        self.loss_cycle_A = self.crit_mse(cycle_A, self.real_A)
        self.loss_cycle_B = self.crit_mse(cycle_B, self.real_B)

        # Identity loss
        ident_A = self.gen_A(self.real_A)
        ident_B = self.gen_B(self.real_B)

        self.loss_ident_A = self.crit_mse(ident_A, self.real_A)
        self.loss_ident_B = self.crit_mse(ident_B, self.real_B)

        # MSE loss
        loss_mse = (self.loss_cycle_A + self.loss_ident_A +
                    self.loss_cycle_B + self.loss_ident_B)

        # GAN loss
        loss_dis_A, _ = self.crit_dis(
            dis=self.dis_A,
            img_real_dst=self.fake_A,
            enc=self.enc)

        loss_dis_B, _ = self.crit_dis(
            dis=self.dis_B,
            img_real_dst=self.fake_B,
            enc=self.enc)

        loss_dis = loss_dis_A + loss_dis_B

        loss_G = loss_mse * self.mse_weight + loss_dis * self.adv_weight

        if self.training:
            loss_G.backward()

        # Get values for visualization
        self.loss_cycle_A = self.loss_cycle_A.data.item()
        self.loss_cycle_B = self.loss_cycle_B.data.item()
        self.loss_ident_A = self.loss_ident_A.data.item()
        self.loss_ident_B = self.loss_ident_B.data.item()

    def backward_D(self):

        loss_dis_A, self.losses_adv_A = self.crit_dis(
            dis=self.dis_A,
            img_real_dst=self.real_A, 
            img_fake_dst=self.fake_A.detach(),
            enc=self.enc)

        loss_dis_B, self.losses_adv_B = self.crit_dis(
            dis=self.dis_B,
            img_real_dst=self.real_B, 
            img_fake_dst=self.fake_B.detach(),
            enc=self.enc)

        loss_D = loss_dis_A + loss_dis_B

        if self.training:
            loss_D.backward()

    def train(self, mode=True):
        """Doesn't change encoder mode"""

        self.training = mode
        
        self.gen_A.train(mode)
        self.gen_B.train(mode)
        self.dis_A.train(mode)
        self.dis_B.train(mode)

        return self