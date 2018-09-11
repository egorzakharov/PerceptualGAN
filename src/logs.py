from tensorboardX import SummaryWriter
import torch
from torchvision.utils import make_grid
import os
import numpy as np



class Logs(object):

    def __init__(self, model, opt):
        super(Logs, self).__init__()

        self.model = model
        self.batch_size = opt.batch_size
        self.log_dir = os.path.join('runs', opt.experiment_name)

        # Get number of discrims for each domain
        self.n_discs = {}

        if hasattr(self.model, 'dis_A'):
            self.n_discs['A'] = 1
        else:
            self.n_discs['A'] = 0
        
        if hasattr(self.model, 'dis_B'):
            self.n_discs['B'] = 1
        else:
            self.n_discs['B'] = 0

        self.writer = SummaryWriter(self.log_dir)
        
        self.n_iter = 0

        # Initialize dictionary of losses
        self.losses = {}

        for domain in ['A', 'B']:

            for i in range(1, self.n_discs[domain]+1):

                for loss_type in ['adv', 'aux']:

                    key = 'losses_%s_%s_%d' % (loss_type, domain, i)
                    self.losses[key] = {}

                    for phase in ['train', 'test']:
                        self.losses[key][phase] = []

            for loss_type in ['cycle', 'ident', 'auxil', 'kldiv']:

                key = 'loss_%s_%s' % (loss_type, domain)
                self.losses[key] = {}

                for phase in ['train', 'test']:
                    self.losses[key][phase] = []

    def update_losses(self, phase):

        if hasattr(self.model, 'losses_adv_A'):
            for i, loss in enumerate(self.model.losses_adv_A, 1):
                self.losses['losses_adv_A_%d' % i][phase] += [loss]

        if hasattr(self.model, 'losses_aux_A'):
            for i, loss in enumerate(self.model.losses_aux_A, 1):
                self.losses['losses_aux_A_%d' % i][phase] += [loss]

        if hasattr(self.model, 'losses_adv_B'):
            for i, loss in enumerate(self.model.losses_adv_B, 1):
                self.losses['losses_adv_B_%d' % i][phase] += [loss]

        if hasattr(self.model, 'losses_aux_B'):
            for i, loss in enumerate(self.model.losses_aux_B, 1):
                self.losses['losses_aux_B_%d' % i][phase] += [loss]

        if hasattr(self.model, 'loss_cycle_A'):
            self.losses['loss_cycle_A'][phase] += [self.model.loss_cycle_A]

        if hasattr(self.model, 'loss_cycle_B'):
            self.losses['loss_cycle_B'][phase] += [self.model.loss_cycle_B]

        if hasattr(self.model, 'loss_ident_A'):
            self.losses['loss_ident_A'][phase] += [self.model.loss_ident_A]

        if hasattr(self.model, 'loss_ident_B'):
            self.losses['loss_ident_B'][phase] += [self.model.loss_ident_B]

        if hasattr(self.model, 'loss_auxil_A'):
            self.losses['loss_auxil_A'][phase] += [self.model.loss_auxil_A]

        if hasattr(self.model, 'loss_auxil_B'):
            self.losses['loss_auxil_B'][phase] += [self.model.loss_auxil_B]

        if hasattr(self.model, 'loss_kldiv_A'):
            self.losses['loss_kldiv_A'][phase] += [self.model.loss_kldiv_A]

        if hasattr(self.model, 'loss_kldiv_B'):
            self.losses['loss_kldiv_B'][phase] += [self.model.loss_kldiv_B]

    def update_tboard(self, n_iter):

        # Average losses
        for key in self.losses.keys():
            for phase in ['train', 'test']:
                if self.losses[key][phase]:
                    self.losses[key][phase] = np.mean(self.losses[key][phase])

        for domain in ['A', 'B']:

            for i in range(1, self.n_discs[domain]+1):

                for key in ['losses_adv_%s' % domain, 'losses_aux_%s' % domain]:

                    if hasattr(self.model, key):
                        key += '_%d' % i
                        self.writer.add_scalars(key, self.losses[key], n_iter)

            for key in ['loss_cycle_%s' % domain, 'loss_ident_%s' % domain,
                        'loss_auxil_%s' % domain, 'loss_kldiv_%s' % domain]:

                if hasattr(self.model, key):
                    self.writer.add_scalars(key, self.losses[key], n_iter)

        # Clear losses
        for key in self.losses.keys():
            for phase in ['train', 'test']:
                self.losses[key][phase] = []

        tensor = []

        for i in range(self.batch_size):

            if hasattr(self.model, 'real_A'):
                tensor += [self.model.real_A[i][None].cpu()]

            if hasattr(self.model, 'fake_B'):
                tensor += [self.model.fake_B[i][None].cpu()]

            if hasattr(self.model, 'real_B'):
                tensor += [self.model.real_B[i][None].cpu()]

            if hasattr(self.model, 'fake_A'):
                tensor += [self.model.fake_A[i][None].cpu()]

        tensor = torch.cat(tensor, 0)

        if tensor.min() < 0:
            tensor = (tensor + 1.) / 2.
        else:
            factor, _ = tensor.max(2, keepdim=True)
            factor, _ = factor.max(3, keepdim=True)
            tensor /= factor
        
        nrow = tensor.shape[0]//self.batch_size
        if self.batch_size > 1: nrow *= 2

        images = make_grid(tensor, nrow, pad_value=1)

        self.writer.add_image('images', images, n_iter)

    def close(self):

        self.writer.export_scalars_to_json(self.log_dir + '/all_scalars.json')
        self.writer.close()