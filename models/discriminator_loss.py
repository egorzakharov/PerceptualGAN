import torch
from torch import nn
from torch.autograd import Variable
from torch.autograd import grad
from src import utils



class DiscriminatorLoss(nn.Module):

    def __init__(self, opt):
        super(DiscriminatorLoss, self).__init__()

        self.gpu_id = opt.gpu_ids[0]

        # Adversarial criteria for the predictions
        if opt.dis_adv_loss_type == 'gan':
            self.crit = nn.BCEWithLogitsLoss()
        elif opt.dis_adv_loss_type == 'lsgan':
            self.crit = nn.MSELoss()

        # Targets for criteria
        self.labels_real = []
        self.labels_fake = []

        # Iterate over discriminators to inialize labels
        for size in opt.dis_output_sizes:

            shape = (opt.batch_size, 1, size, size)
            
            self.labels_real += [Variable(torch.ones(shape).cuda(self.gpu_id))]
            self.labels_fake += [Variable(torch.zeros(shape).cuda(self.gpu_id))]

    def __call__(self, dis, img_real_dst, img_fake_dst=None, 
                 aux_real_dst=None, img_real_src=None, enc=None):

        # Preds for real (during dis backprop) or fake (during gen backprop)
        outputs_real = dis(img_real_dst, img_real_src, enc)

        # Preds for fake during dis backprop
        if img_fake_dst is not None:
            
            outputs_fake = dis(img_fake_dst, img_real_src, enc)

        loss = 0 # loss
        
        losses_adv = [] # losses for each discriminator output

        for i in range(len(outputs_real)):
            
            losses_adv += [self.crit(outputs_real[i], self.labels_real[i])]

            if img_fake_dst is not None:

                losses_adv[-1] += self.crit(outputs_fake[i], self.labels_fake[i])

                losses_adv[-1] *= 0.5

            loss += losses_adv[-1]

        # Get loss values
        losses_adv = [loss_adv.data.item() for loss_adv in losses_adv]

        losses_adv = [sum(losses_adv)]

        return loss, losses_adv
