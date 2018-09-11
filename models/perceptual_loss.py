import os
import torch
import torch.nn as nn
from src.vgg_utils import get_vgg19
import numpy as np



def get_pretrained_net(name, path):
    """ Load pretrained network """

    if name == 'vgg19_caffe':
        os.system('wget -O vgg19_caffe.pth --no-check-certificate -nc https://www.dropbox.com/s/xlbdo688dy4keyk/vgg19-caffe.pth?dl=1')
        vgg = get_vgg19(name, path)
    elif name == 'vgg19_pytorch':
        vgg = get_vgg19(name, path)
    elif name == 'vgg19_pytorch_modified':
        # TODO: correct wget
        vgg = get_vgg19(name, path)
    else:
        assert False, 'Wrong pretrained network name'

    return vgg


class FeatureExtractor(nn.Module):
    """ 
        Assumes input image is
        if `input_range` is 'sigmoid' -- in range [0,1]
                            'tanh'                [-1, 1]
    """
    def __init__(
        self, 
        input_range='sigmoid',
        net_type='vgg19_pytorch_modified',
        preprocessing_type='corresponding',
        layers = '1,6,11,20,29',
        net_path='.'):
        super(FeatureExtractor, self).__init__()

        # Get preprocessing for input range
        if input_range == 'sigmoid':
            self.preprocess_range = lambda x: x
        elif input_range == 'tanh':
            self.preprocess_range = lambda x: (x + 1.) / 2.
        else:
            assert False, 'Wrong input_range'
        self.preprocessing_type = preprocessing_type
        
        # Get preprocessing for pretrained nets
        if preprocessing_type == 'corresponding':

            if 'caffe' in net_type:
                self.preprocessing_type = 'caffe'
            elif 'pytorch' in net_type:
                self.preprocessing_type = 'pytorch'
            else:
                assert False, 'Unknown net_type'
        
        # Store preprocessing means and std
        if self.preprocessing_type == 'caffe':

            self.vgg_mean = nn.Parameter(torch.FloatTensor([103.939, 116.779, 123.680]).view(1, 3, 1, 1))
            self.vgg_std = None

        elif self.preprocessing_type == 'pytorch':

            self.vgg_mean = nn.Parameter(torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.vgg_std = nn.Parameter(torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        else:

            assert False, 'Unknown preprocessing_type'

        net = get_pretrained_net(net_type, net_path)

        # Split net into segments
        self.blocks = nn.ModuleList()
        layers_indices = [int(i) for i in layers.split(',')]
        layers_indices.insert(0, -1)
        for i in range(len(layers_indices)-1):
            layers_i = []
            for j in range(layers_indices[i]+1, layers_indices[i+1]+1):
                layers_i += [net[j]]
            self.blocks += [nn.Sequential(*layers_i)]
        
        self.eval()

    def forward(self, input):

        input = input.clone()
        input = self.preprocess_range(input)

        if self.preprocessing_type == 'caffe':

            r, g, b = torch.chunk(input, 3, dim=1)
            bgr = torch.cat([b, g, r], 1)
            out = bgr * 255 - self.vgg_mean

        elif self.preprocessing_type == 'pytorch':

            input = input - self.vgg_mean
            input = input / self.vgg_std

        output = input
        outputs = []
        
        for block in self.blocks:
            output = block(output)
            outputs.append(output)

        return outputs


class Matcher(nn.Module):
    def __init__(
        self, 
        matching_type='features',
        matching_loss='L1',
        average_loss=True):
        super(Matcher, self).__init__()

        # Matched statistics
        if matching_type == 'features':
            self.get_stats = self.gram_matrix
        elif matching_type == 'features':
            self.get_stats = lambda x: x

        # Loss function
        matching_loss = matching_loss.lower()
        if matching_loss == 'mse':
            self.criterion = nn.MSELoss()
        elif matching_loss == 'smoothl1':
            self.criterion = nn.SmoothL1Loss()
        elif matching_loss == 'l1':
            self.criterion = nn.L1Loss()
        self.average_loss = average_loss

    def gram_matrix(self, input):

        b, c, h, w = input.size()
        features = input.view(b, c, h*w)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)

        return gram

    def __call__(self, input_feats, target_feats):

        input_stats = [self.get_stats(features) for features in input_feats]
        target_stats = [self.get_stats(features) for features in target_feats]

        loss = 0

        for input, target in zip(input_stats, target_stats):
            loss += self.criterion(input, target.detach())
        
        if self.average_loss:
            loss /= len(input_stats)

        return loss


class PerceptualLoss(nn.Module):
    def __init__(
        self, 
        input_range='sigmoid',
        net_type='vgg19_pytorch_modified', 
        preprocessing_type='corresponding', 
        matching_loss='L1',
        match=[
            {'matching_type': 'features',
             'layers': '1,6,11,20,29'}],
        average_loss=True,
        extractor=None):
        super(PerceptualLoss, self).__init__()

        self.average_loss = average_loss

        self.matchers = []
        layers = '' # Get layers needed for all matches
        for m in match:
            self.matchers += [Matcher(
                m['matching_type'], 
                matching_loss, 
                average_loss)]
            layers += m['layers'] + ','
        layers = layers[:-1] # Remove last ','
        layers = np.asarray(layers.split(',')).astype(int)
        layers = np.unique(layers) # Unique layers needed to compute

        # Find correspondence between layers and matchers
        self.layers_idx_m = []
        for m in match:
            layers_m = [int(i) for i in m['layers'].split(',')]
            layers_idx_m = []
            for l in layers_m:
                layers_idx_m += [np.argwhere(layers == l)[0, 0]]
            self.layers_idx_m += [layers_idx_m]
        layers = ','.join(layers.astype(str))

        if extractor is None:
            self.extractor = FeatureExtractor(
                input_range, 
                net_type, 
                preprocessing_type, 
                layers)
        else:
            self.extractor = extractor

    def forward(self, input, target):

        input_feats = self.extractor(input)
        target_feats = self.extractor(target)

        loss = 0
        for i, m in enumerate(self.matchers):
            input_feats_m = [input_feats[j] for j in self.layers_idx_m[i]]
            target_feats_m = [target_feats[j] for j in self.layers_idx_m[i]]
            loss += m(input_feats_m, target_feats_m)

        if self.average_loss:
            loss /= len(self.matchers)

        return loss