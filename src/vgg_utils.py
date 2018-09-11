import torch
import torch.nn as nn
from torchvision.models import vgg19
from src import utils



class VGGModified(nn.Module):
    def __init__(self, vgg19_orig, slope):
        super(VGGModified, self).__init__()

        self.features = nn.Sequential()
        
        self.features.add_module(str(0), vgg19_orig.features[0])
        self.features.add_module(str(1), nn.LeakyReLU(slope, True))
        self.features.add_module(str(2), vgg19_orig.features[2])
        self.features.add_module(str(3), nn.LeakyReLU(slope, True))
        self.features.add_module(str(4), nn.AvgPool2d((2,2), (2,2)))

        self.features.add_module(str(5), vgg19_orig.features[5])
        self.features.add_module(str(6), nn.LeakyReLU(slope, True))
        self.features.add_module(str(7), vgg19_orig.features[7])
        self.features.add_module(str(8), nn.LeakyReLU(slope, True))
        self.features.add_module(str(9), nn.AvgPool2d((2,2), (2,2)))

        self.features.add_module(str(10), vgg19_orig.features[10])
        self.features.add_module(str(11), nn.LeakyReLU(slope, True))
        self.features.add_module(str(12), vgg19_orig.features[12])
        self.features.add_module(str(13), nn.LeakyReLU(slope, True))
        self.features.add_module(str(14), vgg19_orig.features[14])
        self.features.add_module(str(15), nn.LeakyReLU(slope, True))
        self.features.add_module(str(16), vgg19_orig.features[16])
        self.features.add_module(str(17), nn.LeakyReLU(slope, True))
        self.features.add_module(str(18), nn.AvgPool2d((2,2), (2,2)))

        self.features.add_module(str(19), vgg19_orig.features[19])
        self.features.add_module(str(20), nn.LeakyReLU(slope, True))
        self.features.add_module(str(21), vgg19_orig.features[21])
        self.features.add_module(str(22), nn.LeakyReLU(slope, True))
        self.features.add_module(str(23), vgg19_orig.features[23])
        self.features.add_module(str(24), nn.LeakyReLU(slope, True))
        self.features.add_module(str(25), vgg19_orig.features[25])
        self.features.add_module(str(26), nn.LeakyReLU(slope, True))
        self.features.add_module(str(27), nn.AvgPool2d((2,2), (2,2)))

        self.features.add_module(str(28), vgg19_orig.features[28])
        self.features.add_module(str(29), nn.LeakyReLU(slope, True))
        self.features.add_module(str(30), vgg19_orig.features[30])
        self.features.add_module(str(31), nn.LeakyReLU(slope, True))
        self.features.add_module(str(32), vgg19_orig.features[32])
        self.features.add_module(str(33), nn.LeakyReLU(slope, True))
        self.features.add_module(str(34), vgg19_orig.features[34])
        self.features.add_module(str(35), nn.LeakyReLU(slope, True))
        self.features.add_module(str(36), nn.AvgPool2d((2,2), (2,2)))
        
        self.classifier = nn.Sequential()
        
        self.classifier.add_module(str(0), vgg19_orig.classifier[0])
        self.classifier.add_module(str(1), nn.LeakyReLU(slope, True))
        self.classifier.add_module(str(2), nn.Dropout2d(p = 0.5))
        self.classifier.add_module(str(3), vgg19_orig.classifier[3])
        self.classifier.add_module(str(4), nn.LeakyReLU(slope, True))
        self.classifier.add_module(str(5), nn.Dropout2d(p = 0.5))
        self.classifier.add_module(str(6), vgg19_orig.classifier[6])

    def forward(self, x):

        return self.classifier(self.features.forward(x))


def get_vgg19(model_name, model_path):
    
    # load base model
    if model_name == 'vgg19_caffe':
        model = vgg19()
    elif model_name == 'vgg19_pytorch':
        model = vgg19(pretrained=True)
    elif model_name == 'vgg19_pytorch_modified':
        model = VGGModified(vgg19(), 0.2)
        model.load_state_dict(torch.load('%s/%s.pkl' % (model_path, model_name))['state_dict'])
    
    # convert model into standard form
    model.classifier = nn.Sequential(utils.View(), *model.classifier._modules.values())
    vgg = model.features
    vgg_classifier = model.classifier
    names = ['conv1_1','relu1_1','conv1_2','relu1_2','pool1',
             'conv2_1','relu2_1','conv2_2','relu2_2','pool2',
             'conv3_1','relu3_1','conv3_2','relu3_2','conv3_3','relu3_3','conv3_4','relu3_4','pool3',
             'conv4_1','relu4_1','conv4_2','relu4_2','conv4_3','relu4_3','conv4_4','relu4_4','pool4',
             'conv5_1','relu5_1','conv5_2','relu5_2','conv5_3','relu5_3','conv5_4','relu5_4','pool5',
             'torch_view','fc6','relu6','drop6','fc7','relu7','drop7','fc8']
    model = nn.Sequential()
    for n, m in zip(names, list(vgg) + list(vgg_classifier)):
        model.add_module(n, m)
    if model_name == 'vgg19_caffe':
        model.load_state_dict(torch.load('%s/%s.pth' % (model_path, model_name)))

    return model