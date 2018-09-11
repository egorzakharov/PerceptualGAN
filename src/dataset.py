import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import Compose, RandomHorizontalFlip, Resize, ToPILImage
from torchvision.transforms import CenterCrop, Lambda, Normalize, ToTensor, RandomCrop
from torchvision.transforms import functional as F
import numpy as np
import os



def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def get_dataloaders(opt):

    dataloaders = [
        DataLoader(
            MyDatasets(opt, 'train'),
            opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            drop_last=True),
        DataLoader(
            MyDatasets(opt, 'test'),
            opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            drop_last=True)]

    return dataloaders


class MyDataset(object):
    """ Adds random transforms and warping """

    def __init__(self, opt, phase, domain):
        super(MyDataset, self).__init__()

        # Read options
        img_path = opt['%s_img_%s_path' % (phase, domain)]
        img_size = opt['image_size']

        # Process images
        if img_path[-3:] == 'txt':

            # Read images from the list
            data_img_path = opt['img_path']
            data_img_list = np.loadtxt(img_path, dtype=str).tolist()

            self.data_img = sorted([
                data_img_path + name for name in data_img_list])

            self.length = len(self.data_img)

        else:

            # Read images from folder
            self.data_img = sorted([
                img_path + name for name in os.listdir(img_path)])

            self.length = len(self.data_img)

        # Read options
        self.img_size = img_size
        self.phase = phase

        # Prepare transformations
        if 'scale' in opt['input_transforms']:
            self.input_transform = Resize(self.img_size)
        else:
            self.input_transform = None

        if 'flip' in opt['input_transforms']:
            self.flip = F.hflip
        else:
            self.flip = None
        
        if 'crop' in opt['input_transforms']:
            self.output_transform = [CenterCrop(self.img_size)]
        else:
            self.output_transform = []

        if 'random_crop' in opt['input_transforms']:
            self.output_transform += [RandomCrop(self.img_size, pad_if_needed=True)]

        self.loader = pil_loader
        self.output_transform += [
            ToTensor(),
            Normalize(
                [0.5]*3, 
                [0.5]*3)]

        self.output_transform = Compose(self.output_transform)
        
    def get_item(self, index, flip=None):

        index = index % self.length
        
        img = self.loader(self.data_img[index])
        
        if not self.input_transform is None:
            img = self.input_transform(img)

        if self.phase == 'train':

            if not self.flip is None and flip:
                img = self.flip(img)

        img = self.output_transform(img)

        return [img]

    def __len__(self):

        return self.length


class MyDatasets(Dataset):

    def __init__(self, opt, phase='train'):
        super(MyDatasets, self).__init__()

        opt = vars(opt)

        self.datasets = []

        for domain in ['A', 'B']:
            if opt['%s_img_%s_path' % (phase, domain)]:
                self.datasets += [MyDataset(opt, phase, domain)]

        self.phase = phase

        self.down = Resize(opt['image_size'], interpolation=Image.BICUBIC)

        # Check if the problem is for aligned synthesis
        self.aligned = opt['model_type'] == 'pix2pix'

        self.length = opt['epoch_len'] * opt['batch_size']

        if self.phase == 'test': self.length //= 10
        
        if self.phase == 'train': 

            self.shuffle()

        elif self.phase == 'test': 

            self.indices = []

            for d in self.datasets:

                length = min(len(d), len(self))

                self.indices += [np.concatenate([
                    np.random.randint(0, len(d), len(self)-length),
                    np.arange(length)], 0)]

            self.flips = [np.zeros(len(self), dtype=bool)] * len(self.datasets)

        # Detect the end of dataset
        self.counter = 0

    def __getitem__(self, index):

        outputs = []

        for i, d in enumerate(self.datasets):

            outputs += d.get_item(self.indices[i][index], self.flips[i][index])

        self.counter += 1

        # Shuffle datasets after each epoch
        if self.counter == len(self):
            if self.phase == 'train': self.shuffle()
            self.counter = 0

        if len(outputs) == 1 and self.aligned:

            # Super resolution
            outputs[0] = ToPILImage()((outputs[0] + 1) / 2)
            outputs.insert(0, self.down(outputs[0]))
            for i, o in enumerate(outputs):
                outputs[i] = ToTensor()(o) * 2 - 1

        return outputs

    def shuffle(self):

        self.indices = []
        self.flips = []

        for d in self.datasets:
            self.indices += [np.random.randint(0, len(d), len(self))]
            self.flips += [np.random.random(len(self)) > 0.5]

        if self.aligned:
            self.indices = [self.indices[0]] * len(self.datasets)
            self.flips = [self.flips[0]] * len(self.datasets)
                
    def __len__(self):

        return self.length
