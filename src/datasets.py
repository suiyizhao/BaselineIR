import glob
import torch
import random

import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class PairedImgDataset(Dataset):
    def __init__(self, data_source, mode, crop=256, random_resize=None):
        if not mode in ['train', 'val', 'test']:
            raise Exception('The mode should be "train", "val" or "test".')
        
        self.random_resize = random_resize
        self.crop = crop
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.img_paths = sorted(glob.glob(data_source + '/' + mode + '/blurry' + '/*/*.*'))
        self.gt_paths = sorted(glob.glob(data_source + '/'  + mode + '/sharp' + '/*/*.*'))

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index % len(self.img_paths)]).convert('RGB')
        gt = Image.open(self.gt_paths[index % len(self.gt_paths)]).convert('RGB')
        
        img = self.transform(img)
        gt = self.transform(gt)
        
        if self.mode == 'train':
            if self.random_resize is not None:
                # random resize
                scale_factor = random.uniform(self.crop/self.random_resize, 1.)
                img = F.interpolate(img.unsqueeze(0), scale_factor=scale_factor, align_corners=False, mode='bilinear', recompute_scale_factor=False).squeeze(0)
                gt = F.interpolate(gt.unsqueeze(0), scale_factor=scale_factor, align_corners=False, mode='bilinear', recompute_scale_factor=False).squeeze(0)
            
            # crop
            h, w = img.size(1), img.size(2)
            offset_h = random.randint(0, max(0, h - self.crop - 1))
            offset_w = random.randint(0, max(0, w - self.crop - 1))

            img = img[:, offset_h:offset_h + self.crop, offset_w:offset_w + self.crop]
            gt = gt[:, offset_h:offset_h + self.crop, offset_w:offset_w + self.crop]
        
            # horizontal flip
            if random.random() < 0.5:
                idx = [i for i in range(img.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                img = img.index_select(2, idx)
                gt = gt.index_select(2, idx)
        
        return img, gt

    def __len__(self):
        return max(len(self.img_paths), len(self.gt_paths))
    

class SingleImgDataset(Dataset):
    def __init__(self, data_source):
        
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        self.img_paths = sorted(glob.glob(data_source + '/' + 'test' + '/blurry' + '/*/*.*'))

    def __getitem__(self, index):
        
        path = self.img_paths[index % len(self.img_paths)]
        
        img = Image.open(path).convert('RGB')
        
        img = self.transform(img)
        
        return img, path

    def __len__(self):
        return len(self.img_paths)