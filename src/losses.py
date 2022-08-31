import math
import torch
import torch.fft
import torch.nn as nn

import torch.nn.functional as F
from torch.autograd import Variable

class LossCont(nn.Module):
    def __init__(self):
        super(LossCont, self).__init__()
        self.criterion = nn.L1Loss()
        
    def forward(self, imgs, gts):
        return self.criterion(imgs, gts)

class LossFreqReco(nn.Module):
    def __init__(self):
        super(LossFreqReco, self).__init__()
        self.criterion = nn.L1Loss()
        
    def forward(self, imgs, gts):
        imgs = torch.fft.rfftn(imgs, dim=(2,3))
        _real = imgs.real
        _imag = imgs.imag
        imgs = torch.cat([_real, _imag], dim=1)
        gts = torch.fft.rfftn(gts, dim=(2,3))
        _real = gts.real
        _imag = gts.imag
        gts = torch.cat([_real, _imag], dim=1)
        return self.criterion(imgs, gts)