import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .utils import *


    
def get_cnnmodel_2d(modeltype='shallow',hdim = 128,num_channel=3):    
    if modeltype == 'shallow':
        return cnn_shallow_2d(hdim=hdim,
                              num_channel=num_channel)
        
    if modeltype == 'deep':
        return cnn_deep_2d(hdim=hdim,
                           num_channel=num_channel)    
        

class Conv2dResBlock(nn.Module):
    def __init__(self, in_channel=3, out_channel=128):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 5, 1, 2, groups=in_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 5, 1, 2, groups=in_channel),
            nn.ReLU()
        )
        self.final_relu = nn.ReLU()

    def forward(self, x):
        shortcut = x
        output = self.convs(x)
        output = self.final_relu(output + shortcut)
        return output
    


    
class cnn_shallow_2d(nn.Module):
    def __init__(self,hdim = 128,num_channel=3):
        super(cnn_shallow_2d,self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(hdim + hdim, hdim, 1, 1, 0),
            Conv2dResBlock(hdim, hdim),
            Conv2dResBlock(hdim, hdim),
            Conv2dResBlock(hdim, hdim),            
            nn.Conv2d(hdim, 2*num_channel, 1, 1, 0)
        )
        
        return 
        
    def forward(self,hfeature):
        return self.net(hfeature)
        

class cnn_deep_2d(nn.Module):
    def __init__(self,hdim = 128,num_channel=3):
        super(cnn_deep_2d,self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(hdim + hdim, hdim, 1, 1, 0),
            Conv2dResBlock(hdim, hdim),
            Conv2dResBlock(hdim, hdim),
            Conv2dResBlock(hdim, hdim),
            Conv2dResBlock(hdim, hdim),
            nn.Conv2d(hdim, 2*num_channel, 1, 1, 0)
        )
        
        return 
        
    def forward(self,hfeature):
        return self.net(hfeature)
        
    