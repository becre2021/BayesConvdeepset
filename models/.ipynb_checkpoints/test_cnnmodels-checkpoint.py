import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .utils import *



def get_cnnmodels(modeltype='shallow'):
    if modeltype == 'shallow':
        return cnn_shallow()
    if modeltype == 'deep':
        return cnn_deep()

    
    
#points_per_unit=32    #for gp models
points_per_unit=64   #synthetic
#points_per_unit=24*12    #synthetic

class cnn_shallow(nn.Module):
    def __init__(self):
        super(cnn_shallow,self).__init__()
        
        self.out_dims = 8         
        #self.out_dims = 2*8                 
        self.multiplier = 2**3
        self.points_per_unit=points_per_unit

        
        cnn = nn.Sequential(
            nn.Conv1d(8, 16, 5, 1, 2),            
            nn.ReLU(),            
            nn.Conv1d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, 1, 2),
            nn.ReLU(),
            nn.Conv1d(16, 16, 5, 1, 2),
            nn.ReLU(),            
            nn.Conv1d(16, 8, 5, 1, 2),
        )                
        self.cnn = init_sequential_weights(cnn)    
        
            
    
    def forward(self,x):
        #print('x.shape,self.cnn(x).shape,pad_concat(x, self.cnn(x)).shape')        
        #print(x.shape,self.cnn(x).shape,pad_concat(x, self.cnn(x)).shape)
        
        return self.cnn(x) #out_dims = 8
        #return pad_concat(x, self.cnn(x)) #outs_dims = 2*8
    
    
    
    
    
    
    
    
    
    
    
    
class cnn_deep(nn.Module):
    def __init__(self,in_channels=8):
        super(cnn_deep,self).__init__()
        self.activation = nn.ReLU()
        
        self.in_channels = in_channels        
        self.out_channels = in_channels
        self.out_dims = 2*in_channels 
        
        #multiplier = 2**6
        #points_per_unit=64
        #self.ngrid = 
        
        self.multiplier = 2**6
        self.points_per_unit = points_per_unit
        
        
        #self.num_halving_layers = 6

        self.l1 = nn.Conv1d(in_channels=self.in_channels,
                            out_channels=self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l2 = nn.Conv1d(in_channels=self.in_channels,
                            out_channels=2 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l3 = nn.Conv1d(in_channels=2 * self.in_channels,
                            out_channels=2 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l4 = nn.Conv1d(in_channels=2 * self.in_channels,
                            out_channels=4 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l5 = nn.Conv1d(in_channels=4 * self.in_channels,
                            out_channels=4 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)
        self.l6 = nn.Conv1d(in_channels=4 * self.in_channels,
                            out_channels=8 * self.in_channels,
                            kernel_size=5, stride=2, padding=2)

        for layer in [self.l1, self.l2, self.l3, self.l4, self.l5, self.l6]:
            init_layer_weights(layer)

        self.l7 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=4 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l8 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=4 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l9 = nn.ConvTranspose1d(in_channels=8 * self.in_channels,
                                     out_channels=2 * self.in_channels,
                                     kernel_size=5, stride=2, padding=2,
                                     output_padding=1)
        self.l10 = nn.ConvTranspose1d(in_channels=4 * self.in_channels,
                                      out_channels=2 * self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)
        self.l11 = nn.ConvTranspose1d(in_channels=4 * self.in_channels,
                                      out_channels=self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)
        self.l12 = nn.ConvTranspose1d(in_channels=2 * self.in_channels,
                                      out_channels=self.in_channels,
                                      kernel_size=5, stride=2, padding=2,
                                      output_padding=1)

        for layer in [self.l7, self.l8, self.l9, self.l10, self.l11, self.l12]:
            init_layer_weights(layer)

    def forward(self, x):
        """Forward pass through the convolutional structure.

        Args:
            x (tensor): Inputs of shape `(batch, n_in, in_channels)`.

        Returns:
            tensor: Outputs of shape `(batch, n_out, out_channels)`.
        """
        h1 = self.activation(self.l1(x))
        h2 = self.activation(self.l2(h1))
        h3 = self.activation(self.l3(h2))
        h4 = self.activation(self.l4(h3))
        h5 = self.activation(self.l5(h4))
        h6 = self.activation(self.l6(h5))
        h7 = self.activation(self.l7(h6))

        h7 = pad_concat(h5, h7)
        h8 = self.activation(self.l8(h7))
        h8 = pad_concat(h4, h8)
        h9 = self.activation(self.l9(h8))
        h9 = pad_concat(h3, h9)
        h10 = self.activation(self.l10(h9))
        h10 = pad_concat(h2, h10)
        h11 = self.activation(self.l11(h10))
        h11 = pad_concat(h1, h11)
        h12 = self.activation(self.l12(h11))

        #print('pad_concat(x, h12).shape')        
        #print(pad_concat(x, h12).shape)        
        return pad_concat(x, h12)
        

