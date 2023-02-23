"""
reference: https://github.com/cambridge-mlg/convcnp/blob/master/convcnp/conv_cnp.py
"""

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from attrdict import AttrDict  
from .test_cnnmodels import get_cnnmodels
from .utils import gaussian_logpdf, init_sequential_weights, to_multiple



def to_numpy(x):
    """Convert a PyTorch tensor to NumPy."""
    return x.squeeze().detach().cpu().numpy()

def compute_dists(x, y):
    """Fast computation of pair-wise distances for the 1d case.

    Args:
        x (tensor): Inputs of shape (batch, n, 1).
        y (tensor): Inputs of shape (batch, m, 1).

    Returns:
        tensor: Pair-wise distances of shape (batch, n, m).
    """
    return (x - y.permute(0, 2, 1)) ** 2


def to_multiple(x, multiple):
    """Convert `x` to the nearest above multiple.

    Args:
        x (number): Number to round up.
        multiple (int): Multiple to round up to.

    Returns:
        number: `x` rounded to the nearest above multiple of `multiple`.
    """
    if x % multiple == 0:
        return x
    else:
        return x + multiple - x % multiple


    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
# ----------------------------------
# baseline model
# ----------------------------------
class Convcnp(nn.Module):
    """One-dimensional ConvCNP model.

    Args:
        rho (function): CNN that implements the translation-equivariant map rho.
        points_per_unit (int): Number of points per unit interval on input.
            Used to discretize function.
    """

    #def __init__(self, in_channels, rho, points_per_unit,rbf_init_l=None, nbasis=5):
    def __init__(self, in_dims=1,out_dims=1,num_channels=3,cnntype='shallow',init_lengthscale=0.1, nbasis=5):
        
        super(Convcnp, self).__init__()
        self.activation = nn.Sigmoid()
        self.sigma_fn = nn.Softplus()        
        
        self.modelname = 'base'
        self.num_samples = 1
        self.gppriorscale = 0.0 
        
        
        self.in_dims = in_dims
        self.out_dims= out_dims
        self.num_channels = num_channels

        self.encoder = ConvDeepSet(in_channels = self.num_channels,
                                   out_channels = 8,
                                   init_lengthscale=init_lengthscale)
          

        self.cnntype = cnntype
        self.cnn = get_cnnmodels(cnntype) 

    
        #self.nbasis = 5
        self.nbasis = nbasis 
        #linear = nn.Sequential(nn.Linear(8,self.num_channels*self.nbasis))
        cnn_linear = nn.Sequential(nn.Linear(self.cnn.out_dims,self.num_channels*self.nbasis))        
        self.cnn_linear = init_sequential_weights(cnn_linear)        
        
        self.mean_layer = FinalLayer(in_channels=self.num_channels,
                                     nbasis = self.nbasis,
                                     init_lengthscale=init_lengthscale)

        self.logstd_layer = FinalLayer(in_channels=self.num_channels,
                                       nbasis = self.nbasis,
                                       init_lengthscale=init_lengthscale)

        

    def compute_xgrid(self,x,y,x_out,x_thres=1.0):
        x_min = min(torch.min(x).cpu().numpy(),torch.min(x_out).cpu().numpy()) - x_thres
        x_max = max(torch.max(x).cpu().numpy(),torch.max(x_out).cpu().numpy()) + x_thres        
        num_points = int(to_multiple(self.cnn.points_per_unit * (x_max - x_min),self.cnn.multiplier))             
        x_grid = torch.linspace(x_min , x_max , num_points).to(x.device)
            
        # context
        nb,npoints,nchannel = x.size()        
        x_grid = x_grid[None, :, None].repeat(nb, 1, nchannel)
        return x_grid
        
        
        

    def compute_hgrid(self,h):
        h = self.activation(h) 
        h = h.permute(0, 2, 1) #(nbatch, noutchannel, ngrid)
        nb,_,ngrid = h.size()    
        h = self.cnn(h)
        h_grid = h.reshape(nb, -1, ngrid).permute(0, 2, 1) #(nbatch, ngrid,2*noutchannel)
        nb,ngrid,_= h_grid.size()
        h_grid = h_grid.reshape(-1,h_grid.size(-1))
        h_grid = self.cnn_linear(h_grid).reshape(nb,ngrid,self.nbasis,-1)                
        return h_grid
                

        
    def forward(self, x, y, x_out):
        """Run the model forward.

        Args:
            x (tensor): Observation locations of shape (batch, data, features).
            y (tensor): Observation values of shape (batch, data, outputs).
            x_out (tensor): Locations of outputs of shape (batch, data, features).
            
        Returns:
            tuple[tensor]: Means and standard deviations of shape (batch_out, channels_out).
        """
        
        #print('x.size(),x_out.size()')        
        #print(x.size(),x_out.size())
        
        
        nb,npoints,nchannel = x.size()        
        x_grid = self.compute_xgrid(x,y,x_out)
        #print(x_grid.size())
        concat_n_h1h0,n_h1,h1,h0 = self.encoder(x,y,x_grid)        
          
        #print(x_grid.shape,concat_n_h1h0.shape)
        h_grid = self.compute_hgrid(concat_n_h1h0)
        pmu = self.mean_layer(x_grid,h_grid,x_out)
        plogstd =  self.logstd_layer(x_grid,h_grid,x_out)
        #return pmean, 0.01+0.99*F.softplus(pstd)
        #return pmean, 0.1+0.9*F.softplus(pstd)
    
        outs = AttrDict()
        outs.pymu = pmu
        outs.pystd = 0.1+0.9*F.softplus(plogstd)
        #outs.regloss = None       
        return outs        
    
    
    
    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()for param in self.parameters()])


    
    def sample_functionalfeature(self,x,y,xout,numsamples=1):
        nb,npoints,nchannel = x.size()        
        x_grid = self.compute_xgrid(x,y,xout)
        concat_n_h1h0,n_h1,h1,h0 = self.encoder(x,y,x_grid)        
        return n_h1,x_grid 



###########################################    
# encoder and decoder
###########################################

import math
pi = math.pi

eps = 1e-6
class ConvDeepSet(nn.Module):
    """One-dimensional ConvDeepSet module. Uses an RBF kernel for psi(x, x').

    Args:
        out_channels (int): Number of output channels.
        init_lengthscale (float): Initial value for the length scale.
    """

    def __init__(self, in_channels,out_channels, init_lengthscale=1.0,min_init_lengthscale=1e-6):
        super(ConvDeepSet, self).__init__()
        self.in_channels = in_channels        
        self.out_channels = out_channels
        self.g = self.build_weight_model()               
        self.sigma = nn.Parameter(torch.log(min_init_lengthscale+init_lengthscale*torch.ones(self.in_channels)), requires_grad=True)        
        #self.mu = nn.Parameter(np.log(1* torch.rand(self.in_channels)), requires_grad=True)
        self.sigma_fn = torch.exp

    def build_weight_model(self):
        """Returns a point-wise function that transforms the
        (in_channels + 1)-dimensional representation to dimensionality
        out_channels.

        Returns:
            torch.nn.Module: Linear layer applied point-wise to channels.
        """
        #model = nn.Sequential(nn.Linear(self.in_channels, self.out_channels))
        model = nn.Sequential(nn.Linear(2*self.in_channels, self.out_channels))
        
        init_sequential_weights(model)
        return model
    
        
    
    def compute_rbf(self,x1,x2=None):
        if x2 is None:
            x2 = x1            
        # Compute shapes.            
        nbatch,npoints,nchannel = x1.size()
        scales = (self.sigma).exp()[None, None, None, :]                
        
        #compute rbf over multiple channels
        dists = x1.unsqueeze(dim=2) - x2.unsqueeze(dim=1)        
        factors = 1
        if dists.size(-1) != scales.size(-1):
            factors = scales.size(-1) // dists.size(-1) 
            dists = dists.repeat(1,1,1,factors)
        #print(dists.size(),scales.size())
        
        
        dists /= (scales + eps)
        wt = torch.exp(-0.5*dists**2)   
        return wt,factors

        
        
        
    def forward(self, context_x, context_y, x_grid):
        """Forward pass through the layer with evaluations at locations t.

        Args:
            x (tensor): Inputs of observations of shape (n, 1).
            y (tensor): Outputs of observations of shape (n, in_channels).
            t (tensor): Inputs to evaluate function at of shape (m, 1).
        Returns:
            tensor: Outputs of evaluated function at z of shape
                (m, out_channels).
        """


        nbatch,npoints,nchannel = context_x.size()
        wt,factors = self.compute_rbf(context_x,x_grid)
        h0 = wt.sum(dim=1)

        if factors > 1:
            context_y = context_y.repeat(1,1,factors)
        
        h1 = (context_y.unsqueeze(dim=-2)*wt).sum(dim=1)
        n_h1 = h1/(h0+eps)        
        
        y_out = torch.cat((h0, n_h1), dim=-1)    #(nbatch, ngrid,2*noutchannel)         
        y_out = y_out.view(-1,2*self.in_channels)  #(nbatch, ngrid,2*noutchannel) 
        y_out = self.g(y_out)
        y_out = y_out.view(nbatch, -1, self.out_channels) #(nbatch, ngrid,noutchannel)

        #return y_out,h1,h0
        return y_out,n_h1,h1,h0
    

    
    

class FinalLayer(nn.Module):
    """One-dimensional Set convolution layer. Uses an RBF kernel for psi(x, x').

    Args:
        in_channels (int): Number of inputs channels.
        init_lengthscale (float): Initial value for the length scale.
    """

    def __init__(self,  in_channels=1, out_channels = 1, nbasis = 1, init_lengthscale = 1.0,min_init_lengthscale=1e-6):
        super(FinalLayer, self).__init__()
        
        #self.out_channels = in_channels             
        #self.in_channels_hidden = in_channels*self.out_channels       
        self.sigma_fn = torch.exp        
        self.nbasis = nbasis
        self.in_channels = in_channels        
        self.out_channels = out_channels               
                
        #self.g = self.build_weight_model()            
        linear = nn.Sequential(nn.Linear(self.nbasis, self.out_channels))        
        self.g = init_sequential_weights(linear)
            

        
        self.sigma = nn.Parameter(np.log(min_init_lengthscale+init_lengthscale*torch.ones(self.nbasis,self.in_channels)), requires_grad=True)    
        #self.mu = nn.Parameter(np.log(1* torch.rand(self.nbasis,self.in_channels)), requires_grad=True)
        
        
    
    def compute_rbf(self,x1,x2=None):
        if x2 is None:
            x2 = x1            
        # Compute shapes.            
        nbatch,npoints,nchannel = x1.size()
        
        #compute rbf over multiple channels
        dists = x1.unsqueeze(dim=2) - x2.unsqueeze(dim=1)
        dists = dists.unsqueeze(dim=-2).repeat(1,1,1,self.nbasis,1)        
        scales = self.sigma.exp()[None, None, None, :,:]  
        
        dists = dists/(scales + eps)
        wt = torch.exp(-0.5*dists**2)   
        return wt
        
        
    #nbasis == 5 case    
    def forward(self, x_grid, h_grid, target_x ,iterratio=None):
        """Forward pass through the layer with evaluations at locations t.

        Args:
            x (tensor): Inputs of observations of shape (n, 1).
            y (tensor): Outputs of observations of shape (n, in_channels).
            t (tensor): Inputs to evaluate function at of shape (m, 1).

        Returns:
            tensor: Outputs of evaluated function at z of shape
                (m, out_channels).
        """

        nb,ntarget,nchannel = target_x.size()        
        _,ngrid,nbasis,_ = h_grid.size() #(nb,ngrid,nbasis,nchannels)
                
        wt = self.compute_rbf(x_grid,target_x)
        #wt = self.compute_sm(x_grid,target_x)
                
        h = h_grid[:,:,None,:] #(nb,ngrid,1,nbasis,nchannels)
        h_out = (h*wt).sum(dim=1) #(nb,ntarget,nbasis,nchannels)
        
        
        h_out = h_out.transpose(-2,-1) #(nb,ntarget,nchannels,nbasis)
        #h_out = self.g(h_out).squeeze() #(nb,ntarget,nchannels,1)
        h_out = self.g(h_out) #(nb,ntarget,nchannels,1)
        
        if h_out.dim() == 4:
            h_out = h_out[...,0]
        return h_out
    
    
    
#----------------------------------------------
#loss function
#----------------------------------------------

from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
def compute_loss_baseline( pred_mu, pred_std, target_y, intrain=True,reduce=True):    
    """ compute loss for conditional Np models

    Args:
        pred_mu:  (nbatch,ndata,nchannel)
        pred_std: (nbatch,ndata,nchannel)
        target_y: (nbatch,ndata,nchannel)
    Returns:
        logloss: 1

    """

    #print('pred_mu.shape,pred_std.shape,target_y.shape')    
    #print(pred_mu.shape,pred_std.shape,target_y.shape)
    
    
    p_yCc = Normal(loc=pred_mu, scale=pred_std)    
    log_p = p_yCc.log_prob(target_y)          # size = [batch_size, *]        
    
    if intrain:
        reduced_log_p = log_p.sum(dim=(-2,-1))  # size = [batch_size]
        #neglogloss = -reduced_log_p  #averages each loss over batches 
        neglogloss = -reduced_log_p 
        
    else:
        reduced_log_p = log_p.mean(dim=(-2,-1))
        #neglogloss = -reduced_log_p.mean()  #averages each loss over batches 
        neglogloss = -reduced_log_p
   
    if reduce:
        return  neglogloss.mean() 
    else:
        return  neglogloss 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    
if __name__ == "__main__":
    model_m = ConvCNP_Multi(rho = UNet(),points_per_unit=128)
