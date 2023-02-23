import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F   

from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from attrdict import AttrDict

from .utils import init_sequential_weights, to_multiple        
from .test_gpsampler4 import Independent_GPsampler
from .test_cnnmodels import get_cnnmodels


def collapse_z_samples(t):
    """Merge n_z_samples and batch_size in a single dimension."""
    n_z_samples, batch_size, *rest = t.shape
    return t.contiguous().view(n_z_samples * batch_size, *rest)


def replicate_z_samples(t, n_z_samples):
    """Replicates a tensor `n_z_samples` times on a new first dim."""
    nb,*nleft = t.shape
    t.unsqueeze_(dim=1)
    return t.expand(nb, n_z_samples, *nleft)
    #return t.repeat(1, n_z_samples, *nleft)


    
    
eps=1e-6    
num_basis = 5
num_fourierbasis = 10
loglik_err = 0.1


class ICGP_Convnp(nn.Module):
    def __init__(self,in_dims: int = 1,out_dims:int = 1, num_channels: int=3,
                      cnntype='shallow', num_postsamples=10,init_lengthscale=1.0):
        super(ICGP_Convnp,self).__init__()
        
        self.modelname = 'gpind'       
        self.in_dims = in_dims
        self.out_dims = out_dims       
        self.num_channels = num_channels
        self.num_samples = num_postsamples
        
        
        self.activation = nn.Sigmoid()
        
        self.cnntype = cnntype
        self.cnn = get_cnnmodels(cnntype)



        #print('updated ind gp prior ')    
        self.gpsampler = Independent_GPsampler(in_dims=in_dims,
                                               out_dims=out_dims,
                                               num_channels=num_channels,
                                               num_fourierbasis = num_fourierbasis,
                                               scales=init_lengthscale,
                                               loglik_err=loglik_err,
                                               points_per_unit=self.cnn.points_per_unit,
                                               multiplier=self.cnn.multiplier)

    
   
            
        self.gppriorscale = self.gpsampler.prior_scale  
    
    
        self.gp_linear = init_sequential_weights(nn.Sequential(nn.Linear(2*self.num_channels,8)))
        #self.gp_linear = nn.Sequential(nn.Conv1d(2*self.num_channels, 8, 1, groups=self.num_channels ))            


        #num_base = 5        
        self.num_basis = num_basis
        #self.num_basis = 2*num_basis        
        self.num_features = num_channels*num_basis
        cnn_linear = nn.Sequential(nn.Linear(self.cnn.out_dims,self.num_features))              
        self.cnn_linear = init_sequential_weights(cnn_linear)
        
        
        self.smoother = ConvDeepset(in_dims=self.in_dims,
                                    out_dims=self.out_dims,
                                    num_basis=self.num_basis,
                                    num_channels=num_channels,
                                    length_scales =0.1)        
        
        
        self.pred_linear = init_sequential_weights(nn.Sequential(nn.Linear(self.num_features,2*self.num_channels)))
        

    def forward(self,xc,yc,xt,yt=None,iterratio=None):        

        nb,ndata,ndim,nchannel = xc.size()
        _ ,ndata2,_,_ = xt.size()
        
        #print('iterratio {:.2f}'.format(iterratio))
        
        #gpouts = self.gpsampler.sample_posterior(xc,yc,xt,reorder=False,numsamples=self.num_samples,iterratio=iterratio)       
        gpouts = self.gpsampler.sample_posterior(xc,yc,xt,reorder=False,numsamples=self.num_samples)       
        
        post_samples = gpouts.posterior_samples
        density = gpouts.density
        xa_samples = gpouts.xa_samples     
        #print('xa_samples.shape {}'.format(xa_samples.shape))
        
        
        density_samples = replicate_z_samples(density,self.num_samples)       
        features = torch.cat([post_samples,density_samples],dim=-1)
        features = self.gp_linear(features)
        features = self.activation(features) #act on 
        

        _,_,ndata,nchannel = features.size()
        features = features.reshape(-1,ndata,nchannel)
        
        features_update = self.cnn(features.permute(0,2,1))
        features_update = self.cnn_linear(features_update.permute(0,2,1))
        features_update = features_update.reshape(nb,self.num_samples,ndata,self.num_basis,self.num_channels)
            
            
        xt = replicate_z_samples(xt,self.num_samples)        
        xa_samples = replicate_z_samples(xa_samples,self.num_samples)
        #print('xa_samples.shape {}'.format(xa_samples.shape))
        
        xa_samples = xa_samples.unsqueeze(-1).repeat(1,1,1,1,self.num_channels)
        #print('xa_samples.shape {}'.format(xa_samples.shape))
        
        xt = collapse_z_samples(xt)
        xa_samples = collapse_z_samples(xa_samples)       
        features_update = collapse_z_samples(features_update)
        
        #smooth feature
        smoothed_features_update = self.smoother(xa_samples,features_update,xt )              
        smoothed_features_update = smoothed_features_update.reshape(nb,self.num_samples,ndata2,-1)
        smoothed_features_update = smoothed_features_update.permute(1,0,2,3)
        
        #predict        
        features_out = self.pred_linear(smoothed_features_update)                
        pmu,plogstd = features_out.split((self.num_channels,self.num_channels),dim=-1)            

 
        outs = AttrDict()
        outs.pymu = pmu
        outs.pystd = 0.1+0.9*F.softplus(plogstd)
        outs.regloss = gpouts.regloss
        outs.gpouts = gpouts

        return outs
       
            
    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()for param in self.parameters()])


    
    def sample_functionalfeature(self,xc,yc,xt,numsamples=1):
        #xa,post_samples,density,yc_samples = self.gpsampler.sample_posterior(xc,yc,xt,numsamples=numsamples,reorder=True)
        gpouts = self.gpsampler.sample_posterior(xc,yc,xt,numsamples=numsamples,reorder=True)
        #return post_samples,xa
        return gpouts.posterior_samples,gpouts.xa_samples




    

class ConvDeepset(nn.Module):
    #def __init__(self,in_channels=1,out_channles=1):
    def __init__(self,in_dims=1,out_dims=1,num_channels=3,num_basis=5,length_scales=0.1,min_length_scales=1e-6):
        
        super(ConvDeepset,self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.num_channels = num_channels
        self.num_basis = num_basis        
        #self.log_std = nn.Parameter( torch.log(min_length_scales + length_scales*torch.ones(in_dims,num_channels,num_basis)) )    
        self.log_std = nn.Parameter( torch.log(min_length_scales + length_scales*torch.ones(in_dims,num_basis,num_channels)) )    
        
    
        
    # in : [64, 320, 1, 3]), [64, 47, 1, 3] ,nbasis=5 
    # out :[64, 320,47,5,3]
    def compute_rbf(self,x1,x2=None):
        if x2 is None:
            x2 = x1            
        
        nbatch,npoints,ndim,nchannel = x1.size()
        x1 = x1.unsqueeze(dim=-2)
        x2 = x2.unsqueeze(dim=-2)        
        x1 = x1.repeat(1,1,1,self.num_basis,1)
        x2 = x2.repeat(1,1,1,self.num_basis,1)
        
        std = self.log_std.exp()[None,None,...]         
        x1 = x1/(std+eps)   #(nb,ndata1,ndim,nbasis,nchannel)
        x2 = x2/(std+eps)   #(nb,ndata2,ndim,nbasis,nchannel)

        square_term = (x1**2).sum(dim=2).unsqueeze(dim=2) + (x2**2).sum(dim=2).unsqueeze(dim=1)
        product_term = torch.einsum('bnmjl,bmkjl->bnkjl',x1,x2.permute(0,2,1,3,4))
        dist_term = square_term -2*product_term        
        return torch.exp(-0.5*dist_term) 

    
    # features : [64, 320,1,3]
    # outs :     [64,,47,5,3]    
    def forward(self,xa,features,xt):
        """
        """
        nb,ndata1,ndim,nchannel = xa.size()
        _ ,ndata2, _  , _ = xt.size()        
        wt = self.compute_rbf(xa,xt)      #   [64, 320,47,5,3]
        smoothed_features = (features[:,:,None,:]*wt).sum(dim=1)       
        return smoothed_features 

    
    
    def extra_repr(self):
        line = 'C_in={}, C_out={}, '.format(self.in_dims, self.out_dims)
        #line += 'coords_dim={}, nbhd={}, sampling_fraction={}, mean={}'.format(self.coords_dim, self.num_nbhd, self.sampling_fraction, self.mean)
        return line

    

    
    
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

def compute_loss_gp( pred_mu,pred_std, target_y , intrain=True ,reduce=True):
    """ compute loss for latent Np models
    
    Args:
        pred_mu:  (nsamples,nbatch,ndata,nchannel)
        pred_std: (nsamples,nbatch,ndata,nchannel)
        target_y: (nbatch,ndata,nchannel)
    Returns:
        neglogloss: 1
    
    """
    #(nsamples,nb,ndata,nchannels) 
    p_yCc = Normal(loc=pred_mu, scale=pred_std)                
    
    if intrain:
        #sum over channels and targets          
        sumlogprob = p_yCc.log_prob(target_y).sum(dim=(-1,-2))   #(nsamples,nb)   
        logmeanexp_sumlogprob= torch.logsumexp(sumlogprob, dim=0) -  math.log(sumlogprob.size(0)) 
        #neglogloss = -logmeanexp_sumlogprob.mean() #mean over batches
        neglogloss = -logmeanexp_sumlogprob
        
    else :
        #sum over channels and targets 
        meanlogprob = p_yCc.log_prob(target_y).sum(dim=(-1,-2))   #(nsamples,nb) 
        logmeanexp_sumlogprob= torch.logsumexp(meanlogprob, dim=0) -  math.log(meanlogprob.size(0))     
        #neglogloss = -logmeanexp_sumlogprob.mean() #mean over batches
        neglogloss = -logmeanexp_sumlogprob #mean over batches        
        mean_factor = np.prod(list(target_y.shape[1:]))
        neglogloss /= mean_factor
        
    if reduce:    
        return neglogloss.mean()
    else:
        return neglogloss

