import numpy as np
import torch
import torch.nn as nn
import random
from torch.distributions import MultivariateNormal
from attrdict import AttrDict
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform




def channel_last(x):
    return x.transpose(1, 2).transpose(2, 3)    
    
    
class Convcnp_2d(nn.Module):
    def __init__(self, in_dim = 2,
                       out_dim = 1,
                       num_channel=3,
                       hdim=128,
                       cnntype = 'shallow'):
        
        super(Convcnp_2d,self).__init__()
        self.modelname = 'base'
        self.in_dim = 2
        self.out_dim = 1
        self.num_channel = num_channel
        self.hdim = hdim
        self.conv_theta = nn.Conv2d(num_channel, hdim, 9, 1, 4)
        
        
        self.cnntype = cnntype
        self.phicnn = get_cnnmodel_2d(modeltype = cnntype,num_channel=num_channel,hdim=hdim)

        self.pos = nn.Softplus()
        self.num_sample = 1
        
        
        
    def forward(self, density,signal):

        # self.conv_theta.abs_constraint()
        density_prime = self.conv_theta(density)
        signal_prime = self.conv_theta(signal)
        h = torch.cat([signal_prime, density_prime], dim=1)

        f = self.phicnn(h)
        mean, std = f.split(self.num_channel, 1)  #(nb,nchannel,ndata,ndata)
        #std = 0.1 + 0.9*self.pos(std)         #(nb,nchannel,ndata,ndata)
        std = 0.01 + 0.99*self.pos(std)         #(nb*num_sample,nchannel,ndata,ndata)
        

        outs = AttrDict()
        #outs.pymu = mean.permute(0,2,3,1)   #(nb,ndata,ndata,nchannel)
        #outs.pystd = std.permute(0,2,3,1)   #(nb,ndata,ndata,nchannel) 
        outs.pymu = mean   #(nb,nchannel,ndata,ndata)
        outs.pystd = std   #(nb,nchannel,ndata,ndata) 
        return outs
    
    
    @property
    def num_params(self):
        """Number of parameters."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])
    
    
    
    
    
    
    
from .test_gpsampler9_2d_v3 import get_gpsampler_2d
from .test_cnnmodel_latest import get_cnnmodel_2d
    

def collapse_z_samples(t):
    """Merge n_z_samples and batch_size in a single dimension."""
    n_z_samples, batch_size, *rest = t.shape
    return t.contiguous().view(n_z_samples * batch_size, *rest)


def replicate_z_samples(t, n_z_samples):
    """Replicates a tensor `n_z_samples` times on a new first dim."""
    nb,*nleft = t.shape
    t = t.unsqueeze(dim=1)
    return t.expand(nb, n_z_samples, *nleft)



eps=1e-6    
#likerr_scale = 1e-2
likerr_scale = 1e-3

max_freq = 5.
#hdim = 20
points_per_unit=64
multiplier=2**6
num_basis = 5

class TDSP_Convnp_2d(nn.Module):
    def __init__(self, in_dim = 2,
                       out_dim = 1,               
                       num_mixture = 2,
                       num_channel=3,
                       num_sample = 4,
                       max_freq = 5.,
                       priorscale = 0.1,
                       use_weightnet = True,                                
                       hdim_sampler = 32, 
                       tempering = 1e-1,
                       hdim_cnn = 128,
                       cnntype = 'shallow',
                       solve='cg'):
        
        super(TDSP_Convnp_2d,self).__init__()
        self.modelname = 'gpdep'
        self.in_dim = in_dim
        self.out_dim = out_dim    
        self.num_sample = num_sample
        
        self.num_channel = num_channel
        self.hdim_sampler = hdim_sampler
        self.hdim_cnn = hdim_cnn
        
        
        

        self.gpsampler = get_gpsampler_2d(num_mixture = num_mixture,
                                         num_channel = num_channel,
                                         likerr_scale= likerr_scale,
                                         max_freq = max_freq,         
                                         hdim = hdim_sampler,
                                         use_weightnet = use_weightnet,
                                         tempering=tempering,
                                         priorscale = priorscale)
        self.gpsampler.solvemode = solve
        self.conv_theta = nn.Conv2d(num_channel, hdim_cnn, 9, 1, 4)
                
        self.cnntype = cnntype
        self.conv_phi = get_cnnmodel_2d(modeltype = cnntype,num_channel=num_channel,hdim=hdim_cnn)
        self.pos = nn.Softplus()

        
        
    def forward(self, density,signal,num_sample=None):

        gpouts = self.gpsampler.sample_posterior(density,signal,num_sample=self.num_sample)        
        post_sampled = collapse_z_samples( gpouts.post_grid.permute(1,0,2,3,4).contiguous())
        density_sampled = replicate_z_samples(density,self.num_sample)
        density_sampled = collapse_z_samples(density_sampled.permute(1,0,2,3,4).contiguous())
        
        #consider concat new figures 
        
        density_prime = self.conv_theta(density_sampled)
        signal_prime =  self.conv_theta(post_sampled)
        h = torch.cat([signal_prime, density_prime], dim=1)
        f = self.conv_phi(h)
        
        mean, logstd = f.split(self.num_channel, 1)  #(nb*num_sample,nchannel,ndata,ndata)
        #std = 0.1 + 0.9*self.pos(logstd)         #(nb*num_sample,nchannel,ndata,ndata)
        std = 0.01 + 0.99*self.pos(logstd)         #(nb*num_sample,nchannel,ndata,ndata)

        mean = mean.reshape(self.num_sample,-1,*mean.shape[-3:])
        std = std.reshape(self.num_sample,-1,*std.shape[-3:])

        outs = AttrDict()
        outs.pymu = mean   #(num_sample,nb,nchannel,ndata,ndata)
        outs.pystd = std   #(num_sample,nb,nchannel,ndata,ndata) 
        outs.gpouts = gpouts
        return outs

    
    @property
    def num_params(self):
        """Number of parameters."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
import math
import torch.nn.functional as F    
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.distributions.categorical import Categorical
from torch.distributions import kl_divergence as kldiv



#-----------------------------------------------------------------------
# update for masked datasets
#-----------------------------------------------------------------------
def compute_latentgp_loss_2d(  outs, target_y , masked_y = None ,reduce=True, tempering_prior=1e0, eps=1e-4):
    dataloss = compute_latentgp_dataloss_2d( outs,target_y, masked_y = masked_y , reduce=reduce)
    regloss = compute_latentgp_regloss_2d(outs.gpouts, target_y , masked_y = masked_y, reduce=reduce, tempering_prior=tempering_prior, eps=eps)
    return dataloss,regloss

    
    
#def compute_latentgp_dataloss_2d( outs, target_y , masked_y = None ,intrain = True,reduce=True, eps=1e-4):
def compute_latentgp_dataloss_2d( outs, target_y , masked_y = None ,reduce=True, eps=1e-4):    
    """ compute loss for latent Np models
    
    Args:        
        pred_mu:  (nsamples,nbatch,nchannel,ndata1,ndata2)
        pred_std: (nsamples,nbatchh,nchannel,ndata1,ndata2)
        target_y: (nbatchh,nchannel,ndata1,ndata2)
        
        masked_y: (nbatch,nchannel,ndata1,ndata2) or None
    Returns:
        neglogloss: 1
    
    """
    imag_dim = [-1,-2,-3]
 


    #(nsamples,nb,ndata,nchannels) 
    pred_mu,pred_std = outs.pymu, outs.pystd
    if masked_y is not None:
        target_y = target_y*masked_y
        pred_mu = pred_mu*masked_y[None,...]
        pred_std = torch.clamp(pred_std*masked_y[None,...],min=eps)
        weight = masked_y[None,...]        #(ns,nb,nchannel,ndata1,ndata2)
    else:
        weight = 1.
        
    # eval likelihood
    p_yCc = Normal(loc=pred_mu, scale=pred_std)                
    
    #sum over channels and targets 
    if masked_y is not None:            
        sumlogprob = (weight*p_yCc.log_prob(target_y)).sum(dim=imag_dim)   #(nsamples,nb)           
        meanlogprob = sumlogprob/weight.sum(dim=imag_dim) 

    else:
        meanlogprob = (weight*p_yCc.log_prob(target_y)).mean(dim=imag_dim)   #(nsamples,nb)           

    logmeanexp_logprob= torch.logsumexp(meanlogprob, dim=0) -  math.log(meanlogprob.size(0)) 

    
    
            
    neglogloss = -logmeanexp_logprob.mean(dim=0) #mean over samples                
    if reduce:    
        return neglogloss.mean()  #mean over batches
    else:
        return neglogloss


    
def compute_latentgp_regloss_2d(gpouts,target_y , masked_y = None,  reduce=True, tempering_prior=1e0, eps=1e-4):
    
    """ compute loss for latent Np models    
    Args:        
        emp_mu:  (nbatch,nmixture,nchannel,ndata1,ndata2)
        emp_std: (nbatchh,nmixture,nchannel,ndata1,ndata2)
        target_y: (nbatchh,nchannel,ndata1,ndata2)
        
        masked_y: (nbatch,nchannel,ndata1,ndata2) or None
    Returns:
        neglogloss: 1
    
    """
    grid_dim = [-1,-2]

    pred_mu,pred_std = gpouts.post_empdist #torch.Size([6, 9, 3, 32, 32])    
    if masked_y is not None:
        target_y = target_y*masked_y        
        pred_mu = pred_mu*masked_y[:,None,...]
        pred_std = torch.clamp(pred_std*masked_y[:,None,...],min=eps)
        weight = masked_y[:,None,...]        #(nb,1,nchannel,ndata1,ndata2)
    else:
        weight = 1.
        
    empdist = Normal(loc=pred_mu, scale=pred_std+eps)             
    emp_logprob = empdist.log_prob(target_y[:,None,...])  #(nb,nm,nch,ndat1,ndat2)
    
    #print('emp_logprob.shape {}'.format(emp_logprob.shape))
    
    #if masked_y is not None:
    emp_logprob_sum = (weight*emp_logprob).sum(dim=grid_dim)  #(nb,nm,nch)
    emp_logprob_sum = emp_logprob_sum - emp_logprob_sum.max(dim=1,keepdim=True)[0]
    emp_logprob_sum = emp_logprob_sum / tempering_prior
    #print('emp_logprob_sum.shape {}'.format(emp_logprob_sum.shape))
    
    appr_logits = F.softmax(emp_logprob_sum,dim=1).clamp(min=eps,max=1 - eps)        
    
    #print('appr_logits.shape {}'.format(appr_logits.shape))
    
    appr_logits = appr_logits.permute(0,2,1)  #(nb,nch,nm)

    learnable_logits = gpouts.mixweight
    q_cat = Categorical(probs=learnable_logits+eps)
    p_cat = Categorical(probs=appr_logits+eps)
    
    regloss = kldiv(q_cat,p_cat).sum(dim=-1) #(nb)
    #print('regloss.shape {}'.format(regloss.shape))
    if reduce:        
        return regloss.mean()
    else:
        return regloss
    
    
    
    
#def compute_baseline_loss( outs, target_y, masked_y = None, intrain=True,reduce=True,tasktype='2d'):    
def compute_baseline_loss( outs, target_y, masked_y = None,reduce=True,tasktype='2d'):    
    
    """ compute loss for conditional Np models

    Args:
        pred_mu:  (nbatch,ndata,nchannel) or (nbatch,ndata,ndata,nchannel)
        pred_std: (nbatch,ndata,nchannel) or (nbatch,ndata,ndata,nchannel)
        target_y: (nbatch,ndata,nchannel)or (nbatch,ndata,ndata,nchannel)
    Returns:
        logloss: 1

    """
    # 1d: multi-channel time-series, 2d : image completion
    assert tasktype in ['1d','2d']
    
    if tasktype == '1d':
        dim_task = (-2,-1)
    if tasktype == '2d':
        dim_task = (-3,-2,-1)

        
    pred_mu, pred_std = outs.pymu, outs.pystd              
    if masked_y is not None:
        target_y = target_y*masked_y        
        pred_mu = pred_mu*masked_y
        pred_std = torch.clamp(pred_std*masked_y,min=eps)
        weight = masked_y       #(nb,nchannel,ndata1,ndata2)
    else:
        weight = 1.

    
    p_yCc = Normal(loc=pred_mu, scale=pred_std)    
    log_p = p_yCc.log_prob(target_y)          # size = [batch_size, *]        
    
    
    if masked_y is not None:            
        reduced_log_p = (weight*log_p).sum(dim=dim_task)  # size = [batch_size]
        reduced_log_p = reduced_log_p/(weight.sum(dim=dim_task))
    else:
        reduced_log_p = log_p.mean(dim=dim_task)  # size = [batch_size]

                                       
    neglogloss = -reduced_log_p    
    if reduce:
        return  neglogloss.mean(), torch.tensor(0.0) 
    else:
        return  neglogloss, torch.zeros_like(neglogloss).to(target_y.device) 
        #return  neglogloss, torch.tensor(0.0)         
    
    
    
    
    
    



    
    
    
    