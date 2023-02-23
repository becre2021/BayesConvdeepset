import numpy as np
import math
import itertools as it


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

from attrdict import AttrDict

from .test_gpsampler8_1d_v3 import sample_gumbel_softmax,Stkernel_basis,Effcient_GPsampler
from .test_cnnmodel_latest import Conv2dResBlock
from .utils import  to_multiple,init_sequential_weights

import math
pi2 = 2*math.pi


# def to_multiple(x, multiple):
#     """Convert `x` to the nearest above multiple.

#     Args:
#         x (number): Number to round up.
#         multiple (int): Multiple to round up to.

#     Returns:
#         number: `x` rounded to the nearest above multiple of `multiple`.
#     """
#     if x % multiple == 0:
#         return x
#     else:
#         return x + multiple - x % multiple

# def init_sequential_weights(model, bias=0.0):
#     """Initialize the weights of a nn.Sequential model with Glorot
#     initialization.

#     Args:
#         model (:class:`nn.Sequential`): Container for model.
#         bias (float, optional): Value for initializing bias terms. Defaults
#             to `0.0`.

#     Returns:
#         (nn.Sequential): model with initialized weights
#     """
#     for layer in model:
#         if hasattr(layer, 'weight'):
#             nn.init.xavier_normal_(layer.weight, gain=1)
#         if hasattr(layer, 'bias'):
#             nn.init.constant_(layer.bias, bias)
#     return model


class transinvariant_cnn(nn.Module):
    def __init__(self,num_channel=3,
                      num_mixture=9,
                      hdim=20,
                      eps=1e-6):
        
        super(transinvariant_cnn,self).__init__()
        
        self.num_channel = num_channel
        self.num_mixture = num_mixture
        self.hdim = hdim

        conv_weightnet = [nn.Conv2d(self.num_channel, self.hdim , 9, 1, 4),
                          Conv2dResBlock(self.hdim,self.hdim),
                          Conv2dResBlock(self.hdim,self.hdim),                          
                          nn.Conv2d(self.hdim ,self.num_mixture, 9, 1, 4),
                         ]
        #self.conv_weightnet = nn.Sequential(*conv_weightnet)
        self.conv_weightnet = init_sequential_weights(nn.Sequential(*conv_weightnet))
        
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        #self.relu = nn.ReLU() 
        self.tanh = nn.Tanh()
        self.mlp = nn.Linear(self.num_mixture,self.num_mixture*self.num_channel)
        
        
        
        
    def forward(self,signal):
        nb = signal.size(0)
        #outs = self.conv_processing(signal)
        outs = self.conv_weightnet(signal)
        outs = self.pooling(outs).squeeze()
        outs = outs.unsqueeze(dim=-1) if self.num_mixture == 1 else outs        
        #if self.num_mixture == 1:
        #    outs = outs.unsqueeze(dim=-1)
            
        outs = self.mlp(outs).reshape(nb,self.num_channel,self.num_mixture)
        outs = self.tanh(outs)
        return outs
           

        
                
def imag_to_vec(imag):
    nb,nchannel,ngrid1,ngrid2 = imag.shape
    return imag.reshape(nb,nchannel,-1).permute(0,2,1)

def vec_to_imag(vec):
    nb,nchannel,ngrid1,ngrid2 = imag.shape
    return imag.reshape(nb,nchannel,-1).permute(0,2,1)



class Effcient_GPsampler_imag(Effcient_GPsampler):
    def __init__(self,kernel,
                      use_weightnet = True, 
                      hdim=10,
                      tempering=1e-1,
                      priorscale=0.1):
                          
            
        super(Effcient_GPsampler_imag,self).__init__(kernel,
                                                   use_weightnet = use_weightnet, 
                                                   hdim=hdim,
                                                   tempering=tempering,
                                                   )
        
        
        self.kernel = kernel
        self.num_channel = self.kernel.num_channel
        self.num_mixture = self.kernel.num_mixture

        # cnn
        #self.point_per_unit = points_per_unit
        #self.multiplier = multiplier
        
        # use priorassign net
        self.hdim = hdim         
        self.use_weightnet = use_weightnet
        if self.use_weightnet:            
            self.weight_net = transinvariant_cnn(num_channel=self.num_channel,
                                                 num_mixture=self.num_mixture,
                                                 hdim = self.hdim)

            
            
        else:            
            self.weight_net = None
        
        self.tempering = tempering
        self.mixweight = None
        self.mixweight_sample = None        
        
        self.solvemode = None
        #self.cgsolver = LinearCG_Solver.apply

        self.kernelsize = (5,5)
        self.priorscale = priorscale
        return 
    
    

    def bound_hypparams(self,eps=1e-6):
        """
        bound_std = [1.,2.] --> bound_invstd = [.5,1.] 
        """        
        device = self.kernel.logmu.device
        with torch.no_grad():
            #for i,(i_mu,i_std) in enumerate(zip(self.kernel.logmu,self.kernel.logstd)):
            for _,(i_mu,i_mu_bound,i_std,i_std_bound) in enumerate(zip(self.kernel.logmu,self.kernel.bound_logmu,self.kernel.logstd,self.kernel.bound_logstd)):
                
                i_mu.data.clip_(i_mu_bound[...,0].to(device),i_mu_bound[...,1].to(device))
                i_std.data.clip_(i_std_bound[...,0].to(device),i_std_bound[...,1].to(device))

            self.kernel.loglikerr.data.clip_(self.kernel.bound_loglikerr[0].to(device),
                                             self.kernel.bound_loglikerr[1].to(device)) 
                        
        return
    
    


    def build_gridvec_from_imag(self,density,x1_bound=[0.0,1.0],x2_bound=[0.0,1.0],gridratio=1.):        
        nb,nchannel,ngrid1,ngrid2 = density.shape
        self.ngrid = (ngrid1,ngrid2)
        
        #nb,nchannel,ngrid1,ngrid2 = 6,3,32,48
        xgrid1 = torch.linspace( *x1_bound, int(gridratio*ngrid1)).to(density.device)
        xgrid2 = torch.linspace( *x2_bound, int(gridratio*ngrid2)).to(density.device)
        xgrid_list = [xgrid1[None,:,None,None].repeat(nb,1,1,nchannel),xgrid2[None,:,None,None].repeat(nb,1,1,nchannel)]
        xgrid_vec = torch.cartesian_prod(xgrid1,xgrid2)[None,:,:,None,].repeat(nb,1,1,nchannel)    
        #xgrid_imag = (xgrid1[None,:]*xgrid2[:,None])[None,None,...].repeat(nb,nchannel,1,1)

        #return xgrid_vec,xgrid_imag,xgrid_list
        return xgrid_vec


    def build_ctvec_from_imag(self,density,signal):
        density_vec = imag_to_vec(density)
        signal_vec = imag_to_vec(signal)

        idx_nb,idx_ch,idx_data = torch.where(density_vec==1)
        chosen_idx = [torch.where((ith_density_vec == 1).sum(dim=1) > 0 )[0] for ith_density_vec in density_vec]
        chosen_idxnum = [len(i_idx) for i_idx in chosen_idx]
        mindata_batch = min(chosen_idxnum)

        xgrid_vec = self.build_gridvec_from_imag(density)
        xc_vec = [xgrid_vec[i,i_idx[:mindata_batch],...].unsqueeze(dim=0).detach().clone() for i,i_idx in enumerate(chosen_idx)]
        yc_vec = [signal_vec[i,i_idx[:mindata_batch],...].unsqueeze(dim=0).detach().clone() for i,i_idx in enumerate(chosen_idx)]
        xc_vec = torch.cat(xc_vec,dim=0)
        yc_vec = torch.cat(yc_vec,dim=0)
        return xc_vec,yc_vec,xgrid_vec

    

    #def build_mixweight(self,xc,yc,num_sample=10,eps=1e-6):
    def build_mixweight(self,density,signal,num_sample=10,eps=1e-6):                
        logit = self.weight_net(signal) #logweight
        logit = logit / (self.tempering+eps)
        mixweight = F.softmax(logit, dim=-1) 
        mixweight_sample = sample_gumbel_softmax(logit,num_sample=num_sample)        
        
        #gpsampler.Kcc.shape torch.Size([8, 64, 64, 3, 5])        
        #self.Kcc = Kxx
        self.mixweight = mixweight
        self.mixweight_sample = mixweight_sample
        return mixweight,mixweight_sample

    

    def sample_prior(self,density,signal,num_sample=10):                
        xc,yc,xgrid = self.build_ctvec_from_imag(density,signal)
        self.xgrid = xgrid
        #prior_grid, priormix_grid = self.kernel.sample_randomfunction_weight(self.mixweight_sample,xgrid,num_sample=num_sample,use_newsample=True)
        priormix_grid = self.kernel.sample_randomfunction(xgrid,num_sample=num_sample,use_newsample=True)

        return priormix_grid,(density,signal,xc,yc,xgrid)        
        #return prior_grid,priormix_grid,(xc,yc,xgrid)

        



    #def build_update_function(self,xc,yc,xt,xgrid,num_sample=10):
    def build_update_function(self,density,signal,xc,yc,xgrid,num_sample=10,eps=1e-4):
        
        #prepare update function on grid and target
        nb,ndata,ndim,nchannel = xc.size()

        #update term on grid: weighted and not weighted                            
        #(nb,nsample,ndata,mixture,nchannel)
        priormix_xc = self.kernel.sample_randomfunction(xc,num_sample=num_sample,use_newsample=False)        
        Kccmix = self.kernel.build_Kxx(xc,addnoise=True)
        Kccmix = Kccmix.permute(0,4,1,2,3)   #(nb,nchannel,nmixture,ndata,ndata)            
        
        likerr = self.kernel.loglikerr.exp()
        noise2 = torch.randn(nb,1,1,1,1).to(xc.device)*likerr[None,None,None,None,:]
        delta_ycmix = yc[:,None,:,None,:] - (priormix_xc + noise2)        

               
        Kgcmix = self.kernel.build_Kxt(xgrid,xc)        
        Kgcmix = Kgcmix.permute(0,4,1,2,3)   #(nb,nchannel,nmixture,ngrid,ndata) 
                 
        #mixweight, mixweight_sample = self.build_mixweight(Kgcmix,yc,num_sample=num_sample)
        mixweight, mixweight_sample = self.build_mixweight(density,signal,num_sample=num_sample)


        # -----------------------------
        # reflect weight
        # -----------------------------        
        prior_xc = self.kernel.reflect_weightsample_randomfunction(mixweight_sample,priormix_xc)                    
        noise1 = torch.randn_like(prior_xc).to(xc.device)*likerr[None,None,None,:]
        delta_yc = yc[:,None,:,:] - (prior_xc + noise1)  
                                        
            
        Kcc = self.kernel.reflect_weight_Kxt(mixweight,Kccmix) #(nb,nchannel,ndata,ndata)
        Kgc = self.kernel.reflect_weight_Kxt(mixweight,Kgcmix)
        
        # ---------------------------        
        
        
        #solve linear system
        if self.solvemode == 'chol':
            Lcc = torch.linalg.cholesky(Kcc)
            Lccmix = torch.linalg.cholesky(Kccmix) 
            
            Kccinvyc = torch.cholesky_solve(delta_yc.permute(0,3,2,1),Lcc,upper=False)  #(nb,nchannel,ndata,nsample)            
            Kccmixinvyc = torch.cholesky_solve(delta_ycmix.permute(0,4,3,2,1),Lccmix,upper=False)  #(nb,nchannel,nmix,ndata,nsample)
            
        if self.solvemode == 'cholappr':
            Lcc = torch.linalg.cholesky(Kcc)
            Lccmix = torch.linalg.cholesky(Kccmix) 
            
            
            likerr = (2*self.kernel.loglikerr).exp()
            D1 = 1/(1+likerr[None,:,None,None]+eps )
            D2 = 1/(1+likerr[None,:,None,None,None]+eps)
            
            Kccinvyc = D1*delta_yc.permute(0,3,2,1)
            Kccmixinvyc = D2*delta_ycmix.permute(0,4,3,2,1)
            
            #print('D1.shape,D2.shape')            
            #print(D1.shape,D2.shape)
            #print('Lcc.shape,Lccmix.shape,delta_yc.permute(0,3,2,1).shape,delta_ycmix.permute(0,4,3,2,1).shape')            
            #print(Lcc.shape,Lccmix.shape,delta_yc.permute(0,3,2,1).shape,delta_ycmix.permute(0,4,3,2,1).shape)
        
        elif self.solvemode == 'linsolve':
            Kccinvyc = torch.linalg.solve(Kcc,delta_yc.permute(0,3,2,1))
            Kccmixinvyc = torch.linalg.solve(Kccmix,delta_ycmix.permute(0,4,3,2,1))        
        
        
        elif self.solvemode == 'cg':
            Kccinvyc = self.cgsolver(Kcc,delta_yc.permute(0,3,2,1))
            Kccmixinvyc = self.cgsolver(Kccmix,delta_ycmix.permute(0,4,3,2,1))
            
        else:
            pass

            
        # prepare update term
        update_grid = torch.einsum('bsxy,bsyz->bsxz',Kgc,Kccinvyc ).permute(0,3,2,1)   #(nb,nsample,ndata,nchannel)
        updatemix_grid = torch.einsum('bcmxy,bcmyz->bcmxz',Kgcmix,Kccmixinvyc ).permute(0,4,3,2,1)  #(nb,nsample,ndata,nmix,nchannel)
        
        return update_grid,updatemix_grid
    

    

    #def sample_posterior(self,density,signal,num_sample=10, scale=0.1 ,eps=1e-4):
    def sample_rescaled_prior(self,density,signal,num_sample=10 ,eps=1e-4):
        
        nb,nchannel,ngrid1,ngrid2 = density.shape
        #signal,static_mean,static_std = self.compute_static(density,signal)        
        #self.static_mean = static_mean
        #self.static_std = static_std
    
        priormix_grid,dataformat_vec = self.sample_prior(density,signal,num_sample=num_sample)        
        mixweight, mixweight_sample = self.build_mixweight(density,signal,num_sample=num_sample)
        
        #reflect weight
        prior_grid = self.kernel.reflect_weightsample_randomfunction(self.mixweight_sample,priormix_grid)            
        prior_grid = prior_grid.permute(0,1,3,2).contiguous().reshape(nb,num_sample,nchannel,ngrid1,ngrid2)
        priormix_grid = priormix_grid.permute(0,1,3,4,2).contiguous().reshape(nb,num_sample,self.num_mixture,nchannel,ngrid1,ngrid2)
        
        scale_signal = (signal[signal != 0].abs()).mean()
        #scale_signal = (signal.abs()).mean()        
        scale_priormix = priormix_grid.abs().mean()        
        scale_prior = prior_grid.abs().mean()
        rescale_for_priormix = (scale_signal/(scale_priormix+eps)).detach().clone()
        rescale_for_prior = (scale_signal/(scale_prior+eps)).detach().clone()
        rescaled_priormix_grid = self.priorscale*rescale_for_priormix*priormix_grid        
        rescaled_prior_grid = self.priorscale*rescale_for_prior*prior_grid

        return rescaled_prior_grid, rescaled_priormix_grid, prior_grid, priormix_grid
    
    
        
    def build_update_approximation(self,signal,rescaled_priormix_grid,rescaled_prior_grid,num_sample=10,eps=1e-4):
        nb,nchannel,ngrid1,ngrid2 = signal.shape

        delta_signal = signal[:,None,:,:,:] - rescaled_prior_grid
        delta_signal = delta_signal.reshape(-1,*delta_signal.shape[-2:])[:,None,...]               
        
        filterweight = self.build_filterweight(signal)
        self.filterweight = filterweight
        filtered_delta_signal = F.conv2d(delta_signal,filterweight,stride=1)
        padsize = (filterweight.size(-2)//2,filterweight.size(-2)//2,filterweight.size(-1)//2,filterweight.size(-1)//2)
        filtered_delta_signal = F.pad(filtered_delta_signal, padsize,'constant',0)
        
        updatemix_grid = filtered_delta_signal.reshape(nb, num_sample, nchannel,*filtered_delta_signal.shape[-3:])        
        update_grid = (self.mixweight_sample[...,None,None]*updatemix_grid).sum(dim=-3)
        
        updatemix_grid = updatemix_grid.permute(0,1,3,2,4,5).contiguous() 
        return updatemix_grid,update_grid

    
    #poor results
#     def build_update_approximation(self,density,signal,rescaled_priormix_grid,rescaled_prior_grid,num_sample=10,eps=1e-4):
#         nb,nchannel,ngrid1,ngrid2 = signal.shape

#         delta_signal = signal[:,None,:,:,:] - rescaled_prior_grid
#         delta_signal = delta_signal*density[:,None,:,:,:] # different poitns.
        
#         delta_signal = delta_signal.reshape(-1,*delta_signal.shape[-2:])[:,None,...]               
#         #print('delta_signal.shape {}'.format(delta_signal.shape))
        
#         #delta_signal.reshape(-1,*delta_signal.shape[-2:])[:,None,...]               
        
#         filterweight = self.build_filterweight(signal)
#         self.filterweight = filterweight
#         filtered_delta_signal = F.conv2d(delta_signal,filterweight,stride=1)
#         padsize = (filterweight.size(-2)//2,filterweight.size(-2)//2,filterweight.size(-1)//2,filterweight.size(-1)//2)
#         filtered_delta_signal = F.pad(filtered_delta_signal, padsize,'constant',0)
        
#         updatemix_grid = filtered_delta_signal.reshape(nb, num_sample, nchannel,*filtered_delta_signal.shape[-3:])        
#         update_grid = (self.mixweight_sample[...,None,None]*updatemix_grid).sum(dim=-3)
        
#         updatemix_grid = updatemix_grid.permute(0,1,3,2,4,5).contiguous() 
#         return updatemix_grid,update_grid

    
    def compute_static(self,density,signal,eps=1e-4):
        with torch.no_grad():
            ##v1
            #signal_mean = signal.sum(dim=(-1,-2),keepdim=True)/(density.sum(dim=(-1,-2),keepdim=True)+eps)
            #signal_square = (signal**2).sum(dim=(-1,-2),keepdim=True)/(density.sum(dim=(-1,-2),keepdim=True)+eps)
            #signal_std = torch.sqrt(signal_square - (signal_mean**2))
            #static_mean =  signal_mean.detach().clone()
            #static_std =  signal_std.detach().clone()
            #signal = (signal - signal_mean*density)/(signal_std*density+eps)        
            
            #v2 --> this is better normalization compared to v1
            signal_mean =  signal.mean(dim=(-1,-2),keepdim=True).detach().clone()
            signal_std =  signal.std(dim=(-1,-2),keepdim=True).detach().clone()
            signal = (signal - signal_mean)/(signal_std+eps)        
        return signal,signal_mean,signal_std
    
        
        
        
    def correct_post(self,density,signal,post_grid,postmix_grid,eps=1e-4,allow_err=1e-2):
        post_grid = torch.clamp(post_grid,min=signal.min().item()-allow_err, max=signal.max().item()+allow_err)
        postmix_grid = torch.clamp(postmix_grid,min=signal.min().item()-allow_err, max=signal.max().item()+allow_err)
        
                                                                                          
        return post_grid, postmix_grid
        
        
    def sample_posterior(self,density,signal,num_sample=10 ,eps=1e-4,allow_err=1e-2):
        
        nb,nchannel,ngrid1,ngrid2 = density.shape
        
        n_signal,static_mean,static_std = self.compute_static(density,signal)                
        rescaled_prior_grid, rescaled_priormix_grid, prior_grid, priormix_grid = self.sample_rescaled_prior(density,n_signal,num_sample=num_sample-1)
        
        updatemix_grid,update_grid = self.build_update_approximation(n_signal,rescaled_priormix_grid,rescaled_prior_grid,num_sample=num_sample-1)         
        #updatemix_grid,update_grid = self.build_update_approximation(density,n_signal,rescaled_priormix_grid,rescaled_prior_grid,num_sample=num_sample-1)                
        post_grid =  rescaled_prior_grid + update_grid        
        postmix_grid = rescaled_priormix_grid + updatemix_grid        
        
        post_grid = static_mean[:,None,:,:,:]  +  (static_std[:,None,:,:,:])*post_grid  #(nb,ns,,nchannel,ngrid1,ngrid2)   
        postmix_grid = static_mean[:,None,None,:,:,:]  +  (static_std[:,None,None,:,:,:])*postmix_grid  #(nb,nmix,ns,nchannel,ngrid1,ngrid2)   
        post_grid, postmix_grid = self.correct_post(density,signal,post_grid,postmix_grid)
        
        post_grid = torch.cat([signal[:,None,:,:,:],post_grid],dim=1)        
        
        
        postmix_mu,postmix_std = postmix_grid.mean(dim=1), postmix_grid.std(dim=1) 
        
        outs = AttrDict()
        outs.prior_grid = prior_grid        
        outs.rescaled_prior_grid = rescaled_prior_grid        
        outs.priormix_grid = priormix_grid        
        outs.rescaled_priormix_grid = rescaled_priormix_grid        
        
        outs.update_grid = update_grid            
        outs.updatemix_grid = updatemix_grid      
        
        outs.post_grid = post_grid #(nb,ns,nchannel,ngrid1,ngrid2)        
        outs.postmix_grid = postmix_grid  #(nb,ns,nmixture,nchannel,ngrid1,ngrid2)
        
        
        #outs.postmix_tar = postmix_target
        outs.post_empdist = (postmix_mu.detach().clone(),postmix_std.detach().clone())  #(nb,nmixture,nchannel,ngrid1,ngrid2)
        outs.mixweight = self.mixweight
        outs.mixweight_sample = self.mixweight_sample        
        outs.filterweight = self.filterweight

        return outs 
    
    
    
    
    
    


    

    def build_filterweight(self,signal):
        nb,nchannel,*ngrid = signal.shape
        
        def build_garborweight(x,mu,std):
            length_filter = len(x)
            
            cos_right_term = torch.cos(pi2*x[None,:]*mu[:,None])
            cos_left_term = torch.flip(cos_right_term[:,1:],dims=(-1,))
            cos_term = torch.cat([cos_left_term,cos_right_term],dim=-1)

            exp_right_term = torch.exp(-0.5*(pi2*x[None,:]*std[:,None])**2) 
            exp_left_term = torch.flip(exp_right_term[:,1:],dims=(-1,))
            exp_term = torch.cat([exp_left_term,exp_right_term],dim=-1)

            #exp_cos_term = (exp_term*cos_term).T
            exp_cos_term = (exp_term*cos_term).T/length_filter            

            return exp_cos_term        
        
        
        mu, std = self.kernel.logmu.exp().T, self.kernel.logstd.exp().T
        
        x0 =torch.linspace(0,1,ngrid[0]).to(signal.device)
        x0 = x0[:(self.kernelsize[0]//2+1)]
        x1 = torch.linspace(0,1,ngrid[1]).to(signal.device)
        x1 = x1[:(self.kernelsize[1]//2+1)]        
                
        weight_dim1 = build_garborweight(x0,mu[0],std[0])
        weight_dim2 = build_garborweight(x1,mu[1],std[1])
        weight = weight_dim1[:,None,:]*weight_dim2[None,:,:]
        weight =  weight.permute(2,0,1)[:,None,:,:]        
        
        return weight    
    
    
    def build_smoothdensity(self,density,eps=1e-4):
        nb,nchannel,*ngrid = density.shape

        filterweight = self.build_filterweight(density)
        self.filterweight = filterweight
        density_filtered = density.reshape(-1,*density.shape[-2:])[:,None,...]                        
        density_filtered = F.conv2d(density_filtered,filterweight,stride=1)
        padsize = (self.kernelsize[0]//2,self.kernelsize[0]//2,self.kernelsize[1]//2,self.kernelsize[1]//2)
        density_filtered = F.pad(density_filtered, padsize,'constant',0)

        density_filtered = density_filtered.reshape(nb,-1,*density_filtered.shape[-3:])
        density_filtered = density_filtered.permute(0,2,1,3,4).abs()
        
        #magnitude = density_filtered.mean(dim=(-1,-2),keepdim=True).detach().clone()    
        #density_filtered = density_filtered + magnitude*density[:,None,...]
        #density_filtered = (density_filtered / (density_filtered.amax(dim=(-2,-1),keepdim=True) +eps) )

        return density_filtered


    
    
    
    
    
    
    
    

    
def get_gpsampler_2d(num_mixture: int = 3,
                     num_channel: int = 3,
                     likerr_scale: float = 1e-2,
                     max_freq : float = 10.,                       
                     use_weightnet : bool = True,
                     hdim: int = 32,
                     tempering: float = 1e-1,    
                     priorscale: float = 0.1
                     ):    
    #num_channel = 3
    #num_mixture = 3
    #max_freq = 10.

    kernel = Stkernel_basis(num_mixture=num_mixture,
                            num_channel=num_channel,
                            in_dims=2,
                            likerr_scale=likerr_scale,
                            max_freq=max_freq)

    gpsampler = Effcient_GPsampler_imag(kernel,
                                         use_weightnet = use_weightnet, 
                                         hdim=hdim,
                                         tempering=tempering,
                                         priorscale = priorscale)

    gpsampler.solvemode = 'chol'
    #gpsampler.solvemode = 'linsolve'
    #gpsampler.solvemode = 'cg-autodiff'    
    return gpsampler
    
    
        