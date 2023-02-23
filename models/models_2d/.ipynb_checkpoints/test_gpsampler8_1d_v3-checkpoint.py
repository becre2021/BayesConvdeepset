
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import numpy as np
import math
import itertools as it

from attrdict import AttrDict
from .utils import to_multiple





# #chosen parameters folder setting, but does not recognize the task yet
class transinv_mlp_1d(nn.Module):
    def __init__(self,in_dim=1,num_channel=1,num_mixture=1,hdim=10,eps=1e-6):
        
        super(transinv_mlp_1d,self).__init__()
        
        self.in_dim = in_dim
        self.hdim = hdim
        self.num_channel = num_channel 
        self.num_mixture = num_mixture

        

        #self.fc1 = nn.Linear(num_mixture+1,hdim*num_channel)    
        self.fc1 = nn.Linear(num_mixture,hdim)            
        self.fc2 = nn.Linear(num_channel*hdim,hdim)    
        self.fc3 = nn.Linear(hdim,hdim)            
        self.fc4 = nn.Linear(hdim,num_mixture*num_channel)    

        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)    
        nn.init.xavier_uniform_(self.fc4.weight)    
        #nn.init.xavier_uniform_(self.fc5.weight)    



    def forward(self,Kgcmix,yc):
        
        """
        args:
            #xc: (nb,ndata,ndim,nchannel)
            Kcc: (nb,ndata,ndata,nchannel,nmixture)
            yc: (nb,ndata,nchannel)            
        return :
            logit: (nb,nchannel,nmixuture)

        """
        
        nb,ndata,nchannel=yc.shape        
        
        #print('Kgcmix.shape {}, yc.shape {}'.format(Kgcmix.shape,yc.shape))
        #[16, 1, 3, 384, 21],[16, 21, 1]        
        yc_ = yc.permute(0,2,1).contiguous()[:,:,None,None,:]
        feature_xc = (Kgcmix*yc_).sum(dim=-1) #(nb,nchannel,nmix,ngrid)        
        feature_xc = feature_xc.permute(0,1,3,2).contiguous() #(nb,nchannel,ngrid,nmix)
        
        h = F.relu(self.fc1(feature_xc))  #(nb,ndata,nchannel,nmixture+1) 
        h = (h.mean(dim=-2)).reshape(nb,-1)    #(nb,nchannel*hdim)                         
        h = F.relu(self.fc2(h)) 
        h = F.relu(self.fc3(h))         
        #h = F.relu(self.fc4(h))                 
        logit = self.fc4(h)         
        logit = logit.reshape(nb,self.num_channel,-1)              
        return logit
    

class transinv_cnn_1d(nn.Module):
    def __init__(self,in_dim=1,num_channel=1,num_mixture=1,hdim=10,eps=1e-6):
        
        super(transinv_cnn_1d,self).__init__()
        
        self.in_dim = in_dim
        self.hdim = hdim
        self.num_channel = num_channel
        self.num_mixture = num_mixture


        cnn_list = [nn.Conv1d(num_channel*num_mixture, hdim*num_mixture, 5, stride=2, groups = num_mixture),
                    nn.ReLU(),
                    nn.Conv1d(hdim*num_mixture, hdim*num_mixture, 5, stride=2, groups = 1),
                    nn.ReLU(),                    
                    nn.Conv1d(hdim*num_mixture, hdim*num_mixture, 5, stride=2, groups = 1),
                    nn.ReLU(),                    
                    nn.Conv1d(hdim*num_mixture, hdim*num_mixture, 5, stride=2, groups = 1),                    
                   ]
        self.cnn = nn.Sequential(*cnn_list)
        self.pooling = nn.AdaptiveAvgPool1d(1)        
        self.logit = nn.Linear(hdim*num_mixture,num_mixture*num_channel ) 
        
                
    def forward(self,Kgc,yc):
        
        """
        args:
            Kcc: (nb,nchannel,nmixture,ngrid,ndata)
            yc: (nb,ndata,nchannel)            
        return :
            logit: (nb,nchannel,nmixuture)

        """
        #([16, 384, 21, 1, 3])
        nb,ndata,nchannel=yc.shape                
        yc = yc.permute(0,2,1).contiguous()
        feature_xyc = (Kgc*yc[:,:,None,None,:]).sum(dim=-1)   #(nb,nchannel,nmixutre,ngrid)        
        feature_xyc = torch.tensor_split(feature_xyc,feature_xyc.size(1),dim=1)   
        feature_xyc = torch.cat(feature_xyc,dim=2).squeeze() #(nb,nchannel*nmixutre,ngrid)
        
        h = self.cnn(feature_xyc)
        h = self.pooling(h).squeeze()
        #print(h.shape)
        logit = self.logit(h)         
        logit = logit.reshape(nb,self.num_channel,-1)              
        return logit    



class transinv_cnn_1d_simple(nn.Module):
    def __init__(self,in_dim=1,num_channel=1,num_mixture=1,hdim=10,eps=1e-6):
        
        super(transinv_cnn_1d_simple,self).__init__()
        
        self.in_dim = in_dim
        self.hdim = hdim
        self.num_channel = num_channel
        self.num_mixture = num_mixture


        cnn_list = [nn.Conv1d(num_channel, hdim, 5, stride=2, groups = 1),
                    nn.ReLU(),
                    nn.Conv1d(hdim, hdim, 5, stride=2, groups = 1),
                    nn.ReLU(),                    
                    nn.Conv1d(hdim, hdim, 5, stride=2, groups = 1),
                    nn.ReLU(),                    
                    nn.Conv1d(hdim, hdim, 5, stride=2, groups = 1),                    
                   ]
        self.cnn = nn.Sequential(*cnn_list)
        self.pooling = nn.AdaptiveAvgPool1d(1)        
        self.logit = nn.Linear(hdim,num_mixture*num_channel ) 
        
                
    #def forward(self,Kgc,yc):
    def forward(self,datarepr):
        
        """
        args:
            datarepr: (nb,ngrid,nchannel)
        return :
            logit: (nb,nchannel,nmixuture)

        """
        #([16, 384, 21, 1, 3])
        #nb,ndata,nchannel=yc.shape                
        #yc = yc.permute(0,2,1).contiguous()
        
        nb,ndata,nchannel=datarepr.shape                        
        datarepr = datarepr.permute(0,2,1).contiguous() #(nb,nchannel,ngrid)        
        h = self.cnn(datarepr)
        h = self.pooling(h).squeeze()
        #print(h.shape)
        logit = self.logit(h)         
        logit = logit.reshape(nb,self.num_channel,-1)              
        return logit    
    
    

def sample_gumbel_softmax(logits, num_sample=1, temperature=1e-1,training=True,eps=1e-8):           
    """
    args:
        logits: Tensor of "log" probs, : (nb,nchannel,nmixture) 
                temperature = scalar
        
        Output: Tensor of values sampled from Gumbel softmax.
                These will tend towards a one-hot representation in the limit of temp -> 0
                shape = BS x k
    """
    def sample_gumbel(samples_shape, eps=1e-8):
        #shape = logits.shape
        unif = torch.rand(samples_shape)
        g = -torch.log(-torch.log(unif + eps))
        return g.float()
        
    nb,nchannel,nmixture = logits.shape                
    if training :
        g = sample_gumbel((nb,num_sample,nchannel,nmixture)).to(logits.device)                              
        #g = sample_gumbel((nb,num_sample,nchannel,1)).to(logits.device)                              
        
    else:
        g = sample_gumbel((nb,num_sample,nchannel,1)).to(logits.device)                             

    logits = logits[:,None,:,:]
    h = (g + logits)/temperature
    h_max = h.max(dim=-1, keepdim=True)[0]
    h = h - h_max
    y = F.softmax(h,dim=-1).clamp(min=eps,max=1 - eps)    
    return y



pi2 = 2*math.pi
eps=1e-4
pi2repuse=True
zitter=1e-4
num_fourierbasis = 10 #preivous work
#'sm','sinc'
kerneltype = 'sm'  
#kernel_type = 'sinc'  #'sm','sinc'

class Stkernel_basis(nn.Module):     
    def __init__(self,kerneltype = kerneltype,
                      num_mixture=5,
                      num_channel=3,
                      in_dims=1,
                      max_freq=5.,
                      likerr_scale = 1e-3,                 
                      num_fourierbasis = num_fourierbasis):
        
        super(Stkernel_basis, self).__init__()
        
        self.in_dims = in_dims
        self.likerr_scale = likerr_scale
        self.freq_bound =  np.array([eps,max_freq])
        self.num_mixture = num_mixture
        self.num_channel = num_channel        
        self.set_initparam()

        # randomprior function
        self.num_fourierbasis = num_fourierbasis 
        self.sample_w = None
        self.sample_b = None
        self.sample_weight = None           
        self.normal0 = Normal(loc=0.0,scale=1.0)
        self.uniform0 = Uniform(0,1)            
        #self.needsample=False
        self.cached_sample = False
        self.kerneltype = kerneltype
        
        return 
                
        
    def set_initparam(self,eps=eps):      
        if self.in_dims == 1:
            self._set_initparam_1d()
        if self.in_dims == 2:
            self._set_initparam_2d()        
            ## for 2d, we set num_mixture for each dim, total mixutre = mixture**2
            self.num_mixture = self.num_mixture**2            
        return 
        
    def _set_initparam_1d(self,eps=1e-6):        

        #min_freq,max_freq = self.freq_bound         
        min_freq,max_freq = eps, self.num_mixture+eps #more effective than previous        
        
        if self.num_mixture <= 1:
            #logmu = eps + centeredfreq.reshape(-1,1).repeat(1,self.in_dims) +  .01*torch.rand(self.num_mixture,self.in_dims)   
            #logmu[0] = 1e-6*torch.ones(self.in_dims)             
            logmu = eps*torch.rand(1,self.in_dims)      

            #scales = 0.25*(centeredfreq[1]-centeredfreq[0])
            #scale = 0.75*(centeredfreq[1]-centeredfreq[0])
            scale = 0.5

            logstd = eps + scale*torch.ones(1,self.in_dims)            
            self.logmu =    nn.Parameter( torch.log( logmu )) 
            self.logstd =   nn.Parameter( torch.log( logstd ))                 

            # bound params
            delta_mu = 0.25*(max_freq/self.num_mixture)
            delta_std = 0.1*scale        
            bound_mu = torch.cat([ (logmu - delta_mu)[...,None],(logmu + delta_mu)[...,None]],dim=-1)
            bound_mu = torch.clamp(bound_mu,min=eps)
            bound_std = torch.cat([ (logstd - delta_std)[...,None],(logstd + delta_std)[...,None]],dim=-1)
            bound_std = torch.clamp(bound_std,min=eps)        
            self.bound_logmu = torch.log(bound_mu+eps) 
            self.bound_logstd = torch.log(bound_std+eps)
            
        else:
            centeredfreq=torch.linspace(min_freq,max_freq,self.num_mixture)
            logmu = eps + centeredfreq.reshape(-1,1).repeat(1,self.in_dims) +  .01*torch.rand(self.num_mixture,self.in_dims)   
            logmu[0] = 1e-6*torch.ones(self.in_dims)             
            logmu = logmu.sort(dim=0)[0]        

            #scales = 0.25*(centeredfreq[1]-centeredfreq[0])
            #scale = 0.75*(centeredfreq[1]-centeredfreq[0])
            scale = 0.5*(centeredfreq[1]-centeredfreq[0])

            logstd = eps + scale*torch.ones(self.num_mixture,self.in_dims) +  .01*scale*torch.randn(self.num_mixture,self.in_dims)                  
            self.logmu =    nn.Parameter( torch.log( logmu )) 
            self.logstd =   nn.Parameter( torch.log( logstd ))                 

            # bound params
            delta_mu = 0.25*(max_freq/self.num_mixture)
            delta_std = 0.1*scale        
            bound_mu = torch.cat([ (logmu - delta_mu)[...,None],(logmu + delta_mu)[...,None]],dim=-1)
            bound_mu = torch.clamp(bound_mu,min=eps)
            bound_std = torch.cat([ (logstd - delta_std)[...,None],(logstd + delta_std)[...,None]],dim=-1)
            bound_std = torch.clamp(bound_std,min=eps)        
            self.bound_logmu = torch.log(bound_mu+eps) 
            self.bound_logstd = torch.log(bound_std+eps)


        loglikerr = eps + self.likerr_scale*torch.ones(self.num_channel)
        self.loglikerr = nn.Parameter(torch.log(loglikerr))                  
        # bound params        
        bound_likerr = torch.tensor([0.05,1.]) 
        self.bound_loglikerr = torch.log(bound_likerr + eps)                
        #lengthscale = max_freq*torch.ones(1,self.in_dims)
        #lengthscale = 100*torch.ones(1,self.in_dims)                
        lengthscale = 10*torch.ones(1,self.in_dims)        
        self.loglengthscale = nn.Parameter(torch.log(lengthscale + eps))     
        
        return 
    
   
    
    
    def _set_initparam_2d(self,eps=1e-6):        
        #min_freq,max_freq = self.freq_bound         
        min_freq,max_freq = eps, self.num_mixture+eps #more effective than previous        
        
        
        if self.num_mixture <= 1:
            logmu = eps*torch.ones(1,self.in_dims)
            logstd= 1*torch.ones(1,self.in_dims) 
        
            delta_freq = 1.
            #scale = 0.75*delta_freq                 
            scale = 0.5*delta_freq                 
            
            self.logmu =    nn.Parameter( torch.log( logmu )) 
            self.logstd =   nn.Parameter( torch.log( logstd ))                 
            
            
        else:
            centeredfreq=torch.linspace(min_freq,max_freq,self.num_mixture)            
            centeredfreq[0]=eps*torch.ones(1) 
            logmu = eps + torch.cartesian_prod(centeredfreq,centeredfreq) + .1*torch.rand(self.num_mixture**2,self.in_dims)
            logmu[0] = eps*torch.ones(self.in_dims)

            delta_freq = (centeredfreq[1]-centeredfreq[0])
            #scale = 0.25*delta_freq             
            #scale = 0.75*delta_freq     
            scale = 0.5*delta_freq     
            
            centeredfreq2=torch.ones(self.num_mixture)
            logstd = eps + scale*torch.cartesian_prod(centeredfreq2,centeredfreq2) + .1*scale*torch.rand(self.num_mixture**2,self.in_dims)    
            self.logmu =    nn.Parameter( torch.log( logmu )) 
            self.logstd =   nn.Parameter( torch.log( logstd ))                 

            
        delta_mu = 0.1*delta_freq*torch.ones(1,self.in_dims)   
        delta_std = 0.1*scale*torch.ones(1,self.in_dims)                    
        bound_mu = torch.cat([ (logmu - delta_mu)[...,None],(logmu + delta_mu)[...,None]],dim=-1)
        bound_mu = torch.clamp(bound_mu,min=eps)
        bound_std = torch.cat([ (logstd - delta_std)[...,None],(logstd + delta_std)[...,None]],dim=-1)
        bound_std = torch.clamp(bound_std,min=eps)        
        self.bound_logmu = torch.log(bound_mu+eps) 
        self.bound_logstd = torch.log(bound_std+eps)


        loglikerr = eps + self.likerr_scale*torch.ones(self.num_channel)
        self.loglikerr= nn.Parameter(torch.log(loglikerr))                
        bound_likerr = torch.tensor([0.05,1.]) 
        self.bound_loglikerr = torch.log(bound_likerr + eps)                


        lengthscale = max_freq*torch.ones(1,self.in_dims)
        self.loglengthscale = nn.Parameter(torch.log(lengthscale + eps))     
        
        return 

    
    
    
    def param_transform(self):
        #mu,inv_std,likerr = self.logmu.exp(),1/(self.logstd.exp()+eps),self.loglikerr.exp()
        #return mu,inv_std,likerr
        mu,std,likerr = self.logmu.exp(),self.logstd.exp(),self.loglikerr.exp()
        return mu,std,likerr
    
    
    
    def build_Kxt(self,xc,xt=None):
        if self.kerneltype == 'sm':
            return self.build_sm_Kxt(xc,xt)
        if self.kerneltype  == 'sinc':
            return self.build_sinc_Kxt(xc,xt)

        
#     #def forward(self,xc,xt=None):
    def build_rbf_Kxt(self,xc,xt=None):                
        assert len(xc.shape) == 4                            
        if xt is None:
            xt = xc
            nb,ndata,ndim,nchannel = xc.size()         
            ndata2=ndata
        else:
            nb,ndata,ndim,nchannel = xc.size()                            
            _,ndata2,_,_ = xt.size()                            


        #(nb,nmixture,ndata,ndim,nchannel)
        xc_ = xc.unsqueeze(dim=1)
        xt_ = xt.unsqueeze(dim=1)

        #setup param for density
        length_scale = self.loglengthscale.exp()
        length_scale_ = length_scale[None,:,None,:,None].repeat(1,1,1,1,self.num_channel)
        
        #compute kernel
        exp_xc_, exp_xt_  = xc_*length_scale_, xt_*length_scale_
        exp_term_xc2_ = torch.pow(exp_xc_,2).sum(dim=-2)[:,:,:,None,:] 
        exp_term_xt2_ = torch.pow(exp_xt_,2).sum(dim=-2)[:,:,None,:,:]
        cross_term_ = torch.einsum('bmadk,bmcdk->bmack',exp_xc_,exp_xt_)    
        exp_term = exp_term_xc2_ + exp_term_xt2_ -2*cross_term_    

        # outs :  #(nb,mixture,ncontext,ntarget,nchannel)     
        #global pi2 = pi2.to(xc.device)        
        if pi2repuse : 
            outs = torch.exp(-0.5*(pi2**2)*exp_term ) 
        else :
            outs = torch.exp(-0.5*exp_term )
        return outs 

    
    
#     #def forward(self,xc,xt=None):
    def build_sm_Kxt(self,xc,xt=None):        
        """
        args
            xc : (nb,ncontext,ndim,nchannel)
            xt : (nb,ntarget,ndim,nchannel)
        return
            Kxt : (nb,nmixutures,nxc,nxt,nchannel)
        """

        assert len(xc.shape) == 4                            
        if xt is None:
            xt = xc
            nb,ndata,ndim,nchannel = xc.size()         
            ndata2=ndata
        else:
            nb,ndata,ndim,nchannel = xc.size()                            
            _,ndata2,_,_ = xt.size()                            


        #(nb,nmixture,ndata,ndim,nchannel)
        xc_ = xc.unsqueeze(dim=1)
        xt_ = xt.unsqueeze(dim=1)

        #setup param
        #mu,inv_std = self.logmu.exp(),1/(self.logstd.exp()+eps)
        mu,std,_ = self.param_transform()
        mu_=mu[None,:,None,:,None]
        std_= std[None,:,None,:,None]         
            
        #compute kernel
        exp_xc_, exp_xt_  = xc_*std_, xt_*std_
        cos_xc_ ,cos_xt_ = xc_*mu_, xt_*mu_

        exp_term_xc2_ = torch.pow(exp_xc_,2).sum(dim=-2)[:,:,:,None,:] 
        exp_term_xt2_ = torch.pow(exp_xt_,2).sum(dim=-2)[:,:,None,:,:]
        cross_term_ = torch.einsum('bmadk,bmcdk->bmack',exp_xc_,exp_xt_)    
        exp_term = exp_term_xc2_ + exp_term_xt2_ -2*cross_term_    
        cos_term = cos_xc_.sum(dim=-2)[:,:,:,None,:] - cos_xt_.sum(dim=-2)[:,:,None,:,:]


        # outs :  #(nb,mixture,ncontext,ntarget,nchannel)     
        #global pi2 = pi2.to(xc.device)        
        if pi2repuse : 
            outs = torch.exp(-0.5*(pi2**2)*exp_term )*torch.cos(pi2*cos_term) 
        else :
            outs = torch.exp(-0.5*exp_term )*torch.cos(cos_term)
        return outs 



    #def forward(self,xc,xt=None):
    def build_sinc_Kxt(self,xc,xt=None):        
        """
        args
            xc : (nb,ncontext,ndim,nchannel)
            xt : (nb,ntarget,ndim,nchannel)
        return
            Kxt : (nb,nmixutures,nxc,nxt,nchannel)
        """

        assert len(xc.shape) == 4                            
        if xt is None:
            xt = xc
            nb,ndata,ndim,nchannel = xc.size()         
            ndata2=ndata
        else:
            nb,ndata,ndim,nchannel = xc.size()                            
            _,ndata2,_,_ = xt.size()                            


        #setup param
        #mu,inv_std = self.logmu.exp(),1/(self.logstd.exp()+eps)
        mu,std,_ = self.param_transform()
 

        #(nb,nmixture,ndata,ndim,nchannel)
        xc_ = xc.unsqueeze(dim=1)
        xt_ = xt.unsqueeze(dim=1)             
        #print('xc_.shape {}, xt_.shape {}'.format(xc_.shape,xt_.shape))
        
        std_ = std[None,:,None,None,:,None] 
        #(nb,nmixture,ndata,ndata2,ndim,nchannel)
        delta_xc = (xc_[:,:,:,None,:,:] - xt_[:,:,None,:,:,:])*std_        
        #(nb,nmixture,ndata,ndata2,nchannel)
        sinc_term = torch.sinc(delta_xc).prod(dim=-2) 
        
        mu_=mu[None,:,None,:,None]
        cos_xc_ ,cos_xt_ = xc_*mu_, xt_*mu_
        cos_term = cos_xc_.sum(dim=-2)[:,:,:,None,:] - cos_xt_.sum(dim=-2)[:,:,None,:,:]

        # outs :  #(nb,mixture,ncontext,ntarget,nchannel)     
        #global pi2 = pi2.to(xc.device)        
        if pi2repuse : 
            outs = sinc_term*torch.cos(pi2*cos_term) 
        else :
            outs = sinc_term*torch.cos(cos_term)
        return outs 
    
    def build_Kxx(self,xc,addnoise=False,zitter=zitter):
        """
        xc:
            xc : (nb,ndata,ndim,nchannel)
        args:
            Kxx : (nb,nmixutre,ndata,ndata,nchannel)
        """
        nb,ndata,ndim,nchannel = xc.shape
        Kxx = self.build_Kxt(xc) #(nb,nmixutre,ndata,ndata,nchannel)
        
        eye = torch.eye(ndata).to(xc.device)
        if addnoise:
            zitter = zitter+(2*self.loglikerr).exp()            
        return Kxx + zitter*eye[None,None,:,:,None]
        
        #return Kxx

       
    def reflect_weight_Kxt(self,weight,Kxt):
        """
        args
            weights: (nb,nchannel,nmixtures)
            Kxt : (nb,nchannel,nmixutures,nxc,nxt)
        return
            Kxt : (nb,nchannel,nxc,nxt)        
        """
        if weight.dim() == 2:
            weight = weight[None,:,:]
        weight = weight[:,:,:,None,None]
        WKxt = (weight*Kxt).sum(dim=2)                            
        return WKxt  


    
    
    #def sample_w_b(self,xc,yc,nsamples,eps=1e-6):    
    def sample_w_b(self,xc,num_sample,eps=1e-6):    
       
        """
        return :
            self.w_mu : nparams
            sample_w  : (nb,nfourifbasis*num_sample,indim,nchannels)
            sample_b  : (nb,nfouribasis*num_sample,indim,nchannels) 
            sample_weight : (nb,nfouribasis*num_sample,indim,1) 
        """        
        nb = xc.size(0)        
        # (num_mixture,num_dim)
        #mu,inv_std = self.logmu.exp(),1/(self.logstd.exp()+eps)        
        mu,std = self.logmu.exp(), self.logstd.exp()        
        
        # impose depedency over channels                    

        if self.kerneltype == 'sm':        
            eps1 = self.normal0.sample((nb,num_sample*self.num_fourierbasis,self.num_mixture,self.in_dims)).to(mu.device)
        
        if self.kerneltype == 'sinc':        
            eps1 = self.uniform0.sample((nb,num_sample*self.num_fourierbasis,self.num_mixture,self.in_dims)).to(mu.device)
            eps1 = eps1 - 0.5
        
        
        eps2 = self.uniform0.sample((nb,num_sample*self.num_fourierbasis,self.num_mixture,1)).to(mu.device)           
        
        sample_w = mu[None,None,:,:] + std[None,None,:,:]*eps1    #(nb,nsamples*nfouierbasis,num_mixture,in_dims)  
        sample_w = torch.clamp(sample_w,min=eps) #set spectral poinst to be postivie
        sample_b = eps2                                                #(nb,nsamples*nfouierbasis,num_mixture,in_dims)

        sample_weight = self.normal0.sample((nb,num_sample*self.num_fourierbasis,self.num_mixture,1)).to(mu.device) 
        #reduce randomness over mixture        
        #sample_weight = self.normal0.sample((nb,num_sample*self.num_fourierbasis,1,self.num_channel)).to(mu.device) 
        
        #return sample_w,sample_b,random_w,logits_samples
        self.sample_w = sample_w
        self.sample_b = sample_b
        self.sample_weight = sample_weight  #(nb,ns*nf,nm,1)         
        self.cached_sample = True
        self.num_sample = num_sample
        return 

    

    def build_Phix(self,xc,num_sample=10,use_newsample=True):        
        """
        args 
            xc: (nb,ndata,ndim,nchannel)
        return 
            Psi: #(nb,ndata,num_sample*num_fourier,nmixture,nchannel) 
        """        
        nb,ndata,ndim,nchannel = xc.shape
        if use_newsample:
            self.sample_w_b(xc,num_sample,eps=1e-6)
        sample_w,sample_b,sample_weight = self.sample_w, self.sample_b, self.sample_weight
            
        w = sample_w.permute(0,1,3,2).contiguous()
        b = sample_b.permute(0,1,3,2).contiguous()
            
        xc = xc[:,None,:,:,:].repeat(1,w.size(1),1,1,1)
        w = w[...,None].repeat(1,1,1,1,nchannel)
        b = b[...,None].repeat(1,1,1,1,nchannel)

        #pi2 = pi2.to(xc.device)                
        if pi2repuse:            
            xcdotw_b = pi2*torch.einsum('bsndc,bsdmc->bsnmc',xc,w) + pi2*b
        else:
            xcdotw_b = torch.einsum('bsndc,bsdmc->bsnmc',xc,w) + pi2*b
        
        normalizer = np.sqrt(2/self.num_fourierbasis)
        Phi = normalizer*torch.cos(xcdotw_b) 
        Phi = Phi.permute(0,2,1,3,4).contiguous()                    
        return Phi



    
    #def sample_prior_independent(self,xc,numsamples=10,newsample=False):            
    def sample_randomfunction(self,xc,num_sample=10,use_newsample=True):           
        """
        args 
            xc: (nb,ndata,ndim,nchannel)
        return 
            random_priorfunction: #(nb,numsamples,ndata,nmixture,nchannel) 
        """
        nb,ndata,ndim,nchannel = xc.shape        
        Phi = self.build_Phix(xc,num_sample=num_sample,use_newsample=use_newsample)    #(nb,ndata,ns*nf,nmixture,nchannel)         
        Phi = Phi.permute(0,2,1,3,4)                     #(nb,ns*nf,ndata,nmixture,nchannel)   
                                  
        sample_weight = self.sample_weight #(nb,ns*nf,nm,1)        
        sum_costerm = Phi*sample_weight[:,:,None,:,:]
        sum_costerm = sum_costerm.reshape(nb,num_sample,-1,ndata,self.num_mixture,nchannel)  #(nb,ns,nf,ndata,nmixture,nchannel)  
        random_function = (sum_costerm.sum(dim=2))  #(nb,ns,ndata,nmixture,nchannel)        
        return random_function

    
    
    #def sample_randomfunction_weight(self,weight,xc,num_sample=10):    
    def reflect_weightsample_randomfunction(self,mixweight_sample,random_function):            
        """
        args :
            mixweight_sample: (nb,nsample,nchannel,nmixture)
            # mixweight_sample torch.Size([16, 10, 1, 4])
            random_function :  (nb,nsample,ndata,mixture,nchannel)
            # priormix_xc torch.Size([16, 10, 24, 4, 1])
        return : 
            random_funciton_weight: (nb,nsample,ndata,nchannel)            
        """        
#         random_function = random_function.permute(0,1,2,4,3).contiguous()        
#         mixweight_sample = torch.nan_to_num(mixweight_sample.sqrt())
#         random_function_weight = (random_function*mixweight_sample[:,:,None,:,:]).sum(dim=-2) #(nb,numsamples,ndata,nchannel)

        #random_function = random_function.permute(0,1,2,4,3).contiguous()        
        mixweight_sample = torch.nan_to_num(mixweight_sample.sqrt())
        mixweight_sample = mixweight_sample.permute(0,1,3,2).contiguous()        
        random_function_weight = (random_function*mixweight_sample[:,:,None,:,:]).sum(dim=-2) #(nb,numsamples,ndata,nchannel)
        
        return random_function_weight         
    

    
    
    
    def extra_repr(self) -> str:
#         mu,inv_std,likerr = self.param_transform()
#         outs = 'mixture={}, outchannel={}, mu={}, invstd={}, likerrstd={}, freqbound={}'.format(self.num_mixture,self.num_channel,mu,inv_std,likerr,self.freq_bound)
        mu,std,likerr = self.param_transform()
        outs = 'mixture={}, outchannel={},\n mu={},\n std={},\n likerrstd={},\n lenthscale={},\n boundmu={}, \n boundstd={}'.format(self.num_mixture,self.num_channel,mu,std,likerr,self.loglengthscale.exp(),self.bound_logmu.exp(),self.bound_logstd.exp())


        return outs 



from models.utils import to_multiple
from attrdict import AttrDict

eps=1e-4
class Effcient_GPsampler(nn.Module):
    def __init__(self,kerenl,
                      use_weightnet = True, 
                      hdim=10,
                      tempering=1e-1,
                      points_per_unit=None,
                      multiplier=None,
                ):
                          
        print('current gp sampler v3')            
        super(Effcient_GPsampler,self).__init__()
        self.kernel = kerenl
        self.num_channel = self.kernel.num_channel
        self.num_mixture = self.kernel.num_mixture

        # cnn
        self.point_per_unit = points_per_unit
        self.multiplier = multiplier
        
        # use priorassign net
        self.hdim = hdim         
        self.use_weightnet = use_weightnet
        if self.use_weightnet:

#             self.weight_net =  transinv_mlp_1d(in_dim=1,
#                                                num_channel=self.num_channel,
#                                                num_mixture=self.num_mixture,
#                                                hdim=self.hdim
#                                               )
                
            
#             self.weight_net =  transinv_cnn_1d(in_dim=1,
#                                                num_channel=self.num_channel,
#                                                num_mixture=self.num_mixture,
#                                                hdim=self.hdim
#                                               )

            self.weight_net =  transinv_cnn_1d_simple(in_dim=1,
                                               num_channel=self.num_channel,
                                               num_mixture=self.num_mixture,
                                               hdim=self.hdim
                                              )

        else:            
            self.weight_net = None
        
        self.tempering = tempering
        self.mixweight = None
        self.mixweight_sample = None        
        
        self.solvemode = None
        #self.cgsolver = LinearCG_Solver.apply
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
    
    
    
    def build_xgrid(self,xc,yc,xt,x_thres=1.0):
        nb,_,ndim,nchannel=xc.size()         
        x_min = min(torch.min(xc).cpu().numpy(),torch.min(xt).cpu().numpy()) - x_thres
        x_max = max(torch.max(xc).cpu().numpy(),torch.max(xt).cpu().numpy()) + x_thres
        num_points = int(to_multiple(self.point_per_unit * (x_max - x_min),self.multiplier))                     
        xgrid = torch.linspace(x_min,x_max, num_points)
        #xgrid = xgrid.reshape(1,num_points,1).repeat(nb,1,ndim).to(xc.device)
        xgrid = xgrid.reshape(1,num_points,1,1).repeat(nb,1,ndim,nchannel).to(xc.device)        
        return xgrid    
    



    def build_mixweight(self,datarepr,num_sample=10,eps=1e-6):
        #Kgc = Kgc.permute(0,2,3,4,1)
        """
        args
            datarepr : (nb,ngrid,nchannel)            
        """
        #Kgcmix.shape torch.Size([16, 1, 4, 384, 24])
        #print('self.tempering {}'.format(self.tempering))
        #logit = self.weight_net(Kgc.detach().clone(),yc) 
        #logit = self.weight_net(datarepr) 
        logit = self.weight_net(datarepr.detach().clone()) 
        
        
        logit_max = logit.max(dim=-1, keepdim=True)[0]
        logit = (logit - logit_max) / self.tempering
        mixweight = F.softmax(logit, dim=-1)         
        #mixweight = F.softmax(logit/10., dim=-1)         
        mixweight_sample = sample_gumbel_softmax(logit,temperature=1e-1,num_sample=num_sample)        
        
        self.mixweight = mixweight
        self.mixweight_sample = mixweight_sample
        self.num_sample = num_sample
        return mixweight,mixweight_sample

    
    
    def build_deterministic_repr(self,xc,yc,xgrid,eps=1e-4):
        # ---------------------------
        # density
        # ---------------------------        
        Kgcrbf =  self.kernel.build_rbf_Kxt(xgrid,xc) #(nb,nmixutre,ngrid,ndata,nchannel)
        Kgcrbf = Kgcrbf.permute(0,4,1,2,3).contiguous()  #(nb,nchannel,nmixutre,ngrid,ndata)
        density = Kgcrbf[:,:,0,:,:].sum(dim=-1).permute(0,2,1).contiguous()  #(nb,nchannel,ngrid,ndata) -> #(nb,ngrid,nchannel)
        
        yc_ = yc.permute(0,2,1).contiguous() #(nb,nchannel,ndata)
    
        datarepr = (Kgcrbf[:,:,0,:,:]*yc_[:,:,None,:]).sum(dim=-1)
        datarepr = datarepr.permute(0,2,1).contiguous() #(nb,ngrid,nchannel)
        datarepr = datarepr/(density+eps) #(nb,ngrid,nchannel)
                
        return density,datarepr 

        
        
    def sample_prior(self,xc,yc,xt,num_sample=10):        
        xgrid = self.build_xgrid(xc,yc,xt)
        self.xgrid = xgrid
        priormix_grid = self.kernel.sample_randomfunction(xgrid, num_sample=num_sample, use_newsample=True)
        priormix_target = self.kernel.sample_randomfunction(xt, num_sample=num_sample, use_newsample=False)        
        #return prior_grid,xgrid,prior_target,xt
        return priormix_grid, xgrid, priormix_target,xt

    
    
    
    def build_update_function(self,xc,yc,xt,xgrid,priormix_grid,num_sample=10):
        
        priormix_xc = self.kernel.sample_randomfunction(xc,num_sample=num_sample, use_newsample=False)
        #priormix_xt = self.kernel.sample_randomfunction(xt,num_sample=num_sample, use_newsample=False)
        
        likerr = self.kernel.loglikerr.exp()
        noise2 = torch.randn_like(priormix_xc).to(xc.device)*likerr[None,None,None,None,:]
        delta_ycmix = yc[:,None,:,None,:] - (priormix_xc + noise2)     #(nb,ns,ndata,nmixture,nchannel)   
  
        #prepare gram matrix for grid and target
        # Kxx: (nb,ndata,ndata,nchannel), Kxxmix : (nb,nmixture,ndata,ndata,nchannel)
        Kccmix = self.kernel.build_Kxx(xc, addnoise=True) 
        Kccmix = Kccmix.permute(0,4,1,2,3).contiguous()
        #self.Kccmix = Kccmix
        
        Kgcmix = self.kernel.build_Kxt(xgrid, xc)     
        Kgcmix = Kgcmix.permute(0,4,1,2,3).contiguous()
        self.Kgcmix = Kgcmix
                
        Ktcmix = self.kernel.build_Kxt(xt,xc)        
        Ktcmix = Ktcmix.permute(0,4,1,2,3).contiguous()
        #print('Kccmix.shape {}, Kgcmix.shape {}'.format(Kccmix.shape,Kgcmix.shape))
            
        # compute density and allociatoin prob
        density,datarepr = self.build_deterministic_repr(xc,yc,xgrid) #(nb,ngrid,nchannel)
        #mixweight, mixweight_sample = self.build_mixweight(Kgcmix,yc,num_sample=num_sample)
        mixweight, mixweight_sample = self.build_mixweight(datarepr,num_sample=num_sample)
        
        #print('mixweight.shape {}, mixweight_sample {}, priormix_xc {}'.format(mixweight.shape,mixweight_sample.shape,priormix_xc.shape))

        # -----------------------------
        # reflect weight
        # -----------------------------        
        prior_xc = self.kernel.reflect_weightsample_randomfunction(mixweight_sample,priormix_xc)                    
        noise1 = torch.randn_like(prior_xc).to(xc.device)*likerr[None,None,None,:]
        delta_yc = yc[:,None,:,:] - (prior_xc + noise1)  
                                        
            
        Kcc = self.kernel.reflect_weight_Kxt(mixweight,Kccmix) #(nb,nchannel,ndata,ndata)
        Kgc = self.kernel.reflect_weight_Kxt(mixweight,Kgcmix)
                
        
        
        #solve linear system
        if self.solvemode == 'chol':
            Lcc = torch.linalg.cholesky(Kcc)
            Kccinvyc = torch.cholesky_solve(delta_yc.permute(0,3,2,1).contiguous(),Lcc,upper=False)  #(nb,nchannel,ndata,nsample)            
            Lccmix = torch.linalg.cholesky(Kccmix) 
            Kccmixinvyc = torch.cholesky_solve(delta_ycmix.permute(0,4,3,2,1).contiguous(),Lccmix,upper=False)  #(nb,nchannel,nmix,ndata,nsample)
            
        
        elif self.solvemode == 'linsolve':
            Kccinvyc = torch.linalg.solve(Kcc,delta_yc.permute(0,3,2,1).contiguous())
            Kccmixinvyc = torch.linalg.solve(Kccmix,delta_ycmix.permute(0,4,3,2,1).contiguous())        
        
                
        elif self.solvemode == 'cg':            
            Kccinvyc = self.cgsolver(0.5*(Kcc + Kcc.permute(0,1,3,2)),delta_yc.permute(0,3,2,1).contiguous())
            Kccmixinvyc = self.cgsolver(0.5*(Kccmix + Kccmix.permute(0,1,2,4,3)),delta_ycmix.permute(0,4,3,2,1).contiguous())
            
        else:
            pass
        
            
        # prepare update term
        update_grid = torch.einsum('bsxy,bsyz->bsxz',Kgc,Kccinvyc ).permute(0,3,2,1).contiguous()   #(nb,nsample,ndata,nchannel)
        #updatemixterm_grid = torch.einsum('bcmxy,bcmyz->bcmxz',Kgcmix,Kccmixinvyc ).permute(0,4,3,2,1)  #(nb,nsample,ndata,nmix,nchannel)
        
        #updateterm_target = torch.einsum('bsxy,bsyz->bsxz',Ktc,Kccinvyc ).permute(0,3,2,1)   #(nb,nsample,ndata,nchannel)
        updatemix_target = torch.einsum('bcmxy,bcmyz->bcmxz',Ktcmix,Kccmixinvyc ).permute(0,4,3,2,1).contiguous()   #(nb,nsample,ndata,nchannel)        
        
        
        return update_grid, updatemix_target , density,datarepr

        

    
    
    def sample_posterior(self,xc,yc,xt,num_sample=10):
        #prior_gridpriormix_grid,xgrid, prior_target,priormix_target,xt = self.sample_prior(xc,yc,xt,num_sample=num_sample)
        priormix_grid, xgrid, priormix_target, xt = self.sample_prior(xc,yc,xt,num_sample=num_sample)                           
            
        #update_grid,_,updatemix_target,_,Kccinvyc = self.build_update_function(xc,yc,xt,xgrid,num_sample=num_sample)
        update_grid, updatemix_target, density,datarepr = self.build_update_function(xc,yc,xt,xgrid,priormix_grid, num_sample=num_sample)
        
        #reflect weight
        prior_grid = self.kernel.reflect_weightsample_randomfunction(self.mixweight_sample,priormix_grid)            

        #sample posterior function on grid, and cat deterministic inputs
        post_grid = prior_grid + update_grid
        post_grid = torch.cat([datarepr[:,None,:,:],post_grid],dim=1) #(nb,nsample+1,ngrid,nchannel)
        
        #sample posterior function on target and distribution        
        postmix_target = priormix_target + updatemix_target   #(nb,nsample,nmixture,ngrid,nchannel)
        postmixmu_target,postmixstd_target = postmix_target.mean(dim=1), postmix_target.std(dim=1)  #(nb,nmixture,ngrid,nchannel)
                
        
        
        outs = AttrDict()
        outs.xgrid = xgrid
        outs.density = density.abs()
        outs.datarepr = datarepr
        
        
        outs.prior_grid = prior_grid        
        outs.priormix_grid = priormix_grid        
        
        outs.post_grid = post_grid
        outs.postmix_target = postmix_target

        outs.post_empdist = (postmixmu_target.detach().clone(),postmixstd_target.detach().clone())     
        outs.mixweight = self.mixweight
        outs.mixweight_sample = self.mixweight_sample #(nb,nsample,nchannel,nmixture)
        
        #outs.Kccmix = self.Kccmix
        #outs.Kgcmix = self.Kgcmix
        #outs.Ktg = Ktg
        
        return outs 
    
    
    
    
    
    
def get_gpsampler_1d(kerneltype : str = 'sm',
                     num_mixture: int = 1,
                     num_channel: int = 1,
                     likerr_scale: float = 1e-2,
                     max_freq : float = 5.,                       
                     use_weightnet :bool = True,
                     hdim: int = 20,
                     tempering: float = 1e-1,                     
                     points_per_unit : int = 64,
                     multiplier: int = 2**8,                   
                     ):
    
    kernel = Stkernel_basis(kerneltype = kerneltype,
                            num_mixture=num_mixture,
                            num_channel=num_channel,
                            in_dims=1,
                            likerr_scale=likerr_scale,
                            max_freq=max_freq)

    gpsampler = Effcient_GPsampler(kernel,
                              use_weightnet = use_weightnet, 
                              hdim=hdim,
                              tempering=tempering,
                              points_per_unit=points_per_unit,
                              multiplier=multiplier)    

    gpsampler.solvemode = 'chol'
    #gpsampler.solvemode = 'linsolve'
    #gpsampler.solvemode = 'cg-autodiff'    
    return gpsampler
    