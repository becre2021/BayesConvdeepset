from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import itertools as it


from .utils import to_multiple
from attrdict import AttrDict


__all__= ['Spikeslab_GPsampler']



pi2repuse = True
class transinvariant_mlp(nn.Module):
    def __init__(self,in_dims=1,num_channels=1,hdims=10,num_mixtures=1,eps=1e-6):
        super(transinvariant_mlp,self).__init__()
        
        self.in_dims = in_dims
        self.hdims = hdims
        self.num_channels = num_channels
        self.num_mixtures = num_mixtures

        self.fc1 = nn.Linear(num_mixtures+1,hdims)    
        self.fc2 = nn.Linear(hdims*num_channels,hdims)    
        self.fc3 = nn.Linear(hdims,hdims)    
        self.fc4 = nn.Linear(hdims,hdims)            
        self.fc5 = nn.Linear(hdims,num_mixtures*num_channels)    

        
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)    
        nn.init.xavier_uniform_(self.fc4.weight)    
        nn.init.xavier_uniform_(self.fc5.weight)    


    
    def forward(self,xc,yc,param_list=[]):
        """
        args:
            xc: (nb,ndata,ndim,nchannel)
            yc: (nb,ndata,nchannel)            
        return :
            loglogits: (nb,nmixuture)

        """
        
        nb,ndata,ndim,nchannel=xc.shape        
        
        #(nb,ndata,ndata,nchannel,nmixture)
        Kcc = eval_smkernel_batch(param_list,xc=xc,xt=None,likerr_bound=None)
        feature_xc = (Kcc*yc[:,None,:,:,None]).sum(dim=2)   #(nb,ndata,nchannel,nmixture)        
        transinv_feature = torch.cat([feature_xc,yc.unsqueeze(dim=-1)],dim=-1) #(nb,ndata,nchannel,nmixture+1)                
        h = F.relu(self.fc1(transinv_feature))  #(nb,ndata,nchannel,nmixture+1) 
        h = (h.mean(dim=1)).reshape(nb,-1)    #(nb,num_mixtures*num_channels)                         
        h = F.relu(self.fc2(h)) 
        h = F.relu(self.fc3(h)) 
        h = F.relu(self.fc4(h))         
        loglogits = self.fc5(h)         
        loglogits = loglogits.reshape(nb,self.num_channels,-1)      
        
        #self.param_list = param_list
        return loglogits
        
        
    def compute_feature(self,xc,yc,param_list=[]):
        """
        args:
            xc: (nb,ndata,ndim,nchannel)
            yc: (nb,ndata,nchannel)            
        return :
            loglogits: (nb,nmixuture)

        """
        
        nb,ndata,ndim,nchannel=xc.shape        
        
        #(nb,ndata,ndata,nchannel,nmixture)
        
        Kcc = eval_smkernel_batch(param_list,xc=xc,xt=None,likerr_bound=None)
        feature_xc = (Kcc*yc[:,None,:,:,None]).sum(dim=2)   #(nb,ndata,nchannel,nmixture)        
        transinv_feature = torch.cat([feature_xc,yc.unsqueeze(dim=-1)],dim=-1) #(nb,ndata,nchannel,nmixture+1)                
        h = F.relu(self.fc1(transinv_feature))  #(nb,ndata,nchannel,nmixture+1) 
        h = (h.mean(dim=1)).reshape(nb,-1)    #(nb,num_mixtures*num_channels)                         
        return h,feature_xc
        
    
    
    
    
    
def sample_gumbel(samples_shape, eps=1e-20):
    #shape = logits.shape
    unif = torch.rand(samples_shape)
    g = -torch.log(-torch.log(unif + eps))
    return g.float()

def sample_gumbel_softmax(logits, nb=1, nsamples=1, temperature=1.0,training=True):    
    """
        Input:
        logits: Tensor of "log" probs, shape = (nb,nch,nmix)
        temperature = scalar        
        Output: Tensor of values sampled from Gumbel softmax.
                These will tend towards a one-hot representation in the limit of temp -> 0
                shape = BS x k
    """
        
    if logits.dim() == 3:
        nb,nchannel,nmixture = logits.shape                
        if training :
            g = sample_gumbel((nb,nsamples,nchannel,nmixture)).to(logits.device)                               
        else:
            g = sample_gumbel((nb,nsamples,nchannel,1)).to(logits.device)                             
            
        logits = logits[:,None,:,:]


    h = (g + logits)/temperature
    h_max = h.max(dim=-1, keepdim=True)[0]
    #h_max = h.max()
    h = h - h_max
    cache = torch.exp(h)
    y = cache / cache.sum(dim=-1, keepdim=True)
    return y




    
    
pi2 = 2*math.pi
eps=1e-6
pi2repuse=True
    
def eval_smkernel_batch(param_list,xc,xt=None, likerr_bound=[.1,1.0], zitter=1e-4):

    """
    inputs:
        #xc : (nb,ncontext,ndim,nchannel)
        xc : (nb,ncontext,ndim)
        
    outputs:
        Kxz : (nb,ncontext,ncontext,nchannel)      # assume y-value is 1-d      
    """
    
    assert len(xc.shape) == 4                            
    if xt is None:
        xt = xc
        nb,ndata,ndim,nchannel = xc.size()         
        ndata2=ndata
    else:
        nb,ndata,ndim,nchannel = xc.size()                            
        _,ndata2,_,_ = xt.size()                            

    xc_ = xc.unsqueeze(dim=1)
    xt_ = xt.unsqueeze(dim=1)
              
    assert len(param_list) == 4    
    
    #(nmixutre,ndim),(nmixutre,ndim),(nmixutre),(1)
    mu,inv_std,logits,likerr = param_list         
    mu_=mu[None,:,None,:,None]
    inv_std_=inv_std[None,:,None,:,None] 
    
    #(nb,nmixture,ndata,ndim,nchannel)
    xc_ = xc.unsqueeze(dim=1)
    xt_ = xt.unsqueeze(dim=1)

    exp_xc_ = xc_*inv_std_
    exp_xt_ = xt_*inv_std_
    cos_xc_ = xc_*mu_
    cos_xt_ = xt_*mu_

    exp_term_xc2_ = torch.pow(exp_xc_,2).sum(dim=-2)[:,:,:,None,:] 
    exp_term_xt2_ = torch.pow(exp_xt_,2).sum(dim=-2)[:,:,None,:,:]
    cross_term_ = torch.einsum('bmadk,bmcdk->bmack',exp_xc_,exp_xt_)    
    exp_term = exp_term_xc2_ + exp_term_xt2_ -2*cross_term_    
    cos_term = cos_xc_.sum(dim=-2)[:,:,:,None,:] - cos_xt_.sum(dim=-2)[:,:,None,:,:]

    
    # outs :  #(nb,mixture,ncontext,ntarget,nchannel)     
    outs = torch.exp(-0.5*(pi2**2)*exp_term )*torch.cos(pi2*cos_term) if pi2repuse else torch.exp(-0.5*exp_term )*torch.cos(cos_term)
        

        
    if logits is None:
        outs =  outs.permute(0,2,3,4,1)        
        if ndata == ndata2: 
            noise_eye = (zitter)*(torch.eye(ndata)[None,:,:,None,None]).to(xc.device)
            outs = outs + noise_eye     
            return outs #(nb,ncontext,ncontext,nchannel,mixture)  
        else:
            return outs #(nb,ncontext,ntarget,nchannel,mixture)        
            
    else:        
        if logits.dim() == 2:
            #(nchannle,nmixture)         
            logits_ =logits.permute(1,0)[None,:,None,None,:]
            weighted_outs = (outs*logits_).sum(dim=1)                    

        if logits.dim() == 3:
            #(nb,nchannle,nmixture) --> (nb,nmixture,1,1,nchannle)                     
            logits_ = logits.permute(0,2,1)[:,:,None,None,:]
            #(nb,ncontext,ntarget,nchannel)                                     
            weighted_outs = (outs*logits_).sum(dim=1)                
            
        if ndata == ndata2: 
            likerr_ = likerr[None,None,None,:]
            likerr_ = torch.clamp(likerr_,min=likerr_bound[0],max=likerr_bound[1])                
            noise_eye = (zitter+likerr_**2)*(torch.eye(ndata)[None,:,:,None]).to(xc.device)
            weighted_outs = weighted_outs + noise_eye     
                        
            likerr__ = likerr[None,None,None,None,:]
            likerr__ = torch.clamp(likerr__,min=likerr_bound[0],max=likerr_bound[1])                            
            noise_eye2 = (zitter+likerr__**2)*(torch.eye(ndata)[None,None,:,:,None]).to(xc.device)            
            outs = outs + noise_eye2     
            
            return weighted_outs,outs

        else:
            #(nb,ncontext,ntarget,nchannel), (nb,ncontext,ntarget,nchannel,mixture)              
            return weighted_outs,outs
        



        
#class NeuralSpikeslab_GPsampler(nn.Module):        
class Spikeslab_GPsampler(nn.Module):        
   
    def __init__(self,in_dims=1,
                      out_dims=1,
                      num_channels=3, 
                      num_fourierbasis = 10,num_sampleposterior=10 ,
                      scales=.5,
                      loglik_err=1e-2,
                      eps=1e-6,
                      points_per_unit=64,
                      multiplier=2**3,
                      useweightnet = True,hdims=10 ):
        
        #super(NeuralSpikeslab_GPsampler, self).__init__()
        super(Spikeslab_GPsampler, self).__init__()
        
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.num_channels = num_channels
        self.num_fourierbasis = num_fourierbasis
        
        self.normal0 = Normal(loc=0.0,scale=1.0)
        self.uniform0 = Uniform(0,1)
            
        self.points_per_unit = points_per_unit  
        self.multiplier = multiplier        
        self.regloss = 0.0
        
        
        
        
        self.useweightnet = useweightnet       
        
        
        if num_channels==1:
            num_mixtures=3                        
            self.weight_net = transinvariant_mlp(in_dims=in_dims,
                                                hdims=hdims,
                                                num_channels=num_channels,
                                                num_mixtures=num_mixtures) 


        if num_channels==2:
            num_mixtures=4                        
            self.weight_net = transinvariant_mlp(in_dims=in_dims,
                                                hdims=hdims,
                                                num_channels=num_channels,
                                                num_mixtures=num_mixtures) 

            
        if num_channels == 3:
            #------------------------------------------------------
            # this setting obatins the improved results for multichannel experiments.
            # this reported results are obatinedb by following settings.
            #------------------------------------------------------
            num_mixtures=5
            self.weight_net = transinvariant_mlp(in_dims=in_dims,
                                                hdims=hdims,
                                                num_channels=num_channels,
                                                num_mixtures=num_mixtures)
            
            
            #------------------------------------------------------
            # for illustration figure
            #------------------------------------------------------            
#             num_mixtures=2
#             self.weight_net = transinvariant_mlp(in_dims=in_dims,
#                                                 hdims=hdims,
#                                                 num_channels=num_channels,
#                                                 num_mixtures=num_mixtures)
            
        self.num_mixtures = num_mixtures
        self.set_initparams(scales=scales,loglik_err = loglik_err,eps =eps)
        
          
            
        
        self.w=None
        self.b=None        
        self.normalizer=None
        self.random_w=None        
        #self.prior_scale = 0.5        
        self.prior_scale = 1.0              
        
        self.use_constrainedprior = False #True means density aware prior 
        self.tempering0 = 1e-1
        
        print('spikeslab version 7 with tempering {}'.format(self.tempering0))
        self.param_list = None
        
        return 

    

    def bound_hypparams(self,eps=1e-6):
        """
        bound_std = [1.,2.] --> bound_invstd = [.5,1.] 
        """        
        bound_logmu = np.log(self.bound_mu+eps)
        bound_logstd = np.log(self.bound_std+eps)        
        #print(self.bound_std )
        
        with torch.no_grad():
            self.logmu.data.clip_(bound_logmu[0],bound_logmu[1])            
            self.logstd.data.clip_(bound_logstd[0],bound_logstd[1])            
        return

    
    
    
    def set_initparams(self,scales=1.0,loglik_err=1e-2,eps=1e-6):
        loglogits = eps  +  1.*torch.ones(self.num_channels,self.num_mixtures) + .1*torch.rand(self.num_channels,self.num_mixtures)

        if self.num_channels == 1:                    
            maxfreq = 5
            centeredfreq=torch.linspace(0,maxfreq,self.num_mixtures)
            logmu = eps + centeredfreq.reshape(-1,1).repeat(1,self.in_dims) +  .1*torch.rand(self.num_mixtures,self.in_dims)   
            logmu[0] = eps*torch.ones(self.in_dims)             
            logmu = logmu.sort(dim=0)[0]        
            
            logstd = eps + scales*torch.ones(self.num_mixtures,self.in_dims)        
            
            self.bound_mu = np.array([eps,maxfreq])
            self.bound_std = np.array([1,5])

            
        if self.num_channels == 2:                    
            maxfreq = 5
            centeredfreq=torch.linspace(0,maxfreq,self.num_mixtures)
            logmu = eps + centeredfreq.reshape(-1,1).repeat(1,self.in_dims) +  .1*torch.rand(self.num_mixtures,self.in_dims)   
            logmu[0] = eps*torch.ones(self.in_dims)             
            logmu = logmu.sort(dim=0)[0]        
            
            logstd = eps + scales*torch.ones(self.num_mixtures,self.in_dims)        
            self.bound_mu = np.array([eps,maxfreq])
            self.bound_std = np.array([1,5])
            

        #-----------------------------
        #settings for reported results
        #-----------------------------                
        if self.num_channels == 3:        
            maxfreq = 5
            centeredfreq=torch.linspace(0,maxfreq,self.num_mixtures)                        
            logmu = eps + centeredfreq.reshape(-1,1).repeat(1,self.in_dims) +  .1*torch.rand(self.num_mixtures,self.in_dims)              
            logmu[0] = eps*torch.ones(self.in_dims) 
            logmu = logmu.sort(dim=0)[0]                                
            
            logstd = eps + scales*torch.ones(self.num_mixtures,self.in_dims)        
        
            self.bound_mu = np.array([eps,maxfreq])
            self.bound_std = np.array([1,5])


        loglik = eps + loglik_err*torch.ones(self.num_channels)
        self.loglogits =    nn.Parameter( torch.log( loglogits ) )  #much powerful                                                        
        self.logmu =    nn.Parameter( torch.log( logmu )) 
        self.logstd =   nn.Parameter( torch.log( logstd ))                   
        self.loglik =   nn.Parameter( torch.log( loglik ))
        self.loglik_bound = [0.1*self.loglik.exp().min().item(),10*self.loglik.exp().max().item()]
        return 
    
    
    
    
    def build_xgrid(self,xc,xt,x_thres=1.0):
        nb,_,ndim,nchannel=xc.size()         
        x_min = min(torch.min(xc).cpu().numpy(),torch.min(xt).cpu().numpy()) - x_thres
        x_max = max(torch.max(xc).cpu().numpy(),torch.max(xt).cpu().numpy()) + x_thres
        num_points = int(to_multiple(self.points_per_unit * (x_max - x_min),self.multiplier))                     
        xgrid = torch.linspace(x_min,x_max, num_points)
        xgrid = xgrid.reshape(1,num_points,1).repeat(nb,1,ndim).to(xc.device)
        return xgrid
        
    

    def sample_w_b(self,xc,yc,nsamples,eps=1e-6):    
       
        """
        self.w_mu : nparams
        sample_w  : (nb,nfourifbasis,indim,nchannels)
        sample_b  : (nb,nfouribasis,indim,nchannels)        
        """        
        nb = xc.size(0)
        
        # (num_mixtures,num_dim)
        mu,inv_std = self.logmu.exp(),1/(self.logstd.exp()+eps)
        
        eps1 = self.normal0.sample((nb,nsamples*self.num_fourierbasis,self.num_mixtures,self.in_dims)).to(mu.device)
        eps2 = self.uniform0.sample((nb,nsamples*self.num_fourierbasis,self.num_mixtures,1)).to(mu.device)           
        
        
        random_w = self.normal0.sample((nb,nsamples*self.num_fourierbasis,self.num_mixtures,1)).to(mu.device) #impose depedency over channels
        sample_w = mu[None,None,:,:] + inv_std[None,None,:,:]*eps1    #(nb,nsamples*nfouierbasis,num_mixtures,in_dims)                
        sample_b = eps2                                                #(nb,nsamples*nfouierbasis,num_mixtures,in_dims)

        self.w = sample_w
        self.b = sample_b
        self.random_w = random_w           
        return sample_w,sample_b,random_w

    
    
    def sample_logits(self,xc,yc,nsamples,eps=1e-6,tempering0=1e1):        
        nb = xc.size(0)

        if self.useweightnet:
            mu,inv_std = self.logmu.exp(),1/(self.logstd.exp()+eps)
            
            #dictionary kernel are used for computing Kcc().sum(dim=2) without learnable kernel parameter learning
            param_list = (mu.detach().clone(),inv_std.detach().clone(),None,None)
            loglogits = self.weight_net(xc,yc,param_list=param_list)
            
            if self.tempering0 is None:
                self.tempering0 = tempering0
            self.neural_loglogits=loglogits / self.tempering0
            logits = F.softmax(self.neural_loglogits  , dim=-1)    #(nb,nchannle,nmixture)   
            self.neural_logits= logits 

            logits_samples = sample_gumbel_softmax(self.neural_loglogits,
                                                   nb=nb,
                                                   nsamples=nsamples, 
                                                   temperature= 1.,
                                                   training = self.training)
            

        else:
            logits = self.loglogits / self.tempering0      #(num_channels,num_mixtures)             
            logits_samples = sample_gumbel_softmax(logits,
                                                   nb=nb,
                                                   nsamples=nsamples,
                                                   temperature= 1.,
                                                   training = self.training)

            
        self.logits_samples= logits_samples
        return logits_samples
    
    
    
    
    def sample_prior_shared(self,xc,yc,xt,numsamples=10,reorder=False):        
        
        """
        inputs:
            xc : (nb,ncontext,ndim,nchannel)
            xt : (nb,ntarget,ndim,nchannel)        
        outputs:
            xa_samples : (nb,nchannel*(ncontext+ntarget),ndim)
            Psi_pred    : (nb,nchannel*(ncontext+ntarget),nchannel)      # assume y-value is 1-d      
        """
        nb = xc.size(0)
        # -----------------------------------
        # prepare cosine features
        # -----------------------------------        
        xa_samples =self.build_xgrid(xc,xt)
        w,b,random_w = self.sample_w_b(xc,yc,numsamples)    
        
        w = w.permute(0,1,3,2)
        b = b.permute(0,1,3,2)

        xa_samples_ = xa_samples[:,None,:].repeat(1,w.size(1),1,1)
        xadotw = torch.einsum('bjxy,bjyz->bjxz',xa_samples_,w)        
        cos_interms = pi2*xadotw + pi2*b  if pi2repuse else  xadotw + pi2*b                  
        Psi = torch.cos(cos_interms)            
        nb,_,ndata2,nmixture = Psi.shape

        
        Psi = Psi[...,None].repeat(1,1,1,1,self.num_channels)
        random_w = random_w[:,:,None,:,:]
        sum_costerm =  (Psi*random_w).reshape(nb,numsamples,self.num_fourierbasis,ndata2,nmixture,self.num_channels)
        sum_costerm = sum_costerm.sum(dim=2)
        normalizer = np.sqrt(2/self.num_fourierbasis)        
        prior_samples = normalizer*sum_costerm        #(nb,numsamples,ndata2,nmixture,nchannel)

        
        
        # -----------------------------------
        # prepare logits features
        # -----------------------------------                
        logits_samples = self.sample_logits(xc,yc,numsamples,eps=1e-6)
        
        #logits_samples : (nb,numsamples,nchannel,nmixture) --> (nb,numsamples,1,nmixture,nchannel)         
        logits_samples_ = logits_samples.permute(0,1,3,2)[:,:,None,:,:]
        w_prior_samples = (prior_samples*logits_samples_).sum(dim=-2)                
        #return w_prior_samples,xa_samples                 
        return w_prior_samples,xa_samples,prior_samples                 


 

    #def sample_prior_independent(self,xc,yc,numsamples=10,newsample=False):            
    def sample_prior_independent(self,xc,numsamples=10,newsample=False):            
        
        nb,ndata,ndim,nchannel = xc.shape
        w,b,random_w,logits_samples = self.w, self.b,self.random_w,self.logits_samples 
            
            
        w = w.permute(0,1,3,2)
        b = b.permute(0,1,3,2)
            
        xc_ = xc[:,None,:,:,:].repeat(1,w.size(1),1,1,1)
        w_ =w[...,None].repeat(1,1,1,1,nchannel)
        b_ =b[...,None].repeat(1,1,1,1,nchannel)

        if pi2repuse:            
            xcdotw_b = pi2*(torch.einsum('bsndc,bsdmc->bsnmc',xc_,w_) + b_)
        else:
            xcdotw_b = torch.einsum('bsndc,bsdmc->bsnmc',xc_,w_) + pi2*b_
        
        Psi = torch.cos(xcdotw_b)
        sum_costerm = Psi*random_w[:,:,None,:,:]
        sum_costerm_ = sum_costerm.reshape(nb,numsamples,-1,ndata,self.num_mixtures,nchannel)
        normalizer = np.sqrt(2/self.num_fourierbasis)
        prior_samples = (sum_costerm_.sum(dim=2))*normalizer  #(nb,numsamples,ndata,nmixture,nchannel)
        
        
        #logits_samples : (nb,numsamples,nchannel,nmixture) --> (nb,numsamples,1,nmixture,nchannel) 
        logits_samples_ = logits_samples.permute(0,1,3,2)[:,:,None,:,:]
        w_prior_samples = (prior_samples*logits_samples_).sum(dim=-2)                                    
        
        #(nb,numsamples,ndata,nchannel),(nb,numsamples,ndata,nmixture,nchannel)         
        return w_prior_samples,prior_samples



    
    
    
    
    def prepare_updateterms(self,xc,yc,xa_shared=None,xt=None,numsamples=1):
        nb,ndata,ndim,nchannel = xc.shape

        likerr = self.loglik.exp()
        likerr_bound = self.loglik_bound

        
        mu = self.logmu.exp()
        inv_std  = 1/(self.logstd.exp()+eps)        
        logits = self.neural_logits 

        
        param_list = (mu,inv_std,logits,likerr)
        self.param_list = param_list
        xa_shared_ = xa_shared[...,None].repeat(1,1,1,nchannel)
        WK_cc,K_cc = eval_smkernel_batch(param_list,xc,  likerr_bound=likerr_bound)                
        WK_ac,_ = eval_smkernel_batch(param_list,xa_shared_, xc,  likerr_bound=likerr_bound)
        WK_cc_ = WK_cc.permute(0,3,1,2) 
        WK_ac_ = WK_ac.permute(0,3,1,2) #(nb,nchannel,ndata2,ndata)
        
        
        L = torch.linalg.cholesky(WK_cc_ )                  
        #(nb,numsamples,ndata,nchannel),(nb,numsamples,ndata,nmixture,nchannel)         
        w_prior_ind,prior_ind = self.sample_prior_independent(xc,numsamples=numsamples)                
        w_prior_ind =  w_prior_ind +  likerr[None,None,None,:]*torch.randn_like(w_prior_ind).to(xc.device)
        density_term = WK_ac_.sum(dim=-1).permute(0,2,1)    
                
        delta_yc = yc[:,None,:,:]  - w_prior_ind
        delta_yc = delta_yc.permute(0,3,2,1)
        Kinvyc = torch.cholesky_solve(delta_yc,L,upper=False)        
        update_term_shared = torch.einsum('bnac,bncs->bnas',WK_ac_,Kinvyc).permute(0,3,2,1)        
        
        
        _,K_tc = eval_smkernel_batch(param_list, xt.clone(), xc.clone(),  likerr_bound=likerr_bound) 
    
        #K_cc.shape,K_tc.shape,prior_ind.shape,yc.shape
        #torch.Size([4, 5, 20, 20, 3]) torch.Size([4, 5, 40, 20, 3]) torch.Size([4, 5, 20, 5, 3]) torch.Size([4, 20, 3])    
        prior_ind =  prior_ind +  likerr[None,None,None,None,:]*torch.randn_like(prior_ind).to(xc.device)    
        delta_yc2 =  yc[:,None,:,None,:]  - prior_ind
        delta_yc2 = delta_yc2.permute(0,3,4,2,1) #(nb,nmixture,nchannel,ndata,nsamples)    
        K_tc2 = K_tc.permute(0,1,4,2,3)    #(nb,nmixture,nchannel,ndata,nsamples)  
        K_cc2 = K_cc.permute(0,1,4,2,3)   #(nb,nmixture,nchannel,ndata,nsamples)  
    
        L2 = torch.linalg.cholesky(K_cc2)   
        Kinvyc2 = torch.cholesky_solve(delta_yc2,L2,upper=False)        
        update_term_target = torch.einsum('bmntc,bmncs->bmnts',K_tc2,Kinvyc2)  
        update_term_target = update_term_target.permute(0,4,3,1,2) #(nb,nsamples,ndata,nmixture,nchannel)
        #update_term_target.shape
        #torch.Size([4, 5, 3, 20, 5])
        return update_term_shared,density_term,update_term_target 
    
    
    

    def sample_posterior(self,xc,yc,xt,numsamples=1,reorder=False,iterratio=None,use_constrainedprior=True):
        
        w_prior_shared, xa_shared , prior_shared = self.sample_prior_shared(xc,yc,xt,numsamples=numsamples)                
        w_update_term_shared, density_term, update_term_target = self.prepare_updateterms(xc,yc,
                                                                                          xa_shared=xa_shared,
                                                                                          xt=xt,numsamples=numsamples)               
        posterior_shared =  w_prior_shared + w_update_term_shared
                
            
        #(nb,numsamples,ndata,nchannel),(nb,numsamples,ndata,nmixture,nchannel)                         
        w_prior_target ,prior_target = self.sample_prior_independent(xt,numsamples=numsamples)               
        
        #update_term_target.shape
        #torch.Size([4, 5, 3, 20, 5])
        #posterior_target = prior_target + update_term_target[:,:,:,None,:]
        
        #revision for individual prior
        posterior_target = prior_target + update_term_target

        
        outs = AttrDict()
        outs.xa_samples = xa_shared    #(nb,ndata2,ndim)                  
        outs.prior_samples =  prior_shared      
        outs.wprior_samples = w_prior_shared
        outs.posterior_samples= posterior_shared         
        outs.posterior_target= posterior_target 
        outs.neural_loglogits = self.neural_loglogits
        outs.neural_logits = self.neural_logits
        outs.tempering0 = self.tempering0
        
        outs.density = density_term 
        outs.regloss = 0.0        
        return outs    

    
    
    
    
    
    