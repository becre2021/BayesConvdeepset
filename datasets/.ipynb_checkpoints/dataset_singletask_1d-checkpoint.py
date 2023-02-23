import numpy as np
import torch 
import torch.nn as nn




#--------------------------------------------------------------------------------
# simulation utils
#--------------------------------------------------------------------------------

task_list = ['rbf','matern','weaklyperiodic','sawtooth']
def prepare_mixed_1dtask(data_name='rbf',
                    testtype='inter',
                    nbatch = 32,
                    batch_npoints=(64,64),
                    train_range = [-5,5],
                    test_range = [-5,5], 
                    nchannels=3,
                    noise_true = True,
                    eps=1e-4, 
                    intrain=True):

    assert data_name in task_list
    
    x_range = test_range
    intervals = train_range
    ncontext,ntarget = batch_npoints
        
    xf,yf = generate_1dtask_mixed(nb=nbatch, tasktype=data_name, x_range=x_range)
    context_x,context_y,target_x,target_y,full_x,full_y = prepare_batchset(xf,yf,
                                                                           nb = nbatch,
                                                                           intervals=intervals,
                                                                           ncontext=ncontext,
                                                                           ntarget=ntarget,
                                                                           testtype=testtype,
                                                                           intrain=intrain)

    return context_x,context_y,target_x,target_y,full_x,full_y    



#--------------------------------------------------------------------------------
# simulation utils
#--------------------------------------------------------------------------------

#def generate_1dtask_mixed(nb=2, tasktype='rbf',x_range=[0,6], ntotal=250,nchannels=1,noise_true = True ,eps=1e-4):
def generate_1dtask_mixed(nb=2, tasktype='rbf',x_range=[0,10], ntotal=250,nchannels=1,noise_true = True ,eps=1e-4):
    xf = np.linspace(x_range[0]+eps,x_range[1]-eps,ntotal).reshape(-1,1)
    xf = np.repeat(xf,nchannels,axis=1)
    
    if tasktype in ['rbf','matern','weaklyperiodic']:
        kernel = kernel_list(kerneltype=tasktype)
        yf = kernel.sample_posterior(xf,nb=nb)
                    
    if tasktype == 'sawtooth':    
        yf = [sawtooth_varying(xf)[None,:,:] for j in range(nb)]
        yf = np.concatenate(yf,axis=0)
         
 
    if isinstance(yf, np.ndarray):
        yf = torch.from_numpy(yf).float()
 
    if isinstance(xf, np.ndarray):
        xf = torch.from_numpy(xf).float()
        xf = xf[None,:,:].repeat(nb,1,1) 
            
    return xf,yf



    
def filter_intervals(full_x,full_y,intervals=[-1,1],ncontext=64,ntarget=64,testtype='inter',intrain=True):
    
    context_x,context_y = [],[]
    target_x,target_y = [],[]    
    nobs,nchannels = full_x.size()
    for i in range(nchannels):
        
        if len(np.array(intervals).shape) == 1:     
            interval_true = (full_x[:,i]>intervals[0])*(full_x[:,i]<=intervals[1])   #inintervals
        
        # when the observation regions is different across the channels
        if len(np.array(intervals).shape) > 1:
            assert(np.array(intervals).shape[0] == nchannels)
            interval_true = (full_x[:,i]>intervals[i][0])*(full_x[:,i]<=intervals[i][1])   #inintervals

        
        interval_idx = np.where(interval_true)[0]             #interval in 
        interval_notidx = np.where(~interval_true)[0]         #iterval out

        #print(interval_idx,interval_notidx )
        
        if intrain:
            if testtype == 'extra':
                idxc = np.random.permutation(len(interval_idx))            
                chosen_idxc = np.sort(interval_idx[idxc[:ncontext]])
                chosen_idxt = np.sort(interval_idx[idxc[ncontext:ncontext+ntarget]])
                
            elif testtype == 'inter':
                idxt = np.random.permutation(len(interval_notidx))
                chosen_idxc = np.sort(interval_notidx[idxt[:ncontext]])
                chosen_idxt = np.sort(interval_notidx[idxt[ncontext:ncontext+ntarget]])

            else:
                idxf = np.random.permutation(len(full_x[:,i]))
                chosen_idxc = np.sort(idxf[:ncontext])
                chosen_idxt = np.sort(idxf[ncontext:ncontext+ntarget])                
            
        else: 

            #-------------------------------------------------
            #beyond range conctext,target
            #-------------------------------------------------
            if testtype == 'extra':                
                #idxc = np.random.permutation(len(interval_idx))
                idxt = np.random.permutation(len(interval_notidx))
                #chosen_idxc = np.sort(interval_idx[idxc[:ncontext]])    #interval as context
                
                chosen_idxc = np.sort(interval_notidx[idxt[:ncontext]])
                chosen_idxt = np.sort(interval_notidx[idxt[ncontext:ncontext+ntarget]])  #not interval as target
                
            elif testtype == 'inter':
                #idxc = np.random.permutation(len(interval_notidx))
                idxt = np.random.permutation(len(interval_idx))
                #chosen_idxc = np.sort(interval_notidx[idxc[:ncontext]])  #not interval as context

                chosen_idxc = np.sort(interval_idx[idxt[:ncontext]])      #interval as target                
                chosen_idxt = np.sort(interval_idx[idxt[ncontext:ncontext+ntarget]])      #interval as target
                
            else:
                idxf = np.random.permutation(len(full_x[:,i]))
                chosen_idxc = np.sort(idxf[:ncontext])
                chosen_idxt = np.sort(idxf[ncontext:ncontext+ntarget])                



        i_context_x,i_target_x = full_x[chosen_idxc,i],full_x[chosen_idxt,i]
        i_context_y,i_target_y = full_y[chosen_idxc,i],full_y[chosen_idxt,i]


        #print('i_context_x.size(),i_context_y.size(),i_target_x.size(),i_target_y.size()')        
        #print(i_context_x.size(),i_context_y.size(),i_target_x.size(),i_target_y.size())
        
        
        
        context_x.append(i_context_x[:,None])
        context_y.append(i_context_y[:,None])
        target_x.append(i_target_x[:,None])
        target_y.append(i_target_y[:,None])


    context_x = torch.cat(context_x,dim=1)
    context_y = torch.cat(context_y,dim=1)
    target_x = torch.cat(target_x,dim=1)
    target_y = torch.cat(target_y,dim=1)
    
    
    return context_x,context_y,target_x,target_y



def prepare_batchset(xf,yf,nb=8,intervals=[0,4],ncontext=64,ntarget=64,testtype='extra',intrain=True):
    xc_list,yc_list,xt_list,yt_list=[],[],[],[]
    for j in range(nb):
        xc,yc,xt,yt = filter_intervals(xf[j,:,:],yf[j,:,:],intervals=intervals,ncontext=ncontext,ntarget=ntarget,testtype=testtype,intrain=intrain)
        #xc,yc,xt,yt = filter_intervals(xf[0,:,:],yf[0,:,:],intervals=[0,4],ncontext=64,ntarget=64,testtype='extra',intrain=False)
        xc_list.append(xc.unsqueeze(dim=0))
        yc_list.append(yc.unsqueeze(dim=0))
        xt_list.append(xt.unsqueeze(dim=0))
        yt_list.append(yt.unsqueeze(dim=0))

    xc_list = torch.cat(xc_list,dim=0) 
    yc_list = torch.cat(yc_list,dim=0) 
    xt_list = torch.cat(xt_list,dim=0) 
    yt_list = torch.cat(yt_list,dim=0) 

    return xc_list.float(),yc_list.float(),xt_list.float(),yt_list.float(),xf.float(),yf.float()











#--------------------------------------------------------------------------------
# simulation function
#--------------------------------------------------------------------------------

def _rand(val_range, *shape):
    lower, upper = val_range
    return lower + np.random.rand(*shape) * (upper - lower)


def sin_varying(x,freq_dist=(2., 4),shift_dist=(-2, 2),mag_dist=(0.5, 1.5),amp = 1):    
    freq = _rand(freq_dist)
    shift = _rand(shift_dist)
    mag = _rand(mag_dist)
        
    yout =  1*mag*np.sin( 2 * np.pi *(0.1 + freq)*(x            + shift ) )   
    eps = 0.1*np.random.normal(size=x.shape)
    
    print(yout.shape,eps.shape)
    yout += eps
    
    #dep
    #f1 = lambda x : 1*(1+mag)*np.sin( (2.1 + freq)*(x            + phase0 ) )     
    #noise_x = lambda x : 0.1*np.random.normal(size=len(x))   
    #yout = f1(t1)+noise_x(t1)         
    return yout
    
    
#def sawtooth_varying(x,freq_dist=(3, 5),shift_dist=(-5, 5),trunc_dist=(10, 20),amp_dist = (0.5, 1.5)):  #too freqeucny
def sawtooth_varying(x,freq_dist=(1, 2),shift_dist=(-1, 1),trunc_dist=(10, 20),amp_dist = (0.8, 1.2)):  #too freqeucny
    
    """
    args:
        x : (nb,ndim)
    return 
        y : (nb,ndim)
    """
        
    freq = _rand(freq_dist)
    shift = _rand(shift_dist)
    amp = _rand(amp_dist)
    trunc = np.random.randint(trunc_dist[0], trunc_dist[1] + 1)
    
    x = x + shift    
    k = np.arange(1, trunc + 1)[None, :]    
    yout = 0.5 * amp - amp / np.pi *np.sum((-1) ** k * np.sin(2 * np.pi * k * freq * x) / k, axis=1,keepdims=True)        
    #eps = 0.01*np.random.normal(size=x.shape)
    #yout += eps
    
    return yout
     


from math import pi
pi2 = 2*pi
class kernel_list(nn.Module):
    def __init__(self, input_dims=None, active_dims=None, kerneltype="rbf"):
        super(kernel_list, self).__init__()

        if  kerneltype == 'rbf':
            self.l_dist = (1.1,2.1)
            self.sigma = torch.tensor([1.]).float()
        if  kerneltype == 'matern':
            self.nu = 2.5            
            #self.l = torch.tensor([l]).float()
            self.l_dist = (0.19,0.21)
            sigma = 1.
            self.sigma = torch.tensor([sigma]).float()
            
        if  kerneltype == 'weaklyperiodic':
            #self.freq_dist = (7.5,8.5) requires many obs
            self.freq_dist = (2.,3.)
             
        self.kerneltype = kerneltype
            
 
    def sample_posterior(self,x,nb=16,noise_true = True, noisestd=0.1 ,zitter=1e-4,varying=False):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
            x = x[None,:,:].repeat(nb,1,1) 

        if self.kerneltype == 'rbf':
            Kxx = self.K_rbf(x)
            
        if self.kerneltype == 'matern':
            Kxx = self.K_matern(x)
            
        if self.kerneltype == 'weaklyperiodic':
            Kxx = self.K_weaklper(x)
        
        
        Kf = Kxx + zitter*torch.eye(x.size(1))          
        Lf = torch.cholesky(Kf)

        nb,ndata,_ = Lf.shape
        gaussian_nosie = torch.randn(nb,ndata,1)
        sample_yf = Lf.bmm(gaussian_nosie)

        if noise_true:
            sample_yf += noisestd*torch.randn_like(sample_yf) 
        
        return sample_yf


    def K_rbf(self, X1, X2=None):
        """
        args:
            X1 : (nb,ndata,1)
            X2 : (nb,ndata2,1)            
        return:    
        """
        if X2 is None:
            X2 = X1
            
        self.l = _rand(self.l_dist)                
        dist = (X1 - X2.permute(0,2,1))/self.l       
        return (self.sigma**2 )* torch.exp(-0.5*dist**2)

        
        self.nu = nu
        #self.l = torch.tensor([l]).float()
        self.l_dist = (0.19,0.21)
        self.sigma = torch.tensor([sigma]).float()
            
        
 
    def K_matern(self, X1, X2=None):
        """
        args:
            X1 : (nb,ndata,1)
            X2 : (nb,ndata2,1)            
        return:    
        """
        if X2 is None:
            X2 = X1
            
        self.l = _rand(self.l_dist)        
        
        dist = torch.abs(X1 - X2.permute(0,2,1))/self.l       
        #dist = torch.abs(torch.tensordot(distance(X1,X2), 1.0/self.l, dims=1))
        if self.nu == 0.5:
            constant = 1.0
        elif self.nu == 1.5:
            constant = 1.0 + np.sqrt(3.0)*dist
        elif self.nu == 2.5:
            constant = 1.0 + np.sqrt(5.0)*dist + 5.0/3.0*dist**2
        return (self.sigma**2 )* constant * torch.exp(-np.sqrt(self.nu*2.0)*dist)
  

    def K_weaklper(self, X1, X2=None ):
        """
        args:
            X1 : (nb,ndata,1)
            X2 : (nb,ndata2,1)            
        return:    
        """

        if X2 is None:
            X2 = X1

        freq1 = _rand(self.freq_dist)
        freq2 = _rand(self.freq_dist) 
        f1_X1 = torch.cos(pi2*freq1*X1)
        f1_X2 = torch.cos(pi2*freq1*X2)        
        f2_X1 = torch.sin(pi2*freq1*X1)
        f2_X2 = torch.sin(pi2*freq1*X2)

        f1_dist = (f1_X1 - f1_X2.permute(0,2,1))**2
        f2_dist = (f2_X1 - f2_X2.permute(0,2,1))**2
        x_dist = (X1 - X2.permute(0,2,1))**2

        outs = torch.exp(-0.5*f1_dist -0.5*f2_dist - (1/32)*x_dist)
        return outs










