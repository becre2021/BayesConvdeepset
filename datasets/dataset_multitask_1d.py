import numpy as np
import torch 
import torch.nn as nn


#tasktype_list = ['sin3_dep','sin3_ind','sin4','mosm','lmc']
tasktype_list = ['sin3_dep','sin3_ind','sin3','sin4','mosm','lmc','mosmvarying']

class motask_generator(object):
    def __init__(self,tasktype,testtype,nchannels,train_range,test_range,dep=True):
        
        assert tasktype in tasktype_list
        self.tasktype = tasktype
        self.testtype = testtype
        self.nchannels = nchannels
        self.train_range = train_range
        self.test_range = test_range        
        self.full_range = [min(test_range[0],train_range[0]),max(test_range[1],train_range[1])]
        self.dep = dep
        
        self.f = None
        self.build_taskfunction(tasktype)
 

    def build_taskfunction(self,tasktype):

        if tasktype == 'sin3':
            if self.dep:
                self.f = sin_3channels_dep         
            else:
                self.f = sin_3channels_ind
            self.nchannels = 3
                    
        elif tasktype == 'mosm':        
            self.f = MOSM(output_dims=self.nchannels, input_dims=1,dep=self.dep)
            
        elif tasktype == 'mosmvarying':        
            self.f = MOSM(output_dims=self.nchannels, input_dims=1,dep=self.dep,varying=True)
            
            
        elif tasktype == 'lmc':        
            kernel_list = [Matern(nu=0.5,sigma=0.9, l=.5,input_dims=1),
                           Matern(nu=1.5,sigma=0.9, l=.5,input_dims=1),
                           Matern(nu=2.5,sigma=0.9, l=.5,input_dims=1)]            

            lmc = LMC(kernel_list=kernel_list,output_dims=self.nchannels, input_dims=1,Rq=1,dep=self.dep)
            self.f = lmc
            
            
        else:
            pass

        return 
    
    
    def prepare_task(self,nbatch,ncontext,ntarget,train_range = None,test_range=None,noise_true=True,intrain = True):
        #def gen_task(data_type):
        if train_range is None:
            train_range = self.train_range
        
        if test_range is None:
            test_range = self.test_range
        
        if self.tasktype in ['mosm','lmc'] :
            context_x,context_y,target_x,target_y,full_x,full_y = prepare_gptask(self.f,
                                                                                 testtype=self.testtype,
                                                                                 nbatch = nbatch,
                                                                                 batch_npoints=(ncontext,ntarget), 
                                                                                 train_range = train_range ,
                                                                                 test_range = test_range,
                                                                                 nchannels=self.nchannels,
                                                                                 noise_true = noise_true,
                                                                                 intrain = intrain,
                                                                                 varying=False)
   
        if self.tasktype in ['mosmvarying'] :
            
            #mosm varying samples mean parameters from [0.1,2] uniformly 
            context_x,context_y,target_x,target_y,full_x,full_y = prepare_gptask(self.f,
                                                                                 testtype=self.testtype,
                                                                                 nbatch = nbatch,
                                                                                 batch_npoints=(ncontext,ntarget), 
                                                                                 train_range = train_range ,
                                                                                 test_range = test_range,
                                                                                 nchannels=self.nchannels,
                                                                                 noise_true = noise_true,
                                                                                 intrain = intrain,
                                                                                 varying=True)

        elif self.tasktype in ['sin3','sin4','sin3_dep','sin3_ind'] :
#             context_x,context_y,target_x,target_y,full_x,full_y = prepare_batch(self.tasktype,
#                                                                                   nbatch = nbatch,
#                                                                                   batch_npoints=(ncontext,ntarget),
#                                                                                   train_period = train_range,
#                                                                                   test_period =  test_range,
#                                                                                   nchannels=self.nchannels,
#                                                                                   noise_true = noise_true,
#                                                                                   intrain = intrain)    

            context_x,context_y,target_x,target_y,full_x,full_y = prepare_sintask(self.tasktype,
                                                                                  dep = self.dep,
                                                                                  testtype=self.testtype,
                                                                                  nbatch = nbatch,
                                                                                  batch_npoints=(ncontext,ntarget),
                                                                                  train_range = train_range,
                                                                                  test_range =  test_range,
                                                                                  nchannels=self.nchannels,
                                                                                  noise_true = noise_true,
                                                                                  intrain = intrain)    

        else:
            pass
        
        return context_x,context_y,target_x,target_y,full_x,full_y
        

        
ntotal = 250
def prepare_sintask(data_name,testtype='inter' ,nbatch = 32,batch_npoints=(64,64), train_range = [-5,5],test_range = [-5,5], nchannels=3,noise_true = True,eps=1e-4, intrain=True, dep=True):
    
    context_x,context_y = [],[]
    target_x,target_y = [],[]
    full_x,full_y = [],[]
    #ntotal = int(2.1*(batch_npoints[0]+batch_npoints[1]))

    
    for _ in range(nbatch):
        #rand_freq = 10*(np.random.rand()-0.5)
        #rand_freq = 1.0+ 4.0*(np.random.rand())
        rand_freq = 0.0            
        if testtype == 'inter':
            x_range = train_range
            intervals = test_range
            #ntarget,ncontext = batch_npoints
            ncontext,ntarget = batch_npoints
            
            #i_full_x,i_full_y = gp_sampler(kernel,ntotal,x_range, nchannels,noise_true,eps)
            i_full_x,i_full_y = sin_sampler(data_name,dep,ntotal,x_range, nchannels,noise_true,eps, rand_freq)            
            i_context_x,i_context_y,i_target_x,i_target_y = filter_intervals(i_full_x,i_full_y,intervals,ncontext,ntarget,testtype=testtype,intrain=intrain)
            

        elif testtype == 'extra':
            x_range = test_range
            intervals = train_range
            ncontext,ntarget = batch_npoints
                
            #i_full_x,i_full_y = gp_sampler(kernel,ntotal,x_range, nchannels,noise_true,eps)           
            i_full_x,i_full_y = sin_sampler(data_name,dep,ntotal,x_range, nchannels,noise_true,eps, rand_freq)                        
            i_context_x,i_context_y,i_target_x,i_target_y = filter_intervals(i_full_x,i_full_y,intervals,ncontext,ntarget,testtype=testtype,intrain=intrain)

        else:
            x_range = [min(test_range[0],train_range[0]),max(test_range[1],train_range[1])]
            ncontext,ntarget = batch_npoints
            #ntotal = 3*(batch_npoints[0]+batch_npoints[1])
            #ntotal = 2*(batch_npoints[0]+batch_npoints[1])
            #ntotal = 4*(batch_npoints[0]+batch_npoints[1])
               
            i_full_x,i_full_y = sin_sampler(data_name,dep,ntotal,x_range, nchannels,noise_true,eps, rand_freq)   
            
            
            idxf = [np.random.permutation(len(i_full_x[:,i])) for i in range(i_full_x.size(1))]
            chosen_idx_c = [np.sort(iidxf[:ncontext]) for iidxf in idxf] 
            chosen_idx_t = [np.sort(iidxf[ncontext:ncontext+ntarget]) for iidxf in idxf] 
    
            i_context_x = torch.cat([i_full_x[iidx_c,ii].reshape(-1,1)  for ii,iidx_c in enumerate(chosen_idx_c)],dim=1)
            i_context_y = torch.cat([i_full_y[iidx_c,ii].reshape(-1,1)  for ii,iidx_c in enumerate(chosen_idx_c)],dim=1) 
            i_target_x = torch.cat([i_full_x[iidx_t,ii].reshape(-1,1)  for ii,iidx_t in enumerate(chosen_idx_t)],dim=1) 
            i_target_y = torch.cat([i_full_y[iidx_t,ii].reshape(-1,1)  for ii,iidx_t in enumerate(chosen_idx_t)],dim=1) 
            
        context_x.append( i_context_x.unsqueeze(dim=0)  )
        context_y.append( i_context_y.unsqueeze(dim=0) )
        target_x.append( i_target_x.unsqueeze(dim=0)  )
        target_y.append( i_target_y.unsqueeze(dim=0)  )
        full_x.append( i_full_x.unsqueeze(dim=0) )
        full_y.append( i_full_y.unsqueeze(dim=0) )
                
            
            
    context_x = torch.cat(context_x,dim=0).float()
    context_y = torch.cat(context_y,dim=0).float()
    target_x = torch.cat(target_x,dim=0).float()
    target_y = torch.cat(target_y,dim=0).float()
    full_x = torch.cat(full_x,dim=0).float()
    full_y = torch.cat(full_y,dim=0).float()        
    return context_x,context_y,target_x,target_y,full_x,full_y





#ntotal = 250
ntotal = 300
def prepare_gptask(kernel,testtype='inter' ,nbatch = 32,batch_npoints=(64,64), train_range = [-5,5],test_range = [-5,5], nchannels=3,noise_true = True,eps=1e-4, intrain=True,varying=False):

    context_x,context_y = [],[]
    target_x,target_y = [],[]
    full_x,full_y = [],[]
    #ntotal = int(2.1*(batch_npoints[0]+batch_npoints[1]))
    
    for _ in range(nbatch):
                    
        if testtype == 'inter':
            x_range = train_range
            intervals = test_range
            #ntarget,ncontext = batch_npoints
            ncontext,ntarget = batch_npoints
            
            
            i_full_x,i_full_y = gp_sampler(kernel,ntotal,x_range, nchannels,noise_true,eps,varying)
            i_context_x,i_context_y,i_target_x,i_target_y = filter_intervals(i_full_x,i_full_y,intervals,ncontext,ntarget,testtype=testtype,intrain=intrain)
            

        elif testtype == 'extra':
            x_range = test_range
            intervals = train_range
            ncontext,ntarget = batch_npoints
                
            i_full_x,i_full_y = gp_sampler(kernel,ntotal,x_range, nchannels,noise_true,eps,varying)
            i_context_x,i_context_y,i_target_x,i_target_y = filter_intervals(i_full_x,i_full_y,intervals,ncontext,ntarget,testtype=testtype,intrain=intrain)

        
        else:
            x_range = [min(test_range[0],train_range[0]),max(test_range[1],train_range[1])]
            ncontext,ntarget = batch_npoints
            #ntotal = 3*(batch_npoints[0]+batch_npoints[1])
            #ntotal = 4*(batch_npoints[0]+batch_npoints[1])
            #ntotal = 250
            
            i_full_x,i_full_y = gp_sampler(kernel,ntotal,x_range, nchannels,noise_true,eps,varying)    
            
            
            # filtering process
            idxf = [np.random.permutation(len(i_full_x[:,i])) for i in range(i_full_x.size(1))]
            chosen_idx_c = [np.sort(iidxf[:ncontext]) for iidxf in idxf] 
            chosen_idx_t = [np.sort(iidxf[ncontext:ncontext+ntarget]) for iidxf in idxf] 
    
            i_context_x = torch.cat([i_full_x[iidx_c,ii].reshape(-1,1)  for ii,iidx_c in enumerate(chosen_idx_c)],dim=1)
            i_context_y = torch.cat([i_full_y[iidx_c,ii].reshape(-1,1)  for ii,iidx_c in enumerate(chosen_idx_c)],dim=1) 
            i_target_x = torch.cat([i_full_x[iidx_t,ii].reshape(-1,1)  for ii,iidx_t in enumerate(chosen_idx_t)],dim=1) 
            i_target_y = torch.cat([i_full_y[iidx_t,ii].reshape(-1,1)  for ii,iidx_t in enumerate(chosen_idx_t)],dim=1) 
                
        context_x.append( i_context_x.unsqueeze(dim=0)  )
        context_y.append( i_context_y.unsqueeze(dim=0) )
        target_x.append( i_target_x.unsqueeze(dim=0)  )
        target_y.append( i_target_y.unsqueeze(dim=0)  )
        full_x.append( i_full_x.unsqueeze(dim=0) )
        full_y.append( i_full_y.unsqueeze(dim=0) )
                
            
            
    context_x = torch.cat(context_x,dim=0).float()
    context_y = torch.cat(context_y,dim=0).float()
    target_x = torch.cat(target_x,dim=0).float()
    target_y = torch.cat(target_y,dim=0).float()
    full_x = torch.cat(full_x,dim=0).float()
    full_y = torch.cat(full_y,dim=0).float()        
    return context_x,context_y,target_x,target_y,full_x,full_y


    
def filter_intervals(full_x,full_y,intervals=[-1,1],ncontext=64,ntarget=64,testtype='inter',intrain=True):
    # debuggin checked goods
    #print('filter intervals')
    #print('testtype,intrain' )    
    #print(testtype,intrain )
    #print('')
    
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






################################################################
# synthetic dataset
################################################################


sigma_eps_std = 0.1
def gp_sampler(kernel,ntotal,x_range=[-3,3], nchannels=3,noise_true = True ,eps=1e-4,varying=False):
    if varying:
        kernel.reset_param()
    
    xf = torch.linspace(x_range[0]+eps,x_range[1]-eps,ntotal).view(-1,1) 
    xf = xf.repeat(1,nchannels)
    Kf = compute_multioutput_K(kernel.Ksub,xf)
    Lf = torch.cholesky(Kf)

    sample_yf = Lf.mm(torch.randn(Lf.size(0),1)).reshape(nchannels,-1).transpose(0,1)

    if noise_true:
        sample_yf += sigma_eps_std*torch.randn_like(sample_yf) 
        
    return xf,sample_yf





def sin_sampler(data_name,dep,ntotal,x_range=[-3,3], nchannels=3,noise_true = True ,eps=1e-4, rand_freq = 0.0):
    xf = np.linspace(x_range[0]+eps,x_range[1]-eps,ntotal).reshape(-1,1)
    xf = np.repeat(xf,nchannels,axis=1)
    
    if data_name == 'sin3' and dep == True:
        f = sin_3channels_dep
        full_x,full_y = f(xf[:,0],xf[:,1],xf[:,2],noise_true,freq = rand_freq)
        
    if data_name == 'sin3' and dep == False:
        f = sin_3channels_ind
        full_x,full_y = f(xf[:,0],xf[:,1],xf[:,2],noise_true,freq = rand_freq)
    
    elif data_name == 'sin4':
        f = sin_4channels    
        full_x,full_y = f(xf[:,0],xf[:,1],xf[:,2],xf[:,3],noise_true,freq = rand_freq)
    else:
        pass
    
    full_x = torch.tensor(full_x).float()
    full_y = torch.tensor(full_y).float()
    return full_x,full_y


def sin_3channels_ind(t1,t2,t3,noise_true = True,freq=0.0):
    t1.sort(),t2.sort(),t3.sort()    
    phase0 = 2*(np.random.rand()-0.5)
    phase1 = 2*(np.random.rand()-0.5)
    phase2 = 2*(np.random.rand()-0.5)


    phase0 = 2*(np.random.rand()-0.5)
    phase1 = 2*(np.random.rand()-0.5)
    phase2 = 2*(np.random.rand()-0.5)
    #freq = 5.0*(np.random.rand())          
    freq = 0.0
    mag = 0.5*(np.random.rand()-0.5)
    #dep
    f1 = lambda x : 1*(1+mag)*np.sin( (2.1 + freq)*(x            + phase0 ) ) 
    f2 = lambda x : 2*(1+mag)*np.sin( (4.1 + 2*freq)*(x - 0.5    + phase1))
    f3 = lambda x : 3*(1+mag)*np.sin( (6.1 + 3*freq)*(x - 1.0   + phase2))

    
    noise_x = lambda x : 0.1*np.random.normal(size=len(x))
    x_out = np.concatenate([t1[:,None],t2[:,None],t3[:,None]],axis=-1)
    if noise_true:
        y_out = np.concatenate([ (f1(t1)+noise_x(t1))[:,None],(f2(t2)+noise_x(t2))[:,None],(f3(t3)+noise_x(t3))[:,None]],axis=-1)        
    else:
        y_out = np.concatenate([ f1(t1)[:,None],f2(t2)[:,None],f3(t3)[:,None] ],axis=-1)
            
    return x_out,y_out





def sin_3channels_dep(t1,t2,t3,noise_true = True,freq=0.0):
    t1.sort(),t2.sort(),t3.sort()    

    
    phase0 = 2*(np.random.rand()-0.5)
    phase1 = 2*(np.random.rand()-0.5)
    phase2 = 2*(np.random.rand()-0.5)
    freq = 5.0*(np.random.rand())          
    mag = 0.5*(np.random.rand()-0.5)
    #dep
    f1 = lambda x : 1*(1+mag)*np.sin( (2.1 + freq)*(x            + phase0 ) ) 
    f2 = lambda x : 2*(1+mag)*np.sin( (4.1 + 2*freq)*(x - 0.5    + phase1))
    f3 = lambda x : 3*(1+mag)*np.sin( (6.1 + 3*freq)*(x - 1.0   + phase2))

    
    noise_x = lambda x : 0.1*np.random.normal(size=len(x))
    
    
    x_out = np.concatenate([t1[:,None],t2[:,None],t3[:,None]],axis=-1)
    if noise_true:
        y_out = np.concatenate([ (f1(t1)+noise_x(t1))[:,None],(f2(t2)+noise_x(t2))[:,None],(f3(t3)+noise_x(t3))[:,None]],axis=-1)        
    else:
        y_out = np.concatenate([ f1(t1)[:,None],f2(t2)[:,None],f3(t3)[:,None] ],axis=-1)
            
    return x_out,y_out






################################################################
# mogp sampler
################################################################

import torch
import numpy as np
import torch.nn as nn

def distance(X1, X2=None):
    # X1 is NxD, X2 is MxD, then ret is NxMxD
    if X2 is None:
        X2 = X1
    return X1.unsqueeze(1) - X2

def squared_distance(X1, X2=None):
    # X1 is NxD, 
    # X2 is MxD, 
    # ouput : NxMxD
    if X2 is None:
        X2 = X1
    #return (X1.unsqueeze(1) - X2)**2  # slower than cdist for large X
    return torch.cdist(X2.T.unsqueeze(2), X1.T.unsqueeze(2)).T**2



def compute_multioutput_K(k_sub,x1,x2=None,eps=5e-4):
    if x2 is None:
        x2 = x1
        x2_none = True
    else:
        x2_none = False
    output_dims = x1.size(1)
    
    K = []
    k_dict = {}    
    if x2_none:
        for i in range(output_dims):
            k_i = []
            for j in range(output_dims):
                if i==j:
                    ksub_ij = k_sub(i,i,x1[:,i],x2[:,i])          
                    if ksub_ij.is_cuda and x2_none:
                        ksub_ij += eps*torch.eye(x1[:,i].size(0)).cuda()
                    else:
                        ksub_ij += eps*torch.eye(x1[:,i].size(0))
                elif i<j:
                    ksub_ij = k_sub(i,j,x1[:,i],x2[:,j])                
                    k_dict[(i,j)] = ksub_ij
                else:
                    ksub_ij = k_dict[(j,i)].T
                k_i.append(ksub_ij)
            K.append(torch.cat(k_i,dim=-1))
    else:
        for i in range(output_dims):
            k_i = []
            for j in range(output_dims):
                if i==j:
                    ksub_ij = k_sub(i,i,x1[:,i],x2[:,i])          
                else:
                    ksub_ij = k_sub(i,j,x1[:,i],x2[:,j])                
                    k_dict[(i,j)] = ksub_ij
                k_i.append(ksub_ij)
            K.append(torch.cat(k_i,dim=-1))
        
    del k_dict
    K = torch.cat(K,dim=0)
    return K






class Matern(nn.Module):
    def __init__(self, nu=0.5, sigma = 0.1, l=0.1, input_dims=None, active_dims=None, name="Matérn"):
        super(Matern, self).__init__()

        if nu not in [0.5, 1.5, 2.5]:
            raise ValueError("nu parameter must be 0.5, 1.5, or 2.5")

        #l = 0.01 +.09*torch.rand(input_dims)
        #sigma = 1.+2*torch.rand(1)

        #self.l = Parameter(l, lower=1e-6)
        #self.sigma = Parameter(sigma, lower=1e-6)

        
        self.nu = nu
        #self.l = nn.Parameter(torch.tensor([l]).float())
        #self.sigma = nn.Parameter(torch.tensor([sigma]).float())
        self.l = torch.tensor([l]).float()
        self.sigma = torch.tensor([sigma]).float()
 
    def K(self, X1, X2=None):
        # X has shape (data_points,input_dims)
        #X1,X2 = self._check_input(X1,X2)
        if X2 is None:
            X2 = X1

        dist = torch.abs(torch.tensordot(distance(X1,X2), 1.0/self.l, dims=1))
        if self.nu == 0.5:
            constant = 1.0
        elif self.nu == 1.5:
            constant = 1.0 + np.sqrt(3.0)*dist
        elif self.nu == 2.5:
            constant = 1.0 + np.sqrt(5.0)*dist + 5.0/3.0*dist**2
        return self.sigma**2 * constant * torch.exp(-np.sqrt(self.nu*2.0)*dist)

    def show_params(self):
        print('nu : {} '.format(self.nu))
        print('l : {} '.format(self.l))
        print('sigma : {} '.format(self.sigma))
        return



class LMC(nn.Module):
    def __init__(self, kernel_list, output_dims, input_dims, Q=None, Rq=1, name="LMC",dep=True):
        super(LMC, self).__init__()

        if Q is None:
            Q = len(kernel_list)
        weight = torch.rand(output_dims, Q, Rq)
        #print(weight.size())
        #self.weight = nn.Parameter(weight)

        self.dep = dep
        
        if Q==3 and Rq == 1:
            w1 = torch.tensor([[1.0],[0.0],[0.0]])[None,:,:]
            w2 = torch.tensor([[0.0],[1.0],[0.0]])[None,:,:]
            w3 = torch.tensor([[0.0],[0.0],[1.0]])[None,:,:]
            weight_ind = torch.cat([w1,w2,w3],dim=0)            
            
                    
            w1 = torch.tensor([[0.9],[0.9],[0.0]])[None,:,:]
            w2 = torch.tensor([[0.0],[-0.9],[-0.9]])[None,:,:]
            w3 = torch.tensor([[-0.9],[0.0],[0.9]])[None,:,:]
            weight_dep = torch.cat([w1,w2,w3],dim=0)            
                        
            
            #self.weight = nn.Parameter(weight_ind)
            if self.dep:
                print('weight dependecny true')
                #self.weight = nn.Parameter(weight_dep)
                self.weight = weight_dep               
            else:
                print('weight dependecny false')                
                #self.weight = nn.Parameter(weight_ind)
                self.weight = weight_ind
                
        #kernels = self._check_kernels(kernels, Q)        
        self.kernels = kernel_list
        self.transform = nn.Softplus()

        
        
    def reset_param(self):
        #self.mean += 0.5*torch.rand_like(self.mean)
        #print('varying mean {}'.format(self.mean))        
        return 
        
        
    def __getitem__(self, key):
        return self.kernels[key]

    def show_params(self):
        for i,ik in enumerate(self.kernels):
            print('{} compomnet for LMC'.format(i))
            ik.show_params()
            print('\n')
        
        print('lmc weights : {} \n'.format(self.weight))
        
        
    def Ksub(self, i, j, X1, X2=None):
        if X1.dim() == 1:
            X1 = X1.unsqueeze(dim=1)        
        if X2 is None:
            X2 = X1
        else:
            if X2.dim() == 1:
                X2 = X2.unsqueeze(dim=1)
                    
        # X has shape (data_points,input_dims)
        weight = torch.sum(self.weight[i] * self.weight[j], dim=1)  # Q
        kernels = torch.stack([kernel.K(X1,X2) for kernel in self.kernels], dim=2)  # NxMxQ
        return torch.tensordot(kernels, weight, dims=1)

    

    
    
    
    
    
#class IndependentMultiOutputKernel(MultiOutputKernel):
class IndependentMultiOutputKernel(nn.Module):
    def __init__(self, kernel_lists, output_dims=None, name="IMO"):
        if output_dims is None:
            output_dims = len(kernel_lists)
        #super(IndependentMultiOutputKernel, self).__init__(output_dims, name=name)
        #self.kernels = self._check_kernels(kernels, output_dims)
        self.kernels = kernel_lists #kernel list
        

    def __getitem__(self, key):
        return self.kernels[key]
    
    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        if i == j:
            return self.kernels[i](X1, X2)
        else:
            if X2 is None:
                X2 = X1
            return torch.zeros(X1.shape[0], X2.shape[0], device=config.device, dtype=config.dtype)


        
class MOSM(nn.Module):    
    def __init__(self, output_dims, input_dims, active_dims=None, name="mosm",dep=True , varying=False):
        #super(MOSM, self).__init__(output_dims, input_dims, active_dims, name)
        super(MOSM, self).__init__()
        
        self.dep = dep
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.varying = varying
    
        #magnitude_ = [0.5,0.5,0.5]        
        magnitude_ = [0.5,0.5,0.5]                
        magnitude = torch.tensor(magnitude_)
        mean_ = np.array([[0.1],[3.],[5.]])       
        mean = torch.tensor(mean_).float()
        
        #variance_ = [[0.1],[0.1],[0.1]]
        variance_ = [[0.1],[0.1],[0.1]] #v8
        variance = torch.tensor(variance_) 
        
        delay = .5*torch.rand(output_dims, input_dims)
        phase = .0*torch.rand(output_dims)
    
    
        self.magnitude = magnitude
        self.mean = mean
        self.variance = variance
        if 1 < output_dims:
            self.delay = delay
            self.phase = phase
            
        self.twopi = np.power(2.0*np.pi,float(self.input_dims)/2.0)
        self.transform = nn.Softplus()

        
        
    def reset_param(self):
        mean_ = np.array([[0.1],[3.],[5.]])
        mean_pertube = mean_ +  0.5*np.random.randn(*mean_.shape)  #v11
        mean_pertube = np.abs(mean_pertube)
        self.mean = torch.tensor(mean_pertube).float()               
        return 
        
        
        
        
        
        
    def show_params(self):
        print('self.magnitude {}'.format(self.magnitude))                
        print('self.mean {}'.format(self.mean))        
        print('self.variance {}'.format(self.variance))        
        print('self.delay {}'.format(self.delay))        
        print('self.phase {}'.format(self.phase))        
        print('\n')
        
        return 

        
    def Ksub(self, i, j, X1, X2=None):
        # X has shape (data_points,input_dims)
        #tau = self.distance(X1,X2)  # NxMxD

        if X1.dim() == 1:
            X1 = X1.unsqueeze(dim=1)        
        if X2 is None:
            X2 = X1
        else:
            if X2.dim() == 1:
                X2 = X2.unsqueeze(dim=1)
                    
        
        magnitude = self.transform(self.magnitude)
        mean = self.transform(self.mean)        
        variance = self.transform(self.variance)        
        delay = self.transform(self.delay)
        phase = self.transform(self.phase)


        tau = distance(X1,X2)  # NxMxD
       
        if i == j:
            variance = variance[i]
            alpha = magnitude[i]**2 * self.twopi * variance.prod().sqrt()  # scalar
        else:
            inv_variances = 1.0/(variance[i] + variance[j])  # D
            diff_mean = mean[i] - mean[j]  # D
            magnitude = magnitude[i]*magnitude[j]*torch.exp(-np.pi**2 * diff_mean.dot(inv_variances*diff_mean))  # scalar

            mean = inv_variances * (variance[i]*mean[j] + variance[j]*mean[i])  # D
            variance = 2.0 * variance[i] * inv_variances * variance[j]  # D
            delay = delay[i] - delay[j]  # D
            phase = phase[i] - phase[j]  # scalar
        
        
        if i == j:
            exp = torch.exp(-0.5*torch.tensordot(tau**2, variance, dims=1))  # NxM
            cos = torch.cos(2.0*np.pi * torch.tensordot(tau, mean[i], dims=1))  # NxM
            outs= alpha * exp * cos
        else:
            alpha = magnitude * self.twopi * variance.prod().sqrt()  # scalar
            exp = torch.exp(-0.5 * torch.tensordot((tau+delay)**2, variance, dims=1))  # NxM
            cos = torch.cos(2.0*np.pi * torch.tensordot(tau+delay, mean, dims=1) + phase)  # NxM            
            outs = alpha * exp * cos
            if self.dep == False:
                outs = torch.zeros_like(outs)

        return outs        
        
        
        
    
    