import numpy as np
import torch
import torch.nn as nn

from torch import triangular_solve,cholesky
from torch.distributions.multivariate_normal import MultivariateNormal as MVN
from torch.distributions.normal import Normal

from ..datasets.dataset_multitask_1d import Matern,LMC,MOSM
from ..datasets.dataset_multitask_1d import sigma_eps_std
from ..datasets.dataset_multitask_1d import compute_multioutput_K
from .utils import gaussian_logpdf







zitter = 1e-6
class GP_Baseline(nn.Module):
    def __init__(self,in_dims=1,out_dims=1,num_channels=3,kerneltype='kernel',dep=True):
        super(GP_Baseline,self).__init__()
        
        self.in_dims = in_dims 
        self.out_dims = out_dims
        self.num_channels = num_channels
        
        self.kerneltype = kerneltype
        self.dep = dep
        self.noise_std = sigma_eps_std
        if kerneltype == 'mosm':
            self.kernel = MOSM(output_dims=self.num_channels, input_dims=self.in_dims,dep=self.dep)
        elif kerneltype == 'lmc':
            kernel_list = [Matern(nu=0.5,sigma=0.9, l=.5,input_dims=1),
                           Matern(nu=1.5,sigma=0.9, l=.5,input_dims=1),
                           Matern(nu=2.5,sigma=0.9, l=.5,input_dims=1)]            
            self.kernel = LMC(kernel_list=kernel_list,output_dims=self.num_channels, input_dims=1,Rq=1,dep=self.dep)
        else :
            print('current kerneltype: {}'.format(kerneltype))

        
    def forward(self,context_x,context_y,target_x):
        """ 1d inputs 
        context_x: (nb,ndata,nchannel)
        context_y: (nb,ndata,nchannel)        
        target_x:  (nb,ndata,nchannel)        
        """
        
        
        nb,ntarget,nchannel = target_x.size()

        b_mean,b_cov = [],[]
        b_mean2,b_cov2 = [],[]        
        for i_context_x,i_context_y,i_target_x in zip(context_x,context_y,target_x):    
            K_cc = compute_multioutput_K(self.kernel.Ksub,i_context_x)
            #print('K_cc.size(): {}'.format(K_cc.size()))
            
            K_ct = compute_multioutput_K(self.kernel.Ksub,i_context_x,i_target_x)

            i_c_y = i_context_y.transpose(0,1).reshape(-1,1)
            L = torch.linalg.cholesky(K_cc)
            A = triangular_solve(K_ct,L,upper=False)[0]   #A = trtrs(L, k_xs)
            V = triangular_solve(i_c_y,L,upper=False)[0]
            mean_f = torch.mm(torch.transpose(A, 0, 1), V)        
            
            # diag cov
            var_f1 = compute_multioutput_K(self.kernel.Ksub,i_target_x).diag()         
            var_f2 = torch.sum(A * A, 0)
            var_f = (var_f1 - var_f2).reshape(-1,1) 
            b_mean.append(mean_f.view(nchannel,-1).T[None,:,:])
            b_cov.append(var_f.view(nchannel,-1).T[None,:,:])

            # full cov
            var_f11 = compute_multioutput_K(self.kernel.Ksub,i_target_x)                        
            var_f22 = torch.mm(A.t(), A)
            cov_f = (var_f11 - var_f22) 
            b_mean2.append(mean_f[None,:])
            b_cov2.append(cov_f[None,:,:])                
                

        pred_mean1 = torch.cat(b_mean,dim=0)    
        pred_std1 = torch.cat(b_cov,dim=0)
        pred_std1 = torch.clamp(pred_std1,min=1e-8,max=None).sqrt()        
        pred_std1 += self.noise_std*torch.ones_like(pred_std1)
        #print('self.noise_std*torch.ones_like(pred_std1)')
        #print(self.noise_std*torch.ones_like(pred_std1))
        
        pred_mean2 = torch.cat(b_mean2,dim=0)    
        pred_cov2 = torch.cat(b_cov2,dim=0)
        pred_cov2 += (self.noise_std**2+zitter)*torch.eye(pred_cov2.size(1))[None,:,:]
        #print('(self.noise_std**2)*torch.eye(pred_cov2.size(1))[None,:,:]')
        #print((self.noise_std**2)*torch.eye(pred_cov2.size(1))[None,:,:])

        return pred_mean1,pred_std1,pred_mean2,pred_cov2

    
    
    def compute_logprob_batch(self,context_x,context_y,target_x,target_y):
        """ 1d inputs assumptions
        """
        
        #print('context_x.size(),context_y.size(),target_x.size(),target_y.size()')
        #print(context_x.size(),context_y.size(),target_x.size(),target_y.size())
        
        diag_mean,diag_std,full_mean,full_cov = self.forward(context_x,context_y,target_x)
        
        #print('diag_mean.size(),diag_std.size(),full_mean.size(),full_cov.size()')
        #print(diag_mean.size(),diag_std.size(),full_mean.size(),full_cov.size())        
        
        #logprob for full cov 
        mvn = MVN(full_mean.squeeze(),full_cov)
        target_y_= target_y.permute(0,2,1).reshape(target_y.size(0),-1)
        full_logprob = mvn.log_prob(target_y_).mean()

        #logprob for diag std         
        normal = Normal(loc=diag_mean, scale=diag_std)    
        diag_logprob = normal.log_prob(target_y).sum(dim=(-2,-1)).mean()

        #print('full_logprob,diag_logprob')
        #print(full_logprob,diag_logprob)
        #print('')        
        return full_logprob,diag_logprob
    
    
    
    
#     def compute_logprob_task(self,set_dict_epoch):    
#         full_logprob = []
#         diag_logprob = []    
#         ntask = set_dict_epoch['context_x'].size(0)    
#         for ith in range(ntask):        
#             context_x,context_y = set_dict_epoch['context_x'][ith],set_dict_epoch['context_y'][ith]
#             target_x,target_y = set_dict_epoch['target_x'][ith],set_dict_epoch['target_y'][ith]        
#             full_logprob_nb,diag_logprob_nb = self.compute_logprob_batch(context_x,context_y,target_x,target_y)

#             full_logprob.append(full_logprob_nb.cpu().data.numpy())    
#             diag_logprob.append(diag_logprob_nb.cpu().data.numpy())    

#         avg_f_ll,std_f_ll = np.array(full_logprob).mean().round(2),(np.array(full_logprob).std()/np.sqrt(ntask)).round(2)
#         avg_d_ll,std_d_ll = np.array(diag_logprob).mean().round(2),(np.array(diag_logprob).std()/np.sqrt(ntask)).round(2)    
#         return (avg_f_ll,std_f_ll),(avg_d_ll,std_d_ll)
    
    
    def compute_logprob_task(self,dataset_list,normalization=False):    
        full_logprob = []
        diag_logprob = []    
        ntask = len(dataset_list)   
        #for ith_data in range(dataset_list):        
        for ith_data in dataset_list:                    
            #context_x,context_y = set_dict_epoch['context_x'][ith],set_dict_epoch['context_y'][ith]
            #target_x,target_y = set_dict_epoch['target_x'][ith],set_dict_epoch['target_y'][ith]        
            xc,yc,xt,yt,xf,yf = ith_data
                
            #full_logprob_nb,diag_logprob_nb = self.compute_logprob_batch(context_x,context_y,target_x,target_y)
            full_logprob_nb,diag_logprob_nb = self.compute_logprob_batch(xc,yc,xt,yt)                        
            if normalization:
                full_logprob_nb /= yt.size(-1)*yt.size(-2)
                diag_logprob_nb /= yt.size(-1)*yt.size(-2)
                
            full_logprob.append(full_logprob_nb.cpu().data.numpy())    
            diag_logprob.append(diag_logprob_nb.cpu().data.numpy())    

        avg_f_ll,std_f_ll = np.array(full_logprob).mean().round(2),(np.array(full_logprob).std()/np.sqrt(ntask)).round(2)
        avg_d_ll,std_d_ll = np.array(diag_logprob).mean().round(2),(np.array(diag_logprob).std()/np.sqrt(ntask)).round(2)    
        return (avg_f_ll,std_f_ll),(avg_d_ll,std_d_ll)

    
    
    
        
#     def forward(self,context_x,context_y,target_x, diag = True):
#         nb,ntarget,nchannel = target_x.size()

#         b_mean,b_var = [],[]
#         for i_context_x,i_context_y,i_target_x in zip(context_x,context_y,target_x):    
#             K_cc = compute_multioutput_K(self.kernel.Ksub,i_context_x)
#             print('K_cc.size(): {}'.format(K_cc.size()))
            
#             K_ct = compute_multioutput_K(self.kernel.Ksub,i_context_x,i_target_x)

#             i_c_y = i_context_y.transpose(0,1).reshape(-1,1)
#             L = cholesky(K_cc)
#             A = triangular_solve(K_ct,L,upper=False)[0]   #A = trtrs(L, k_xs)
#             V = triangular_solve(i_c_y,L,upper=False)[0]
#             mean_f = torch.mm(torch.transpose(A, 0, 1), V)        
            
#             if diag:
#                 var_f1 = compute_multioutput_K(self.kernel.Ksub,i_target_x).diag()         
#                 var_f2 = torch.sum(A * A, 0)
#                 var_f = (var_f1 - var_f2).reshape(-1,1) 
#                 b_mean.append(mean_f.view(nchannel,-1).T[None,:,:])
#                 b_var.append(var_f.view(nchannel,-1).T[None,:,:])
                
#             else:
#                 var_f1 = compute_multioutput_K(self.kernel.Ksub,i_target_x)                        
#                 var_f2 = torch.mm(A.t(), A)
#                 var_f = (var_f1 - var_f2) 

#                 print('mean_f.shape,var_f.shape')                
#                 print(mean_f.shape,var_f.shape)
                
#                 b_mean.append(mean_f[None,:])
#                 b_var.append(var_f[None,:,:])
                
#         if diag:
#             pred_mean = torch.cat(b_mean,dim=0)    
#             pred_std = torch.cat(b_var,dim=0).sqrt()
#         else:
#             pred_mean = torch.cat(b_mean,dim=0)    
#             pred_std = torch.cat(b_var,dim=0)

#         return pred_mean,pred_std














# def gp_predict_batch(context_x,context_y,target_x, Ksub, diag = True):

#     nb,ntarget,nchannel = target_x.size()
    
#     b_mean,b_var = [],[]
#     for i_context_x,i_context_y,i_target_x in zip(context_x,context_y,target_x):    
#         K_cc = compute_multioutput_K(Ksub,i_context_x)
#         K_ct = compute_multioutput_K(Ksub,i_context_x,i_target_x)
        
#         i_c_y = i_context_y.transpose(0,1).reshape(-1,1)
#         L = cholesky(K_cc)
#         A = triangular_solve(K_ct,L,upper=False)[0]   #A = trtrs(L, k_xs)
#         V = triangular_solve(i_c_y,L,upper=False)[0]
#         mean_f = torch.mm(torch.transpose(A, 0, 1), V)        
#         if diag:
#             var_f1 = compute_multioutput_K(Ksub,i_target_x).diag()         
#             var_f2 = torch.sum(A * A, 0)
#             var_f = (var_f1 - var_f2).reshape(-1,1) 
#         else:
#             var_f1 = compute_multioutput_K(Ksub,i_target_x)                        
#             var_f2 = torch.mm(A.t(), A)
#             var_f = (var_f1 - var_f2) 
            
#         b_mean.append(mean_f.view(nchannel,-1).T[None,:,:])
#         b_var.append(var_f.view(nchannel,-1).T[None,:,:])
        
#     b_mean = torch.cat(b_mean,dim=0)    
#     if diag:
#         b_std = torch.cat(b_var,dim=0).sqrt()
#     else:
#         raise NotImplementedError
        
#     return b_mean,b_std
        
    
    
    
    
# def validate_oracle_epochs_with_dict(set_dict_epoch,Ksub=None,test_range=None):
#     # large is better 
#     #model.eval()
#     likelihoods = []
    
#     ntask = set_dict_epoch['context_x'].size(0)    
#     for ith in range(ntask):        
        
#         context_x,context_y = set_dict_epoch['context_x'][ith],set_dict_epoch['context_y'][ith]
#         target_x,target_y = set_dict_epoch['target_x'][ith],set_dict_epoch['target_y'][ith]
#         #y_mean,y_std = gp_predict_batch(context_x.cuda(),context_y.cuda(),target_x.cuda(),Ksub=Ksub)
#         y_mean,y_std = gp_predict_batch(context_x,context_y,target_x,Ksub=Ksub)
        
#         obj = gaussian_logpdf(target_y, y_mean, y_std, 'batched_mean')        
#         #obj = gaussian_logpdf(target_y.cuda(), y_mean, y_std, 'batched_mean')        
#         likelihoods.append(obj.cpu().data.numpy())        
                
#     avg_ll,std_ll = np.array(likelihoods).mean().round(2),(np.array(likelihoods).std()/np.sqrt(ntask)).round(2)
#     return avg_ll,std_ll       



