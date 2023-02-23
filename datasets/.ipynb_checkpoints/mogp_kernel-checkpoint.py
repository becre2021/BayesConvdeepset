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
    
    output_dims = x1.size(1)
    K = []
    k_dict = {}
    for i in range(output_dims):
        k_i = []
        for j in range(output_dims):
            if i==j:
                #ksub_ij = mosm.Ksub(i,i,x1[:,i])           
                ksub_ij = k_sub(i,i,x1[:,i],x2[:,i])            
                if ksub_ij.is_cuda:
                    ksub_ij += eps*torch.eye(x1[:,i].size(0)).cuda()
                else:
                    ksub_ij += eps*torch.eye(x1[:,i].size(0))
            elif i<j:
                #ksub_ij = mosm.Ksub(i,j,x1[:,i],x1[:,j])
                ksub_ij = k_sub(i,j,x1[:,i],x2[:,j])                
                k_dict[(i,j)] = ksub_ij
            else:
                ksub_ij = k_dict[(j,i)].T
            k_i.append(ksub_ij)
        K.append(torch.cat(k_i,dim=-1))
    
    del k_dict
    K = torch.cat(K,dim=0)
    return K


def build_mogp_kernel(kernel_type='mosm',nchannels=3):
    # nchannels = 3
    #kernel_type = 'mosm'

    if kernel_type == 'mosm':
        kernel = MOSM(output_dims=nchannels, input_dims=1)        
        #kernel_list = []
        #kernel = MOSM(output_dims=nchannels, input_dims=1)            
        pass        
    elif kernel_type == 'lmc':
        pass        
    elif kernel_type == 'csm':
        pass
    elif kernel_type == 'conv':
        pass
    else:
        pass

    return kernel






class Matern(nn.Module):
    def __init__(self, nu=0.5, input_dims=None, active_dims=None, name="Matérn"):
        super(Matern, self).__init__()

        if nu not in [0.5, 1.5, 2.5]:
            raise ValueError("nu parameter must be 0.5, 1.5, or 2.5")

        l = 0.1 +.9*torch.rand(input_dims)
        sigma = 1.+2*torch.rand(1)

        self.nu = nu
        #self.l = Parameter(l, lower=1e-6)
        #self.sigma = Parameter(sigma, lower=1e-6)
        self.l = nn.Parameter(l)
        self.sigma = nn.Parameter(sigma)

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
    def __init__(self, output_dims, input_dims, active_dims=None, name="MOSM"):
        #super(MOSM, self).__init__(output_dims, input_dims, active_dims, name)
        super(MOSM, self).__init__()
        

        self.input_dims = input_dims
        self.output_dims = output_dims
        
        # TODO: incorporate mixtures?
        # TODO: allow different input_dims per channel
        magnitude = 0.1+0.9*torch.rand(output_dims)
        mean = 0.1+0.9*torch.rand(output_dims, input_dims)
        variance = 0.005*torch.rand(output_dims, input_dims)
        delay = 0*torch.rand(output_dims, input_dims)
        phase = .1*torch.rand(output_dims)
                
        #self.magnitude = Parameter(magnitude, lower=config.positive_minimum)
        #self.mean = Parameter(mean, lower=config.positive_minimum)
        #self.variance = Parameter(variance, lower=config.positive_minimum)
        self.magnitude = nn.Parameter(magnitude)
        self.mean = nn.Parameter(mean)
        self.variance = nn.Parameter(variance)
        
        
        if 1 < output_dims:
            #self.delay = Parameter(delay)
            #self.phase = Parameter(phase)
            self.delay = nn.Parameter(delay)
            self.phase = nn.Parameter(phase)
            
            
        self.twopi = np.power(2.0*np.pi,float(self.input_dims)/2.0)
        self.transform = nn.Softplus()

        
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
            return alpha * exp * cos
        else:
            alpha = magnitude * self.twopi * variance.prod().sqrt()  # scalar
            exp = torch.exp(-0.5 * torch.tensordot((tau+delay)**2, variance, dims=1))  # NxM
            cos = torch.cos(2.0*np.pi * torch.tensordot(tau+delay, mean, dims=1) + phase)  # NxM
            return alpha * exp * cos

        
class LMC(nn.Module):
    def __init__(self, kernel_list, output_dims, input_dims, Q=None, Rq=1, name="LMC"):
        super(LMC, self).__init__()

        if Q is None:
            Q = len(kernel_list)
        weight = torch.rand(output_dims, Q, Rq)
        print(weight.size())
        self.weight = nn.Parameter(weight)

        #kernels = self._check_kernels(kernels, Q)        
        self.kernels = kernel_list
        self.transform = nn.Softplus()

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

    
