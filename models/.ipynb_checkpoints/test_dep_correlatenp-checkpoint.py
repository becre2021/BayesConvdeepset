from .test_cnnmodels import get_cnnmodels
from .test_ind_correlatenp import eps,num_basis,num_fourierbasis,loglik_err
from .test_ind_correlatenp import ICGP_Convnp,ConvDeepset,compute_loss_gp
from .test_gpsampler7  import Spikeslab_GPsampler
    
    

class DCGP_Convnp(ICGP_Convnp):
    def __init__(self,in_dims: int = 1,out_dims:int = 1, num_channels: int=3,
                      cnntype='shallow',num_postsamples=10,init_lengthscale=1.0):
        super(DCGP_Convnp,self).__init__(in_dims,out_dims, num_channels,
                                          cnntype,num_postsamples,init_lengthscale)
        
        self.modelname = 'gpdep'        

        
        #print('Neuralspikeslab prior ')    
        self.gpsampler =  Spikeslab_GPsampler(in_dims=in_dims,
                                            out_dims=out_dims,
                                            num_channels=num_channels, 
                                            num_fourierbasis = num_fourierbasis,
                                            scales=init_lengthscale,
                                            loglik_err=loglik_err,
                                            points_per_unit=self.cnn.points_per_unit,
                                            multiplier=self.cnn.multiplier,
                                            useweightnet = True)

        
       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    