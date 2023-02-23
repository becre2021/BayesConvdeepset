# see example for wandb : https://github.com/wandb/examples
import wandb        
import os
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from attrdict import AttrDict
import argparse
import time 
from datetime import datetime
import random
import copy 
from collections import OrderedDict    




def build_metric():    
    #metric = AttrDict()
    #metric._setattr('_sequence_type', list)
    metric = OrderedDict()
    metric['train_ll'] = []
    metric['val_ll'] = []
    metric['itertime'] = []
    
    print(metric)
    return metric 
    
    
    
def build_density(image,mainidx,bgidx,share_channel=True,p=[0.1,0.1],intrain=True):

    #print('p {}'.format(p))
    
    ng1,ng2 = image.shape[-2:]
    
    mask_list = []
    mask_list2 = []
    
    bgidx_list = []
    mainidx_list = []
    
    for i_image in image:
        bg_idx = (i_image == 0).prod(0)
        main_idx = torch.ones_like(bg_idx) - bg_idx    
        bgidx_list.append(bg_idx[None,...].unsqueeze(dim=0))
        mainidx_list.append(main_idx[None,...].unsqueeze(dim=0))

        grid1,grid2 = torch.where(main_idx == 1)
        #---------------------------------------
        # shared mask
        #---------------------------------------        
        if share_channel:
            numel = len(grid1)
            tmp_zeros = torch.zeros(ng1,ng2)        
            chosenidx = sorted(np.random.choice(np.arange(numel),int(numel*p[0]) )) 
            tmp_zeros[(grid1[chosenidx],grid2[chosenidx])]= 1.
            mask = tmp_zeros.unsqueeze(dim=0).repeat(3,1,1)

            
            tmp_zeros2 = torch.zeros(ng1,ng2)        
            chosenidx2 = sorted(np.random.choice(np.arange(numel),int(numel*p[1]) )) 
            tmp_zeros2[(grid1[chosenidx2],grid2[chosenidx2])]= 1.
            mask2 = tmp_zeros2.unsqueeze(dim=0).repeat(3,1,1)
            
        #---------------------------------------
        # unshared mask
        #---------------------------------------                    
        else:
            numel = len(grid1)        
            tmp_zeros = []
            for k in range(3):
                k_tmp_zeros = torch.zeros(ng1,ng2)
                k_chosenidx = sorted(np.random.choice(np.arange(numel),int(numel*p[0]) )) #shared mask 
                k_tmp_zeros[(grid1[k_chosenidx],grid2[k_chosenidx])]= 1.
                #k_mask = 
                tmp_zeros.append(k_tmp_zeros.unsqueeze(dim=0))
            mask = torch.cat(tmp_zeros,dim=0)
        
        mask_list.append(mask[None,...])
        mask_list2.append(mask2[None,...])

    mask1,mask2 =  torch.cat(mask_list,dim=0),torch.cat(mask_list2,dim=0)        
    if intrain:
        mask2 = torch.clamp(mask1 + mask2,min=0.0,max=1.)
    return mask1,mask2




proposed_model_list = ['gpind','gpdep','gpdep2']
p_list,pratio_list = [0.01,0.05,0.10,0.20],np.array([0.4,0.3,0.2,0.1])
pratio_list = pratio_list/pratio_list.sum()

def pretrain_epoch_pair(dataset,
                         model,
                         opt,
                         lossfun,
                         current_iter=1,
                         total_iter = 500,
                         p_list = p_list,
                         eps=1e-4):    
    
    model.train()
    opt.zero_grad()
    loss_list,dataloss_list,regloss_list = [],[],[]
    
    #dict_keys(['imag', 'label', 'mainidx', 'bgidx'])        
    imag_list = dataset['imag']
    label_list = dataset['label']
    mainidx_list = dataset['mainidx']
    bgidx_list = dataset['bgidx']
    
    for imag,label,mainidx,bgidx in zip(imag_list,label_list,mainidx_list,bgidx_list):

        density_c,density_t = build_density(imag,mainidx,bgidx,share_channel=True, p=np.random.choice(p_list,2,p=pratio_list ) ,intrain=True)                
        signal = imag*density_c
        outs = model(density_c.cuda(),signal.cuda())                    
        dataloss, regloss = lossfun( outs, (density_t*imag).cuda(), masked_y = (density_t*mainidx).cuda(),reduce=True )        
        


        loss = regloss                
        loss_list.append(loss.cpu().data.numpy())                      
        dataloss_list.append(dataloss.cpu().data.numpy())                              
        regloss_list.append(regloss.cpu().data.numpy())        

        
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        if model.modelname in ['gpdep']:
            model.gpsampler.bound_hypparams()
        
        
    ntask = len(imag_list)
    mu_ll,std_ll = np.array(loss_list).mean(), (np.array(loss_list).std()/np.sqrt(ntask))
    mu_data,std_data = np.array(dataloss_list).mean(), (np.array(dataloss_list).std()/np.sqrt(ntask))
    mu_reg,std_reg = np.array(regloss_list).mean(), (np.array(regloss_list).std()/np.sqrt(ntask))
    
    return np.array((mu_ll,std_ll)).round(3),np.array((mu_data,std_data)).round(3),np.array((mu_reg,std_reg)).round(3)






proposed_model_list = ['gpind','gpdep','gpdep2']
def train_epoch_pair(dataset,
                     model,
                     opt,
                     lossfun,
                     current_iter=1,
                     total_iter = 500,
                     reglambda=1e-1,
                     p_list=p_list,                     
                     eps=1e-4):    
    
    
    
    model.train()
    opt.zero_grad()
    loss_list,dataloss_list,regloss_list = [],[],[]
    
    imag_list = dataset['imag']
    label_list = dataset['label']
    mainidx_list = dataset['mainidx']
    bgidx_list = dataset['bgidx']

    for imag,label,mainidx,bgidx in zip(imag_list,label_list,mainidx_list,bgidx_list):
        
        density_c,density_t = build_density(imag,mainidx,bgidx,share_channel=True, p=np.random.choice(p_list,2,p=pratio_list ) ,intrain=True)                
        signal = imag*density_c
        outs = model(density_c.cuda(),signal.cuda())                    
        dataloss, regloss = lossfun( outs, (density_t*imag).cuda(), masked_y = (density_t*mainidx).cuda(),reduce=True)        
        
        normalize = (dataloss/(regloss+eps)).abs().detach().clone()
        loss = dataloss + reglambda*normalize*regloss        
        
        loss_list.append(loss.cpu().data.numpy())                      
        dataloss_list.append(dataloss.cpu().data.numpy())                              
        regloss_list.append(regloss.cpu().data.numpy())        

        
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        if model.modelname in ['gpdep']:
            model.gpsampler.bound_hypparams()
        
        
    ntask = len(imag_list)
    mu_ll,std_ll = np.array(loss_list).mean(), (np.array(loss_list).std()/np.sqrt(ntask))
    mu_data,std_data = np.array(dataloss_list).mean(), (np.array(dataloss_list).std()/np.sqrt(ntask))
    mu_reg,std_reg = np.array(regloss_list).mean(), (np.array(regloss_list).std()/np.sqrt(ntask))
    
    #return (mu_ll,std_ll),(mu_data,std_data),(mu_reg,std_reg)
    return np.array((mu_ll,std_ll)).round(3),np.array((mu_data,std_data)).round(3),np.array((mu_reg,std_reg)).round(3)





def validate_epoch_pair(dataset,
                        model,
                        lossfun,
                        p_list=p_list,
                        eps=1e-4,
                        reduce=True):
    # large is better 
    model.eval()
    loss_list,regloss_list = [],[]
    
    imag_list = dataset['imag']
    label_list = dataset['label']
    mainidx_list = dataset['mainidx']
    bgidx_list = dataset['bgidx']

    for imag,label,mainidx,bgidx in zip(imag_list,label_list,mainidx_list,bgidx_list):     


        density_c,density_t = build_density(imag,mainidx,bgidx,share_channel=True, p=p_list ,intrain=False)                
        signal = imag*density_c
        outs = model(density_c.cuda(),signal.cuda())                    
        #dataloss, regloss = lossfun( outs, (density_t*imag).cuda(), masked_y = (density_t*mainidx).cuda(), intrain=False,reduce=True)        
        dataloss, regloss = lossfun( outs, (density_t*imag).cuda(), masked_y = (density_t*mainidx).cuda(),reduce=True)        

        
        loss_list.append( -dataloss.cpu().data.numpy())        
    
    if reduce:
        ntask = len(imag_list)
        avg_ll,std_ll = np.array(loss_list).mean(), (np.array(loss_list).std()/np.sqrt(ntask))
        return np.array((avg_ll,std_ll)).round(3)     
    else:
        return loss_list


    


    
    

# from models.test_gpsampler9_2d_latentnp import  Convcnp_2d, compute_baseline_loss
# from models.test_gpsampler9_2d_latentnp import  TDSP_Convnp_2d, compute_latentgp_loss_2d
from models.models_2d.test_gpsampler9_2d_latentnp import  Convcnp_2d, compute_baseline_loss
from models.models_2d.test_gpsampler9_2d_latentnp import  TDSP_Convnp_2d, compute_latentgp_loss_2d



def get_model(args):        


    if args.modelname == 'gpdep':       
        
        model = TDSP_Convnp_2d(in_dim=2,
                               out_dim=1,
                               num_mixture=args.nmixture,
                               num_channel=args.nchannel,
                               num_sample=args.nsample,
                               priorscale = args.priorscale,
                               use_weightnet=args.useweightnet,
                               tempering=args.tempering,
                               cnntype=args.cnntype,
                               solve=args.solve)
        
        lossfun = compute_latentgp_loss_2d     

        
    if args.modelname == 'base':
        model = Convcnp_2d(in_dim=2,
                           out_dim=1,
                            num_channel=args.nchannel,
                            cnntype=args.cnntype)        
        lossfun = compute_baseline_loss
        
        
        
      
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay = args.weightdecay)                
    return model,opt,lossfun

   
    

def make_expinfo(args):
    args_dict = args.__dict__
    model_info = ''
    #tasktypesingletask_testtypeextra_depTrue

    for ith_keys in args_dict:
        if ith_keys in ['msg','printfreq','gentime','tasktype','testtype','dep','useweightnet','kerneltype','solve','nchannel']:
            pass
        else:
            model_info += ith_keys + str(args_dict[ith_keys]) + '_'
            
    return model_info
    

    
    
        
def main():

    #----------------------------------------
    # argin
    #----------------------------------------    
    #parser = argparse.ArgumentParser(description='exp1-synthetics-singletask')
    parser = argparse.ArgumentParser(description='exp3-imag-completion')
    
    # model configuration
    parser.add_argument('--modelname',type=str, default='gpdep') #--modelname gpdep2
    parser.add_argument('--nmixture',type=int, default=3)  #--numQ 3  to check whether numQ is 
    parser.add_argument('--nchannel',type=int, default=1)  #--numQ 3  to check whether numQ is 
    parser.add_argument('--nsample', type=int, default=5)
    parser.add_argument('--useweightnet', action='store_true')
    parser.add_argument('--tempering', type=float, default=1e-1)
    parser.add_argument('--cnntype', type=str, default='deep')
    parser.add_argument('--kerneltype', type=str, default='sm')
    parser.add_argument('--priorscale', type=float, default=0.1)
    parser.add_argument('--anphdim', type=int, default=32)
    
    parser.add_argument('--solve', type=str, default='cg')    
    parser.add_argument('--convcnpinitl', type=float, default=0.1)
    
    

    # training configuration
    parser.add_argument('--nepoch', type=int, default=500) #iterations
    parser.add_argument('--lr', type=float, default= 1e-3) #iterations
    parser.add_argument('--weightdecay', type=float, default= 1e-4)
    parser.add_argument('--reglambda',type=float, default=.5) #--modelname gpdep2

    
    # exp configuration
    parser.add_argument('--tasktype', type=str, default='celeba32') # sin3,sin4,mogp,lmc,convk,
    #parser.add_argument('--testtype', type=str, default='extra') # inter,extra
    parser.add_argument('--dep', action='store_true')
    parser.add_argument('--datav',type=int, default=1) #--modelname gpdep2
    parser.add_argument('--runv',type=int, default=1) #--modelname gpdep2    
    parser.add_argument('--printfreq',type=int, default=20) #--modelname gpdep2
    parser.add_argument('--randomseed',type=int,default=1111)    
    parser.add_argument('--msg',type=str,default='none')
    parser.add_argument('--gentime',type=str,default='none')


    args = parser.parse_args()   
    gentime = datetime.today().strftime('%m%d%H%M')    
    args.gentime = gentime
    
    
    if args.modelname in ['anp','cnp','base']:
        args.nsample = 1

    torch.manual_seed(args.randomseed)
    random.seed(args.randomseed)
    np.random.seed(args.randomseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



    expinfo = make_expinfo(args)
    print('#'+'-'*100)
    print(expinfo[:-1])
    print('#'+'-'*100)
    print('\n')
    
    
    
    #-----------------------------
    # dataset path and save path
    #-----------------------------            
    # dataset path
    #dataset_path = './regression_singlechannel/'
    dataset_path = './download/processed_{}/'.format(args.tasktype)
    save_dir = './task_completion_imag_new/param{}_datav{}_lr{}/'.format(args.tasktype,args.datav,args.lr)
    
    
    os.makedirs(save_dir) if not os.path.exists(save_dir) else 1        
    save_filename = expinfo[:-1]
    saved_modelparam_path = save_dir + save_filename
    saved_modelparam_path += '_' + 'gentime'+args.gentime
    #aved_modelparam_path += '.pth'

    
    #-----------------------------
    # wandbi name assignment
    #-----------------------------    
    config = copy.deepcopy(args.__dict__)
    wandb.init( project="bconvcnp-imag-new1",
                notes="datav{}, msg:{}".format(args.datav,args.msg),            
                config = config,           
                reinit= True)
    wandb.run.name = expinfo[:-1]


    
    #---------------------------
    # get model and metric table
    #---------------------------
    model,opt,lossfun = get_model(args)
    metric = build_metric()    
    print(model)

    
    #-----------------------------------------------------
    # pretrain for initilaization
    #-----------------------------------------------------
    if model.modelname in ['gpdep']:
        #------------------------------------------
        # pretrain
        #------------------------------------------    
        pre_opt = torch.optim.Adam(model.gpsampler.parameters(), lr=args.lr,weight_decay = args.weightdecay)                
        numrep = 3
        for k in range(1,numrep+1):
            pretrain_idx = np.random.choice(np.arange(1,args.nepoch + 1),args.nepoch//5,replace=False)
            
            for i,idx in enumerate(pretrain_idx):   
                epoch_start = time.time()        
                trainset_path= dataset_path + 'set_{}'.format(idx)               
                pretrain_set = torch.load(trainset_path + '.db')

                _,_,pretr_reg = pretrain_epoch_pair(pretrain_set,
                                            model,
                                            pre_opt,
                                            lossfun, 
                                            current_iter = idx,
                                            total_iter = len(pretrain_idx))      

                epoch_end = time.time()
                takentime = np.array(epoch_end - epoch_start).round(3)

                if (i%args.printfreq) ==0 or (i== 1) or (i==len(pretrain_idx)):                
                    print('pretrain [{}/{}]| epochs [{}/{}] | pretr loss ({:.3f},{:.3f}) \t {:.3f}(sec)'.format(k,numrep,i,len(pretrain_idx),pretr_reg[0],pretr_reg[1],takentime ) )                       
            print('\n')

        argsdict = AttrDict()
        for ith_key in args.__dict__:
            argsdict[ith_key]=args.__dict__[ith_key]            

        state_dict = copy.deepcopy(model.state_dict())
        opt_dict = copy.deepcopy(opt.state_dict())                

        saved_dict = {'epoch': i + 1,
                     'state_dict': state_dict,
                     'optimizer': opt_dict,
                     'args_dict':argsdict,                             
                     'metric_dict':metric
                     }
        
    
    #------------------------------------------
    #start training
    #------------------------------------------
    
    #print(model)
    #opt = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay = args.weightdecay)    :  #not ncessarnily needed               
    wandb.run.save()
    wandb.watch(model)
    
    torch.autograd.set_detect_anomaly(True)    
    best_loss = -np.inf    
    learningtime = 0.0
    args_dict_history = {}
    obsepochs = 20                     
    for i in range(1,args.nepoch + 1):   

        epoch_start = time.time()
        
        # get_train set
        #trainset_path= dataset_path + 'syndata_{}_v{}/dep{}_{}_{}'.format(args.tasktype,args.datav, args.dep, args.testtype, i)      
        trainset_path= dataset_path + 'set_{}'.format(i)               
        #train_set = torch.load(trainset_path + '.db')['imag']
        train_set = torch.load(trainset_path + '.db')
        
        # train dataset per epoch
        tr_loss, tr_data, tr_reg = train_epoch_pair(train_set,
                                                    model,
                                                    opt,
                                                    lossfun, 
                                                    current_iter = i ,
                                                    total_iter = args.nepoch,
                                                    reglambda = args.reglambda )        
        epoch_end = time.time()
        
        


        
        if i%args.printfreq ==0 or i== 1:                
            #validation_path = dataset_path + 'valset_{}'.format(-32)                           
            validation_path = dataset_path + 'valset_{}'.format(-16)                           
            
            valid_set = torch.load(validation_path + '.db')            
            # validate trained model
            
            
            #val_data = validate_epoch_pair( valid_set,model,lossfun )            
            #p_list_small = = [0.05,0.2,0.4]
            print('-'*100)
            print('validation')
            for kk,i_p in enumerate(p_list):
                if kk == 0:
                    #valloss_list = validate_epoch_pair( valid_set,model,lossfun,p_list = [i_p,p_list[-1]] )                                   
                    valloss_list = validate_epoch_pair( valid_set,model,lossfun,p_list = [i_p,0.5] )                                   
                    
                else:
                    #valloss_list += validate_epoch_pair( valid_set,model,lossfun,p_list = [i_p,p_list[-1]] )                                   
                    valloss_list += validate_epoch_pair( valid_set,model,lossfun,p_list = [i_p,0.5] )                                   

                    
            avg_ll,std_ll = np.array(valloss_list).mean(),np.array(valloss_list).std()
            val_data = np.array((avg_ll,std_ll)).round(3)  
            
            # --------------------------------------
            # record metric
            # --------------------------------------            
            metric['train_ll'].append(tr_data.tolist())            
            metric['val_ll'].append(val_data.tolist())                        
            takentime = np.array(epoch_end - epoch_start).round(3)
            metric['itertime'].append( takentime )            
            learningtime += takentime

                     


            if best_loss < val_data[0]:
                best_loss = val_data[0]

                argsdict = AttrDict()
                for ith_key in args.__dict__:
                    argsdict[ith_key]=args.__dict__[ith_key]            
                
                #attrmetric = AttrDict()
                #for ith_key in metric.__dict__:
                #    attrmetric[ith_key]=metric.__dict__[ith_key]                            
                saved_epoch = i
                state_dict = copy.deepcopy(model.state_dict())
                opt_dict = copy.deepcopy(opt.state_dict())                
                
                
                if i <= 20:
                    obsepochs = 20                 
                if (i <= 40) and (i > 20):
                    obsepochs = 40 
                if (i <= 60) and (i > 40):
                    obsepochs = 60 
                if (i <= 80) and (i > 60):
                    obsepochs = 80                     
                if (i <= 100) and (i > 80):
                    obsepochs = 100      

                args_dict_history[obsepochs] = state_dict
            else:
                args_dict_history[obsepochs] = args_dict_history[obsepochs-20]
                
                #pass
 
            saved_dict = {'epoch': i + 1,
                         'best_acc_top1': best_loss,                         
                         'state_dict': state_dict,
                         'optimizer': opt_dict,
                         'args_dict':argsdict,                             
                         'state_dict_history':args_dict_history,                                                       
                         'metric_dict':metric
                         }
            #torch.save(saved_dict,saved_modelparam_path)
            

       
            current_savedfilename = saved_modelparam_path  + '.pth'                
            torch.save(saved_dict,current_savedfilename)
            torch.cuda.empty_cache()
                                
            
#             print('epochs [{}/{}] | tr loss ({:.3f},{:.3f}), val data ({:.3f},{:.3f}), \t saved_param: {} saved at epochs {} with best val loss {:.3f} \t {:.3f}(sec)'.format(i,args.nepoch,tr_loss[0],tr_loss[1],val_data[0],val_data[1],saved_modelparam_path,saved_epoch,best_loss,epoch_end-epoch_start) )                       
            print('epochs [{}/{}] | tr loss ({:.3f},{:.3f}), val data ({:.3f},{:.3f}) with best val loss {:.3f} \t {:.3f}(sec)'.format(i,args.nepoch,tr_loss[0],tr_loss[1],val_data[0],val_data[1],best_loss,takentime) )     
            print('saved at {}'.format(current_savedfilename))
                

            #wandbi tarinining check
            wandb.log({"tr_loss-intrain": tr_loss[0],
                       'tr_data-intrain':tr_data[0],                       
                       'tr_reg-intrain':tr_reg[0],
                       "val_data-intrain": val_data[0],
                       'current_epoch':i,
                       'learningtime':learningtime,
                      })


        torch.cuda.empty_cache()

    return 







if __name__ == "__main__":
    main()


 