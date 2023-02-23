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


    

def merge_allset_1d(xc,yc,xt,yt):
    xct = torch.cat([xc,xt],dim=1)
    yct = torch.cat([yc,yt],dim=1)
    xct,s_idx =torch.sort(xct,dim=1)

    if len(xc.size()) == 3:
        yct = torch.gather(yct,1,s_idx)    
    if len(xc.size()) == 4:
        yct = torch.gather(yct,1,s_idx[:,:,0,:])
    return xct,yct





proposed_model_list = ['gpind','gpdep','gpdep2']
#def train_epoch_pair(batch_dataset_pair,model,opt,lossfun,current_iter=1,total_iter = 500,eps=1e-4,start_temp=1e1,final_temp=1e-1):    

def pretrain_epoch_pair(batch_dataset_pair,
                         model,
                         opt,
                         lossfun,
                         current_iter=1,
                         total_iter = 500,
                         eps=1e-4):    
    
    model.train()
    opt.zero_grad()
    loss_list,dataloss_list,regloss_list = [],[],[]
    
    for dataset_pair in batch_dataset_pair:
        xc,yc,xt,yt = dataset_pair[:4]     
                
        
        if model.modelname in proposed_model_list and len(xc.size()) == 3:        
            xc,xt=xc.unsqueeze(dim=-2),xt.unsqueeze(dim=-2)        
        #----------------------------------------------------------------    
        #is necessar?
        xt,yt = merge_allset_1d(xc,yc,xt,yt)
        #----------------------------------------------------------------    
        #predict & train models
        outs = model(xc.cuda(),yc.cuda(),xt.cuda())    
        dataloss,regloss = lossfun( outs, yt.cuda(), intrain=True) 
        #dataloss,regloss = lossfun( outs, yt.cuda(), intrain= False) 
                
        loss = regloss
        
        
        loss_list.append(loss.cpu().data.numpy())                      
        dataloss_list.append(dataloss.cpu().data.numpy())                              
        regloss_list.append(regloss.cpu().data.numpy())        

        
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        if model.modelname in ['gpdep']:
            model.gpsampler.bound_hypparams()
        
        
    ntask = len(batch_dataset_pair)
    mu_ll,std_ll = np.array(loss_list).mean(), (np.array(loss_list).std()/np.sqrt(ntask))
    mu_data,std_data = np.array(dataloss_list).mean(), (np.array(dataloss_list).std()/np.sqrt(ntask))
    mu_reg,std_reg = np.array(regloss_list).mean(), (np.array(regloss_list).std()/np.sqrt(ntask))
    
    #return (mu_ll,std_ll),(mu_data,std_data),(mu_reg,std_reg)
    return np.array((mu_ll,std_ll)).round(3),np.array((mu_data,std_data)).round(3),np.array((mu_reg,std_reg)).round(3)




#1e1,1e0
def train_epoch_pair(batch_dataset_pair,
                     model,
                     opt,
                     lossfun,
                     current_iter=1,
                     total_iter = 500,
                     reglambda = 0.1,
                     eps=1e-4):    
    
    
    
    model.train()
    opt.zero_grad()
    loss_list,dataloss_list,regloss_list = [],[],[]
    
    for dataset_pair in batch_dataset_pair:
        xc,yc,xt,yt = dataset_pair[:4]     
        
        
        if model.modelname in proposed_model_list and len(xc.size()) == 3:        
            xc,xt=xc.unsqueeze(dim=-2),xt.unsqueeze(dim=-2)        
        #is necessar?
        xt,yt = merge_allset_1d(xc,yc,xt,yt)
        #----------------------------------------------------------------    
        #predict & train models
        outs = model(xc.cuda(),yc.cuda(),xt.cuda())    
        dataloss,regloss = lossfun( outs, yt.cuda(), intrain=True) 
        
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
        
        
    ntask = len(batch_dataset_pair)
    mu_ll,std_ll = np.array(loss_list).mean(), (np.array(loss_list).std()/np.sqrt(ntask))
    mu_data,std_data = np.array(dataloss_list).mean(), (np.array(dataloss_list).std()/np.sqrt(ntask))
    mu_reg,std_reg = np.array(regloss_list).mean(), (np.array(regloss_list).std()/np.sqrt(ntask))
    
    #return (mu_ll,std_ll),(mu_data,std_data),(mu_reg,std_reg)
    return np.array((mu_ll,std_ll)).round(3),np.array((mu_data,std_data)).round(3),np.array((mu_reg,std_reg)).round(3)


def validate_epoch_pair(batch_dataset_pair,model,lossfun):
    # large is better 
    model.eval()
    loss_list,regloss_list = [],[]
    
    for dataset_pair in batch_dataset_pair:
        xc,yc,xt,yt,xf,yf = dataset_pair        

        if model.modelname in proposed_model_list and len(xc.size()) == 3:        
            xc,xt=xc.unsqueeze(dim=-2),xt.unsqueeze(dim=-2)        
        
        outs = model(xc.cuda(),yc.cuda(),xt.cuda())    
        dataloss,regloss = lossfun( outs, yt.cuda(), intrain=True)         
        loss_list.append( -dataloss.cpu().data.numpy())        
        
    ntask = len(batch_dataset_pair)
    avg_ll,std_ll = np.array(loss_list).mean(), (np.array(loss_list).std()/np.sqrt(ntask))
    return np.array((avg_ll,std_ll)).round(3)     
    


    




from models.test_cnp import RegressionANP, RegressionCNP
from models.test_baseline_1d import Convcnp,compute_loss_baseline
from models.test_baselinelatent_1d import Convcnp_latent,compute_loss_baselinelatent
from models.test_gpsampler8_1d_latentnp import TDSP_Convnp_1d,compute_latentgp_loss_1d
#from models.test_gpsampler8_1d_latentnp_v2 import TDSP_Convnp_1d,compute_latentgp_loss_1d #failed
        

def get_model(args):   

    if args.modelname == 'gpdep':        

        model = TDSP_Convnp_1d(in_dim=1,
                               out_dim=1,
                               kerneltype = args.kerneltype,
                               num_mixture=args.nmixture,
                               num_channel=args.nchannel,
                               num_sample=args.nsample,
                               use_weightnet=args.useweightnet,
                               tempering=args.tempering,
                               cnntype=args.cnntype)
        
        lossfun = compute_latentgp_loss_1d     

        
    if args.modelname == 'base':
        model = Convcnp(in_dim=1,
                        out_dim=1,
                        num_channel=args.nchannel,
                        init_lengthscale=args.convcnpinitl,
                        cnntype=args.cnntype)        
        lossfun = compute_loss_baseline

        
    if args.modelname == 'baselatent':
        model = Convcnp_latent(in_dim=1,
                                out_dim=1,
                                num_channel=args.nchannel,
                                num_sample=args.nsample,
                                init_lengthscale=args.convcnpinitl,
                                cnntype=args.cnntype)        
        lossfun = compute_loss_baselinelatent
        
        
        
    if args.modelname == 'anp':
        model = RegressionANP(input_dim=args.nchannel,
                              num_channels=args.nchannel,
                              latent_dim=128)           
        lossfun = compute_loss_baseline
        
        
    if args.modelname == 'cnp':
        model = RegressionCNP(input_dim=args.nchannel,
                              num_channels=args.nchannel,
                              latent_dim=128)      
        lossfun = compute_loss_baseline




      
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay = args.weightdecay)                
    return model,opt,lossfun

   
    
def make_expinfo(args):
    args_dict = args.__dict__
    model_info = ''
    #tasktypesingletask_testtypeextra_depTrue

    for ith_keys in args_dict:
        if ith_keys in ['msg','printfreq','gentime','tasktype','testtype','dep']:
            pass
        else:
            model_info += ith_keys + str(args_dict[ith_keys]) + '_'
                    
    return model_info
    

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
    
 
def main():

    #----------------------------------------
    # argin
    #----------------------------------------    
    #parser = argparse.ArgumentParser(description='exp1-synthetics-singletask')
    parser = argparse.ArgumentParser(description='exp1-single-regresion')


    
    # model configuration
    parser.add_argument('--modelname',type=str, default='gpdep') #--modelname gpdep2
    parser.add_argument('--nmixture',type=int, default=3)  #--numQ 3  to check whether numQ is 
    parser.add_argument('--nchannel',type=int, default=1)  #--numQ 3  to check whether numQ is 
    parser.add_argument('--nsample', type=int, default=5)
    parser.add_argument('--useweightnet', action='store_true')
    parser.add_argument('--tempering', type=float, default=1e-0)
    parser.add_argument('--cnntype', type=str, default='deep')
    parser.add_argument('--kerneltype', type=str, default='sm')
    
    parser.add_argument('--solve', type=str, default='chol')        
    parser.add_argument('--convcnpinitl', type=float, default=0.1)
    
    

    # training configuration
    parser.add_argument('--nepoch', type=int, default=500) #iterations
    parser.add_argument('--lr', type=float, default= 1e-3) #iterations
    parser.add_argument('--weightdecay', type=float, default= 1e-4)
    parser.add_argument('--reglambda',type=float, default=.5) #--modelname gpdep2
    
    # exp configuration
    parser.add_argument('--tasktype', type=str, default='singletask') # sin3,sin4,mogp,lmc,convk,
    parser.add_argument('--testtype', type=str, default='extra') # inter,extra
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
    dataset_path = './download/'

    # savedir path        
    #save_dir = './regression_task_single_ablation_numQ/param_{}_lr{}/'.format(args.tasktype,args.lr)
    #save_dir = './task_regression_singlechannel_new/param_{}_lr{}/'.format(args.tasktype,args.lr)
    save_dir = './task_regression_singlechannel_new/param{}_datav{}_lr{}/'.format(args.tasktype,args.datav,args.lr)
    

            
    os.makedirs(save_dir) if not os.path.exists(save_dir) else 1        
    save_filename = expinfo[:-1]
    saved_modelparam_path = save_dir + save_filename
    saved_modelparam_path += '_' + 'gentime'+args.gentime
    saved_modelparam_path += '.pth'

    
    #-----------------------------
    # wandbi name assignment
    #-----------------------------    
    config = copy.deepcopy(args.__dict__)
    #print(args.__dict__)
    
    wandb.init( project="bconvcnp-synthetic-single-new1",
                notes="datav{}, msg:{}".format(args.datav,args.msg),            
                config = config,           
                reinit= True)
    wandb.run.name = expinfo[:-1]


    
    model,opt,lossfun = get_model(args)
    metric = build_metric()    
    print(model)

    
    
    if model.modelname in ['gpdep']:
        #------------------------------------------
        # pretrain
        #------------------------------------------    
        pre_opt = torch.optim.Adam(model.gpsampler.parameters(), lr=args.lr,weight_decay = args.weightdecay)                

        #pretrain_idx,numrep = np.arange(1,args.nepoch + 1)[::3], 1
        pretrain_idx,numrep = np.arange(1,args.nepoch + 1)[::5], 2

        #pretrain_idx = np.arange(1,args.nepoch + 1)[::20]
        for k in range(1,numrep+1):
            for i,idx in enumerate(pretrain_idx):   
                epoch_start = time.time()        
                trainset_path= dataset_path + 'syndata_{}_v{}/dep{}_{}_{}'.format(args.tasktype,args.datav, args.dep, args.testtype, idx)
                pretrain_set = torch.load(trainset_path + '.db')['train_set']        


                _,_,pretr_reg = pretrain_epoch_pair(pretrain_set,
                                            model,
                                            pre_opt,
                                            lossfun, 
                                            current_iter = i,
                                            total_iter = len(pretrain_idx))      

                epoch_end = time.time()
                takentime = np.array(epoch_end - epoch_start).round(3)

                if i%args.printfreq ==0 or i== 1:                
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
        torch.save(saved_dict,saved_modelparam_path)
            
            

    
    #------------------------------------------
    #start training
    #------------------------------------------
    wandb.run.save()
    wandb.watch(model)
    
    torch.autograd.set_detect_anomaly(True)    
    best_loss = -np.inf    
    learningtime = 0.0
    for i in range(1,args.nepoch + 1):   

        epoch_start = time.time()
        
        # get_train set
        trainset_path= dataset_path + 'syndata_{}_v{}/dep{}_{}_{}'.format(args.tasktype,args.datav, args.dep, args.testtype, i)       
        train_set = torch.load(trainset_path + '.db')['train_set']
        
        # train dataset per epoch
        tr_loss, tr_data, tr_reg = train_epoch_pair( train_set,
                                                    model,
                                                    opt,
                                                    lossfun, 
                                                    current_iter = i,
                                                    reglambda = args.reglambda,
                                                    total_iter = args.nepoch )        
        epoch_end = time.time()

        
        if i%args.printfreq ==0 or i== 1:                
            # get validation set 
            validation_path= dataset_path + 'syndata_{}_v{}/dep{}_{}_{}'.format(args.tasktype,args.datav, args.dep, args.testtype, -128)        
            valid_set = torch.load(validation_path + '.db')['train_set']
            
            # validate trained model
            val_data = validate_epoch_pair( valid_set,model,lossfun )            

            
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
            else:
                pass
                
                

            saved_dict = {'epoch': i + 1,
                         'best_acc_top1': best_loss,                         
                         'state_dict': state_dict,
                         'optimizer': opt_dict,
                         'args_dict':argsdict,                             
                         'metric_dict':metric
                         }
            torch.save(saved_dict,saved_modelparam_path)
            
            
#             print('epochs [{}/{}] | tr loss ({:.3f},{:.3f}), val data ({:.3f},{:.3f}), \t saved_param: {} saved at epochs {} with best val loss {:.3f} \t {:.3f}(sec)'.format(i,args.nepoch,tr_loss[0],tr_loss[1],val_data[0],val_data[1],saved_modelparam_path,saved_epoch,best_loss,epoch_end-epoch_start) )                       
            print('epochs [{}/{}] | tr loss ({:.3f},{:.3f}), val data ({:.3f},{:.3f}) with best val loss {:.3f} \t {:.3f}(sec)'.format(i,args.nepoch,tr_loss[0],tr_loss[1],val_data[0],val_data[1],best_loss,takentime) )                       

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


 