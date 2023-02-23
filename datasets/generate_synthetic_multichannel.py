import os
import time
import torch
import numpy as np
import torch.nn as nn 
import matplotlib.pyplot as plt
#from dataset_multitask_1d import motask_generator
from .dataset_multitask_1d import motask_generator

import argparse
parser = argparse.ArgumentParser(description='exp1')
#parser.add_argument('--num_saved', type=int, default=0) 
parser.add_argument('--ntask', type=int, default=200) 
parser.add_argument('--nbatch', type=int, default=16) 
parser.add_argument('--datav', type=int, default=3) 


parser.add_argument('--tasktype', type=str, default='lmc') #sin3, lmc, mogp
parser.add_argument('--testtype',type=str,default='extra') #inter extra



parser.add_argument('--dep', default=False, action='store_true')
parser.add_argument('--train', default=False, action='store_true')

parser.add_argument('--msg',type=str,default='none')

args = parser.parse_args()   


# parser.add_argument('--num_batches', type=int, default=10000)
# parser.add_argument('--batch_size', type=int, default=16)
# parser.add_argument('--filename', type=str, default='batch')
# parser.add_argument('--X0', type=float, default=50)
# parser.add_argument('--Y0', type=float, default=100)


save_dir = './syndata_{}_v{}'.format(args.tasktype,args.datav)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)           






# #-----------------------------------------
# # globarl configuration
# #-----------------------------------------
# #tasktype = 'lmc'

nchannels = 3
def build_gen_cls(args):    
    tasktype = args.tasktype    
    testtype = args.testtype
    print('-'*100)
    print('build gen cls by tasktyp: {}, testtype: {}, dep: {}, train: {}'.format(args.tasktype,args.testtype,args.dep,args.train))
    print('-'*100)
    
    if args.testtype == 'inter':
        #train_range = [-4,4]
        #test_range = [-2,2]
        #train_range = [0,10]
        #test_range = [0,5]  
        train_range = [0,3]
        test_range = [0,6]
        
    elif args.testtype == 'extra':
        #train_range = [-2,2]
        #test_range = [-4,4]
        train_range = [0,3]
        test_range = [0,6]
        
    else:
        pass

    
    if args.tasktype == 'lmc':
        gen_cls = motask_generator(tasktype=tasktype,testtype=testtype,nchannels=nchannels,train_range=train_range,test_range=test_range,dep=args.dep)
    
    elif args.tasktype == 'mosm':
        gen_cls = motask_generator(tasktype=tasktype,testtype=testtype,nchannels=nchannels,train_range=train_range,test_range=test_range,dep=args.dep)

    elif args.tasktype == 'mosmvarying':
        gen_cls = motask_generator(tasktype=tasktype,testtype=testtype,nchannels=nchannels,train_range=train_range,test_range=test_range,dep=args.dep)

    elif args.tasktype =='sin3':
        gen_cls = motask_generator(tasktype=tasktype,testtype=testtype,nchannels=nchannels,train_range=train_range,test_range=test_range,dep=args.dep)
    
    return gen_cls,train_range,test_range




def set_numdata(datav):
    if datav == 4:
        ncontext = np.random.randint(10,25)    
        ntarget = 50 - ncontext    

    #datav = 5 small context set and target set
    if datav == 5:        
        ncontext = np.random.randint(5,12)    
        ntarget = 25 - ncontext    

        
    if datav == 7:        
        ncontext = np.random.randint(10,50)    
        ntarget = np.random.randint(ncontext,50)   
        
    if datav == 8:        
        ncontext = np.random.randint(10,50)    
        ntarget = np.random.randint(ncontext,50)           
        
    if datav == 9:        
        #ncontext = np.random.randint(5,20)    
        ncontext = np.random.randint(5,25)            
        ntarget = np.random.randint(ncontext,50)           

    if datav == 10:        
        ncontext = np.random.randint(5,10)    
        ntarget = np.random.randint(ncontext,30)           
        
        
    if datav == 11:        
        #ncontext = np.random.randint(5,20)    
        ncontext = np.random.randint(5,25)            
        ntarget = np.random.randint(ncontext,50)           
        
    return ncontext,ntarget







#--------------------------------------------------------------------------------------------------------------------------------------
# construct training set
#--------------------------------------------------------------------------------------------------------------------------------------    
def build_trainset(args,ij,gen_cls,train_range,test_range,nbatches_per=50):            

    batch_trainset = [] 
    tic = time.time()        
    for _ in range(nbatches_per):
        ncontext,ntarget= set_numdata(args.datav)
        xc,yc,xt,yt,xf,yf = gen_cls.prepare_task(nbatch=args.nbatch,
                                                   ncontext=ncontext,
                                                   ntarget=ntarget,
                                                   train_range=gen_cls.train_range,
                                                   test_range=gen_cls.test_range,
                                                   noise_true=True,
                                                   intrain = True)

        #print('inter context {}, target {}'.format(xc.shape,xt.shape))        

        train_set = (xc,yc,xt,yt,xf,yf) 
        batch_trainset.append(train_set)

        #db = {'train_set':train_set, 'valid_set':valid_set,'test_set':test_set}
        #db = {'train_set':train_set, 'valid_set':valid_set}
    db = {'train_set':batch_trainset}

    #save_path_set = './syndata_lmc/{}_{}_{}.db'.format(tasktype , testtype,args.num_saved)
    #save_path_set = './syndata_lmc/{}_{}_{}.db'.format(tasktype ,args.testtype, args.num_saved)
    #save_path_set = './syndata_{}/dep{}_{}_{}.db'.format(tasktype,args.dep ,args.testtype, args.num_saved)


    save_path_set = save_dir + '/dep{}_{}_{}.db'.format(args.dep ,args.testtype, ij)
    torch.save(db, save_path_set)
    print(save_path_set + '_taken {:.2f} (sec)'.format(time.time()-tic ))
    return 






gen_cls,train_range,test_range = build_gen_cls(args)    
for ij in range(1,args.ntask+1):
    build_trainset(args,ij,gen_cls,train_range,test_range)
    
    
    
    
    
    
    
    
    
#--------------------------------------------------------------------------------------------------------------------------------------
# construct valid set
#--------------------------------------------------------------------------------------------------------------------------------------
args.train = False
gen_cls,train_range,test_range = build_gen_cls(args)    

#nvaltask = 32 
nvaltask = 64 
#nvaltask = 128 
#nbatch = 32

batches_inter = []
batches_extra = []


for _ in range(nvaltask):    
    ncontext,ntarget= set_numdata(args.datav)
    

    xc,yc,xt,yt,xf,yf  = gen_cls.prepare_task(nbatch=args.nbatch,
                                               ncontext=ncontext,
                                               ntarget=ntarget,
                                               train_range=gen_cls.train_range,
                                               test_range=gen_cls.test_range,
                                               noise_true=True,
                                               intrain = True) 

    # inter_testset = {'context_x':context_x,
    #                  'context_y':context_y,
    #                  'target_x':target_x,
    #                  'target_y':target_y,
    #                  'full_x':full_x,
    #                  'full_y':full_y}
    inter_testset =  (xc,yc,xt,yt,xf,yf)
    batches_inter.append(inter_testset)

    
    print('inter context {}, target {}'.format(xc.shape,xt.shape))        


    #ncontext = np.random.randint(10,50-10)    
    #ntarget = 50 - ncontext    

    xc,yc,xt,yt,xf,yf = gen_cls.prepare_task(nbatch=args.nbatch,
                                             ncontext=ncontext,
                                            ntarget=ntarget,
                                            train_range=gen_cls.train_range,
                                            test_range=gen_cls.test_range,
                                            noise_true=True,
                                            intrain = False)


    # extra_testset = {'context_x':context_x,
    #                  'context_y':context_y,
    #                  'target_x':target_x,
    #                  'target_y':target_y,
    #                  'full_x':full_x,
    #                  'full_y':full_y}
    extra_testset = (xc,yc,xt,yt,xf,yf )
    batches_extra.append(extra_testset)

    print('outer context {}, target {}'.format(xc.shape,xt.shape))        
    
#db = {'train_set':inter_testset, 'valid_set':extra_testset}
db = {'train_set':batches_inter, 'valid_set':batches_extra}
save_path_set = './syndata_{}_v{}/dep{}_{}_{}.db'.format(args.tasktype,args.datav, args.dep ,args.testtype, -nvaltask)
#save_path_set = './syndata_{}_v{}/dep{}_{}_{}_small.db'.format(tasktype,args.datav, args.dep ,args.testtype, -nvaltask)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)    
torch.save(db, save_path_set)
print(save_path_set)
print('\n'*2)    
    
    
    
    
    
    
    
    

    
#--------------------------------------------------------------------------------------------------------------------------------------
# construct varying context and target valid set
#--------------------------------------------------------------------------------------------------------------------------------------

def build_dataset(args,ncontext,ntarget):
    args.train = False
    gen_cls,train_range,test_range = build_gen_cls(args)    

    #nvaltask = 4
    #nvaltask = 16 
    #nvaltask = 32 
    nvaltask = 64 
    #nvaltask = 256 
    #args.nbatch = 4
    #nbatch = 32

    batches_inter = []
    batches_extra = []
    full_db={}
    for _ in range(nvaltask):
        xc,yc,xt,yt,xf,yf  = gen_cls.prepare_task(nbatch=args.nbatch,
                                                   ncontext=ncontext,
                                                   ntarget=ntarget,
                                                   train_range=gen_cls.train_range,
                                                   test_range=gen_cls.test_range,
                                                   noise_true=True,
                                                   intrain = True) 

        inter_testset =  (xc,yc,xt,yt,xf,yf)
        batches_inter.append(inter_testset)
        print('inter context {}, target {}'.format(xc.shape,xt.shape))        


        xc,yc,xt,yt,xf,yf = gen_cls.prepare_task(nbatch=args.nbatch,
                                                 ncontext=ncontext,
                                                ntarget=ntarget,
                                                train_range=gen_cls.train_range,
                                                test_range=gen_cls.test_range,
                                                noise_true=True,
                                                intrain = False)



        extra_testset = (xc,yc,xt,yt,xf,yf )
        batches_extra.append(extra_testset)
        print('outer context {}, target {}'.format(xc.shape,xt.shape))        

    #db = {'train_set':inter_testset, 'valid_set':extra_testset}
    db = {'train_set':batches_inter, 'valid_set':batches_extra,'ncontext':ncontext,'ntarget':ntarget}
    #save_path_set = './syndata_{}_v{}/dep{}_{}_ncontext{}.db'.format(tasktype,args.datav, args.dep ,args.testtype,ncontext)
    
    
    #save_path_set = './syndata_{}_v{}/dep{}_{}_ncontext{}.db'.format(tasktype,args.datav, args.dep ,args.testtype,ncontext)
    
    save_dir1 = save_dir + '_varyingnct'
    if not os.path.isdir(save_dir1):
        os.makedirs(save_dir1)    
    
    save_filename = 'dep{}_{}_ncontext{}.db'.format(args.dep ,args.testtype,ncontext)
    #save_filename = 'dep{}_{}_ncontext{}_v2.db'.format(args.dep ,args.testtype,ncontext)
    
    save_path_set = os.path.join(save_dir1,save_filename)
    
    #torch.save(db, save_path_set)
    torch.save(db, save_path_set)
    print(save_path_set)
    print('\n'*2)    

    del db
    del xc,yc,xt,yt,xf,yf 
    return 


# ncontext_list = [10,20,30,40,50]
# ntarget = 100
# for ncontext in ncontext_list:
#     build_dataset(args,ncontext,ntarget)



ncontext_list = [5,10,15,20,25,30]
ntarget = 50
for ncontext in ncontext_list:
    build_dataset(args,ncontext,ntarget)





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    