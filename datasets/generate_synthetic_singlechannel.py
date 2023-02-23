import os
import time
import torch
import numpy as np
import torch.nn as nn 
import matplotlib.pyplot as plt

#from dataset_multitask_1d import motask_generator
#from dataset_singletask_1d import prepare_mixed_1dtask, task_list
from .dataset_singletask_1d import prepare_mixed_1dtask, task_list



import argparse
parser = argparse.ArgumentParser(description='exp1')
#parser.add_argument('--num_saved', type=int, default=0) 
parser.add_argument('--ntask', type=int, default=200) 
parser.add_argument('--nbatch', type=int, default=16) 
parser.add_argument('--datav', type=int, default=1) 



parser.add_argument('--tasktype', type=str, default='singletask') #sin3, lmc, mogp
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
#save_dir = './regression_task_single/syndata_{}_v{}'.format(args.tasktype,args.datav)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)           






#-----------------------------------------
# globarl configuration
#-----------------------------------------
#tasktype = 'lmc'

from attrdict import AttrDict


# def build_gen_cls(args):    
#     tasktype = args.tasktype    
#     testtype = args.testtype
#     print('-'*100)
#     print('build gen cls by tasktyp: {}, testtype: {}, dep: {}, train: {}'.format(args.tasktype,args.testtype,args.dep,args.train))
#     print('-'*100)
    
#     if args.testtype == 'inter':
#         #train_range = [-4,4]
#         #test_range = [-2,2]
#         #train_range = [0,10]
#         #test_range = [0,5]  
#         train_range = [0,4]
#         test_range = [0,8]
        
#     elif args.testtype == 'extra':
#         #train_range = [-2,2]
#         #test_range = [-4,4]
#         #train_range = [0,3]
#         #test_range = [0,6]
#         train_range = [0,4]
#         test_range = [0,8]
        
#     else:
#         pass

    
# #     if args.tasktype == 'lmc':
# #         gen_cls = motask_generator(tasktype=tasktype,testtype=testtype,nchannels=nchannels,train_range=train_range,test_range=test_range,dep=args.dep)
    
# #     elif args.tasktype == 'mosm':
# #         gen_cls = motask_generator(tasktype=tasktype,testtype=testtype,nchannels=nchannels,train_range=train_range,test_range=test_range,dep=args.dep)

# #     elif args.tasktype == 'mosmvarying':
# #         gen_cls = motask_generator(tasktype=tasktype,testtype=testtype,nchannels=nchannels,train_range=train_range,test_range=test_range,dep=args.dep)

# #     elif args.tasktype =='sin3':
# #         gen_cls = motask_generator(tasktype=tasktype,testtype=testtype,nchannels=nchannels,train_range=train_range,test_range=test_range,dep=args.dep)
    
#     gen_cls = AttrDict()
#     gen_cls.train_range = train_range
#     gen_cls.test_range = test_range
#     return gen_cls



def set_numdata(datav):
#     if datav == 4:
#         ncontext = np.random.randint(10,25)    
#         ntarget = 50 - ncontext    

#     #datav = 5 small context set and target set
#     if datav == 5:        
#         ncontext = np.random.randint(5,12)    
#         ntarget = 25 - ncontext    

        
#     if datav == 7:        
#         ncontext = np.random.randint(10,50)    
#         ntarget = np.random.randint(ncontext,50)   
        
#     if datav == 8:        
#         ncontext = np.random.randint(10,50)    
#         ntarget = np.random.randint(ncontext,50)           
        
#     if datav == 9:        
#         #ncontext = np.random.randint(5,20)    
#         ncontext = np.random.randint(5,25)            
#         ntarget = np.random.randint(ncontext,50)           

#     if datav == 10:        
#         ncontext = np.random.randint(5,10)    
#         ntarget = np.random.randint(ncontext,30)           
        
        
#     if datav == 11:        
#         #ncontext = np.random.randint(5,20)    
#         ncontext = np.random.randint(5,25)            
#         ntarget = np.random.randint(ncontext,50)           

        
    if datav == 1:  #small set
        #ncontext = np.random.randint(5,20)    
        #ncontext = np.random.randint(5,25)            
        #ntarget = np.random.randint(ncontext,50)           
        ncontext = np.random.randint(5,25)            
        ntarget = np.random.randint(ncontext,50)           
        

    if datav == 2:  #large set       
        #ncontext = np.random.randint(5,20)    
        ncontext = np.random.randint(10,50)            
        ntarget = np.random.randint(ncontext,50)           
        
    return ncontext,ntarget







#--------------------------------------------------------------------------------------------------------------------------------------
# construct training set
#--------------------------------------------------------------------------------------------------------------------------------------
from dataset_singletask_1d import prepare_mixed_1dtask,task_list

train_range = [0,4]
test_range = [0,8]
nchannels = 1





#def build_trainset(args,ij,gen_cls,train_range,test_range,nbatches_per=50):            
def build_trainset(args,ij,train_range,test_range, testtype='extra',nbatches_per=50,nbatch=16):            

    batch_trainset = [] 
    tic = time.time()        
    
    batch_tasklist = np.random.choice(task_list,nbatches_per,p=[0.25,0.25,0.25,0.25])
    
    for kk in range(nbatches_per):
        ncontext,ntarget= set_numdata(args.datav)
        #xc,yc,xt,yt,xf,yf = gen_cls.prepare_task(nbatch=args.nbatch,
        #                                           ncontext=ncontext,
        #                                           ntarget=ntarget,
        #                                           train_range=gen_cls.train_range,
        #                                           test_range=gen_cls.test_range,
        #                                           noise_true=True,
        #                                           intrain = True)

        #current_tasktype = np.random.choice(task_list)
        current_tasktype = batch_tasklist[kk]
        xc,yc,xt,yt,xf,yf = prepare_mixed_1dtask(data_name = current_tasktype,
                                                testtype = testtype,
                                                nbatch = nbatch,
                                                batch_npoints=(ncontext,ntarget),
                                                train_range = train_range,
                                                test_range = test_range, 
                                                nchannels=1,
                                                noise_true = True,
                                                eps=1e-4, 
                                                intrain=True)
        
        train_set = (xc,yc,xt,yt,xf,yf) 
        batch_trainset.append(train_set)
    db = {'train_set':batch_trainset}
    save_path_set = save_dir + '/dep{}_{}_{}.db'.format(args.dep ,args.testtype, ij)
        
    torch.save(db, save_path_set)
    print(save_path_set + '_taken {:.2f} (sec)'.format(time.time()-tic ))
    return 






#gen_cls,train_range,test_range = build_gen_cls(args)   
for ij in range(1,args.ntask+1):
    build_trainset(args,ij,train_range,test_range,nbatches_per=50,nbatch=args.nbatch)
    
    

    
    
    
    
    
    
    
#--------------------------------------------------------------------------------------------------------------------------------------
# construct valid set
#--------------------------------------------------------------------------------------------------------------------------------------
args.train = False
#gen_cls,train_range,test_range = build_gen_cls(args)    


#nvaltask = 32 
#nvaltask = 64 
nvaltask = 128 
#nbatch = 32


testtype='extra'
nbatches_per=50
nbatch=16
#batch_tasklist = np.random.choice(task_list,nvaltask,p=[0.25,0.25,0.25,0.25])

#nvaltask = 128 
num_task = len(task_list)
num_eachtask = nvaltask // num_task 

batch_tasklist = []
for kkk in range(num_task):
    batch_tasklist += [task_list[kkk] for _ in range(num_eachtask) ]    


batches_inter,batches_extra = [],[]    
for kk in range(nvaltask):    
    ncontext,ntarget= set_numdata(args.datav)
    

#     xc,yc,xt,yt,xf,yf  = gen_cls.prepare_task(nbatch=args.nbatch,
#                                                ncontext=ncontext,
#                                                ntarget=ntarget,
#                                                train_range=gen_cls.train_range,
#                                                test_range=gen_cls.test_range,
#                                                noise_true=True,
#                                                intrain = True) 


    #current_tasktype = np.random.choice(task_list)
    current_tasktype = batch_tasklist[kk]
    xc,yc,xt,yt,xf,yf = prepare_mixed_1dtask(data_name = current_tasktype,
                                            testtype = testtype,
                                            nbatch = nbatch,
                                            batch_npoints=(ncontext,ntarget),
                                            train_range = train_range,
                                            test_range = test_range, 
                                            nchannels=1,
                                            noise_true = True,
                                            eps=1e-4, 
                                            intrain=True)

    inter_testset =  (xc,yc,xt,yt,xf,yf)
    batches_inter.append(inter_testset)
    print('inter context {}, target {}'.format(xc.shape,xt.shape))        


    
    
    
    
    

#     xc,yc,xt,yt,xf,yf = gen_cls.prepare_task(nbatch=args.nbatch,
#                                              ncontext=ncontext,
#                                             ntarget=ntarget,
#                                             train_range=gen_cls.train_range,
#                                             test_range=gen_cls.test_range,
#                                             noise_true=True,
#                                             intrain = False)

    xc,yc,xt,yt,xf,yf = prepare_mixed_1dtask(data_name = current_tasktype,
                                            testtype = testtype,
                                            nbatch = nbatch,
                                            batch_npoints=(ncontext,ntarget),
                                            train_range = train_range,
                                            test_range = test_range, 
                                            nchannels=1,
                                            noise_true = True,
                                            eps=1e-4, 
                                            intrain=False)

    extra_testset = (xc,yc,xt,yt,xf,yf )
    batches_extra.append(extra_testset)
    print('outer context {}, target {}'.format(xc.shape,xt.shape))        
    
#db = {'train_set':inter_testset, 'valid_set':extra_testset}
db = {'train_set':batches_inter, 'valid_set':batches_extra}
#save_path_set = './syndata_{}_v{}/dep{}_{}_{}.db'.format(args.tasktype,args.datav, args.dep ,args.testtype, -nvaltask)
save_path_set = './syndata_{}_v{}/dep{}_{}_{}.db'.format(args.tasktype,args.datav, args.dep ,args.testtype, -nvaltask)
#save_path_set = './syndata_{}_v{}/dep{}_{}_{}.db'.format(args.tasktype,args.datav, args.dep ,args.testtype, -nvaltask)


#save_path_set = './syndata_{}_v{}/dep{}_{}_{}_small.db'.format(tasktype,args.datav, args.dep ,args.testtype, -nvaltask)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)    
torch.save(db, save_path_set)
print(save_path_set)
print('\n'*2)    
    
    
    
    
    
    
    
    

    
#--------------------------------------------------------------------------------------------------------------------------------------
# construct varying context and target valid set
#--------------------------------------------------------------------------------------------------------------------------------------


#nvaltask = 32 
#nvaltask = 64 
nvaltask = 128 
#nbatch = 32


testtype='extra'
nbatches_per=50
nbatch=16
#batch_tasklist = np.random.choice(task_list,nvaltask,p=[0.25,0.25,0.25,0.25])
    
testset_dict = {}    
for kth_task in task_list:       
    print('current kth_task: {}'.format(kth_task))
    batches_inter,batches_extra = [],[]    
    for kk in range(nvaltask):    
        ncontext,ntarget= set_numdata(args.datav)


    #     xc,yc,xt,yt,xf,yf  = gen_cls.prepare_task(nbatch=args.nbatch,
    #                                                ncontext=ncontext,
    #                                                ntarget=ntarget,
    #                                                train_range=gen_cls.train_range,
    #                                                test_range=gen_cls.test_range,
    #                                                noise_true=True,
    #                                                intrain = True) 


        #current_tasktype = np.random.choice(task_list)
        current_tasktype = kth_task
        xc,yc,xt,yt,xf,yf = prepare_mixed_1dtask(data_name = current_tasktype,
                                                testtype = testtype,
                                                nbatch = nbatch,
                                                batch_npoints=(ncontext,ntarget),
                                                train_range = train_range,
                                                test_range = test_range, 
                                                nchannels=1,
                                                noise_true = True,
                                                eps=1e-4, 
                                                intrain=True)

        inter_testset =  (xc,yc,xt,yt,xf,yf)
        batches_inter.append(inter_testset)
        print('inter context {}, target {}'.format(xc.shape,xt.shape))        




        xc,yc,xt,yt,xf,yf = prepare_mixed_1dtask(data_name = current_tasktype,
                                                testtype = testtype,
                                                nbatch = nbatch,
                                                batch_npoints=(ncontext,ntarget),
                                                train_range = train_range,
                                                test_range = test_range, 
                                                nchannels=1,
                                                noise_true = True,
                                                eps=1e-4, 
                                                intrain=False)

        extra_testset = (xc,yc,xt,yt,xf,yf )
        batches_extra.append(extra_testset)
        print('outer context {}, target {}'.format(xc.shape,xt.shape))        
    
    #db = {'train_set':inter_testset, 'valid_set':extra_testset}
    #db = {'train_set':batches_inter, 'valid_set':batches_extra}
    testset_dict[kth_task] = {'train_set':batches_inter, 'valid_set':batches_extra}
    print('\n'*5)


    


save_dir1 = save_dir + '_test'
if not os.path.isdir(save_dir1):
    os.makedirs(save_dir1)    

#save_filename = 'dep{}_{}_ncontext{}.db'.format(args.dep ,args.testtype,ncontext)
#save_filename = 'dep{}_{}_ncontext{}_v2.db'.format(args.dep ,args.testtype,ncontext)
save_filename = 'dep{}_{}_nvaltask{}.db'.format(args.dep ,args.testtype,nvaltask)
save_path_set = os.path.join(save_dir1,save_filename)

#torch.save(db, save_path_set)
torch.save(testset_dict, save_path_set)
print(save_path_set)
print('\n'*2)    





















































#save_path_set = './syndata_{}_v{}/dep{}_{}_{}.db'.format(args.tasktype,args.datav, args.dep ,args.testtype, -nvaltask)
#save_path_set = './syndata_{}_v{}/dep{}_{}_{}.db'.format(args.tasktype,args.datav, args.dep ,args.testtype, -nvaltask)
#save_path_set = './syndata_{}_v{}/dep{}_{}_{}.db'.format(args.tasktype,args.datav, args.dep ,args.testtype, -nvaltask)


#save_path_set = './syndata_{}_v{}/dep{}_{}_{}_small.db'.format(tasktype,args.datav, args.dep ,args.testtype, -nvaltask)
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)    
# torch.save(db, save_path_set)
# print(save_path_set)
# print('\n'*2)    









# #def build_dataset(args,ncontext,ntarget):
# def prepare_testset(args,ncontext,ntarget):
    
#     args.train = False
#     gen_cls,train_range,test_range = build_gen_cls(args)    

#     #nvaltask = 4
#     #nvaltask = 16 
#     #nvaltask = 32 
#     nvaltask = 64 
#     #nvaltask = 256 
#     #args.nbatch = 4
#     #nbatch = 32

#     batches_inter = []
#     batches_extra = []
#     full_db={}
#     for _ in range(nvaltask):
#         xc,yc,xt,yt,xf,yf  = gen_cls.prepare_task(nbatch=args.nbatch,
#                                                    ncontext=ncontext,
#                                                    ntarget=ntarget,
#                                                    train_range=gen_cls.train_range,
#                                                    test_range=gen_cls.test_range,
#                                                    noise_true=True,
#                                                    intrain = True) 

#         inter_testset =  (xc,yc,xt,yt,xf,yf)
#         batches_inter.append(inter_testset)
#         print('inter context {}, target {}'.format(xc.shape,xt.shape))        


#         xc,yc,xt,yt,xf,yf = gen_cls.prepare_task(nbatch=args.nbatch,
#                                                  ncontext=ncontext,
#                                                 ntarget=ntarget,
#                                                 train_range=gen_cls.train_range,
#                                                 test_range=gen_cls.test_range,
#                                                 noise_true=True,
#                                                 intrain = False)



#         extra_testset = (xc,yc,xt,yt,xf,yf )
#         batches_extra.append(extra_testset)
#         print('outer context {}, target {}'.format(xc.shape,xt.shape))        

#     #db = {'train_set':inter_testset, 'valid_set':extra_testset}
#     db = {'train_set':batches_inter, 'valid_set':batches_extra,'ncontext':ncontext,'ntarget':ntarget}
#     #save_path_set = './syndata_{}_v{}/dep{}_{}_ncontext{}.db'.format(tasktype,args.datav, args.dep ,args.testtype,ncontext)
    
    
#     #save_path_set = './syndata_{}_v{}/dep{}_{}_ncontext{}.db'.format(tasktype,args.datav, args.dep ,args.testtype,ncontext)
    
#     save_dir1 = save_dir + '_varyingnct'
#     if not os.path.isdir(save_dir1):
#         os.makedirs(save_dir1)    
    
#     save_filename = 'dep{}_{}_ncontext{}.db'.format(args.dep ,args.testtype,ncontext)
#     #save_filename = 'dep{}_{}_ncontext{}_v2.db'.format(args.dep ,args.testtype,ncontext)
    
#     save_path_set = os.path.join(save_dir1,save_filename)
    
#     #torch.save(db, save_path_set)
#     torch.save(db, save_path_set)
#     print(save_path_set)
#     print('\n'*2)    

#     del db
#     del xc,yc,xt,yt,xf,yf 
#     return 


# # ncontext_list = [10,20,30,40,50]
# # ntarget = 100
# # for ncontext in ncontext_list:
# #     build_dataset(args,ncontext,ntarget)



# ncontext_list = [5,10,15,20,25,30]
# ntarget = 50
# for ncontext in ncontext_list:
#     build_dataset(args,ncontext,ntarget)





    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    