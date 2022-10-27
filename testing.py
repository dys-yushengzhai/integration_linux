import numpy as np
import networkx as nx
import scipy
from layers import *
from config import config_gap
from losses import *
import random
import torch
import losses
from models import *
from utils_file.utils import *
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from QGraph.v0_1.graph_coarsen import graph_coarsen


torch.manual_seed(1763)
np.random.seed(453658)
random.seed(41884)
torch.cuda.manual_seed(9597121)
# # data
# A = input_matrix()
# print(A.toarray())

# A = torch_sparse.remove_diag(st.from_scipy(A)).to_symmetric()
# A = st.set_value(A, torch.ones_like(A.storage._value),layout='coo')
# # rowcols = np.array([row,col])
# # edges = torch.tensor(rowcols,dtype=torch.long)
# row = A.storage._row
# col = A.storage._col

# edge_index = torch.vstack((row,col))
# x = torch.randn(A.size(dim=0),2)
# dataset = []
# dataset.append(Data(x=x,edge_index=edge_index))
# loader = DataLoader(dataset,shuffle=False)
# device = 'cpu'
# se_params = {'l':32,'pre':2,'post':2,'coarsening_threshold':2,'activation':'tanh','lins':[16,32,32,16,16]}
# model = ModelSpectral_1026(se_params, device)

# for d in loader:
#     print(model.forward(d))

mode = 'train'
 
data_train = 'imagesensor'
data_test = 'imagesensor'
data = data_train if mode=='train' else data_test

n_max = 500000000
n_min = 100

model = 'spectral for graph embedding'
config  = config_gap(data=data,batch_size=1,mode=mode,n_max=n_max,n_min=n_min)
config.data = data
config.model = model


# print the message of dataset
print_message(data,mode, config)

# spectral embedding optimizer == se_opt(dict)(lr,weight_decay)
config.se_opt = {'lr':0.0005,'weight_decay':5e-6}
# partitioning embdding optimizer == pm_opt(dict)(lr,weight_decay)
config.pe_opt = {'lr':0.001,'weight_decay':5e-6}
# whether to run spectral embedding
config.is_se = True
# whether to run partitiong embedding 
config.is_pe = True
config.hyper_para_loss_embedding = 1
config.hyper_para_loss_normalized_cut = 0.1
config.se_params = {'l':32,'pre':2,'post':2,'coarsening_threshold':2,'activation':'tanh','lins':[16,32,32,16,16]}
config.pe_params = {'l':32,'pre':4,'post':4,'coarsening_threshold':2,'activation':'tanh','lins':[16,16,16,16,16]}
config.se_epoch = 2 # 120 80(0.001)
config.pe_epoch = 10 # 150 # 100(0.0005)
config.se_epoch_train = 2
config.pe_epoch_train = 150

config.se_train_savepath = 'spectral_weights/spectral_weights_'+data_train+'_'+str(config.se_epoch_train)+'.pt'
config.pe_train_savepath  = 'partitioning_weights/partitioning_weights_'+data_train+'_'+str(config.se_epoch_train)+'_'+str(config.pe_epoch_train)+'.pt'
config.se_test_savepath = 'spectral_weights/spectral_weights_'+data_test+'.pt'
config.pe_test_savepath = 'partitioning_weights/partitioning_weights_'+data_test+'.pt'

device = config.device

f = ModelSpectral_1026(config.se_params,device).to(device)
f.train()
print('Number of parameters:',sum(p.numel() for p in f.parameters()))
print('')
optimizer = torch.optim.Adam(f.parameters(),**config.se_opt)
loss_fn = loss_embedding
print('Start spectral embedding module')
print(' ')





for i in range(config.se_epoch):
    for d in config.loader:
        d = d.to(device)
        L = laplacian(d).to(device)
        x = f(d)
        _,_,_,cuts,t,ias,ibs,ia,ib= best_part(x,d,2)
        print('cut: ',cuts)
        print('ia: ',len(ia))
        print('ib: ',len(ib))
        loss = loss_fn(x,L,config.hyper_para_loss_embedding)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch:',i,'   Loss:',loss)
        
torch.save(f.state_dict(), config.se_train_savepath)
print('Model saved')
print('')

i = 0
cluster_info,edge_info,gc = f.get_info()
for i in range(len(cluster_info)):
    print(i)
    print(cluster_info[i].shape)
    