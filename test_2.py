# import os.path as osp

# import torch
# import torch.nn.functional as F
# from sklearn.metrics import f1_score

# from torch_geometric.data import Batch
# from torch_geometric.datasets import PPI
# from torch_geometric.loader import ClusterData, ClusterLoader, DataLoader
# from torch_geometric.nn import BatchNorm, SAGEConv

# path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'pyg_data','PPI')
# train_dataset = PPI(path, split='train')
# val_dataset = PPI(path, split='val')
# test_dataset = PPI(path, split='test')

# print(train_dataset)

# train_data = Batch.from_data_list(train_dataset)
# print(train_data)

# cluster_data = ClusterData(train_data, num_parts=50, recursive=False,
#                            save_dir=train_dataset.processed_dir)

# train_loader = ClusterLoader(cluster_data, batch_size=1, shuffle=False,
#                              )

# for d in train_loader:
#     print(d)

# print()

# import numpy as np
# import networkx as nx
# import scipy
# from layers import *
# from config import config_gap
# from losses import *
# import random
# import torch
# import losses
# from models import *
# from utils_file.utils import *
# from torch_geometric.loader import DataLoader
# from torch_geometric.data import Data
# from torch.utils.tensorboard import SummaryWriter
# import pickle
# from kmeans_pytorch import kmeans
# import os.path as osp
# from torch_geometric.datasets import Planetoid

# torch.set_printoptions(precision=32)

# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)),'data', 'pyg_data',dataset)
# dataset = Planetoid(path, dataset)
# data = dataset[0]
# print(data)

# class Net(torch.nn.Module):
#     super().__init__()
#     pool_ratios = [0.8,0.8]


# import os.path as osp

# import torch
# import torch.nn.functional as F

# from torch_geometric.datasets import Planetoid
# from torch_geometric.nn import GraphUNet
# from torch_geometric.utils import dropout_edge

# dataset = 'Cora'
# path = osp.join(osp.dirname(osp.realpath(__file__)),'data', 'pyg_data',dataset)
# dataset = Planetoid(path, dataset)
# data = dataset[0]

# class Net(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         pool_ratios = [2000 / data.num_nodes, 0.5]
#         self.unet = GraphUNet(dataset.num_features, 32, dataset.num_classes,
#                               depth=3, pool_ratios=pool_ratios)

#     def forward(self,data):
#         edge_index, _ = dropout_edge(data.edge_index, p=0.2,
#                                      force_undirected=True,
#                                      training=self.training)
#         x = F.dropout(data.x, p=0.92, training=self.training)

#         x = self.unet(x, edge_index)
#         return F.log_softmax(x, dim=1)

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model,data = Net().to(device),data.to(device)
# optimizer = torch.optim.Adam(model.parameters(),lr = 0.01,weight_decay=0.001)

# model.forward(data)


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
from torch.utils.tensorboard import SummaryWriter
import pickle
from kmeans_pytorch import kmeans
from torch_geometric.nn import GraphUNet
from torch_geometric.nn import GCNConv

torch.set_printoptions(precision=32)

torch.manual_seed(1763)
np.random.seed(453658)
random.seed(41884)
torch.cuda.manual_seed(9597121)

mode = 'train'

data_train = 'imagesensor'
data_test = 'IMDB'
data = data_train if mode=='train' else data_test

model = 'spectral for graph embedding'

n_max = 500000000
n_min = 100

config  = config_gap(data=data,batch_size=1,mode=mode,n_max=n_max,n_min=n_min)
print_message(data,mode, config)

class Net(torch.nn.Module):
    def __init__(self,in_channel,hidden_channel,out_channel,depth,pool_ratios):
        super().__init__()
        self.unet = GraphUNet(in_channel, hidden_channel, out_channel, depth = depth,pool_ratios=pool_ratios)
        self.lins1 = nn.Linear(2,16)
        self.lins2 = nn.Linear(16,32)
        self.lins3 = nn.Linear(32,32)
        self.final = nn.Linear(32,2)
    def forward(self,data):
        x = self.unet(data.x,data.edge_index)
        x = torch.tanh(x)
        x = self.lins1(x)
        x = torch.tanh(x)
        x = self.lins2(x)
        x = torch.tanh(x)
        # x = self.lins3(x)
        # x = torch.tanh(x)
        x = self.final(x)
        x,_=torch.linalg.qr(x,mode='reduced')
        _,_,_,cuts,t,ias,ibs,ia,ib= best_part(x,d,2)
        print('cut: ',cuts)
        print('ia: ',len(ia))
        print('ib: ',len(ib))
        return x



def train(config):
    pass

pool_ratios = [0.95,0.85,0.85]

f = Net(2, 16, 2, 25, pool_ratios)
f.train()
print('Number of parameters:',sum(p.numel() for p in f.parameters()))
print(' ')
optimizer = torch.optim.Adam(f.parameters(),lr=0.0001,weight_decay=5e-5)
loss_fn = loss_embedding
print('Start spectral embedding module')
print(' ')

for i in range(120):
    for d in config.loader:
        d = d
        L = laplacian(d)
        x = f(d)
        loss = loss_fn(x,L,0.5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch:',i,'   Loss:',loss)