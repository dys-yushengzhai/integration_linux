# from __future__ import division
# import torch
# from utils_file.dataset_utils import mixed_dataset
# from utils_file.utils import *
# from utils_file.dataset_utils import *
# import networkx as nx
# from torch_geometric.data import Data, Batch
# from torch_geometric.loader import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# import pickle

# import torch
# import torch.nn as nn
# from layers import *
# from torch_geometric.nn import  avg_pool, graclus
# from torch_geometric.data import Batch
# from layers import SAGEConv
# import pickle
# import random
# from torch_geometric.utils import degree

# import scipy
# from layers import *
# from config import config_gap
# from losses import *
# # data = 'imagesensor'

# # loader = mixed_dataset(data)

# # print(loader)
# torch.set_printoptions(precision=32)
# with open('x_no_pickle.pickle','rb') as f:
#     x_no_pickle = pickle.load(f)

# with open('x_pickle.pickle','rb') as f:
#     x_pickle = pickle.load(f)

# print(x_no_pickle[0][0])

# print(x_pickle[0][0])

# # tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],[1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]])
# # tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],[1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]])


# print(x_no_pickle[1][0])
# print(x_pickle[1][0])

# torch.manual_seed(176364)
# np.random.seed(453658)
# random.seed(41884)
# torch.cuda.manual_seed(9597121)

# for i in range(10):
#     print(torch.randn(1,1))

# y = torch.tensor([[1,0],[1,0],[1,0],[1,0],[1,0]])
# node_num = y.shape[0]
# partition_num = y.shape[1]
# y = torch.sum(y,axis=0)

# print(node_num/partition_num)
# y = y-node_num/partition_num
# print(y)
# print(torch.sum(y.pow(2),axis=0))
# # print(y.size(0)/y.size(1))
# # print(y)

# node_num = y.shape[0]
# print(node_num)
# partition_num = y.shape[1]

# y = torch.sum((torch.sum(y,axis=0)-node_num/partition_num).pow(2),axis=0)/(torch.tensor(node_num,dtype=torch.float32).pow(2)/2.)
# print(y)

# A = input_matrix()
# row = torch.from_numpy(A.row).long()
# col = torch.from_numpy(A.col).long()
# data = torch.from_numpy(A.data)
# edge_index = torch.vstack((row,col))
# print(edge_index)
# matrix = st.from_edge_index(edge_index=edge_index,edge_attr=data)
# print(matrix.sparse_size(dim=0))
# d = degree(edge_index[0],num_nodes=matrix.sparse_size(dim=0))
# d = torch.tensor([2.,2.,4.,3.,3.,2.,2.])
# print((d==0.).any())



# class GNet(nn.modules):
#     def __init__(self,in_dim,n_classes,args):
#         super(GNet,self).__init__()
#         self.n_act = getattr(nn, args.act_n)()
#         self.c_act = getattr(nn, args.act_c)()


# data = 'imagesensor'

# init_plot = False

# config  = config_gap(data=data,batch_size=1,mode='train')
# print(config.dataset)
# for d in config.loader:
#     print(d.edge_index)


# import torch
# import torch.nn.functional as F
# from torch.nn import ModuleList
# from tqdm import tqdm

# from torch_geometric.datasets import Reddit
# from torch_geometric.loader import ClusterData, ClusterLoader, NeighborSampler
# from torch_geometric.nn import SAGEConv
# from utils_file.utils import *

# dataset = Reddit('data/pyg_data/Reddit')
# data = dataset[0]
# print(data)

# cluster_data = ClusterData(data, num_parts=1500, recursive=False,
#                            save_dir=dataset.processed_dir) 

# print(cluster_data)   

# train_loader = ClusterLoader(cluster_data, batch_size=20, shuffle=False,
#                              )


# class Net(torch.nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.convs = ModuleList(
#             [SAGEConv(in_channels, 128),
#              SAGEConv(128, out_channels)])

#     def forward(self,x,edge_index):
#         for i,conv in enumerate(self.convs):
#             x = conv(x,edge_index)
#             if i != len(self.convs) -1:
#                 x = F.relu(x)
#                 x = F.dropout(x, p=0.5, training=self.training)
#         return F.log_softmax(x, dim=-1)
#     def inference(self,x_all):
#         pbar = tqdm(total=x_all.size(0) * len(self.convs))
#         pbar.set_description('Evaluating')

#         # Compute representations of nodes layer by layer, using *all*
#         # available edges. This leads to faster computation in contrast to
#         # immediately computing the final representations of each batch.
#         for i, conv in enumerate(self.convs):
#             xs = []
#             for batch_size, n_id, adj in subgraph_loader:
#                 edge_index, _, size = adj.to(device)
#                 x = x_all[n_id].to(device)
#                 x_target = x[:size[1]]
#                 x = conv((x, x_target), edge_index)
#                 if i != len(self.convs) - 1:
#                     x = F.relu(x)
#                 xs.append(x.cpu())

#                 pbar.update(batch_size)

#             x_all = torch.cat(xs, dim=0)

#         pbar.close()

#         return x_all

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net(dataset.num_features, dataset.num_classes).to(device)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# def train():
#     model.train()

#     total_loss = total_nodes = 0
#     for batch in train_loader:
#         batch = batch.to(device)
#         optimizer.zero_grad()
#         out = model(batch.x, batch.edge_index)
#         loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
#         loss.backward()
#         optimizer.step()

#         nodes = batch.train_mask.sum().item()
#         total_loss += loss.item() * nodes
#         total_nodes += nodes
    
#     return total_loss/total_nodes

# @torch.no_grad()
# def test():  # Inference should be performed on the full graph.
#     model.eval()

#     out = model.inference(data.x)
#     y_pred = out.argmax(dim=-1)

#     accs = []
#     for mask in [data.train_mask, data.val_mask, data.test_mask]:
#         correct = y_pred[mask].eq(data.y[mask]).sum().item()
#         accs.append(correct / mask.sum().item())
#     return accs

# for epoch in range(1,31):
#     loss = train()
#     if epoch % 5 == 0:
#         train_acc, val_acc, test_acc = test()
#         print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
#                 f'Val: {val_acc:.4f}, test: {test_acc:.4f}')
#     else:
#         print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')

# A = input_matrix()
# print(A.toarray())
# A1=A/np.sqrt(A.sum(axis=1))
# A2=A1.T/np.sqrt(A.sum(axis=0))
# A2=A2.T
# print(A2)
# print((A.sum(axis=1)))
# print(A.sum(axis=0))

import torch
import torch.nn.functional as F
from torch_sparse import spspmm

from torch_geometric.nn import GCNConv, TopKPooling
from torch_geometric.utils import (
    add_self_loops,
    remove_self_loops,
    sort_edge_index,
)
from torch_geometric.utils.repeat import repeat
from utils_file.utils import *
import random
import numpy as np
import torchinfo
from torchsummary import summary

torch.manual_seed(1763)
np.random.seed(453658)
random.seed(41884)
torch.cuda.manual_seed(9597121)
class GraphUNet(torch.nn.Module):
    r"""The Graph U-Net model from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = act
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(GCNConv(in_channels, channels, improved=True))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(GCNConv(channels, channels, improved=True))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(GCNConv(in_channels, channels, improved=True))
        self.up_convs.append(GCNConv(in_channels, out_channels, improved=True))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()


    def forward(self, x, edge_index, batch=None):
        """"""
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        
        x = self.down_convs[0](x, edge_index, edge_weight)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index, edge_weight)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        return x


    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight



    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, {self.out_channels}, '
                f'depth={self.depth}, pool_ratios={self.pool_ratios})')

model = GraphUNet(2, 5, 2, 2)
A = input_matrix()
print(A.toarray())

A = torch_sparse.remove_diag(st.from_scipy(A)).to_symmetric()
A = st.set_value(A, torch.ones_like(A.storage._value),layout='coo')
print(A.size(dim=0))
# rowcols = np.array([row,col])
# edges = torch.tensor(rowcols,dtype=torch.long)
row = A.storage._row
col = A.storage._col
edge_index = torch.vstack((row,col))
x = torch.randn(A.size(dim=0),2)
model.forward(x=x,edge_index=edge_index)



"""                                        GCN              
x = tensor([[-3.1371e-01,  1.3259e+00],  ------->   x = tensor([[ 0.3813, -1.2260,  0.3330,  0.3587, -0.8081],
           [ 1.5378e+00,  7.7990e-01],                        [-0.3950,  0.0832, -0.8068, -0.2036,  0.5675],
           [-7.7357e-01,  2.0818e+00],                        [ 0.1947, -1.1110, -0.0187,  0.2518, -0.5228],
           [-1.3409e+00, -4.9539e-01],                        [ 0.3090, -0.5248,  0.4522,  0.2243, -0.5484],
           [-6.5719e-03,  1.7900e-01],                        [ 0.0819, -0.4894, -0.0165,  0.1090, -0.2249],
           [-1.0002e-01,  1.7321e-03],                        [ 0.2685, -0.5973,  0.3380,  0.2149, -0.5086],
           [ 8.1550e-01,  8.8847e-01]])                       [-0.1976, -0.4846, -0.6084, -0.0274,  0.1643]])
                                                                                    |
                                                                                    |
                                                                                    |
                                                                                    |-------------》x = tensor([[0.0117, 0.0000, 0.0000, 0.0000, 0.0174],
                                                                                                                [0.0117, 0.0000, 0.0000, 0.0000, 0.0174],
                                                                                                                [0.0117, 0.0000, 0.0000, 0.0000, 0.0174],
                                                                                                                [0.0117, 0.0000, 0.0000, 0.0000, 0.0174]])




"""









"""
batch = tensor([0, 0, 0, 0, 0, 0, 0])
edge_weight = tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
edge_index = tensor([[0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6],
                     [2, 3, 4, 6, 0, 4, 5, 6, 0, 4, 5, 1, 2, 3, 2, 3, 1, 2]])
x = tensor([[-1.2020e+00, -8.0225e-01],
        [-7.2775e-01,  4.5368e-01],
        [-9.1953e-01,  1.2829e+00],
        [ 3.1200e-04,  1.5939e+00],
        [-4.9886e-01,  2.4894e+00],
        [ 9.9458e-01, -1.4677e+00],
        [ 3.2335e+00,  2.9766e-01]])
x.size(self.node_dim) = 7

out = tensor([[-0.2565,  0.7457,  0.1146,  0.5072,  0.2703],
              [ 0.8121,  0.2360,  0.5125, -0.5713, -0.2839],
              [ 0.4719,  0.1052,  0.2871, -0.3447, -0.1720],
              [ 0.3880,  0.4790,  0.3683, -0.1271, -0.0550],
              [ 0.9425,  1.3738,  0.9656, -0.2248, -0.0873],
              [ 0.0782, -0.3104, -0.0629, -0.1878, -0.1007],
              [ 1.0486, -0.6760,  0.3311, -1.1283, -0.5825]],
       grad_fn=<ScatterAddBackward0>)

theta = Parameter containing:
tensor([[ 0.5293,  0.7412],
        [-0.7858,  0.5801],
        [ 0.0173,  0.5907],
        [-0.7467, -0.3761],
        [-0.3919, -0.1788]], requires_grad=True)

edge_weight = tensor([0.2041, 0.2236, 0.2236, 0.2500, 0.2041, 0.1826, 0.2041, 0.2041, 0.2236,
        0.2000, 0.2236, 0.2236, 0.1826, 0.2000, 0.2041, 0.2236, 0.2500, 0.2041,
        0.5000, 0.5000, 0.3333, 0.4000, 0.4000, 0.5000, 0.5000])



gcn_norm(edge_index,edge_weight,x.size(self.node_dim))

"""

# x = torch.tensor([[-1.2020e+00, -8.0225e-01],
#                   [-7.2775e-01,  4.5368e-01],
#                   [-9.1953e-01,  1.2829e+00],
#                   [ 3.1200e-04,  1.5939e+00],
#                   [-4.9886e-01,  2.4894e+00],
#                   [ 9.9458e-01, -1.4677e+00],
#                   [ 3.2335e+00,  2.9766e-01]])

# theta = torch.tensor([[ 0.5293,  0.7412],
#                       [-0.7858,  0.5801],
#                       [ 0.0173,  0.5907],
#                       [-0.7467, -0.3761],
#                       [-0.3919, -0.1788]])

# edge_weight = torch.tensor([0.2041, 0.2236, 0.2236, 0.2500, 0.2041, 0.1826, 0.2041, 0.2041, 0.2236,
#         0.2000, 0.2236, 0.2236, 0.1826, 0.2000, 0.2041, 0.2236, 0.2500, 0.2041,
#         0.5000, 0.5000, 0.3333, 0.4000, 0.4000, 0.5000, 0.5000])
# edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 0, 1, 2, 3, 4, 5,
#          6],
#         [2, 3, 4, 6, 0, 4, 5, 6, 0, 4, 5, 1, 2, 3, 2, 3, 1, 2, 0, 1, 2, 3, 4, 5,
#          6]])

# s = torch.sparse_coo_tensor(edge_index, edge_weight)
# s = s.to_dense()
# print(s)

# x = x @ theta.t()
# x = s @ x
# print(x)


            
