import torch
import torch.nn as nn
from layers import *
from torch_geometric.nn import  avg_pool, graclus
from torch_geometric.data import Batch
from layers import SAGEConv
import pickle
from QGraph.v0_1.graph_coarsen import graph_coarsen


# Neural network for the embedding module
class ModelSpectral(torch.nn.Module):
    def __init__(self,se_params,device):
        super(ModelSpectral,self).__init__()
        self.l = se_params.get('l')
        self.pre = se_params.get('pre')
        self.post = se_params.get('post')
        self.coarsening_threshold = se_params.get('coarsening_threshold')
        self.activation = getattr(torch, se_params.get('activation'))
        self.lins = se_params.get('lins')

        self.conv_post = nn.ModuleList(
            [SAGEConv(self.l, self.l) for i in range(self.post)]
        )
        self.conv_coarse = SAGEConv(2,self.l)
        self.lins1=nn.Linear(self.l,self.lins[0])
        self.lins2=nn.Linear(self.lins[0],self.lins[1]) 
        self.lins3=nn.Linear(self.lins[1],self.lins[2]) 
        self.final=nn.Linear(self.lins[2],2)
        self.device = device

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        unpool_info = []
        x_info=[]
        cluster_info=[]
        edge_info=[]
        while x.size()[0] > self.coarsening_threshold:
            cluster = graclus(edge_index,num_nodes=x.shape[0])
            cluster_info.append(cluster)
            edge_info.append(edge_index)
            gc = avg_pool(cluster, Batch(batch=batch, x=x, edge_index=edge_index,shuffle=False))
            x, edge_index, batch = gc.x, gc.edge_index, gc.batch
        # coarse iterations
        x=torch.eye(2).to(self.device)
        x=self.conv_coarse(x,edge_index)
        x=self.activation(x)
        while edge_info:
            # un-pooling / interpolation / prolongation / refinement
            edge_index = edge_info.pop()
            cluster = cluster_info.pop()
            output, inverse = torch.unique(cluster, return_inverse=True)
            x = x[inverse]
            # post-smoothing
            for i in range(self.post):
                x = self.activation(self.conv_post[i](x, edge_index))
        x=self.lins1(x)
        x=self.activation(x)
        x=self.lins2(x)
        x=self.activation(x)
        x=self.lins3(x)
        x=self.activation(x)
        x=self.final(x)
        x,_=torch.linalg.qr(x,mode='reduced')
        return x

# Neural network for the partitioning module
class ModelPartitioning(torch.nn.Module):
    def __init__(self,pe_params):
        super(ModelPartitioning,self).__init__()

        self.l = pe_params.get('l')
        self.pre = pe_params.get('pre')
        self.post = pe_params.get('post')
        self.coarsening_threshold = pe_params.get('coarsening_threshold')
        self.activation = getattr(torch, pe_params.get('activation'))
        self.lins = pe_params.get('lins')

        self.conv_first = SAGEConv(1, self.l)
        self.conv_pre = nn.ModuleList(
            [SAGEConv(self.l, self.l) for i in range(self.pre)]
        )
        self.conv_post = nn.ModuleList(
            [SAGEConv(self.l, self.l) for i in range(self.post)]
        )
        self.conv_coarse = SAGEConv(self.l,self.l)
        
        self.lins1=nn.Linear(self.l,self.lins[0])
        self.lins2=nn.Linear(self.lins[0],self.lins[1]) 
        self.lins3=nn.Linear(self.lins[1],self.lins[2]) 
        self.final=nn.Linear(self.lins[4],2)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.activation(self.conv_first(x, edge_index))
        unpool_info = []
        x_info=[]
        cluster_info=[]
        edge_info=[]
        batches=[]
        while x.size()[0] > self.coarsening_threshold:
            # pre-smoothing
            for i in range(self.pre):
                x = self.activation(self.conv_pre[i](x, edge_index))
            # pooling / coarsening / restriction
            x_info.append(x)
            batches.append(batch)
            cluster = graph_coarsen(edge_index[0],edge_index[1],weight=None,num_nodes=x.shape[0])
            cluster_info.append(cluster)
            edge_info.append(edge_index)
            gc = avg_pool(cluster, Batch(batch=batch, x=x, edge_index=edge_index))
            x, edge_index, batch = gc.x, gc.edge_index, gc.batch
        # coarse iterations
        x = self.activation(self.conv_coarse(x,edge_index))
        while edge_info:
            # un-pooling / interpolation / prolongation / refinement
            edge_index = edge_info.pop()
            output, inverse = torch.unique(cluster_info.pop(), return_inverse=True)
            x = (x[inverse] + x_info.pop())/2
            # post-smoothing
            for i in range(self.post):
                x = self.activation(self.conv_post[i](x, edge_index))
        x=self.lins1(x)
        x=self.activation(x)
        x=self.lins2(x)
        x=self.activation(x)
        x=self.lins3(x)
        x=self.activation(x)
        x=self.final(x)
        x=torch.softmax(x,dim=1)
        return x
    
# 2022_10_26
class ModelSpectral_1026(torch.nn.Module):
    def __init__(self,se_params,device):
        super(ModelSpectral_1026,self).__init__()
        self.l = se_params.get('l')
        self.pre = se_params.get('pre')
        self.post = se_params.get('post')
        self.coarsening_threshold = se_params.get('coarsening_threshold')
        self.activation = getattr(torch, se_params.get('activation'))
        self.lins = se_params.get('lins')
        
        self.conv_post = nn.ModuleList(
            [SAGEConv(self.l, self.l) for i in range(self.post)]
            # [ChebConv(self.l, self.l,3) for i in range(self.post)]
        )
        self.conv_coarse = SAGEConv(2,self.l)
        # self.conv_coarse = ChebConv(2,self.l,3)
        self.lins1=nn.Linear(self.l,self.lins[0])
        # self.bn_1=nn.BatchNorm1d(self.lins[0])
        self.lins2=nn.Linear(self.lins[0],self.lins[1]) 
        # self.bn_2=nn.BatchNorm1d(self.lins[1])
        self.lins3=nn.Linear(self.lins[1],self.lins[2]) 
        # self.bn_3=nn.BatchNorm1d(self.lins[2])
        self.final=nn.Linear(self.lins[2],2)
        # self.bn_qr=nn.BatchNorm1d(2)
        self.device = device
        
        self.arr_edge = None
        
    
    def forward(self,graph):
        if self.arr_edge is None:
            x,edge_index,batch=graph.x,graph.edge_index,graph.batch
            cluster_info=[]
            edge_info=[]
            while x.size()[0]>self.coarsening_threshold:
                cluster = graph_coarsen(edge_index[0],edge_index[1],weight=None,num_nodes=x.shape[0])
                cluster_info.append(cluster)
                edge_info.append(edge_index)
                gc = avg_pool(cluster, Batch(batch=batch, x=x, edge_index=edge_index,shuffle=False))
                x, edge_index, batch = gc.x, gc.edge_index, gc.batch
            self.arr_cluster = cluster_info
            self.arr_edge = edge_info
            self.last_gc = gc
        
        
        nMap = len(self.arr_cluster)
        edge_index = self.last_gc.edge_index
        x = torch.eye(2).to(self.device)
        x = self.conv_coarse(x,edge_index)
        x = self.activation(x)
        for map_no in reversed(range(nMap)):
            edge_index = self.arr_edge[map_no]
            cluster = self.arr_cluster[map_no]
            output, inverse = torch.unique(cluster, return_inverse=True)
            x = x[inverse]
            for i in range(self.post):
                x = self.activation(self.conv_post[i](x, edge_index))
        x=self.lins1(x)
        x=self.activation(x)
        x=self.lins2(x)
        x=self.activation(x)
        x=self.lins3(x)
        x=self.activation(x)
        x=self.final(x)
        x,_=torch.linalg.qr(x,mode='reduced')
        
        return x
    
    def get_info(self):
        return self.arr_cluster,self.arr_edge,self.last_gc
        
class ModelPartitioning_1026_1v1(torch.nn.Module):
    def __init__(self,pe_params):
        super(ModelPartitioning_1026_1v1,self).__init__()

        self.l = pe_params.get('l')
        self.pre = pe_params.get('pre')
        self.post = pe_params.get('post')
        self.coarsening_threshold = pe_params.get('coarsening_threshold')
        self.activation = getattr(torch, pe_params.get('activation'))
        self.lins = pe_params.get('lins')

        self.conv_first = SAGEConv(1, self.l)
        self.conv_pre = nn.ModuleList(
            [SAGEConv(self.l, self.l) for i in range(self.pre)]
        )
        self.conv_post = nn.ModuleList(
            [SAGEConv(self.l, self.l) for i in range(self.post)]
        )
        self.conv_coarse = SAGEConv(self.l,self.l)
        
        self.lins1=nn.Linear(self.l,self.lins[0])
        self.lins2=nn.Linear(self.lins[0],self.lins[1]) 
        self.lins3=nn.Linear(self.lins[1],self.lins[2]) 
        self.final=nn.Linear(self.lins[4],2)
        
        self.arr_edge = None

    def forward(self,graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        x = self.activation(self.conv_first(x, edge_index))
        x_origin = x
        if self.arr_edge is None:

            cluster_info = []
            edge_info = []
            while x.size()[0] > self.coarsening_threshold:
                # pooling / coarsening /restriction
                cluster = graph_coarsen(edge_index[0],edge_index[1],weight=None,num_nodes=x.shape[0])
                cluster_info.append(cluster)
                edge_info.append(edge_index)
                gc = avg_pool(cluster, Batch(batch=batch, x=x, edge_index=edge_index))
                x,edge_index,batch = gc.x,gc.edge_index,gc.batch
            self.arr_cluster = cluster_info
            self.arr_edge = edge_info
            self.last_gc = gc
        
        x_info = []
        x = x_origin
        nMap = len(self.arr_cluster)
        for map_no in range(nMap):
            for i in range(self.pre):
                x = self.activation(self.conv_pre[i](x, edge_index))
            x_info.append(x)
            gc = avg_pool(self.arr_cluster[map_no], Batch(batch=batch, x=x, edge_index=self.arr_edge[map_no]))
            x,edge_index,batch = gc.x,gc.edge_index,gc.batch

        self.arr_x = x_info
           
        # coarsen iterations
        x = self.activation(self.conv_coarse(x,edge_index))
        for map_no in reversed(range(nMap)):
            # un-pooling / interpolation / prolongation / refinement
            edge_index = self.arr_edge[map_no]
            cluster = self.arr_cluster[map_no]
            output_index,inverse = torch.unique(cluster,return_inverse=True)
            x = (x[inverse] + self.arr_x[map_no])/2
            # post-smoothing
            for i in range(self.post):
                x = self.activation(self.conv_post[i](x, edge_index))
                    
        x=self.lins1(x)
        x=self.activation(x)
        x=self.lins2(x)
        x=self.activation(x)
        x=self.lins3(x)
        x=self.activation(x)
        x=self.final(x)
        x=torch.softmax(x,dim=1)
        return x
            
        
        
        
    
    
# def down_sampling(graph,coarsening_threshold):
#     x,edge_index,batch = graph.x,graph.edge_index,graph.batch
#     cluster_info = []
#     edge_info = []
#     while x.size()[0]>coarsening_threshold:
#         cluster = graph_coarsen(edge_index[0], edge_index[1],weight=None,num_nodes=x.shape[0])
#         cluster_info.append(cluster)
#         edge_info.append(edge_index)
#         gc = avg_pool(cluster,Batch(batch=batch, x=x, edge_index=edge_index,shuffle=False))
#         x,edge_index,batch = gc.x,gc.edge_index,gc.batch
#     return cluster_info,edge_info,gc