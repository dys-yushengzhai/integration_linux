
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

    def forward(self, graph):        
        if self.arr_edge is None:
            x, edge_index, batch = graph.x, graph.edge_index, graph.batch
            # unpool_info = []
            # x_info=[]        
            cluster_info=[]
            edge_info=[]
            # self.coarsening_threshold = x.size()[0]/2
            while x.size()[0] > self.coarsening_threshold:
                cluster = graclus(edge_index,num_nodes=x.shape[0])
                cluster_info.append(cluster)
                edge_info.append(edge_index)
                gc = avg_pool(cluster, Batch(batch=batch, x=x, edge_index=edge_index,shuffle=False))
                x, edge_index, batch = gc.x, gc.edge_index, gc.batch
            self.arr_cluster = cluster_info
            self.arr_edge = edge_info            
            self.last_gc = gc
        # coarse iterations   
        nMap = len(self.arr_cluster)     
        edge_index = self.last_gc.edge_index
        x=torch.eye(2).to(self.device)
        x=self.conv_coarse(x,edge_index)
        x=self.activation(x)
        for map_no in reversed(range(nMap)):
        # while edge_info:
            # un-pooling / interpolation / prolongation / refinement
            # edge_index = edge_info.pop()
            # cluster = cluster_info.pop()
            edge_index = self.arr_edge[map_no]
            cluster = self.arr_cluster[map_no]
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
        # x=self.bn_qr(x)
        x,_=torch.linalg.qr(x,mode='reduced')
        # x=self.bn_qr(x)
        # x=torch.linalg.svd(x,full_matrices=False).U
        
        return x