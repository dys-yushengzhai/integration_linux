import torch
import torch.nn.functional as F
from torch_sparse import spspmm

from torch_geometric.nn import GCNConv, TopKPooling,SAGPooling,ASAPooling
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
from test_layers import *
torch.manual_seed(1763)
np.random.seed(453658)
random.seed(41884)
torch.cuda.manual_seed(9597121)

# data
A = input_matrix()
print(A.toarray())

A = torch_sparse.remove_diag(st.from_scipy(A)).to_symmetric()
A = st.set_value(A, torch.ones_like(A.storage._value),layout='coo')
# rowcols = np.array([row,col])
# edges = torch.tensor(rowcols,dtype=torch.long)
row = A.storage._row
col = A.storage._col

edge_index = torch.vstack((row,col))
x = torch.randn(A.size(dim=0),2)

# example of TopKPooling(gpool)
print(edge_index.shape)
print(x)

model_topKPooling = TopKPooling(2)
# model_topKPooling.forward(x=x,edge_index=edge_index)

# example of self-attention graph pooling
model_SAGPooling = SAGPooling(2)
# model_SAGPooling.forward(x=x, edge_index=edge_index)

# example of ASAPooling
model_ASAP = ASAPooling(2)
model_ASAP.forward(x=x, edge_index=edge_index)
