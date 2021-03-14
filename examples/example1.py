# 1) Creating Message Passing Networks
# MessagePassing, propagate, message, update

# Implementing the GCN Layer
import os.path as osp
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.datasets import Planetoid

class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = torch.nn.Linear(in_channels, out_channels).cuda()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        # propagate: internally calls message, aggregate, update functions
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset)
dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
data = dataset[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x, edge_index = data.x.to(device), data.edge_index.to(device)

# 실험
edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
lin = torch.nn.Linear(1433, 32).cuda()
x = lin(x)

# edge_index = [2, edge수]
# row, col은 단지 위 edge_index에서 각각 source node, target node 기준으로 [edge수, ], [edge수, ]로 나눈 것
row, col = edge_index
# row, col 중 하나만 사용하면 되므로 degree를 계산함
# 예를 들어 4번 node가 총 6개의 edge를 갖고 있다면 deg[4] = 6
deg = degree(col, x.size(0), dtype=x.dtype)
deg_inv_sqrt = deg.pow(-0.5)
# 모든 edge(self.loop 포함)에 대해 정규화 상수를 생성함
# 지금 예시에서 norm.size() = (13264)
norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

# 전체
conv = GCNConv(1433, 32)
out = conv(x, edge_index)

#----
# 2) Creating your own datasets
# torch_geometric.data.Dataset, torch_geometric.data.InMemoryDataset(CPU memory)
# raw_dir: where the dataset gets downloaded to
# processed_dir: where the processed dataset is being saved
# (1) transform
# (2) pre_transform
# (3) pre_filter

#---
# 3) Advanced Mini-batching
# 근접 행렬은 대각형태로 쌓여있기 때문에 고립된 부분 복수의 부분 그래프를 생성한다.
# data.Dataloader

# Bipartite Graphs
# 이분 그래프에서 source nodes of edges in edge_index는 target nodex of edges in edge_index와 다르게 증가해야 함
# 2 node type: x_s, x_t
import torch_geometric

class BipartiteData(torch_geometric.data.Data):
    def __init__(self, edge_index, x_s, x_t):
        super(BipartiteData, self).__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t

    def __inc__(self, key, value):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value)

# increment source and target nodes of edges in edge_index independently on each other

edge_index = torch.tensor([
    [0, 0, 1, 1],
    [0, 1, 1, 2],
])
x_s = torch.randn(2, 16)  # 2 nodes.
x_t = torch.randn(3, 16)  # 3 nodes.

data = BipartiteData(edge_index, x_s, x_t)
data_list = [data, data]
loader = torch_geometric.data.DataLoader(data_list, batch_size=2)
batch = next(iter(loader))






