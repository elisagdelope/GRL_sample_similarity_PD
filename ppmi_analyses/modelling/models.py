import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, ChebConv, GATConv, GINConv, GATv2Conv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, in_f, h_f, out_f, p_dropout):
        super().__init__()
        torch.manual_seed(42)
        self.lin1 = nn.Linear(in_f, h_f)
        self.lin2 = nn.Linear(h_f, out_f)
        self.bn1 = nn.BatchNorm1d(h_f)
        self.p_dropout = p_dropout

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin2(x)
        return out

class MLP2(torch.nn.Module):
    def __init__(self, in_f, h1_f, h2_f, out_f, p_dropout):
        super().__init__()
        torch.manual_seed(42)
        self.lin1 = nn.Linear(in_f, h1_f)
        self.lin2 = nn.Linear(h1_f, h2_f)
        self.lin3 = nn.Linear(h2_f, out_f)
        self.bn1 = nn.BatchNorm1d(h1_f)
        self.bn2 = nn.BatchNorm1d(h2_f)
        self.p_dropout = p_dropout

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin3(x)
        return out


class GCN(nn.Module):
    def __init__(self, in_f, h_f, out_f, p_dropout):
        super(GCN, self).__init__()
        torch.manual_seed(42)
        self.conv1 = GCNConv(in_f, h_f)
        # FC1
        self.lin1 = nn.Linear(h_f, out_f)
        self.bn1 = nn.BatchNorm1d(h_f)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_weight):
        #x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out


class Cheb_GCNN(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, K, out_f, p_dropout): #, DL1_F, DL2_F
        super(Cheb_GCNN, self).__init__()

        # graph CL1
        self.conv1 = ChebConv(in_channels=in_f, out_channels=CL1_F, K=K)
        # graph CL2
        self.conv2 = ChebConv(in_channels=CL1_F, out_channels=CL2_F, K=K)
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_weight):
        #x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)

        return out # returns the embedding x & prediction out


class Cheb_GCNN_uw(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, K, out_f, p_dropout):  # , DL1_F, DL2_F
        super(Cheb_GCNN_uw, self).__init__()

        # graph CL1
        self.conv1 = ChebConv(in_channels=in_f, out_channels=CL1_F, K=K)
        # graph CL2
        self.conv2 = ChebConv(in_channels=CL1_F, out_channels=CL2_F, K=K)
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index):
        #x, edge_index = data.x, data.edge_index
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out  # returns the embedding x & prediction out


class GCNN(nn.Module):
    def __init__(self, in_f ,CL1_F, CL2_F, out_f, p_dropout):
        super(GCNN, self).__init__()

        # graph CL1
        self.conv1 = GCNConv(in_channels=in_f, out_channels=CL1_F)
        # graph CL2
        self.conv2 = GCNConv(in_channels=CL1_F, out_channels=CL2_F)
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_weight):
        #x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        # node embeddings:
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)

        return out # returns the embedding x & prediction out


class GCNN_uw(nn.Module):
    def __init__(self, in_f ,CL1_F, CL2_F, out_f, p_dropout):
        super(GCNN_uw, self).__init__()

        # graph CL1
        self.conv1 = GCNConv(in_channels=in_f, out_channels=CL1_F)
        # graph CL2
        self.conv2 = GCNConv(in_channels=CL1_F, out_channels=CL2_F)
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index):
        #x, edge_index = data.x, data.edge_index
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)

        return out # returns the embedding x & prediction out

class pw_GCNN(nn.Module):
    def __init__(self, sparse_mask, in_mask_f, out_mask_f, CL1_F, CL2_F, out_f, p_dropout):
        super(pw_GCNN, self).__init__()
        self.sparse_masked = SparseMaskedLinear_v2(in_mask_f, out_mask_f, sparse_mask, device=device)
        # graph CL1
        self.conv1 = GCNConv(in_channels=out_mask_f, out_channels=CL1_F)
        # graph CL2
        self.conv2 = GCNConv(in_channels=CL1_F, out_channels=CL2_F)
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        self.bn_sparse = nn.BatchNorm1d(out_mask_f)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_weight):
        #x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # pw mask:
        x = F.relu(self.sparse_masked(x))
        x= self.bn_sparse(x)
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out # returns the embedding x & prediction out


class pw_GCNN_uw(nn.Module):
    def __init__(self, sparse_mask, in_mask_f, out_mask_f, CL1_F, CL2_F, out_f, p_dropout):
        super(pw_GCNN_uw, self).__init__()
        self.sparse_masked = SparseMaskedLinear_v2(in_mask_f, out_mask_f, sparse_mask, device=device)
        # graph CL1
        self.conv1 = GCNConv(in_channels=out_mask_f, out_channels=CL1_F)
        # graph CL2
        self.conv2 = GCNConv(in_channels=CL1_F, out_channels=CL2_F)
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        self.bn_sparse = nn.BatchNorm1d(out_mask_f)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index):
        #x, edge_index = data.x, data.edge_index
        # pw mask:
        x = self.sparse_masked(x)
        x= self.bn_sparse(x)
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out # returns the embedding x & prediction out


class pw_Cheb_GCNN1(nn.Module):
    def __init__(self, sparse_mask, in_mask_f, out_mask_f, CL1_F, K, out_f, p_dropout):  # , DL1_F, DL2_F
        super(pw_Cheb_GCNN1, self).__init__()
        self.sparse_masked = SparseMaskedLinear_v3(in_mask_f, out_mask_f, sparse_mask, device=device)
        # graph CL1
        self.conv1 = ChebConv(in_channels=out_mask_f, out_channels=CL1_F, K=K)
        # FC1
        self.lin1 = nn.Linear(CL1_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn_sparse = nn.BatchNorm1d(out_mask_f)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_weight):
        #x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # pw mask:
        x = F.relu(self.sparse_masked(x))
        x = self.bn_sparse(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)

        return out  # returns the embedding x & prediction out


class pw_Cheb_GCNN(nn.Module):
    def __init__(self, sparse_mask, in_mask_f, out_mask_f, CL1_F, CL2_F, K, out_f, p_dropout):  # , DL1_F, DL2_F
        super(pw_Cheb_GCNN, self).__init__()
        self.sparse_masked = SparseMaskedLinear_v2(in_mask_f, out_mask_f, sparse_mask, device=device)
        # graph CL1
        self.conv1 = ChebConv(in_channels=out_mask_f, out_channels=CL1_F, K=K)
        # graph CL2
        self.conv2 = ChebConv(in_channels=CL1_F, out_channels=CL2_F, K=K)
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn_sparse = nn.BatchNorm1d(out_mask_f)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_weight):
        #x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # pw mask:
        x = self.sparse_masked(x)
        x = self.bn_sparse(x)
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)

        return out  # returns the embedding x & prediction out


class pw_Cheb_GCNN_uw(nn.Module):
    def __init__(self, sparse_mask, in_mask_f, out_mask_f, CL1_F, CL2_F, K, out_f, p_dropout):  # , DL1_F, DL2_F
        super(pw_Cheb_GCNN_uw, self).__init__()
        self.sparse_masked = SparseMaskedLinear_v2(in_mask_f, out_mask_f, sparse_mask, device=device)
        # graph CL1
        self.conv1 = ChebConv(in_channels=out_mask_f, out_channels=CL1_F, K=K)
        # graph CL2
        self.conv2 = ChebConv(in_channels=CL1_F, out_channels=CL2_F, K=K)
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn_sparse = nn.BatchNorm1d(out_mask_f)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index):
        #x, edge_index = data.x, data.edge_index
        # pw mask:
        x = self.sparse_masked(x)
        x = self.bn_sparse(x)
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)

        return out  # returns the embedding x & prediction out



def broadcast(src, other, dim):
    # Source: torch_scatter
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


class SparseMaskedLinear_v2(nn.Module):
    """ Masked linear layer with sparse mask AND sparse weight matrix (faster and more memory efficient) """
    def __init__(self, in_features, out_features, sparse_mask, bias=True, device=None, dtype=None):
        """
        in_features: number of input features
        out_features: number of output features
        sparse_mask: torch tensor of shape (n_connections, 2), where indices[:, 0] index the input neurons
                     and indices[:, 1] index the output neurons
        """
        # Reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
        self.sparse_mask = sparse_mask
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.nn.init.normal_(torch.empty((sparse_mask.shape[0]), **factory_kwargs)))  # Shape=(n_connections,)
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, **factory_kwargs))
    def forward(self, input):
        # weight shape: (out_features, in_features)
        x = input[:, self.sparse_mask[:, 0]]  # Shape=(batch_size, n_connections)
        src = x * self.weight[None, :]  # Shape=(batch_size, n_connections)
        # Reduce via scatter sum
        out = torch.zeros((x.shape[0], self.out_features), dtype=x.dtype, device=x.device)  # Shape=(batch_size, out_features)
        index = broadcast(self.sparse_mask[:, 1], src, dim=-1)
        out = out.scatter_add_(dim=-1, index=index, src=src)
        if self.use_bias:
            out = out + self.bias
        return out
    def reset_parameters(self):
        nn.init.normal_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

class SparseMaskedLinear_v3(nn.Module):
    """ Masked linear layer with sparse mask AND sparse weight matrix (faster and more memory efficient) """
    def __init__(self, in_features, out_features, sparse_mask, K=1, bias=True, device=None, dtype=None):
        """
        in_features: number of input features
        out_features: number of output features
        sparse_mask: torch tensor of shape (n_connections, 2), where indices[:, 0] index the input neurons
                     and indices[:, 1] index the output neurons
        """
        # Reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
        self.sparse_mask = sparse_mask
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.weight = nn.Parameter(
            torch.nn.init.normal_(torch.empty((sparse_mask.shape[0]), **factory_kwargs)))  # Shape=(n_connections,)
        self.use_bias = bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features*K, **factory_kwargs))
    def forward(self, input):
        # weight shape: (out_features, in_features)
        x = input[:, self.sparse_mask[:, 0]]  # Shape=(batch_size, n_connections)
        src = x * self.weight[None, :]  # Shape=(batch_size, n_connections)
        # Reduce via scatter sum
        out = torch.zeros((x.shape[0], self.out_features*self.K), dtype=x.dtype, device=x.device)  # Shape=(batch_size, out_features)
        index = broadcast(self.sparse_mask[:, 1], src, dim=-1)
        out = out.scatter_add_(dim=-1, index=index, src=src)
        if self.use_bias:
            out = out + self.bias
        return out
    def reset_parameters(self):
        nn.init.normal_(self.weight)
        if self.use_bias:
            nn.init.zeros_(self.bias)

class pw_MLP(nn.Module):
    def __init__(self, sparse_mask, in_mask_f, out_mask_f, h_f, out_f, p_dropout):
        super(pw_MLP, self).__init__()
        torch.manual_seed(42)
        self.sparse_masked = SparseMaskedLinear_v2(in_mask_f, out_mask_f, sparse_mask, device=device)
        self.lin1 = nn.Linear(out_mask_f, h_f)
        self.lin2 = nn.Linear(h_f, out_f)
        self.bn1 = nn.BatchNorm1d(out_mask_f)
        self.bn2 = nn.BatchNorm1d(h_f)
        self.p_dropout = p_dropout

    def forward(self, data):
        #x = data.x
        x = data
        x = self.sparse_masked(x)
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin2(x)
        return out


class pw_MLP2(nn.Module):
    def __init__(self, sparse_mask, in_mask_f, out_mask_f, h1_f, h2_f, out_f, p_dropout):
        super(pw_MLP2, self).__init__()
        torch.manual_seed(42)
        self.sparse_masked = SparseMaskedLinear_v2(in_mask_f, out_mask_f, sparse_mask, device=device)
        self.lin1 = nn.Linear(out_mask_f, h1_f)
        self.lin2 = nn.Linear(h1_f, h2_f)
        self.lin3 = nn.Linear(h2_f, out_f)
        self.bn1 = nn.BatchNorm1d(out_mask_f)
        self.bn2 = nn.BatchNorm1d(h1_f)
        self.bn3 = nn.BatchNorm1d(h2_f)
        self.p_dropout = p_dropout

    def forward(self, data):
        x = data.x
        #x = data
        x = F.relu(self.sparse_masked(x))
        x = self.bn1(x)
        x = self.sparse_masked(x)
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.bn3(x)
        # classifier
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin3(x)
        return out



class GAT(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, heads, out_f, p_dropout):
        super(GAT, self).__init__()
        # graph CL1
        self.gat1 = GATv2Conv(in_channels=in_f, out_channels=CL1_F, heads=heads, edge_dim=1)
        # graph CL2
        self.gat2 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL2_F, heads=1, concat=False)
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F * heads)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        self.p_dropout = p_dropout
    def forward(self, x, edge_index, edge_attr):
        #x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # node embeddings:
        x = self.gat1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.gat2(x, edge_index, edge_attr) #*
        x = F.relu(x)
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out # returns the embedding x & prediction out


class GAT_uw(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, heads, out_f, p_dropout):
        super(GAT_uw, self).__init__()
        # graph CL1
        self.gat1 = GATv2Conv(in_channels=in_f, out_channels=CL1_F, heads=heads)
        # graph CL2
        self.gat2 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL2_F, heads=1, concat=False)
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F * heads)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        self.p_dropout = p_dropout
    def forward(self, x, edge_index):
        #x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # node embeddings:
        x = F.relu(self.gat1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out # returns the embedding x & prediction out


class GIN_uw(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, out_f, p_dropout):  # , DL1_F, DL2_F
        super(GIN_uw, self).__init__()

        # graph CL1
        self.conv1 = GINConv(
            nn.Sequential(nn.Linear(in_f, CL1_F),
                       nn.BatchNorm1d(CL1_F), nn.ReLU(),
                       nn.Linear(CL1_F, CL1_F), nn.ReLU()))
        self.conv2 = GINConv(
            nn.Sequential(nn.Linear(CL1_F, CL2_F),
                       nn.BatchNorm1d(CL2_F), nn.ReLU(),
                       nn.Linear(CL2_F, CL2_F), nn.ReLU()))
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index):
        #x, edge_index = data.x, data.edge_index
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))

        # Classifier
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)

        return out  # returns the embedding x & prediction out


class GCNN3L(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, CL3_F, out_f):
        super(GCNN3L, self).__init__()

        # graph CL1
        self.conv1 = GCNConv(in_channels=in_f, out_channels=CL1_F)
        # graph CL2
        self.conv2 = GCNConv(in_channels=CL1_F, out_channels=CL2_F)
        # graph CL3
        self.conv3 = GCNConv(in_channels=CL2_F, out_channels=CL3_F)
        # FC1
        self.lin1 = nn.Linear(CL3_F, out_f)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # node embeddings:
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))


        out = self.lin1(x)

        return out  # returns the embedding x & prediction out
