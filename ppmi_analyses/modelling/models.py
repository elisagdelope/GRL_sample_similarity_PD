import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, ChebConv, GATConv, GINConv, GATv2Conv, GraphUNet, TransformerConv, GPSConv, GINEConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool, BatchNorm
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

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

    def forward(self, x, edge_index):
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
        return out  # returns the embedding x & prediction out

class Cheb_GCNN_10L_uw(nn.Module):
    def __init__(self, in_f, CL1_F, K, out_f, p_dropout):
        super(Cheb_GCNN_10L_uw, self).__init__()
        # graphCL
        self.conv1 = ChebConv(in_channels=in_f, out_channels=CL1_F, K=K)
        self.conv2 = ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K)
        self.conv3 = ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K)
        self.conv4 = ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K)
        self.conv5 = ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K)
        self.conv6 = ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K)
        self.conv7 = ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K)
        self.conv8 = ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K)
        self.conv9 = ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K)
        self.conv10 = ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K)
        # FC
        self.lin1 = nn.Linear(CL1_F, out_f)
        # BN
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn2 = nn.BatchNorm1d(CL1_F)
        self.bn3 = nn.BatchNorm1d(CL1_F)
        self.bn4 = nn.BatchNorm1d(CL1_F)
        self.bn5 = nn.BatchNorm1d(CL1_F)
        self.bn6 = nn.BatchNorm1d(CL1_F)
        self.bn7 = nn.BatchNorm1d(CL1_F)
        self.bn8 = nn.BatchNorm1d(CL1_F)
        self.bn9 = nn.BatchNorm1d(CL1_F)
        self.bn10 = nn.BatchNorm1d(CL1_F)
        self.p_dropout = p_dropout
    def forward(self, x, edge_index):
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv6(x, edge_index))
        x = self.bn6(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv7(x, edge_index))
        x = self.bn7(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv8(x, edge_index))
        x = self.bn8(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv9(x, edge_index))
        x = self.bn9(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv10(x, edge_index))
        x = self.bn10(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out  # returns the embedding x & prediction out

class Cheb_GCNN_10L(nn.Module):
    def __init__(self, in_f, CL1_F, K, out_f, p_dropout):
        super(Cheb_GCNN_10L, self).__init__()
        # graphCL
        self.conv1 = ChebConv(in_channels=in_f, out_channels=CL1_F, K=K)
        self.conv2 = ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K)
        self.conv3 = ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K)
        self.conv4 = ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K)
        self.conv5 = ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K)
        self.conv6 = ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K)
        self.conv7 = ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K)
        self.conv8 = ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K)
        self.conv9 = ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K)
        self.conv10 = ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K)
        # FC
        self.lin1 = nn.Linear(CL1_F, out_f)
        # BN
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn2 = nn.BatchNorm1d(CL1_F)
        self.bn3 = nn.BatchNorm1d(CL1_F)
        self.bn4 = nn.BatchNorm1d(CL1_F)
        self.bn5 = nn.BatchNorm1d(CL1_F)
        self.bn6 = nn.BatchNorm1d(CL1_F)
        self.bn7 = nn.BatchNorm1d(CL1_F)
        self.bn8 = nn.BatchNorm1d(CL1_F)
        self.bn9 = nn.BatchNorm1d(CL1_F)
        self.bn10 = nn.BatchNorm1d(CL1_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_weight):
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = self.bn3(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv4(x, edge_index, edge_weight))
        x = self.bn4(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv5(x, edge_index, edge_weight))
        x = self.bn5(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv6(x, edge_index, edge_weight))
        x = self.bn6(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv7(x, edge_index, edge_weight))
        x = self.bn7(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv8(x, edge_index, edge_weight))
        x = self.bn8(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv9(x, edge_index, edge_weight))
        x = self.bn9(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv10(x, edge_index, edge_weight))
        x = self.bn10(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out  # returns the embedding x & prediction out


class Cheb_GCNN_50L(nn.Module):
    def __init__(self, in_f, CL1_F, K, out_f, p_dropout):
        super(Cheb_GCNN_50L, self).__init__()
        # graphCL
        self.convs = nn.ModuleList([ChebConv(in_channels=CL1_F if i > 0 else in_f, out_channels=CL1_F, K=K) for i in range(50)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(CL1_F) for _ in range(50)])
        self.lin1 = nn.Linear(CL1_F, out_f)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_weight):
        # node embeddings:
        for i in range(50):
            x = F.relu(self.convs[i](x, edge_index, edge_weight))
            x = self.bns[i](x)
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

class GCNN_10L_uw(nn.Module):
    def __init__(self, in_f, CL1_F, out_f, p_dropout):
        super(GCNN_10L_uw, self).__init__()
        # graphCL
        self.conv1 = GCNConv(in_channels=in_f, out_channels=CL1_F)
        self.conv2 = GCNConv(in_channels=CL1_F, out_channels=CL1_F)
        self.conv3 = GCNConv(in_channels=CL1_F, out_channels=CL1_F)
        self.conv4 = GCNConv(in_channels=CL1_F, out_channels=CL1_F)
        self.conv5 = GCNConv(in_channels=CL1_F, out_channels=CL1_F)
        self.conv6 = GCNConv(in_channels=CL1_F, out_channels=CL1_F)
        self.conv7 = GCNConv(in_channels=CL1_F, out_channels=CL1_F)
        self.conv8 = GCNConv(in_channels=CL1_F, out_channels=CL1_F)
        self.conv9 = GCNConv(in_channels=CL1_F, out_channels=CL1_F)
        self.conv10 = GCNConv(in_channels=CL1_F, out_channels=CL1_F)
        # FC
        self.lin1 = nn.Linear(CL1_F, out_f)
        # BN
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn2 = nn.BatchNorm1d(CL1_F)
        self.bn3 = nn.BatchNorm1d(CL1_F)        
        self.bn4 = nn.BatchNorm1d(CL1_F)
        self.bn5 = nn.BatchNorm1d(CL1_F)
        self.bn6 = nn.BatchNorm1d(CL1_F)
        self.bn7 = nn.BatchNorm1d(CL1_F)
        self.bn8 = nn.BatchNorm1d(CL1_F)
        self.bn9 = nn.BatchNorm1d(CL1_F)
        self.bn10 = nn.BatchNorm1d(CL1_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index):
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv6(x, edge_index))
        x = self.bn6(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv7(x, edge_index))
        x = self.bn7(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv8(x, edge_index))
        x = self.bn8(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv9(x, edge_index))
        x = self.bn9(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv10(x, edge_index))
        x = self.bn10(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out  # returns the embedding x & prediction out

class GCNN_10L(nn.Module):
    def __init__(self, in_f, CL1_F, out_f, p_dropout):
        super(GCNN_10L, self).__init__()
        # graphCL
        self.conv1 = GCNConv(in_channels=in_f, out_channels=CL1_F)
        self.conv2 = GCNConv(in_channels=CL1_F, out_channels=CL1_F)
        self.conv3 = GCNConv(in_channels=CL1_F, out_channels=CL1_F)
        self.conv4 = GCNConv(in_channels=CL1_F, out_channels=CL1_F)
        self.conv5 = GCNConv(in_channels=CL1_F, out_channels=CL1_F)
        self.conv6 = GCNConv(in_channels=CL1_F, out_channels=CL1_F)
        self.conv7 = GCNConv(in_channels=CL1_F, out_channels=CL1_F)
        self.conv8 = GCNConv(in_channels=CL1_F, out_channels=CL1_F)
        self.conv9 = GCNConv(in_channels=CL1_F, out_channels=CL1_F)
        self.conv10 = GCNConv(in_channels=CL1_F, out_channels=CL1_F)
        # FC
        self.lin1 = nn.Linear(CL1_F, out_f)
        # BN
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn2 = nn.BatchNorm1d(CL1_F)
        self.bn3 = nn.BatchNorm1d(CL1_F)        
        self.bn4 = nn.BatchNorm1d(CL1_F)
        self.bn5 = nn.BatchNorm1d(CL1_F)
        self.bn6 = nn.BatchNorm1d(CL1_F)
        self.bn7 = nn.BatchNorm1d(CL1_F)
        self.bn8 = nn.BatchNorm1d(CL1_F)
        self.bn9 = nn.BatchNorm1d(CL1_F)
        self.bn10 = nn.BatchNorm1d(CL1_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_weight):
        # node embeddings:
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = self.bn3(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv4(x, edge_index, edge_weight))
        x = self.bn4(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv5(x, edge_index, edge_weight))
        x = self.bn5(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv6(x, edge_index, edge_weight))
        x = self.bn6(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv7(x, edge_index, edge_weight))
        x = self.bn7(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv8(x, edge_index, edge_weight))
        x = self.bn8(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv9(x, edge_index, edge_weight))
        x = self.bn9(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv10(x, edge_index, edge_weight))
        x = self.bn10(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out  # returns the embedding x & prediction out

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
    def __init__(self, sparse_mask, in_mask_f, out_mask_f, CL1_F, CL2_F, out_f, p_dropout, device):
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
    def __init__(self, sparse_mask, in_mask_f, out_mask_f, CL1_F, CL2_F, out_f, p_dropout, device):
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
    def __init__(self, sparse_mask, in_mask_f, out_mask_f, CL1_F, K, out_f, p_dropout, device):  # , DL1_F, DL2_F
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
    def __init__(self, sparse_mask, in_mask_f, out_mask_f, CL1_F, CL2_F, K, out_f, p_dropout, device):  # , DL1_F, DL2_F
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
    def __init__(self, sparse_mask, in_mask_f, out_mask_f, CL1_F, CL2_F, K, out_f, p_dropout, device):  # , DL1_F, DL2_F
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
    def __init__(self, sparse_mask, in_mask_f, out_mask_f, h_f, out_f, p_dropout, device):
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
    def __init__(self, sparse_mask, in_mask_f, out_mask_f, h1_f, h2_f, out_f, p_dropout, device):
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
        self.gat2 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL2_F, heads=1, concat=False, edge_dim=1)
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
        x = self.gat2(x, edge_index, edge_attr)
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


class GAT_10L_uw(nn.Module):
    def __init__(self, in_f, CL1_F, heads, out_f, p_dropout):
        super(GAT_10L_uw, self).__init__()
        # graphCL
        self.gat1 = GATv2Conv(in_channels=in_f, out_channels=CL1_F, heads=heads)
        self.gat2 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        self.gat3 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        self.gat4 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        self.gat5 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        self.gat6 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        self.gat7 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        self.gat8 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        self.gat9 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        self.gat10 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=1, concat=False)
        # FC
        self.lin1 = nn.Linear(CL1_F, out_f)
        # BN
        self.bn1 = nn.BatchNorm1d(CL1_F * heads)
        self.bn2 = nn.BatchNorm1d(CL1_F * heads)
        self.bn3 = nn.BatchNorm1d(CL1_F * heads)
        self.bn4 = nn.BatchNorm1d(CL1_F * heads)
        self.bn5 = nn.BatchNorm1d(CL1_F * heads)
        self.bn6 = nn.BatchNorm1d(CL1_F * heads)
        self.bn7 = nn.BatchNorm1d(CL1_F * heads)
        self.bn8 = nn.BatchNorm1d(CL1_F * heads)
        self.bn9 = nn.BatchNorm1d(CL1_F * heads)
        self.bn10 = nn.BatchNorm1d(CL1_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index):
        # node embeddings:
        x = F.relu(self.gat1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat3(x, edge_index))
        x = self.bn3(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat4(x, edge_index))
        x = self.bn4(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat5(x, edge_index))
        x = self.bn5(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat6(x, edge_index))
        x = self.bn6(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat7(x, edge_index))
        x = self.bn7(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat8(x, edge_index))
        x = self.bn8(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat9(x, edge_index))
        x = self.bn9(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat10(x, edge_index))
        x = self.bn10(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out  # returns the embedding x & prediction out

class GAT_10L(nn.Module):
    def __init__(self, in_f, CL1_F, heads, out_f, p_dropout):
        super(GAT_10L, self).__init__()
        # graphCL
        self.gat1 = GATv2Conv(in_channels=in_f, out_channels=CL1_F, heads=heads, edge_dim=1)
        self.gat2 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads, edge_dim=1)
        self.gat3 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads, edge_dim=1)
        self.gat4 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads, edge_dim=1)
        self.gat5 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads, edge_dim=1)
        self.gat6 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads, edge_dim=1)
        self.gat7 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads, edge_dim=1)
        self.gat8 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads, edge_dim=1)
        self.gat9 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads, edge_dim=1)
        self.gat10 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=1, concat=False, edge_dim=1)
        # FC
        self.lin1 = nn.Linear(CL1_F, out_f)
        # BN
        self.bn1 = nn.BatchNorm1d(CL1_F * heads)
        self.bn2 = nn.BatchNorm1d(CL1_F * heads)
        self.bn3 = nn.BatchNorm1d(CL1_F * heads)
        self.bn4 = nn.BatchNorm1d(CL1_F * heads)
        self.bn5 = nn.BatchNorm1d(CL1_F * heads)
        self.bn6 = nn.BatchNorm1d(CL1_F * heads)
        self.bn7 = nn.BatchNorm1d(CL1_F * heads)
        self.bn8 = nn.BatchNorm1d(CL1_F * heads)
        self.bn9 = nn.BatchNorm1d(CL1_F * heads)
        self.bn10 = nn.BatchNorm1d(CL1_F)
        self.p_dropout = p_dropout
    def forward(self, x, edge_index, edge_attr):
        # node embeddings:
        x = F.relu(self.gat1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat3(x, edge_index, edge_attr))
        x = self.bn3(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat4(x, edge_index, edge_attr))
        x = self.bn4(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat5(x, edge_index, edge_attr))
        x = self.bn5(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat6(x, edge_index, edge_attr))
        x = self.bn6(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat7(x, edge_index, edge_attr))
        x = self.bn7(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat8(x, edge_index, edge_attr))
        x = self.bn8(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat9(x, edge_index, edge_attr))
        x = self.bn9(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gat10(x, edge_index, edge_attr))
        x = self.bn10(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out  # returns the embedding x & prediction out

class GAT_50L_uw(nn.Module):
    def __init__(self, in_f, CL1_F, heads, out_f, p_dropout):
        super(GAT_50L_uw, self).__init__()
        # graphCL
        self.gats = nn.ModuleList([GATv2Conv(in_channels=CL1_F * heads if i > 0 else in_f, out_channels=CL1_F, heads=heads) for i in range(49)])
        self.gats.append(GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=1, concat=False))
        self.bns = nn.ModuleList([nn.BatchNorm1d(CL1_F * heads) for _ in range(49)])
        self.bns.append(nn.BatchNorm1d(CL1_F))
        self.lin1 = nn.Linear(CL1_F, out_f)
        self.p_dropout = p_dropout

        # self.gat1 = GATv2Conv(in_channels=in_f, out_channels=CL1_F, heads=heads)
        # self.gat2 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat3 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat4 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat5 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat6 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat7 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat8 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat9 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat10 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat11 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat12 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat13 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat14 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat15 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat16 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat17 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat18 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat19 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat10 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat22 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat22 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat23 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat24 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat25 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat26 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat27 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat28 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat29 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat30 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat32 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat32 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat33 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat34 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat35 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat36 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat37 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat38 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat39 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat40 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)        
        # self.gat41 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat42 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat43 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat44 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat45 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat46 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat47 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat48 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat49 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=heads)
        # self.gat50 = GATv2Conv(in_channels=CL1_F * heads, out_channels=CL1_F, heads=1, concat=False)

        # # FC
        # self.lin1 = nn.Linear(CL1_F, out_f)
        # # BN
        # self.bn1 = nn.BatchNorm1d(CL1_F * heads)
        # self.bn2 = nn.BatchNorm1d(CL1_F * heads)
        # self.bn3 = nn.BatchNorm1d(CL1_F * heads)
        # self.bn4 = nn.BatchNorm1d(CL1_F * heads)
        # self.bn5 = nn.BatchNorm1d(CL1_F * heads)
        # self.bn6 = nn.BatchNorm1d(CL1_F * heads)
        # self.bn7 = nn.BatchNorm1d(CL1_F * heads)
        # self.bn8 = nn.BatchNorm1d(CL1_F * heads)
        # self.bn9 = nn.BatchNorm1d(CL1_F * heads)
        # self.bn10 = nn.BatchNorm1d(CL1_F)
        # self.p_dropout = p_dropout

    def forward(self, x, edge_index):
        # node embeddings:
        for i in range(49):
            x = F.relu(self.gats[i](x, edge_index))
            x = self.bns[i](x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)

        x = self.gats[-1](x, edge_index)
        x = self.bns[-1](x)
        out = self.lin1(x)

        # x = F.relu(self.gat1(x, edge_index))
        # x = self.bn1(x)
        # x = F.dropout(x, p=self.p_dropout, training=self.training)
        # x = F.relu(self.gat2(x, edge_index))
        # x = self.bn2(x)
        # x = F.dropout(x, p=self.p_dropout, training=self.training)
        # x = F.relu(self.gat3(x, edge_index))
        # x = self.bn3(x)
        # x = F.dropout(x, p=self.p_dropout, training=self.training)
        # x = F.relu(self.gat4(x, edge_index))
        # x = self.bn4(x)
        # x = F.dropout(x, p=self.p_dropout, training=self.training)
        # x = F.relu(self.gat5(x, edge_index))
        # x = self.bn5(x)
        # x = F.dropout(x, p=self.p_dropout, training=self.training)
        # x = F.relu(self.gat6(x, edge_index))
        # x = self.bn6(x)
        # x = F.dropout(x, p=self.p_dropout, training=self.training)
        # x = F.relu(self.gat7(x, edge_index))
        # x = self.bn7(x)
        # x = F.dropout(x, p=self.p_dropout, training=self.training)
        # x = F.relu(self.gat8(x, edge_index))
        # x = self.bn8(x)
        # x = F.dropout(x, p=self.p_dropout, training=self.training)
        # x = F.relu(self.gat9(x, edge_index))
        # x = self.bn9(x)
        # x = F.dropout(x, p=self.p_dropout, training=self.training)
        # x = F.relu(self.gat10(x, edge_index))
        # x = self.bn10(x)
        # x = F.dropout(x, p=self.p_dropout, training=self.training)
        return out  # returns the embedding x & prediction out




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




class MeanAggMPNN_layer(MessagePassing):
    def __init__(self, in_features, out_features):
        super(MeanAggMPNN_layer, self).__init__(aggr='mean')  # Set aggregation method
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return x_j  # Identity message function

    def update(self, aggr_out):
        return F.relu(self.lin(aggr_out))


class MeanAggMPNN_uw(nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, out_f, p_dropout):
        super(MeanAggMPNN_uw, self).__init__()
        self.conv1 = MeanAggMPNN_layer(in_features=in_f, out_features=CL1_F)
        self.conv2 = MeanAggMPNN_layer(in_features=CL1_F, out_features=CL2_F)
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index):
        x = self.conv1(x=x, edge_index=edge_index)
        x = F.relu(x)
        x = self.conv2(x=x, edge_index=edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out
    

class GUNet_uw(torch.nn.Module):
    def __init__(self, in_f, HL1_F, CL1_F, HL2_F, CL2_F, out_f, d, p_dropout):
        super(GUNet_uw, self).__init__()
        self.unet1 = GraphUNet(in_channels=in_f, hidden_channels= HL1_F, out_channels=CL1_F, depth=d)
        self.unet2 = GraphUNet(in_channels=CL1_F, hidden_channels=HL2_F, out_channels=CL2_F, depth=d)
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index):
        x = F.relu(self.unet1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.unet2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out
    

class GUNet1L_uw(torch.nn.Module):
    def __init__(self, in_f, HL1_F, CL1_F, out_f, d, p_dropout):
        super(GUNet1L_uw, self).__init__()
        self.unet1 = GraphUNet(in_channels=in_f, hidden_channels= HL1_F, out_channels=CL1_F, depth=d)
        self.lin1 = nn.Linear(CL1_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index):
        x = F.relu(self.unet1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)        
        return out

class GUNet(torch.nn.Module):
    def __init__(self, in_f, HL1_F, CL1_F, HL2_F, CL2_F, out_f, d, p_dropout):
        super(GUNet, self).__init__()
        self.unet1 = GraphUNet(in_channels=in_f, hidden_channels= HL1_F, out_channels=CL1_F, depth=d, pool_ratios=0.5)
        self.unet2 = GraphUNet(in_channels=CL1_F, hidden_channels=HL2_F, out_channels=CL2_F, depth=d, pool_ratios=0.5)
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_weight):
        x = F.relu(self.unet1(x, edge_index, edge_weight))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.unet2(x, edge_index, edge_weight))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out

class GTC(torch.nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, heads, out_f, p_dropout):
        super(GTC, self).__init__()
        self.gt1 = TransformerConv(in_channels=in_f, out_channels=CL1_F, heads=heads, edge_dim=1)
        self.gt2 = TransformerConv(in_channels=CL1_F * heads, out_channels=CL2_F, heads=1, concat=False, edge_dim=1)
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F * heads)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.gt1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gt2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out   
     
class GTC_uw(torch.nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, heads, out_f, p_dropout):
        super(GTC_uw, self).__init__()
        self.gt1 = TransformerConv(in_channels=in_f, out_channels=CL1_F, heads=heads, edge_dim=1, lin_edge=None) 
        self.gt2 = TransformerConv(in_channels=CL1_F * heads, out_channels=CL2_F, heads=1, concat=False, edge_dim=1, lin_edge=None)
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F * heads)
        self.bn2 = nn.BatchNorm1d(CL2_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.gt1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.gt2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out

class GPST_lin(torch.nn.Module):
    def __init__(self, in_f, CL1_F, heads, out_f, p_dropout, K):
        super(GPST_lin, self).__init__()
        self.gps1 = GPSConv(channels=in_f, heads=heads, conv=ChebConv(in_channels=in_f, out_channels=in_f, K=K), dropout=p_dropout)
        self.gps2 = GPSConv(channels=CL1_F, heads=1, conv=ChebConv(in_channels=CL1_F, out_channels=CL1_F, K=K), dropout=p_dropout)
        self.lin1 = nn.Linear(in_f, CL1_F)
        self.lin2 = nn.Linear(CL1_F, out_f)
        self.bn1 =  BatchNorm(in_f)
        self.bn2 =  BatchNorm(CL1_F)
        self.p_dropout = p_dropout
    
    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.gps1(x, edge_index)))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.bn2(self.lin1(x)))
        x = F.relu(self.bn2(self.gps2(x, edge_index)))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin2(x)
        return out 
    
class GPST(torch.nn.Module):
    def __init__(self, in_f, heads, out_f, p_dropout, K):
        super(GPST, self).__init__()
        self.gps1 = GPSConv(channels=in_f, heads=heads, conv=ChebConv(in_channels=in_f, out_channels=in_f, K=K), dropout=p_dropout)
        self.gps2 = GPSConv(channels=in_f, heads=1, conv=ChebConv(in_channels=in_f, out_channels=in_f, K=K), dropout=p_dropout)
        self.lin1 = nn.Linear(in_f, out_f)
        self.bn1 = BatchNorm(in_f)
        self.bn2 = BatchNorm(in_f)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index):
        x = F.relu(self.bn1(self.gps1(x, edge_index)))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.bn2(self.gps2(x, edge_index)))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out 

    
class GPST_GINE(torch.nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, heads, out_f, p_dropout, edge_dim):
        super(GPST_GINE, self).__init__()
        mlp1 = torch.nn.Sequential(
            nn.Linear(in_f, CL1_F),
            nn.ReLU(),
            nn.Linear(CL1_F, in_f)
        )
        mlp2 = torch.nn.Sequential(
            nn.Linear(in_f, CL2_F),
            nn.ReLU(),
            nn.Linear(CL2_F, in_f)
        )
        self.gps1 = GPSConv(channels=in_f, heads=heads, conv=GINEConv(nn=mlp1, edge_dim=edge_dim), dropout=p_dropout)
        self.gps2 = GPSConv(channels=in_f, heads=1, conv=GINEConv(nn=mlp2, edge_dim=edge_dim), dropout=p_dropout)
        self.lin1 = nn.Linear(in_f, out_f)
        self.bn1 = BatchNorm(in_f)
        self.bn2 = BatchNorm(in_f)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.bn1(self.gps1(x, edge_index, edge_attr=edge_attr)))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.bn2(self.gps2(x, edge_index, edge_attr=edge_attr)))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out 

class GPST_GINE_lin(torch.nn.Module):
    def __init__(self, in_f, CL1_F, CL2_F, heads, out_f, p_dropout, edge_dim):
        super(GPST_GINE_lin, self).__init__()
        mlp1 = torch.nn.Sequential(
            nn.Linear(in_f, CL1_F),
            nn.ReLU(),
            nn.Linear(CL1_F, in_f)
        )
        mlp2 = torch.nn.Sequential(
            nn.Linear(CL1_F, CL2_F),
            nn.ReLU(),
            nn.Linear(CL2_F, CL1_F)
        )
        self.gps1 = GPSConv(channels=in_f, heads=heads, conv=GINEConv(nn=mlp1, edge_dim=edge_dim), dropout=p_dropout)
        self.gps2 = GPSConv(channels=CL1_F, heads=1, conv=GINEConv(nn=mlp2, edge_dim=edge_dim), dropout=p_dropout)
        self.lin1 = nn.Linear(in_f, CL1_F)
        self.lin2 = nn.Linear(CL1_F, out_f)
        self.bn1 =  BatchNorm(in_f)
        self.bn2 =  BatchNorm(CL1_F)
        self.p_dropout = p_dropout
    
    def forward(self, x, edge_index, edge_attr):
        x = F.relu(self.bn1(self.gps1(x, edge_index, edge_attr=edge_attr)))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.bn2(self.lin1(x)))
        x = F.relu(self.bn2(self.gps2(x, edge_index, edge_attr=edge_attr)))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin2(x)
        return out
    
class GINE(nn.Module):
    def __init__(self, in_f, HL1_F, CL1_F, HL2_F, CL2_F, out_f, p_dropout, edge_dim):
        super(GINE, self).__init__()
        mlp1 = nn.Sequential(
            nn.Linear(in_f, HL1_F),
            nn.ReLU(),
            nn.Linear(HL1_F, CL1_F)
        )
        mlp2 = nn.Sequential(
            nn.Linear(CL1_F, HL2_F),
            nn.ReLU(),
            nn.Linear(HL2_F, CL2_F)
        )      
        # graph CL1
        self.conv1 = GINEConv(nn=mlp1, edge_dim=edge_dim)
        # graph CL2
        self.conv2 = GINEConv(nn=mlp2, edge_dim=edge_dim)
        # FC1
        self.lin1 = nn.Linear(CL2_F, out_f)
        self.bn1 = nn.BatchNorm1d(CL1_F)
        self.bn2 =  BatchNorm(CL2_F)
        self.p_dropout = p_dropout

    def forward(self, x, edge_index, edge_attr):
        # Node embeddings:
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.bn1(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        out = self.lin1(x)
        return out  
    
    