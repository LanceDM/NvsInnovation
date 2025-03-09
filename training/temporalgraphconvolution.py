import torch
import torch.nn as nn
import torch.nn.functional as F

# Temporal Graph Convolution Layer
class TemporalGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A_st):
        super(TemporalGraphConv, self).__init__()
        self.A_st = nn.Parameter(A_st, requires_grad=False)  # Spatio-temporal adjacency matrix
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', (x, self.A_st))  # Graph convolution
        x = self.conv(x)
        return x

# ST-GCN Model with Temporal Graph Convolution
class STGCN_Temporal(nn.Module):
    def __init__(self, in_channels, num_classes, A_st):
        super(STGCN_Temporal, self).__init__()
        self.graph_conv1 = TemporalGraphConv(in_channels, 64, A_st)
        self.graph_conv2 = TemporalGraphConv(64, 128, A_st)
        self.graph_conv3 = TemporalGraphConv(128, 256, A_st)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.graph_conv1(x))
        x = F.relu(self.graph_conv2(x))
        x = F.relu(self.graph_conv3(x))
        x = x.mean(dim=2)  # Pool over time only
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x 
