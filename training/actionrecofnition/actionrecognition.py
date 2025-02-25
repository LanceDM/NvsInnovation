import torch
import torch.nn as nn
import torch.nn.functional as F

# Graph Convolution Layer
class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(GraphConv, self).__init__()
        self.A = nn.Parameter(A, requires_grad=False)  # Non-trainable adjacency matrix
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        x = torch.einsum('nctv,vw->nctw', (x, self.A))  # Spatio-temporal convolution
        x = self.conv(x)
        return x

# ST-GCN Model for Action Recognition
class STGCN_ActionRecognition(nn.Module):
    def __init__(self, in_channels, num_classes, A):
        super(STGCN_ActionRecognition, self).__init__()
        self.graph_conv1 = GraphConv(in_channels, 64, A)
        self.graph_conv2 = GraphConv(64, 128, A)
        self.graph_conv3 = GraphConv(128, 256, A)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.relu(self.graph_conv1(x))
        x = F.relu(self.graph_conv2(x))
        x = F.relu(self.graph_conv3(x))
        x = x.mean(dim=[2, 3])  # Global average pooling
        return x  # Return raw logits (apply softmax outside)
