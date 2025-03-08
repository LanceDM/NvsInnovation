import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from temporalgraphconvolution import STGCN_Temporal

# Generate spatio-temporal adjacency matrix
def generate_temporal_graph(num_joints, num_frames):
    A_spatial = np.eye(num_joints)  # Spatial adjacency
    A_temporal = np.eye(num_frames, k=1) + np.eye(num_frames, k=-1)  # Temporal adjacency
    A_st = np.kron(A_temporal, A_spatial)  # Kronecker product to form spatio-temporal graph
    return torch.tensor(A_st, dtype=torch.float32, requires_grad=False)

# Model setup
num_joints = 18
num_frames = 50
A_st = generate_temporal_graph(num_joints, num_frames)
num_classes = 2  # Seizure / Non-Seizure
model = STGCN_Temporal(in_channels=3, num_classes=num_classes, A_st=A_st)

# Simulated dataset
train_data = torch.rand(100, 3, num_frames, num_joints)
train_labels = torch.randint(0, 2, (100,), dtype=torch.long)
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Testing
sample_input = torch.rand(1, 3, num_frames, num_joints)
prediction = F.softmax(model(sample_input), dim=1)
predicted_class = torch.argmax(prediction, dim=1).item()
print(f'Predicted Action: {"Seizure" if predicted_class == 1 else "Non-Seizure"}')
