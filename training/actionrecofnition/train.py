import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from actionrecognition import STGCN_ActionRecognition

# Generate Adjacency Matrix
def generate_adjacency_matrix(num_joints):
    A = np.eye(num_joints)  # Identity matrix for a simple skeleton graph
    return torch.tensor(A, dtype=torch.float32, requires_grad=False)

num_joints = 18  # Adjust based on dataset
A = generate_adjacency_matrix(num_joints)

# Simulated dataset: (batch_size=100, channels=3, frames=50, joints=18)
train_data = torch.rand(100, 3, 50, 18)
train_labels = torch.randint(0, 2, (100,), dtype=torch.long)  # Ensure long dtype

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# Prepare DataLoader
train_dataset = TensorDataset(train_data, train_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Initialize Model
num_classes = 2  # Seizure / Non-Seizure
model = STGCN_ActionRecognition(in_channels=3, num_classes=num_classes, A=A)

# Define Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(10):  # Train for 10 epochs
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        inputs = inputs.permute(0, 2, 3, 1)  # Ensure correct shape (batch, channels, frames, joints)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Testing
test_sample = torch.rand(1, 3, 50, 18)
test_sample = test_sample.permute(0, 2, 3, 1)  # Ensure correct shape

prediction = F.softmax(model(test_sample), dim=1)
predicted_class = torch.argmax(prediction, dim=1).item()
print(f'Predicted Action: {"Seizure" if predicted_class == 1 else "Non-Seizure"}')
