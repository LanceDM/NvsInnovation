import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from actionrecognition import STGCN_ActionRecognition
from temporalgraphconvolution import STGCN_Temporal
from tqdm import tqdm  # Import tqdm for progress bar

# Generate Adjacency Matrix
def generate_adjacency_matrix(num_joints):
    A = np.eye(num_joints)  # Identity matrix for a simple skeleton graph
    return torch.tensor(A, dtype=torch.float32, requires_grad=False)

def generate_temporal_graph(num_joints, num_frames):
    A_spatial = np.eye(num_joints)  # Identity for spatial adjacency
    A_temporal = np.eye(num_frames) + np.eye(num_frames, k=1) + np.eye(num_frames, k=-1)  # Temporal adjacency
    A_st = np.block([[A_spatial if i == j else np.zeros((num_joints, num_joints))
                      for j in range(num_frames)] for i in range(num_frames)])  # Block diagonal matrix
    return torch.tensor(A_st, dtype=torch.float32, requires_grad=False)

num_joints = 17  # Matches NPZ skeleton keypoints
num_frames = 50
A = generate_adjacency_matrix(num_joints)
A_st = generate_temporal_graph(num_joints, num_frames)

# Simulated dataset: (batch_size=100, channels=3, frames=50, joints=17)
train_data = torch.rand(80, 3, 50, 17)  # 80% for training
train_labels = torch.randint(0, 2, (80,), dtype=torch.long)
val_data = torch.rand(20, 3, 50, 17)  # 20% for validation
val_labels = torch.randint(0, 2, (20,), dtype=torch.long)

# Prepare DataLoaders
train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# Initialize Models
num_classes = 2  # Seizure / Non-Seizure
model_action = STGCN_ActionRecognition(in_channels=3, num_classes=num_classes, A=A)
model_temporal = STGCN_Temporal(in_channels=3, num_classes=num_classes, A_st=A_st)

# Define Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer_action = torch.optim.Adam(model_action.parameters(), lr=0.001)
optimizer_temporal = torch.optim.Adam(model_temporal.parameters(), lr=0.001)

# Training Loop with tqdm progress bar and Validation Step
num_epochs = 10
for epoch in range(num_epochs):  
    for model, optimizer, model_name in [(model_action, optimizer_action, "Action Recognition"),
                                          (model_temporal, optimizer_temporal, "Temporal Convolution")]:
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [{model_name} Training]", leave=True)
        
        for inputs, labels in progress_bar:
            optimizer.zero_grad()
            inputs = inputs.permute(0, 3, 1, 2)  # Correct ST-GCN format (N, C, T, V)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} - {model_name} Training Loss: {avg_loss:.4f}")
        
        # Validation Step
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [{model_name} Validation]", leave=True)
            for inputs, labels in progress_bar:
                inputs = inputs.permute(0, 3, 1, 2)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                predicted = torch.argmax(outputs, dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                progress_bar.set_postfix(val_loss=loss.item())
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total * 100
        print(f"Epoch {epoch+1} - {model_name} Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

print("Training Complete!")

# Testing
test_sample = torch.rand(1, 3, 50, 17)
test_sample = test_sample.permute(0, 3, 1, 2)  # Correct shape

prediction_action = F.softmax(model_action(test_sample), dim=1)
predicted_class_action = torch.argmax(prediction_action, dim=1).item()
print(f'Predicted Action: {"Seizure" if predicted_class_action == 1 else "Non-Seizure"}')

prediction_temporal = F.softmax(model_temporal(test_sample), dim=1)
predicted_class_temporal = torch.argmax(prediction_temporal, dim=1).item()
print(f'Predicted Temporal Conv: {"Seizure" if predicted_class_temporal == 1 else "Non-Seizure"}')
