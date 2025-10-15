"""fraud-detection: A Flower / PyTorch app."""
import os
import pandas as pd
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.preprocessing import StandardScaler

def load_data(partition_id: int, num_partitions: int, batch_size=64):
    DATA_PATH = "/content/creditcard.csv"
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("creditcard.csv not found. Please download it and place it in the correct directory.")

    # Load and preprocess dataset
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X_tensor, y_tensor)
    total_size = len(dataset)
    
    # Partition data
    partition_size = total_size // num_partitions
    start = partition_id * partition_size
    end = total_size if partition_id == num_partitions - 1 else start + partition_size
    
    partition_indices = list(range(start, end))
    partition_dataset = Subset(dataset, partition_indices)
    
    # Split into train/test sets
    train_size = int(0.8 * len(partition_dataset))
    test_size = len(partition_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(partition_dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class Net(nn.Module):
    def __init__(self, input_size=30, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return self.sigmoid(out)

def train(model, trainloader, epochs, lr, device):
    model.to(device)
    criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    running_loss=0.0
    for _ in range(epochs):
      for batch in trainloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss

def test(model, testloader, device):
    model.to(device)
    criterion = nn.BCELoss().to(device)
    correct, loss = 0, 0
    with torch.no_grad():
        for batch in testloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss += criterion(outputs, labels).item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    avg_loss = loss / len(testloader)
    return avg_loss, accuracy
