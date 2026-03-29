import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from model.network import ChessNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

states = np.load("data/states.npy")
values = np.load("data/values.npy")

states = states[:50000]
values = values[:50000]
print(f"Loaded {len(states)} positions")

x = torch.tensor(states)
v = torch.tensor(values)

dataset = TensorDataset(x, v)
loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = ChessNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(5):
    total = 0
    for xb, vb in loader:
        xb, vb = xb.to(device), vb.to(device)
        optimizer.zero_grad()
        _, pred = model(xb)
        loss = loss_fn(pred.squeeze(), vb)
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"Epoch {epoch+1}/5 - Loss: {total/len(loader):.4f}")

torch.save(model.state_dict(), "model/chess_model.pt")
print("Model saved!")

'''python -m model.train_real'''