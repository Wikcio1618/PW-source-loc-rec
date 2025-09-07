import torch
import torch.nn as nn
import torch.optim as optim

from preprocessing import train_data_generator

def train(model, device='cpu', steps=500, lr=1e-3):
    model.to(device)
    model.train()
    
    generator = train_data_generator()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        X_batch, y_batch = next(generator)
        X_tensor = torch.tensor(X_batch, device=device)
        y_tensor = torch.tensor(y_batch, device=device).unsqueeze(1)

        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")
