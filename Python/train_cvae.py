"""
train_cvae.py
-------------
Train a Conditional Variational Autoencoder (CVAE) model using the dataset
prepared from PCM-generated data. The CVAE learns the inverse mapping from
2D feature parameters → 4D state parameters.

Author: Jixin Ding
Date: 2025-10-16
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ==================== CVAE Model Definition ====================
class CVAE(nn.Module):
    def __init__(self, input_dim=4, cond_dim=2, latent_dim=8):
        super(CVAE, self).__init__()
        # Encoder: 4D state conditioned on 2D feature
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        # Decoder: latent + condition → reconstructed 4D
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def encode(self, x, c):
        h = torch.cat([x, c], dim=1)
        h = self.encoder(h)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        h = torch.cat([z, c], dim=1)
        return self.decoder(h)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

# ==================== Training Function ====================
def train_cvae(dataset_path="prepared_dataset.pt", num_epochs=200, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(dataset_path)

    train_loader = DataLoader(
        TensorDataset(data["X_train"], data["Y_train"]),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(data["X_val"], data["Y_val"]),
        batch_size=batch_size, shuffle=False
    )

    model = CVAE(input_dim=4, cond_dim=2, latent_dim=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='sum')

    def loss_function(recon_x, x, mu, logvar):
        mse = criterion(recon_x, x)
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return mse + 0.01 * kld

    print("🚀 Start training CVAE...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            recon_x, mu, logvar = model(x, y)
            loss = loss_function(recon_x, x, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss/len(train_loader.dataset):.4f}")

    torch.save(model.state_dict(), "trained_cvae.pth")
    print("✅ Model training complete and saved as trained_cvae.pth")

if __name__ == "__main__":
    train_cvae()