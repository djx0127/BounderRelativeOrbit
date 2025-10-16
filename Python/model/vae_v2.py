import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

# hyper parameters
latent_dim = 8 # 潜在维度


class Encoder(nn.Module): # 处理以更好拟合更多潜在的分布

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU())
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x, cond):
        x_cond = torch.cat([x, cond], dim=1) # (x,6)
        h = self.net(x_cond) #(x, 2 * latent_dim)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim + 2, 128), nn.ReLU(),
                                 nn.Linear(128, 64), nn.ReLU(),
                                 nn.Linear(64, 4))

    def forward(self, z, cond):
        z_cond = torch.cat([z, cond], dim=1)
        return self.net(z_cond)


class CVAE_v2(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder() # 编码器
        self.decoder = Decoder() # 解码器

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    

    def forward(self, x, cond):
        mu, logvar = self.encoder(x, cond)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, cond)
        return x_recon, mu, logvar
