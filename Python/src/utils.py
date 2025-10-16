import torch
import torch.nn as nn
import torch.nn.functional as F
from model.vae import latent_dim


def loss_fn(x_recon, x_true, mu, logvar):
    recon_loss = F.mse_loss(x_recon, x_true) # 重构值和真实值的均方误差
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) # 分布与标准正态分布的 KL散度
    return recon_loss + kl_loss

def diversity_cvae_loss(x, x_hats, mu, logvar, kl_weight=1.0):
    """
    x: [B, x_dim]
    x_hats: [B, k, x_dim]
    """
    B, k, _ = x_hats.shape
    x_expand = x.unsqueeze(1).repeat(1, k, 1)  # [B, k, x_dim]
    
    recon_errors = ((x_hats - x_expand) ** 2).mean(dim=2)  # [B, k]
    min_recon_loss = recon_errors.min(dim=1).values.mean()  # scalar

    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = min_recon_loss + kl_weight * kl_loss
    return total_loss, min_recon_loss.item(), kl_loss.item()

def generate_k_samples(model, y, k=5):
    """
    对每个样本的 y 生成 k 个不同的 x 重构
    输入:
      y: [B, cond_dim]
    输出:
      x_hats: [B, k, x_dim]
    """
    B = y.size(0)
    y_expand = y.unsqueeze(1).repeat(1, k, 1)  # [B, k, cond_dim]
    z_samples = torch.randn(B, k, latent_dim).to(y.device)  # [B, k, latent_dim]
    y_expand = y_expand.view(B*k,-1)
    z_samples = z_samples.view(B*k,-1)
    x_hat_flat = model.decoder(z_samples,y_expand)  # [B*k, x_dim]
    x_hat = x_hat_flat.view(B, k, -1)  # [B, k, x_dim]
    return x_hat

def gengrate_sample(model, cond, num_samples=100, sample_std=1.0):
    """
    Generate samples from the CVAE model.
    :param model: Trained CVAE model.
    :param cond: Condition input for the decoder.
    :param num_samples: Number of samples to generate.
    :param sample_std: Standard deviation for the latent space sampling.
    :return: Generated samples.
    """
    model.eval()
    cond = torch.tensor(cond, dtype=torch.float32).unsqueeze(0)
    cond = cond.repeat(num_samples, 1)

    z = torch.randn(num_samples, latent_dim) * sample_std
    with torch.no_grad():
        samples = model.decoder(z, cond)
    return samples.numpy()

def gengrate_sample_mle(f_model,
                        target_y,
                        num_candidates=10000,
                        num_sample=20):
    f_model.eval()
    with torch.no_grad():

        x_candidates = torch.rand(num_candidates, 4)  # 假设 [0,1] 区间

        y_preds = f_model(x_candidates)

        target_tensor = torch.tensor(target_y,
                                     dtype=torch.float32).unsqueeze(0)
        errors = torch.norm(y_preds - target_tensor, dim=1)

        top_indices = torch.topk(-errors, k=num_sample).indices
        return x_candidates[top_indices]