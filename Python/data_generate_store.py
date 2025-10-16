"""
data_generate_store.py
----------------------
Generate new 4D spacecraft formation parameters using the trained CVAE model.
Takes 2D feature conditions as input and produces multiple 4D samples.
The results are saved as .mat for MATLAB verification.

Author: [Your Name]
Date: [YYYY-MM-DD]
"""

import torch
import numpy as np
from scipy.io import savemat
from train_cvae import CVAE

def generate_samples(cond_input, num_samples=100, model_path="trained_cvae.pth", dataset_path="prepared_dataset.pt"):
    """
    Generate 4D state samples given 2D feature condition.
    
    Args:
        cond_input (array): shape (2,) representing 2D feature [Td, Ωd].
        num_samples (int): number of 4D state samples to generate.
        model_path (str): trained CVAE model file.
        dataset_path (str): normalization data for de-normalization.
    
    Returns:
        np.ndarray: generated 4D samples (de-normalized).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(dataset_path)
    model = CVAE(input_dim=4, cond_dim=2, latent_dim=8).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cond = torch.tensor(cond_input, dtype=torch.float32, device=device).unsqueeze(0)
    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, 8).to(device)
            x_gen = model.decode(z, cond)
            samples.append(x_gen.cpu().numpy())

    X_gen = np.vstack(samples)

    # Denormalize
    X_min, X_max = data["X_min"], data["X_max"]
    X_gen_denorm = X_gen * (X_max - X_min) + X_min

    savemat("generated_samples.mat", {"Generated_X": X_gen_denorm})
    print(f"✅ Generated {num_samples} samples saved to generated_samples.mat")

    return X_gen_denorm

if __name__ == "__main__":
    # Example: condition = [Td, Ωd]
    condition = [0.5, 0.4]
    generate_samples(condition, num_samples=100)