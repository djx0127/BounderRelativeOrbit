# Semi-Analytical Bounded Formation Configuration Screening (PCM-CVAE)

This repository contains the implementation of the **Semi-Analytical Bounded Formation Configuration Screening Method** proposed in  
*Ding et al., "Semi-analytical Bounded Formation Configuration Screening Method Based on Poincaré Contraction Mapping," IEEE Transactions on Aerospace and Electronic Systems, 2025.*

---

## 🛰️ Overview

This project develops a **semi-analytical framework** that combines **Poincaré Contraction Mapping (PCM)** with a **Conditional Variational Autoencoder (CVAE)** to efficiently screen bounded spacecraft formation configurations.

The workflow consists of **three main stages**:

1. **Bounded Formation Configuration (BFC) Dataset Generation (MATLAB)**  
   - Constructs a dataset of 4D state parameters → 2D feature parameters using the analytical **Poincaré Contraction Mapping (PCM)**.  
   - Generates mappings between system state parameters (4D or higher 6D) and feature parameters (2D crossing period × separation angle).  
   - Output: a 4→2 dimensional dataset used for training.
   - e.g.: Take the BFC in the displaced orbit as an example, the outputs of PCM from 4D (κ, α, Hz, ΔE) parameters to 2D parameters (ΔT, ΔΩ) are generated for training.

2. **Semi-Analytical Screening Model (Python)**  
   - Implements the **CVAE-based inverse mapping** from 2D feature parameters → 4D state parameters.  
   - Trains, validates, and tests the inverse mapping model to reconstruct feasible deputy-state configurations from a chief's feature parameters.  
   - Enables large-scale generation of bounded formations.

3. **PCM-CVAE Validation and Evaluation (MATLAB)**  
   - Re-applies the analytical PCM to verify generated configurations.  
   - Computes accuracy (σT, σΩ) and dispersion (Ma's distance) metrics to evaluate the precision and diversity of generated formations.
   - Visualize the formation configuration in the ECI coordinate system and the chief spacecraft's orbital coordinate system.

---

## 📂 Repository Structure

├── MATLAB/  
│ ├── PCM_Data_Generation/ # Step 1: Generate 4D→2D datasets via Poincaré Contraction Mapping  
│ ├── PCM_CVAE_Validation/ # Step 3: Validation, evaluation and visualization of generated configurations  
│ └── utils/ # Auxiliary MATLAB scripts and plotting functions  
│  
├── Python/  
│ ├── CVAE_Model/ # Step 2: Conditional Variational Autoencoder implementation  
│ ├── dataset_preparation.py # Data normalization, split (train/val/test)  
│ ├── train_cvae.py # Training script  
│ ├── date_generate_store.py # Data genaration and storage  
│ └── requirements.txt # Python dependencies  
│  
└── README.md


---

## ⚙️ Environment Setup

### MATLAB
- MATLAB R2022a or later
- Toolboxes: *Symbolic Math Toolbox*, *Optimization Toolbox*

### Python
- Python ≥ 3.8  
- Required packages:
  ```bash
  pip install torch numpy matplotlib scikit-learn
  ```


## 🚀 Usage

1. Generate PCM Dataset (MATLAB)
```matlab
cd MATLAB/PCM_Data_Generation
run('Generate_PCM_Data.m')
```

* Outputs dataset_4to2.mat containing mappings of 4D→2D parameters. 
* The dynamic environment of the formation can be re-established. Parameters such as domain scope can be modified accordingly. 

2. Train CVAE Model (Python)
```bash
cd Python
python dataset_preparation.py
python train_cvae.py
python date_generate_store.py
```

* This will train the CVAE model on the dataset and save the trained model and data to /Python/CVAE_Model/CVAEOutputData.mat

3. Validate PCM-CVAE (Matlab)
```matlab
cd MATLAB/PCM_CVAE_Validation
run('Evaluate_CVAE_Result.m')
```

* Evaluates configuration accuracy (σT, σΩ) and dispersion (Mahalanobis distance D).

## 📊 Outputs
Dataset: dataset_4to2.mat  
Trained Model: CVAE_Model/saved_model.pt  
Generated Output: CVAE_Model/CVAEOutputData.mat  
Validation Results: accuracy & dispersion plots, formation trajectory visualizations.  

## 🧠 Key References
If you use this code, please cite:  
```pgsql
@article{ding2025pcm_cvae,
  author={Jixin Ding and Ming Xu and Xue Bai and Xiaoyi Wang and Xiao Pan},
  title={Semi-analytical Bounded Formation Configuration Screening Method Based on Poincaré Contraction Mapping},
  journal={IEEE Transactions on Aerospace and Electronic Systems},
  year={2025}
}
```

## 📬 Contact
Jixin Ding (丁纪昕)
School of Astronautics, Beihang University
Email: djx0127@buaa.edu.cn

## 🪐 Acknowledgements

This work is supported by the National Natural Science Foundation of China (No. 124B2048).
