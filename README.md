# Molecular Property Prediction in the Ultra-Low Data Regime

<img width="1395" alt="Screenshot 2025-02-18 at 11 26 14 AM" src="https://github.com/user-attachments/assets/6b0b8acc-caec-425e-a963-4569d4891658" />



This repository contains the official code accompanying the paper **"Molecular Property Prediction in the Ultra-Low Data Regime"** by  
- Basem A. Eraqi<sup>1</sup>, Dmitrii Khizbullin<sup>2</sup>, Shashank S. Nagaraja<sup>1</sup>, and S. Mani Sarathy<sup>1</sup>  
<sup>1</sup>Clean Energy Research Platform, King Abdullah University of Science and Technology, Saudi Arabia  
<sup>2</sup>Center of Excellence in Generative AI, King Abdullah University of Science and Technology, Saudi Arabia  

### Paper Abstract (Brief Overview)

Data scarcity remains a major obstacle to effective machine learning in molecular property prediction and design, affecting fields like pharmaceuticals, solvents, polymers, and energy carriers. While multi-task learning (MTL) leverages shared structure across tasks, it often suffers from **negative transfer** when tasks have imbalanced data distributions. Our **Adaptive Checkpointing with Specialization (ACS)** approach provides a data-efficient training scheme for multi-task graph neural networks, enabling reliable property predictions with extremely limited labeled data. ACS outperforms or matches other state-of-the-art supervised methods and demonstrates its practical utility by predicting 15 sustainable aviation fuel (SAF) properties with as few as 29 labeled samples.

---

## Repository Contents

- **`main.py`**  
  Main script for training and evaluation.
- **`dataset.py`**  
  Pre-processes training data and produce molecular graphs.
- **`checkpointing.py`**  
  Facilitate adaptive and global-loss checkpointing.
- **`metrics_and_losses.py`**  
  Helper functions for training and evaluation.
- **`model.py`**  
  Model architecture selection.
- **`performance_analysis.py`**  
  Post-training evaluation script.
- **`smiles_utils.py`**  
  Helper functions for handling SMILES.

---

## Installation and Setup

### 1. Create (and activate) a new conda environment

```bash
conda create -n acs python=3.9
conda activate acs
```

### 2. Add conda-forge channel (if not already added)

```bash
conda config --add channels conda-forge
conda update -n base --all
conda install -n base mamba
```

### 3. Install PyTorch & PyTorch Geometric

```bash
mamba install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
mamba install -c pyg pyg
```

### 4. Install additional libraries

```bash
mamba install pandas
mamba install matplotlib
pip install tensorboard
pip install rdkit
pip install openpyxl
pip install xlsxwriter
pip install plotly
pip install scikit-learn
pip install tqdm
pip install deepchem
```

---

## Usage

1. **Clone this repository**:
   ```bash
   git clone https://github.com/<YourUsername>/<YourRepoName>.git
   cd <YourRepoName>
   ```

2. **(Optional) Place your data**:  
   - By default, the code looks for data in `data/SAF.xlsx` or one of the recognized `data/*.csv` benchmark datasets.  
   - Adjust or rename your data file as needed.

3. **Run training**:
   ```bash
   python main.py --model-type acs --task-type regression --num-folds 5 --max-steps 50000
   ```
   - Common arguments:
     - `--model-type`: can be `acs` (Adaptive Checkpointing), `mtl`, `mtl-glc`, or `stl`.
     - `--task-type`: `regression` or `classification`.
     - `--num-folds`: number of cross-validation folds (default 5).
     - `--max-steps`: total training steps (default 50000).
     - `--use-scaffold-split`: toggles scaffold-based data splitting.
     - Run `python main.py --help` for more options.

4. **View logs**:  
   TensorBoard logs are saved under `runs/`. Start TensorBoard via:
   ```bash
   tensorboard --logdir runs/
   ```

5. **Results & Analysis**:
   - Results (RMSE, R², or ROC-AUC) and scatter plots are exported to `.xlsx` and `.html` files in each fold’s directory (`runs/<timestamp>/fold_<n>`).
   - A summary of K-fold performance is compiled as `k_fold_performance.xlsx` in the run directory.

---


## Citation

If this work helps your research, please consider citing our paper:

```bibtex
@article{
  title={Molecular Property Prediction in the Ultra-Low Data Regime},
  author={Eraqi, Basem A. and Khizbullin, Dmitrii and Nagaraja, Shashank S. and Sarathy, S. Mani},
  journal={...},
  year={2025}
}
```

---

## License



