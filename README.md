# GRASP: Differentially Private Graph Reconstruction Defense with Structured Perturbation

## 🧠 Overview

This repository provides the official implementation of **GRASP**, a defense framework designed to protect graph structures from **Graph Reconstruction Attacks (GRA)** under **Differential Privacy (DP)** constraints.

GRASP addresses the limitations of unstructured perturbation in existing DP-GNNs by introducing a **structured perturbation** mechanism that mixes **identical and independent noise** using a **Bernoulli process**. This approach effectively disrupts the relative ranking of node embedding similarities, providing stronger protection against GRA while maintaining high utility.

---

## 🛠️ Installation

### Required Dependencies

You can create a `requirements.txt` using the list below:

```txt
torch==2.5.1
torch-geometric==2.6.1
torch-scatter==2.1.2
torch-sparse==0.6.18
torchmetrics==1.5.1
networkx==3.4.2
matplotlib==3.7.2
seaborn==0.13.2
numpy==1.24.2
pandas==2.2.1
scikit-learn==1.4.2
tqdm
```

### Installation

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Code

You can train and evaluate GRASP using `main.py` with the following commands. Below are a few examples for different datasets:

### Cora

```bash
python main.py --gnn gcn --dataset cora --lr 0.001 --local_layers 3 --hidden_channels 512 --weight_decay 5e-4 --dropout 0.7 --rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123 --device 0 --runs 2 --save_model --epochs 3000 --defence_k 1.5 --defence_limit --defence_sigma 1.0 --defence_p 0.7
```

### Citeseer

```bash
python main.py --gnn gcn --dataset citeseer --lr 0.001 --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 --rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123 --device 1 --runs 2 --save_model --epochs 3000 --defence_k 1.5 --defence_limit --defence_sigma 1.0 --defence_p 0.7
```

### Pubmed

```bash
python main.py --gnn gcn --dataset pubmed --lr 0.005 --local_layers 2 --hidden_channels 256 --weight_decay 5e-4 --dropout 0.7 --rand_split --train_prop 0.6 --valid_prop 0.2 --seed 123 --device 6 --runs 2 --save_model --epochs 5000 --defence_k 1.5 --defence_limit --defence_sigma 1.0 --defence_p 0.7
```

You can find similar commands for `amazon-computer`, `amazon-photo`, `coauthor-cs`, `coauthor-physics`, and `wikics` datasets in the repository or by adapting the above patterns.
