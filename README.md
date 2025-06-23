# SplitNN Pseudo-Distributed Training with PyTorch

This repository demonstrates a toy “split learning” (SplitNN) setup using PyTorch’s `torch.distributed` API. The model is partitioned between multiple client processes (front-end MLPs) and a server process (back-end MLP). During training, each client:

1. Performs a forward pass up to a designated “cut layer.”  
2. Sends the intermediate activations (and labels) to the server.  
3. Receives gradients on the cut layer from the server and completes the backward pass locally.  

A similar protocol is used for inference to compute accuracy.

---

## 📦 Prerequisites

- Python 3.7+  
- PyTorch 1.8+  
- scikit-multilearn  
- scikit-learn  
- NumPy  
- Matplotlib  

Install dependencies, for example:

```bash
    pip install torch numpy scikit-multilearn scikit-learn matplotlib
```

---

## 📂 Data Preparation

1. Download the **gisette** dataset in sparse ARFF format from OpenML:  
   https://www.openml.org/data/download/18631146/gisette.arff  
2. Place `gisette.arff` in the same directory as `code.py`.

---

## 🚀 Usage

Simply run:

```bash
    python code.py
```

By default, you should observe final validation accuracies of approximately:

- Client 1: 94%  
- Client 2: 96%  
- Client 3: 96%  

Minor run-to-run variations may occur.

---

## ⚙️ Command-Line Arguments

You can customize hyperparameters and behavior via flags:

| Flag                     | Description                                                                   | Default                 |
|--------------------------|-------------------------------------------------------------------------------|-------------------------|
| `--device`               | Device for computation (`cuda` or `cpu`)                                      | `cuda`                  |
| `--init_method`          | URL for `torch.distributed` init (`tcp://…`)                                  | `tcp://localhost:23456` |
| `--n_epoch`              | Number of training epochs                                                     | `40`                    |
| `--n_data`               | Total number of data instances (exclusive with `n_data_per_client`)           | `None`                  |
| `--n_data_per_client`    | Number of instances per client (exclusive with `n_data`)                      | `None`                  |
| `--test_ratio`           | Fraction of data reserved for testing                                         | `0.2`                   |
| `--seed`                 | Random seed for data splits                                                   | `42`                    |
| `--learning_rate`        | Learning rate                                                                 | `0.001`                 |
| `--optimizer`            | Optimizer (`adam` or `sgd`)                                                   | `adam`                  |
| `--batch_size`           | Batch size                                                                    | `32`                    |
| `--activation`           | Activation function (`relu`, `tanh`, `sigmoid`)                               | `relu`                  |
| `--dropout`              | Dropout rate                                                                  | `0.2`                   |
| `--hidden_layers_client` | Sizes of hidden layers on each client (space-separated list)                  | `[256]`                 |
| `--hidden_layers_server` | Sizes of hidden layers on the server (space-separated list)                   | `[256]`                 |
| `--n_report`             | Report server loss every _n_ requests                                         | `50`                    |
| `--n_client`             | Number of client processes                                                    | `3`                     |
| `--plot_dir`             | File path to save the server’s loss curve                                     | `loss.jpg`              |

Example:

    python code.py --n_epoch 50 --batch_size 64 --hidden_layers_client 512 256 \
                   --hidden_layers_server 256 --learning_rate 0.0005

---

## 🔄 Reproducing Results

1. Ensure `gisette.arff` is in place.  
2. Run `python code.py`.  
3. Observe per-epoch accuracy logs on each client, and a final loss plot `loss.jpg` saved by the server.

---

## 📄 Citation
If you use this code in your research, please consider citing it using the following BibTeX entry.

```bibtex
@software{guo2025splitnn,
  author       = {Guo, Jizhou},
  title        = {{SplitNN Pseudo-Distributed Training with PyTorch}},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15721682},
  url          = {https://doi.org/10.5281/zenodo.15721682}
}
```

## 💡 Notes & Tips

- **Process Launching**  
  The script uses `torch.multiprocessing` to spawn one server (`rank=0`) and _N_ clients (`rank=1…N`).  
- **Model Partition**  
  The client-side MLP input dimension is hard-coded to match the 5 000-feature Gisette dataset. Adapt if you use another dataset.  
- **Communication Backend**  
  We use the Gloo backend for CPU/GPU compatibility. For GPU-only setups, consider NCCL for higher throughput.  
- **Argument Coupling**  
  Only one of `--n_data` or `--n_data_per_client` may be set.

---
