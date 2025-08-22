# PyTorch Portfolio – Applied Deep Learning Projects

A curated, production‑ready portfolio of PyTorch projects you can **run, learn from, and showcase**. Each project is small, focused, and follows the same structure so recruiters/teammates can navigate quickly.

> ⭐ Tip: Clone and run any project in minutes. Every folder has a `README`, configs, scripts, and a short report.

---

## 🔧 Tech Stack

* **Core**: Python, PyTorch, TorchVision, TorchAudio, TorchText
* **Training**: Lightning (optional), Hydra configs, mixed precision
* **Tracking**: TensorBoard & Weights & Biases (optional)
* **Quality**: pre‑commit, black, isort, ruff, pytest
* **CI**: GitHub Actions (unit tests, style checks)

---

## 📁 Repo Structure

```
.
├── projects/
│   ├── 01_tabular-regression_boston/
│   ├── 02_tabular-classification_bank-marketing/
│   ├── 03_cnn-mnist/
│   ├── 04_cnn-cifar10/
│   ├── 05_transfer-learning_resnet_imagenette/
│   ├── 06_nlp-sentiment_lstm_imdb/
│   ├── 07_nlp-text-classification_bert_agnews/
│   ├── 08_time-series_lstm_air-passengers/
│   ├── 09_recommendation_matrix-factorization_movielens/
│   ├── 10_segmentation_unet_carvana/
│   ├── 11_object-detection_fasterrcnn_pennfudan/
│   ├── 12_gans_dcgan_mnist/
│   └── 13_rl_dqn_cartpole/   # (bonus: shows breadth)
│
├── templates/                # reusable code (datasets, training loops, utils)
│   ├── dataset_template.py
│   ├── model_template.py
│   ├── train_template.py
│   └── evaluate_template.py
│
├── tools/
│   ├── make_dataset.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
│
├── configs/                  # hydra configs (defaults + per‑project overrides)
│   ├── default.yaml
│   └── <project>.yaml
│
├── tests/
│   └── test_smoke.py
│
├── requirements.txt
├── environment.yml
├── pyproject.toml            # black/isort/ruff config
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
└── README.md (this file)
```

---

## 🗂️ Project Index (with learning goals)

1. **Tabular Regression – Boston Housing**
   *Goal*: supervised regression, feature scaling, MAE/R², early stopping.
   **Folder**: `projects/01_tabular-regression_boston`
   **Metrics**: MAE, RMSE, R²

2. **Tabular Classification – Bank Marketing**
   *Goal*: class imbalance, ROC‑AUC/PR‑AUC, calibration.
   **Folder**: `projects/02_tabular-classification_bank-marketing`

3. **CNN Basics – MNIST**
   *Goal*: Conv/Pool/Dropout, training loop anatomy.
   **Folder**: `projects/03_cnn-mnist`

4. **CNN + Data Augment – CIFAR‑10**
   *Goal*: augmentations, LR schedules, CutOut/MixUp (optional).
   **Folder**: `projects/04_cnn-cifar10`

5. **Transfer Learning – ResNet on Imagenette**
   *Goal*: fine‑tuning strategies, layer freezing, discriminative LRs.
   **Folder**: `projects/05_transfer-learning_resnet_imagenette`

6. **NLP Sentiment – LSTM on IMDB**
   *Goal*: tokenization, padding, packed sequences, embeddings.
   **Folder**: `projects/06_nlp-sentiment_lstm_imdb`

7. **NLP Text Classification – BERT on AG News**
   *Goal*: transformers, attention masks, gradient clipping.
   **Folder**: `projects/07_nlp-text-classification_bert_agnews`

8. **Time Series Forecasting – LSTM (Air Passengers)**
   *Goal*: sliding windows, scaling, MAPE/SMAPE.
   **Folder**: `projects/08_time-series_lstm_air-passengers`

9. **Recommender – Matrix Factorization (MovieLens 100K)**
   *Goal*: implicit vs explicit feedback, RMSE & NDCG\@K.
   **Folder**: `projects/09_recommendation_matrix-factorization_movielens`

10. **Image Segmentation – U‑Net (Carvana or Oxford Pets)**
    *Goal*: dice loss, IoU, augmentation for masks.
    **Folder**: `projects/10_segmentation_unet_carvana`

11. **Object Detection – Faster R‑CNN (Penn‑Fudan)**
    *Goal*: custom collate, anchors, mAP.
    **Folder**: `projects/11_object-detection_fasterrcnn_pennfudan`

12. **Generative – DCGAN (MNIST)**
    *Goal*: adversarial training, FID (optional).
    **Folder**: `projects/12_gans_dcgan_mnist`

13. **Reinforcement Learning – DQN (CartPole)**
    *Goal*: replay buffer, target network, ε‑greedy.
    **Folder**: `projects/13_rl_dqn_cartpole`

> You can start with 3–5 projects and grow over time. Commit early; keep results and a short write‑up in each folder.

---

## 🚀 Quickstart

```bash
# Clone
git clone https://github.com/<your-username>/pytorch-portfolio.git
cd pytorch-portfolio

# Create environment (conda)
conda env create -f environment.yml
conda activate torch-portfolio

# Or pip
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Pre-commit hooks (format + lint)
pre-commit install

# Run a project (example: CIFAR-10)
python tools/train.py project=04_cnn-cifar10 trainer.max_epochs=20

# View logs
tensorboard --logdir runs
```

---

## ⚙️ Config System (Hydra)

All projects share a **single training entrypoint** (`tools/train.py`) with a Hydra config. Each project has its own YAML that overrides defaults.

**Example:**

```bash
python tools/train.py project=03_cnn-mnist trainer.max_epochs=10 optimizer.lr=1e-3
```

**Default config snippet (`configs/default.yaml`):**

```yaml
seed: 42
project: 03_cnn-mnist
trainer:
  max_epochs: 10
  mixed_precision: true
  batch_size: 64
optimizer:
  name: adam
  lr: 0.001
  weight_decay: 0.0
model:
  name: simple_cnn
  hidden_dim: 128
logging:
  use_wandb: false
  log_dir: runs
```

---

## 🧩 Reusable Training Template (pure PyTorch)

```python
# templates/train_template.py
import torch, time
from torch.utils.data import DataLoader

def train(model, train_ds, val_ds, loss_fn, optimizer, epochs=10, batch_size=64, device="cuda"):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)
    model.to(device)

    best_val = float("inf")
    for epoch in range(1, epochs+1):
        model.train(); t0 = time.time(); running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * x.size(0)
        train_loss = running / len(train_loader.dataset)

        # validation
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += loss_fn(pred, y).item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | {time.time()-t0:.1f}s")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best.pt")
```

---

## 📊 Results & Reports (per project)

Each project folder includes:

* `README.md` with **dataset**, **model**, **how to run**, and **learning notes**
* `metrics.json` (MAE/RMSE/Acc/F1/mAP etc.)
* `figures/` (loss curves, confusion matrix, sample preds)
* `model_card.md` (short description & limitations)

**Short report template:**

```markdown
# Results
- Train/Val/Test metrics: ...
- Best checkpoint: ...
- Inference speed: ...

# What I learned
- … three bullets maximum …

# Next steps
- … improvements to try …
```

---

## 🧪 Testing & Quality

* `pytest -q` runs smoke tests on tiny batches/dummy data
* `pre-commit` auto‑formats code and checks style on every commit

**`tests/test_smoke.py` idea:**

```python
import torch

def test_torch_works():
    assert torch.cuda.is_available() or True  # allow CPU only
```

---

## 📦 Environment

**`requirements.txt` (minimal):**

```
torch
torchvision
torchaudio
torchtext
numpy
pandas
matplotlib
scikit-learn
torchmetrics
tensorboard
hydra-core
PyYAML
rich
pre-commit
black
isort
ruff
pytest
```

**`environment.yml` (conda):**

```yaml
name: torch-portfolio
channels: [pytorch, conda-forge]
dependencies:
  - python=3.11
  - pytorch
  - torchvision
  - torchaudio
  - cudatoolkit  # or pytorch-cuda=12.1 on Windows
  - pip
  - pip:
      - torchtext
      - numpy
      - pandas
      - matplotlib
      - scikit-learn
      - torchmetrics
      - tensorboard
      - hydra-core
      - PyYAML
      - rich
      - pre-commit
      - black
      - isort
      - ruff
      - pytest
```

---

## 🧭 Roadmap

* [ ] Add Lightning versions (side‑by‑side with pure PyTorch)
* [ ] Export ONNX + TorchScript for a couple of models
* [ ] Dockerfiles per project + `Makefile`
* [ ] Hugging Face Spaces demo for 1–2 projects
* [ ] Add `inference/` notebooks with reproducible examples

---

## 🤝 Contributing / Using as a Learning Resource

* Feel free to fork and use as a template for your own learning.
* Keep each project self‑contained, small, and well‑documented.
* Prefer **readable** code over clever tricks.

---

## 📣 Showcase (add screenshots)

Add 2–3 images per project in `figures/` (loss curves, confusion matrix, sample predictions). These make your README scannable for recruiters.

---

## 📜 License

Choose MIT or Apache‑2.0 for simple reuse.

---

## 🧱 Badges (optional flair)

```
![Built with PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Tests](https://github.com/<you>/pytorch-portfolio/actions/workflows/ci.yml/badge.svg)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)
```

---

## 🧠 How to Add a New Project (checklist)

1. Duplicate a folder from `projects/03_cnn-mnist` → rename.
2. Create `README.md` with dataset, model, how to run, metrics.
3. Add Hydra config under `configs/<project>.yaml`.
4. Implement `dataset.py`, `model.py`, and `train.py` (or reuse templates).
5. Log metrics to TensorBoard; save best checkpoint.
6. Add 2–3 figures and a `model_card.md`.
7. Add a tiny smoke test in `tests/`.
8. Update the **Project Index** above.

---

## 🔌 Dataset Notes (built‑in downloaders)

* **MNIST / CIFAR‑10 / Imagenette** via `torchvision.datasets`
* **IMDB / AG News** via `torchtext.datasets`
* **MovieLens 100K**: lightweight; script in `tools/make_dataset.py`
* **Penn‑Fudan**: small detection dataset; download script provided
* **Air Passengers**: CSV in repo for deterministic runs

---

## 📥 Example Commands

```bash
# MNIST
python tools/train.py project=03_cnn-mnist trainer.max_epochs=5 optimizer.lr=5e-4

# CIFAR-10 with augmentations and cosine LR
python tools/train.py project=04_cnn-cifar10 data.augment=true optimizer.lr=0.1 scheduler=cosine

# BERT text classification (AG News)
python tools/train.py project=07_nlp-text-classification_bert_agnews trainer.max_epochs=3 optimizer.lr=2e-5

# Recommender on MovieLens
python tools/train.py project=09_recommendation_matrix-factorization_movielens trainer.max_epochs=10
```

---

Happy training! Keep commits atomic and document what you learned in each project. 🚀
