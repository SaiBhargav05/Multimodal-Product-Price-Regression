# Multimodal Product Price Regression — Amazon ML Challenge 2025

> **Amazon ML Challenge 2025 (India)** · Smart Product Pricing Challenge  
> **Final CV SMAPE: 21.835%** across 5-fold stratified ensemble · A100 80GB · FP16 AMP

---

## Overview

An end-to-end multimodal deep learning system for Amazon product price prediction, fusing **visual** (CLIP image embeddings) and **textual** (LoRA-adapted BERT) modalities via cross-attention, with regex-NER scalar features late-fused into a SwiGLU MLP regression head.

The dataset is the official Amazon ML Challenge 2025 (India) dataset: 75,000 training products and 75,000 test products, each with a catalog text field (item name + description + IPQ) and a product image URL from `m.media-amazon.com`.

---

## Results

| Metric | Value |
|---|---|
| **CV SMAPE (5-fold mean)** | **21.835%** |
| Fold 1 | 21.833% |
| Fold 2 | 22.158% |
| Fold 3 | 21.700% |
| Fold 4 | 21.662% |
| Fold 5 | 21.823% |
| Std across folds | ±0.175% |
| Best single Optuna trial | 23.616% |
| Ensemble improvement over best single trial | **~1.78pp** |

The tight ±0.175% standard deviation across folds confirms stable generalization — the model is not fold-sensitive.

---

## Architecture

```
catalog_content (text)
        │
   [Text Cleaning]                    product image
   HTML strip, regex                       │
   alphanumeric, whitespace            [Download]
        │                                  │
  [BERT base-uncased]              [CLIP ViT-B/32]
  + LoRA (r=32, α=32)              (frozen weights)
  1.07% trainable params                   │
  (1,179,648 / 110,661,888)               │
        │                                  │
   [768-dim text emb]            [768-dim image emb]
   + LayerNorm                   + LayerNorm
        │                                  │
        └──────────┬───────────────────────┘
                   │
        [CrossAttentionFusion]
         MultiheadAttention
         text queries image keys/values
                   │
              [768-dim]
                   │
    [NER scalar features late-fusion]
    ner_count  (pack size, e.g. Pack of 6 → 6.0)
    ner_size   (volume/weight, e.g. 12 Oz → 12.0)
    norm_size  (size / count ratio)
    total_content (size × count)
                   │
              [770-dim]
                   │
          [SwiGLU MLP Head]
          Block 1: 770 → 512 → 256
          Block 2: 256 → 128 → 1
          (SwiGLU activation: SiLU(x·Wgate) × x·Wvalue)
                   │
            [log-price output]
                   │
              [expm1(·)]
                   │
            predicted price ($)
```

---

## Key Design Decisions

### 1. LoRA on BERT (not full fine-tuning)
Applied Low-Rank Adaptation (r=32, α=32) to all BERT attention projection layers. This keeps BERT weights frozen and adds only **1,179,648 trainable parameters (1.07% of 110.6M)** — preventing catastrophic forgetting while adapting to product domain vocabulary.

### 2. Loss Curriculum
Training uses a two-phase loss schedule:
- **Epochs 1–10:** Standard SMAPE loss (stable warmup, prevents early divergence)
- **Epochs 10+:** Weighted SMAPE loss (upweights high-price tail products)

Motivation: The price distribution is heavily right-skewed ($0.13–$2,796; 38% of products under $10, 71% under $25). Standard SMAPE underweights expensive products — the curriculum corrects this without losing stability in early training.

### 3. Log-space regression
All targets transformed as `log1p(price)` before training, predictions back-transformed with `expm1()`. This compresses the 4-order-of-magnitude price range into a learnable regression target and aligns with SMAPE's multiplicative nature.

### 4. NER feature extraction (regex-based)
Two scalar features extracted from raw catalog text:
- **`ner_count`:** Pack size (e.g. "Pack of 6" → 6.0). Present in 15,999 train products.
- **`ner_size`:** Volume/weight (e.g. "12 Oz" → 12.0). Present in 54,987 train products.

Correlation with log-price: count and size features carry meaningful price signal (bulk packs cost more). Late-fused after cross-attention to avoid contaminating the embedding space.

### 5. Image augmentation (train only)
```
Train:     Resize(256) → RandomCrop(224) → RandomHorizontalFlip
           → ColorJitter → RandomAffine → Normalize(CLIP stats)
           → RandomErasing
Inference: Resize(224) → Normalize(CLIP stats)
```

### 6. 5-Fold Stratified CV Ensemble
Price bucketed into 10 log-spaced bins for stratification. Five fold models trained independently; test predictions averaged in log-space before expm1. Ensemble SMAPE (21.835%) outperforms best single Optuna trial (23.616%) by **~1.78 percentage points**.

### 7. Overfit check before full training
Single-sample overfit test (300 iters, lr=1e-4): loss drops from 2.126 → 0.012 by iteration 90, confirming the full backprop chain (LoRA → CrossAttention → SwiGLU) is connected and trainable.

---

## Hyperparameter Tuning (Optuna — 20 Trials)

Search space:
| Parameter | Range |
|---|---|
| Learning rate | 1e-5 to 1e-3 (log scale) |
| Dropout | 0.10 to 0.50 |
| Hidden dim | 256, 512, 768 |
| LoRA rank | 8, 16, 32 |

**Best trial:** lr=4.99e-5, dropout=0.397, hidden_dim=512, SMAPE=23.616%

---

## Model Diagnostics

### SwiGLU Weight Histograms
All weight matrices (attention projections, SwiGLU gate/value/output) show healthy uniform-ish distributions centered around zero — no weight collapse or explosion.

### SwiGLU Gate Activation Firing Rates
- Block 1: 0.289 (28.9% of gates active)
- Block 2: 0.212 (21.2% of gates active)

Both well below the 50% baseline, indicating the gating mechanism is selective — the network is not uniformly activating, which would indicate dead gating. Healthy sparse activation.

### Training Curves (per fold)
All 5 folds show the same pattern: train loss drops steeply in the first 10 epochs (warmup phase), then gradually flattens. Val SMAPE oscillates then stabilizes around 21.7–22.2%. No divergence, no overfitting.

---

## Submission Analysis

| Stat | Train (ground truth) | Test (predicted) |
|---|---|---|
| Count | 74,999 | 75,000 |
| Mean price | $23.65 | $25.85 |
| Median price | $14.00 | $20.69 |
| Max price | $2,796 | $1,130 |
| Min price | $0.13 | $1.04 |
| Negative prices | 0 | 0 |
| Null prices | 0 | 0 |
| Duplicate IDs | — | 0 |

The predicted distribution is slightly shifted right vs. training (median $20.69 vs $14.00) — expected behavior from the weighted SMAPE loss curriculum, which deliberately upweights expensive products. Max predicted price ($1,130) is conservative vs. training max ($2,796), consistent with regression-to-mean behavior on extreme tail values.

---

## Stack

| Component | Tool |
|---|---|
| Framework | PyTorch 2.x |
| Text encoder | `bert-base-uncased` + custom LoRA |
| Image encoder | `openai/clip-vit-base-patch32` (frozen) |
| HPO | Optuna (TPE sampler, 20 trials) |
| Training | FP16 AMP on NVIDIA A100-SXM4-80GB (85.1 GB VRAM) |
| CV strategy | StratifiedKFold (k=5, log-price bins) |
| Environment | Google Colab Pro (A100 High RAM) |
| Data | Amazon ML Challenge 2025 India (official) |

---

## Project Structure

```
├── Amazon_ML_Final_v4.ipynb   # Full training notebook (47 cells)
├── submission.csv              # 75,000 test predictions
├── train.csv                   # Training data (not included, Amazon challenge)
├── test.csv                    # Test data (not included, Amazon challenge)
├── images/                     # Downloaded product images (~75K files)
└── README.md
```

---

## Reproduce

```python
# 1. Install
!pip install transformers optuna torchvision tqdm scikit-learn

# 2. Set HF token (Colab secrets)
# Tools → Secrets → HF_TOKEN

# 3. Place train.csv and test.csv in /content/

# 4. Run all cells in Amazon_ML_Final_v4.ipynb
# Full run time: ~3-4 hours on A100
```

---


