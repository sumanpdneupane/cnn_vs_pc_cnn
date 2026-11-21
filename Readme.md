# Robustness Analysis of Standard CNN and Predictive Coding CNN Models for Image Classification under Gaussian and Salt-and-Pepper Noise

| Part | Section | What it covers |
|---|---|---|
| [**Start**](#start) | [Overview](#overview) | Problem, research question, hypothesis, what we do |
| | [Installation](#installation) | Clone, install, run a notebook |
| | [Data](#data) | Tiny ImageNet; validation images = test set |
| [**What the thesis uses**](#what-the-thesis-uses) | [StandardCNN](#standardcnn) | The baseline CNN |
| | [HPCCNNRaoV2 (the PC-CNN)](#hpccnnraov2-the-pc-cnn) | The predictive-coding CNN |
| | [PCCNNEP](#pccnnep-ep-extra-experiment) | Extra experiment: the CNN trained with Equilibrium Propagation |
| | [Experiments Chosen](#experiments-chosen) | Runs, seeds, settings, augmentation pipeline |
| | [Evaluation](#evaluation-to-do-after-training) | Noise levels, metrics, statistics |
| [**How the choices were made**](#how-the-choices-were-made) | [StandardCNN Selection](#standardcnn-selection) | Architecture and training settings |
| | [Model Size and Computation Cost](#model-size-and-computation-cost) | Why 2.7M and not 8M parameters or a higher-parameter model |
| | [Augmentation Selection](#augmentation-selection) | Why ResizeMix on 50% of images |

## Start

### Overview

**Problem.** Image classifiers are usually tested on clean images, but real images are often noisy (poor light, low-quality cameras, compression, sensor noise). Noise and other corruptions can make a network wrong, so clean accuracy alone does not show how reliable a model is. A Predictive-Coding CNN (PC-CNN) corrects its features with top-down predictions and bottom-up prediction errors over several steps (iterative inference), which may help with noisy images. But there is little direct comparison of a Standard CNN and a PC-CNN with the same structure, both trained without noise and tested under controlled noise.

**Research question.** How do a Standard CNN and a PC-CNN differ in robustness when the Tiny ImageNet validation images get Gaussian or salt-and-pepper noise at test time?

**Hypothesis.** Both kinds of noise will lower the results of both models. The PC-CNN may behave differently from the Standard CNN under noise, because it uses prediction-error correction and iterative inference.

**What we do:**
1. Build two models with the same layers: **StandardCNN** (one feedforward pass) and **HPCCNNRaoV2** (the same CNN + 7 cycles of top-down predictions and bottom-up prediction errors).
2. Train both on the Tiny ImageNet training images (200 classes) without noise (with augmentation, see Experiments Chosen), with the same settings and 3 seeds each.
3. Test on the validation images without noise (baseline).
4. Add Gaussian and salt-and-pepper noise to the validation images at 0.05, 0.1, 0.2, 0.3 and 0.5 (test only, never in training) and test again.
5. Compare the models: accuracy drop, loss increase, robustness retention, corruption error, calibration, statistical tests over the seeds, and compute cost (see Evaluation).

Extra experiment: **PCCNNEP**, the same CNN trained with a predictive-coding learning rule (Equilibrium Propagation) instead of backprop, one seed.

<img src="assets/divider.svg" width="100%" height="2" alt="">

### Installation

**Needs:** Python 3.10, Git and Git LFS (for the checkpoints). A GPU is optional: the code uses CUDA if available, else Apple MPS, else the CPU.

**1. Clone the repository** (the Tiny ImageNet data is included; the checkpoints come through Git LFS):

```commandline
git lfs install
git clone https://github.com/sumanpdneupane/cnn_vs_pc_cnn.git
cd cnn_vs_pc_cnn
git lfs pull
```

**2. Create a virtual environment and install the packages:**

```commandline
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. Start Jupyter from the repository root.** The notebooks import `src.…`, so the root must be on the Python path:

```commandline
PYTHONPATH="$PWD" jupyter lab      # Windows PowerShell: $env:PYTHONPATH = (Get-Location); jupyter lab
```

**4. Run an experiment.** Open a notebook in `experiments_in_seeds/` (see the runs table below) and run all cells. Checkpoints and `metrics.csv` are written to `checkpoints/<model>/<run>/`. To continue a stopped run, set `RESUME = True` in the first code cell. Training time per seed: see Model Size and Computation Cost below.

<img src="assets/divider.svg" width="100%" height="2" alt="">

### Data

- **Tiny ImageNet** (200 classes, 64×64, 500 train + 50 val images per class), in `data/tiny_imagenet/tiny-imagenet-200` (val images sorted into one folder per class).
- **Validation = test set.** The test labels of Tiny ImageNet are not public, so the 10,000 validation images are the test set. The model, its training settings, the augmentation and the evaluated checkpoint were also chosen on them (the same for both models). Evaluation uses the best checkpoint (`best_model.pth`, highest validation accuracy), and noise is added only at test.
- Normalisation: mean/std of the Tiny ImageNet training set (values in Settings, under Experiments Chosen). Validation images: no augmentation. Training augmentation: see the Augmentation pipeline in Experiments Chosen below.
- Code: `src/dataset/`.

| Used for | Paper |
|---|---|
| Dataset | Le & Yang (2015), *Tiny ImageNet Visual Recognition Challenge* — http://cs231n.stanford.edu/reports/2015/pdfs/yle_project.pdf |

<img src="assets/divider.svg" width="100%" height="2" alt="">

## What the thesis uses

### StandardCNN

`src/model/cnn.py`. The baseline, trained with backprop.

Each conv: 3×3, no bias → BatchNorm → LeakyReLU (0.05). Input 3×64×64.

| Block | Layer | Output shape | Receptive field | Parameters |
|---|---|---|---|---|
| level1 | conv1 | 32×64×64 | 3 | 928 |
| | conv2 + max-pool | 64×32×32 | 6 | 18,560 |
| level2 | conv3 | 64×32×32 | 10 | 36,992 |
| | conv4 + max-pool | 128×16×16 | 16 | 73,984 |
| level3 | conv5 | 128×16×16 | 24 | 147,712 |
| | conv6 + max-pool + Dropout2d 0.15 | 256×8×8 | 36 | 295,424 |
| level4 | conv7 | 256×8×8 | 52 | 590,336 |
| | conv8 (dilation 2) + Dropout2d 0.25 | 512×8×8 | 84 | 1,180,672 |
| head | global average pooling | 512 | whole image | 0 |
| | dense 512 (BN, LeakyReLU, dropout 0.35) | 512 | | 263,168 |
| | classifier | 200 | | 102,600 |
| | **Total** | | | **2,710,376** |

Parameters include the BatchNorm γ, β. Receptive field in input pixels.

- Training: SGD (Nesterov, momentum 0.9, lr 0.05, weight decay 1e-4), cosine schedule, label smoothing 0.1, batch 64, 100 epochs. How the model and these settings were chosen: StandardCNN Selection below.

| Used for | Paper |
|---|---|
| 3×3 conv blocks (VGG style) | Simonyan & Zisserman (2015) — https://arxiv.org/abs/1409.1556 |
| BatchNorm | Ioffe & Szegedy (2015) — https://arxiv.org/abs/1502.03167 |
| LeakyReLU | Maas et al. (2013), *Rectifier Nonlinearities Improve Neural Network Acoustic Models* — https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf |
| He (Kaiming) init | He et al. (2015) — https://arxiv.org/abs/1502.01852 |
| Dilated conv | Yu & Koltun (2016) — https://arxiv.org/abs/1511.07122 |
| Dropout / Dropout2d | Srivastava et al. (2014) — https://jmlr.org/papers/v15/srivastava14a.html; Tompson et al. (2015) — https://arxiv.org/abs/1411.4280 |
| Global average pooling | Lin et al. (2014), *Network in Network* — https://arxiv.org/abs/1312.4400 |
| Label smoothing | Szegedy et al. (2016) — https://arxiv.org/abs/1512.00567 |

<img src="assets/divider.svg" width="100%" height="2" alt="">

### HPCCNNRaoV2 (the PC-CNN)

`src/model/hpc_cnn_rao_v2.py` (class `HPCCNNRaoV2`). The Standard CNN (same layers, start weights, dropout and training) plus predictive coding at every conv layer.

- Cycle 0 = one feedforward pass (= the Standard CNN). Then **7 cycles**, in training and at test:
  1. **Top-down:** each layer predicts the layer below, p = Wᵀr, and the layer moves toward the prediction.
  2. **Bottom-up:** the prediction error e = r − p goes up through W and corrects the layer above.
- Tied weights: the prediction uses the transposed conv of the same layer. Update rates learned per channel.
- **Per conv, not per block:** each of the 8 convs has its own top-down prediction (`conv_transpose2d` with that conv's weight). Unpool to the max-pool winners only after conv2, 4, 6.

| Block | Conv | Bottom-up (conv) | Top-down (transposed conv, same weight) | Extra: rate a | Extra: rate b |
|---|---|---|---|---|---|
| level1 | conv1 | image → r1 (32×64×64) | r1 → image | 32 | 32 |
| | conv2 + pool | r1 → r2 (64×32×32) | r2 → unpool → r1 | 64 | 64 |
| level2 | conv3 | r2 → r3 (64×32×32) | r3 → r2 | 64 | 64 |
| | conv4 + pool | r3 → r4 (128×16×16) | r4 → unpool → r3 | 128 | 128 |
| level3 | conv5 | r4 → r5 (128×16×16) | r5 → r4 | 128 | 128 |
| | conv6 + pool | r5 → r6 (256×8×8) | r6 → unpool → r5 | 256 | 256 |
| level4 | conv7 | r6 → r7 (256×8×8) | r7 → r6 | 256 | 256 |
| | conv8 (dilation 2) | r7 → r8 (512×8×8) | r8 → r7 | 512 | – (top layer) |
| | | | | **1,440** | **928** |

**Extra compared with the Standard CNN**
- **Parameters:** only the rates, 2,368 (a = bottom-up step per channel of r1..r8, b = top-down step per channel of r1..r7). Total 2,712,744.
- **Not learned (buffers):** λ per layer (top eigenvalue of WᵀW with the BatchNorm scale included, sets the step a/λ) and its 8 power-iteration vectors.
- **Training:** the same loss, optimizer and schedule; backprop runs through all cycles. Hidden dropout uses one mask for all cycles.

| Used for | Paper |
|---|---|
| Top-down predictions, bottom-up errors, tied weights | Rao & Ballard (1999), *Predictive coding in the visual cortex*, Nature Neuroscience — https://doi.org/10.1038/4580 |
| Cycle schedule, update rates, training through the cycles | Wen et al. (2018), *Deep Predictive Coding Network for Object Recognition*, ICML — https://arxiv.org/abs/1802.04762 |
| Same dropout mask in every cycle | Gal & Ghahramani (2016), *A Theoretically Grounded Application of Dropout in Recurrent Neural Networks* — https://arxiv.org/abs/1512.05287 |

<img src="assets/divider.svg" width="100%" height="2" alt="">

### PCCNNEP (EP, extra experiment)

`src/model/pc_cnn_ep.py` (class `PCCNNEP`). The Standard CNN with its weights learned by centered Equilibrium Propagation: local updates, no backprop. One seed.

- Each batch: free phase (one feedforward pass), then the output is nudged toward the label (+β and −β, β = 0.01). 10 sweeps spread the prediction errors e = state − f(state below) down the layers, and each weight is updated from its local error × input.
- At test: one feedforward pass, like the Standard CNN.
- Difference from the PC-CNN: here predictive coding is the learning rule. In the PC-CNN it is the inference (7 cycles at test).

| Used for | Paper |
|---|---|
| Predictive coding as learning | Whittington & Bogacz (2017), *Neural Computation* — https://doi.org/10.1162/NECO_a_00949 |
| Equilibrium Propagation | Scellier & Bengio (2017), *Frontiers in Computational Neuroscience* — https://doi.org/10.3389/fncom.2017.00024 |
| Centered EP (+β and −β) | Laborieux et al. (2021), *Frontiers in Neuroscience* — https://doi.org/10.3389/fnins.2021.633674 |

<img src="assets/divider.svg" width="100%" height="2" alt="">

### Experiments Chosen

- All 200 classes: notebooks in `experiments_in_seeds/`, checkpoints in `checkpoints/`.
- 10-class test runs: `archive/_03_experiments_10_classes/` (checkpoints in its `checkpoints/`).

| Model | Experiments | Seeds |
|---|---|---|
| StandardCNN | `experiments_in_seeds/cnn/` | 42, 21, 84 |
| HPCCNNRaoV2 (the PC-CNN) | `experiments_in_seeds/hpc_cnn_rao_v2/` | 42, 21, 84 |
| PCCNNEP (EP, extra experiment) | `experiments_in_seeds/pc_cnn_ep/` | 42 |

**Settings** (the same for every model; optimizer and learning rate: see StandardCNN above):

| Setting | Value | Why |
|---|---|---|
| Seeds | 42, 21, 84 (EP: 42 only) | 3 seeds show how much a result changes by chance, and allow paired tests (CNN seed 42 vs PC-CNN seed 42, and so on). Fixed before training, not tuned. EP has one seed because one run takes about 4 days. |
| What a seed sets | start weights, order of the training images, augmentation and dropout randomness | With the same seed, every model starts from the same weights and sees the same batches, so the models differ only in the model itself. |
| Images | all 500 train + 50 val per class, 200 classes (100,000 + 10,000) | The full dataset; the seed does not change which images are used. |
| Normalisation | mean (R, G, B) = 0.4802, 0.4481, 0.3975; std (R, G, B) = 0.2764, 0.2689, 0.2816 | Computed on the Tiny ImageNet training set (`src/utils/utils.py`). The same for training, validation and test, in every model. |
| Batch size | 64 | The same for all models. |
| Epochs | 100 | Cosine learning-rate schedule down to 1e-5 over the 100 epochs. Long enough to converge; 150 epochs cost 1.5× the time for a small gain (see StandardCNN Selection below). |
| Data-loader workers | 4 | Fastest in our test (`00_mac_worker_test.ipynb`, 11-core M3 Pro): 0 / 2 / 4 / 6 / 8 workers = 35 / 54 / 66 / 58 / 49 batches per second. |
| Clean-train check | epoch 1, then every 5 epochs | Accuracy on the training images without augmentation, to watch overfitting. Not every epoch because it costs time. |
| Checkpoints | `latest_checkpoint.pth` every epoch, `best_model.pth` at the best val accuracy | `latest` stores everything to resume exactly (weights, optimizer, schedule, random states). Evaluation uses `best_model.pth`. |

**Augmentation pipeline** (training only, the same for all models; run 39, how it was chosen: Augmentation Selection below). Code: `src/dataset/resizemix.py` and the "Dataset and transforms" cell of each notebook (`before_mix_tf`, `after_mix_tf`, `test_tf`).

| Step | Setting | Value |
|---|---|---|
| 1. Before mixing (`before_mix_tf`) | Crop and flip | RGB → RandomCrop 64, padding 4 (reflect) → horizontal flip, p = 0.5 |
| 2. Mixing (`ResizeMixDataset`) | ResizeMix on 50% of images (`RESIZEMIX_PROB = 0.50`) | Another random training image is shrunk to τ × 64 px, τ ~ U(0.1, 0.8) (`RESIZEMIX_SCALE`), and pasted at a random spot. The other 50% stay unmixed. |
| | Label | λ · own label + (1 − λ) · pasted label, λ = 1 − pasted area / image area (≈ 1 − τ²) |
| 3. After mixing (`after_mix_tf`) | RandAugment | 2 ops, magnitude 5, bilinear |
| 4. Normalise | Mean and std (R, G, B) | From Settings above |
| Loss | Mixed cross-entropy | λ · CE(own label) + (1 − λ) · CE(pasted label), label smoothing 0.1 |
| Validation, test, clean-train check (`test_tf`) | No augmentation | RGB → tensor → normalise (mean/std from Settings). |

<img src="assets/divider.svg" width="100%" height="2" alt="">

### Evaluation (to do, after training)

Every best checkpoint (`best_model.pth`, highest validation accuracy), all seeds, on the 10,000 validation images.

- **Noise, at test only:** Gaussian (standard deviation) and salt-and-pepper (proportion of pixels set to the minimum or maximum) at 0.05, 0.1, 0.2, 0.3 and 0.5. Added in pixel range [0, 1], then clipped, then normalised. The same noisy images for every model.
- **Clean metrics:** accuracy, loss, confusion matrix, precision, recall, F1.
- **Robustness metrics:** accuracy drop, loss increase, RRR (noisy / clean accuracy), CE and mCE (Standard CNN as reference), ECE.
- **Statistics:** paired by seed (CNN 42 vs PC-CNN 42, and so on): paired t-test, Wilcoxon, 95% confidence intervals. With 3 seeds these are weak on their own (Wilcoxon's smallest possible p is 0.25), so also McNemar on the 10,000 images and bootstrap confidence intervals, for each seed pair.
- **Compute cost:** inference time, GPU memory, FLOPs, training time. Measured only when no training is running.
- Tables use all seeds (mean ± SD). Figures for one model (confusion matrix, examples) use seed 42. EP has one seed, so it is descriptive only.

| Used for | Paper |
|---|---|
| CE, mCE | Hendrycks & Dietterich (2019), *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* — https://arxiv.org/abs/1903.12261 |
| ECE | Guo et al. (2017), *On Calibration of Modern Neural Networks* — https://arxiv.org/abs/1706.04599 |
| McNemar test | McNemar (1947), *Psychometrika* — https://doi.org/10.1007/BF02295996; Dietterich (1998), *Neural Computation* — https://doi.org/10.1162/089976698300017197 |
| Wilcoxon signed-rank test | Wilcoxon (1945), *Biometrics Bulletin* — https://doi.org/10.2307/3001968 |
| Bootstrap confidence intervals | Efron (1979), *The Annals of Statistics* — https://doi.org/10.1214/aos/1176344552 |

<img src="assets/divider.svg" width="100%" height="2" alt="">

## How the choices were made

The next three sections explain why the setup above was chosen.

### StandardCNN Selection

How the architecture and the training settings of the Standard CNN were chosen (the final model: StandardCNN above).

All runs: Standard CNN, 200 classes, seed 42, 100 epochs unless noted. Notebooks and checkpoints: `archive/_01_z_cnn/`.

The columns (also used in Model Size and Augmentation Selection):
- **Val:** accuracy on the validation images, mean of the last 5 epochs (96–100; run 11: 146–150).
- **Aug. train:** accuracy on the augmented training images, last epoch.
- **Clean train:** accuracy on the training images without augmentation, last epoch.
- **Gap:** clean train − val at the last epoch, in percentage points. A large gap means overfitting.

**1. Architecture.** Each row changes the model; rows with "the same" keep the model and add mixing. The Training column shows the mixing used. Notebook = folder number / file in `archive/_01_z_cnn/` (click to open):

| Notebook | Model | Training | Parameters | Val (%) | Aug. train (%) | Clean train (%) | Gap (pp) | What we learned |
|---|---|---|---|---|---|---|---|---|
| [_01/01_clean.ipynb](archive/_01_z_cnn/_01_cnn_single_layer/01_clean.ipynb) | 4 convs, one per level, flatten → FC | no mixing | 5.93M | 55.17 | 56.82 | 78.78 | 23.56 | Too shallow |
| [_02/01_clean.ipynb](archive/_01_z_cnn/_02_cnn_double_layer/01_clean.ipynb) | 8 convs, two per level, flatten → FC | no mixing | 9.07M | 60.66 | 80.67 | 95.00 | 34.34 | Two convs per level: +5.5 points, but strong overfitting |
| [_02/04_clean25_mixup375_cutmix375.ipynb](archive/_01_z_cnn/_02_cnn_double_layer/04_clean25_mixup375_cutmix375.ipynb) | the same | 25% clean, 37.5% MixUp, 37.5% CutMix | 9.07M | 62.24 | 57.23 | 86.12 | 23.94 | Mixing: +1.6 points, gap 34 → 24 |
| [_03/01_clean.ipynb](archive/_01_z_cnn/_03_cnn_double_layer_gap/01_clean.ipynb) | 8 convs + global average pooling | no mixing | 4.87M | 60.79 | **88.80** | **97.89** | 37.16 | GAP: the same val with half the parameters |
| [_03/02_clean25_mixup375_cutmix375_drop.ipynb](archive/_01_z_cnn/_03_cnn_double_layer_gap/02_clean25_mixup375_cutmix375_drop.ipynb) | the same + block dropout 0.05 / 0.10 | 25% clean, 37.5% MixUp, 37.5% CutMix | 4.87M | 63.91 | 57.48 | 86.45 | 22.67 | Mixing + dropout: +3.1 points, gap 37 → 23 |
| [_04/01_split450_rand3_7.ipynb](archive/_01_z_cnn/_04_cnn_double_gap_2level_with_3layer_parameter/01_split450_rand3_7.ipynb) | 10 convs (2-2-3-3) + GAP + dense 512 | 25% clean, 37.5% MixUp, 37.5% CutMix | 8.00M | **65.47** | 70.23 | 97.62 | 32.25 | Deeper + dense layer: +1.6 points; the training settings were tested on this model (table 2) |
| [_05/39_resizemix50_seed42.ipynb](archive/_01_z_cnn/_05_cnn/39_resizemix50_seed42.ipynb) | **8 convs, 2.7M (final)** | ResizeMix on 50% of images | **2.71M** | 64.07 | 56.13 | 83.27 | **19.12** | 3× fewer parameters, less overfitting; chosen for cost (see Model Size and Computation Cost below) |

All rows except the final model used 450 train / 100 val images per class (from run 02 on: the official 500 / 50), so the final model is not exactly comparable with them. They used RandAugment (3, 7), label smoothing 0.1, SGD lr 0.05, weight decay 1e-4, and classifier dropout 0.5 (4- and 8-conv models) or 0.2 (10-conv model).

**2. Training settings**, tested on the 10-conv model, one change per run. Only the runs behind a decision; notebooks in `archive/_01_z_cnn/_04_cnn_double_gap_2level_with_3layer_parameter/` (click to open):

| Notebook | Change | Val (%) | Aug. train (%) | Clean train (%) | Gap (pp) |
|---|---|---|---|---|---|
| [01_split450_rand3_7.ipynb](archive/_01_z_cnn/_04_cnn_double_gap_2level_with_3layer_parameter/01_split450_rand3_7.ipynb) | 450 train / 100 val images per class, RandAugment (3, 7) | 65.47 | 70.23 | 97.62 | 32.25 |
| [02_rand3_7.ipynb](archive/_01_z_cnn/_04_cnn_double_gap_2level_with_3layer_parameter/02_rand3_7.ipynb) | official 500 / 50 split | 66.50 | 70.31 | 97.22 | 30.62 |
| [03_rand2_5.ipynb](archive/_01_z_cnn/_04_cnn_double_gap_2level_with_3layer_parameter/03_rand2_5.ipynb) | RandAugment (2, 5) instead of (3, 7) | 66.71 | **73.40** | **98.40** | 31.53 |
| [05_wd1e5.ipynb](archive/_01_z_cnn/_04_cnn_double_gap_2level_with_3layer_parameter/05_wd1e5.ipynb) | weight decay 1e-5 instead of 1e-4 | 64.26 | 70.67 | 98.14 | 33.93 |
| [10_blockdrop015_025.ipynb](archive/_01_z_cnn/_04_cnn_double_gap_2level_with_3layer_parameter/10_blockdrop015_025.ipynb) | from 03: block dropout 0.15 / 0.25 (instead of 0.05 / 0.10) and classifier dropout 0.35 (instead of 0.2) | 66.03 | 64.84 | 92.45 | 26.39 |
| [12_blockdrop020_030.ipynb](archive/_01_z_cnn/_04_cnn_double_gap_2level_with_3layer_parameter/12_blockdrop020_030.ipynb) | block dropout 0.20 / 0.30 | 66.45 | 63.33 | 90.57 | **24.07** |
| [11_blockdrop015_025_e150.ipynb](archive/_01_z_cnn/_04_cnn_double_gap_2level_with_3layer_parameter/11_blockdrop015_025_e150.ipynb) | 10 with 150 epochs | **67.79** | 68.96 | 96.67 | 28.75 |
| [13_adamw.ipynb](archive/_01_z_cnn/_04_cnn_double_gap_2level_with_3layer_parameter/13_adamw.ipynb) | 10 with AdamW (lr 0.001, weight decay 0.01) | 62.85 | 60.96 | 90.44 | 27.29 |
| [14_lamb.ipynb](archive/_01_z_cnn/_04_cnn_double_gap_2level_with_3layer_parameter/14_lamb.ipynb) | 10 with LAMB (lr 0.001, weight decay 0.01) | 63.50 | 59.65 | 88.76 | 25.24 |

**What was kept and why:**

| Setting | Kept | Why |
|---|---|---|
| Data split | official 500 train / 50 val per class | All training images and the official validation images (01 → 02: +1.0). |
| RandAugment | 2 ops, magnitude 5 | A tie with (3, 7) (+0.2), and the milder one. |
| Label smoothing | 0.1 | 0.05 was 0.5 lower. |
| Optimizer | SGD, Nesterov, momentum 0.9 | AdamW and LAMB were 2.5–3.2 points lower. |
| Learning rate | 0.05, cosine down to 1e-5 | Not tuned separately. |
| Weight decay | 1e-4 | 1e-5 was 2.5 lower. 5e-4 was the best on the 8M model (run 15: 67.18, +1.2), but it did not help on the 2.7M model in our test. |
| Dropout | Dropout2d 0.15 / 0.25 after levels 3 / 4, dense 0.35 | Without block dropout (run 08: 65.89) the gap is the largest (33.2). All dropout levels ended within about 1 point (0.10 / 0.20, run 09: 67.09); 0.15 / 0.25 was the base of the later runs (11, 13, 14, 15). |
| Epochs | 100 | Converged by epoch 100: in the final runs the last 10 epochs add only +0.4 to +0.9 points. 150 epochs gave +1.8 on the 8M model for 1.5× the time (PC-CNN: about 114 h instead of 76 h per seed). Shorter runs mislead: run 15 was 10.9 points behind run 10 at epoch 50 but ended 1.2 higher. |
| Batch size | 64 | Not tuned. |

| Used for | Paper |
|---|---|
| SGD | Robbins & Monro (1951), *A Stochastic Approximation Method* — https://doi.org/10.1214/aoms/1177729586 |
| Nesterov momentum | Sutskever et al. (2013), *On the importance of initialization and momentum in deep learning* — http://proceedings.mlr.press/v28/sutskever13.html |
| Cosine learning-rate schedule | Loshchilov & Hutter (2017), *SGDR: Stochastic Gradient Descent with Warm Restarts* — https://arxiv.org/abs/1608.03983 |
| AdamW | Loshchilov & Hutter (2019), *Decoupled Weight Decay Regularization* — https://arxiv.org/abs/1711.05101 |
| LAMB | You et al. (2020), *Large Batch Optimization for Deep Learning: Training BERT in 76 minutes* — https://arxiv.org/abs/1904.00962 |

<img src="assets/divider.svg" width="100%" height="2" alt="">

### Model Size and Computation Cost

**Why a small model (2.7M parameters) and not the 8M model or a higher-parameter model**

An earlier Standard CNN had 8.0M parameters (the 10-conv model in StandardCNN Selection above). We compared it with the 2.7M model (8 convs) that is used for all experiments.

**The two models** (seed 42, 100 epochs, 200 classes; columns as in StandardCNN Selection above):

| Model | Parameters | Compute per image (one forward pass) | Val (%) | Aug. train (%) | Clean train (%) | Gap (pp) |
|---|---|---|---|---|---|---|
| Earlier CNN (10 convs) | 8,003,720 | 1,140M MACs (≈ 2,280M FLOPs) | **66.03** | **64.84** | **92.45** | 26.39 |
| StandardCNN (8 convs, used) | **2,710,376** | **419M MACs (≈ 838M FLOPs)** | 64.07 | 56.13 | 83.27 | **19.12** |

Compute: 1 MAC = one multiply-add = 2 FLOPs; conv and linear layers only, for one 64×64 image. The mixing was not the same (8M: 25% clean, 37.5% MixUp, 37.5% CutMix; 2.7M: ResizeMix on 50% of images), so the differences are approximate. The 8M row is run 10 of the training-settings table in StandardCNN Selection above (same dropout, weight decay, learning rate and optimizer as our model); the best 8M run (weight decay 5e-4) reached 67.18.

**Training time** on a MacBook Pro (Apple M3 Pro: 11-core CPU, 14-core GPU via MPS, 18 GB memory), one run at a time. 8M estimated as 2.7× the time, because it needs 2.7× the compute:

| Model | Per seed (2.7M, used) | All planned runs, 2.7M (used) | All planned runs, 8M (estimated) |
|---|---|---|---|
| StandardCNN, 3 seeds | about 3.7 h | ≈ 0.5 days | about 1.3 days |
| HPCCNNRaoV2 (the PC-CNN), 3 seeds | about 76 h (3.2 days) | ≈ 9.5 days | about 26 days |
| PCCNNEP (EP), 1 seed | about 4 days (estimated) | about 4 days | about 11 days |
| **Total** | | **about 14 days** | **about 38 days** |

**Why the 2.7M model:**

| Reason | Explanation |
|---|---|
| Training time | The PC-CNN runs 7 extra cycles through every conv and EP runs many inference steps per batch, so both get slower as the CNN gets bigger. About 38 days of training was too long before the thesis deadline; about 14 days fits. |
| Less overfitting | The 8M model is 2 points better on validation but overfits more: gap 26.39 vs 19.12 (20.83 even with weight decay 5e-4, run 15), although it mixed more (75% of batches vs 50% of images). |
| The question is relative | The thesis asks how the CNN and the PC-CNN differ in robustness to noise (which one is more robust), with the same backbone. That comparison is fair at any size, so the 8M model's 2 extra points of clean accuracy are not needed. |

**Limitation.** All results are for this 2.7M-parameter model. Whether the robustness difference between the CNN and the PC-CNN changes with model size is not tested here (future work).

<img src="assets/divider.svg" width="100%" height="2" alt="">

### Augmentation Selection

How the training augmentation was chosen. Final choice: **run 39, ResizeMix on 50% of images**; its steps are in the Augmentation pipeline, under Experiments Chosen above.

**How it was chosen.** The Standard CNN overfits: even with MixUp + CutMix (run 33) it gets 84% on the training images (without augmentation) but 64% on validation, a 20-point gap. So we compared mixing augmentations as regularisers. Most errors were on small or off-centre objects, so some runs target those. All runs: the Standard CNN, 100 epochs, seed 42; columns as in StandardCNN Selection. Notebooks and checkpoints: `archive/_01_z_cnn/_05_cnn/`.

| Run | Augmentation | Why we tried it | Val (%) | Aug. train (%) | Clean train (%) | Gap (pp) |
|---|---|---|---|---|---|---|
| 33 | 50% clean, 25% MixUp, 25% CutMix | Standard mixing regularisers | 63.88 | 60.86 | 84.38 | 20.47 |
| 38 | ResizeMix on every image | Whole image shrunk and pasted: the object stays complete, label by area | 62.22 | 41.95 | 75.36 | **13.22** |
| **39** | **ResizeMix on 50% of images** | 38 underfit, so mix only half | **64.07** | 56.13 | 83.27 | 19.12 |
| 60 | Zoom 0.8–1.2 on 50% of images | Objects at different sizes (scale jittering) | 63.26 | 58.45 | 82.72 | 19.36 |
| 61 | 33 with box-aware CutMix | Cut around the object (training boxes), label by the visible object | 63.77 | **61.08** | **84.45** | 20.86 |
| 62 | 39 with box-aware ResizeMix | Label by the visible object | 63.94 | 57.48 | 83.67 | 19.70 |
| 63 | 39 with box-aware CutMix instead of ResizeMix | CutMix label by the visible object | 63.86 | 54.34 | 82.77 | 18.94 |

Aug. train is below val because mixed images are harder (two images, split labels). 38 has the smallest gap only because it underfits: its val is also the lowest.

**Why ResizeMix (39).** All runs except 38 ended within one point, so the best two (33 and 39) were run with 3 seeds (42, 21, 84):

| Run | Seed 42 | Seed 21 | Seed 84 | Mean ± SD |
|---|---|---|---|---|
| 33 MixUp + CutMix | 63.88 | **63.62** | 63.36 | 63.62 ± 0.26 |
| **39 ResizeMix 50%** | **64.07** | 63.52 | **63.84** | **63.81 ± 0.28** |
| Difference (39 − 33) | +0.19 | −0.10 | +0.48 | +0.19 ± 0.29 |

**Accuracy: a tie.** 39 is 0.19 points higher on average, but not on every seed (seed 21: −0.10). Paired t-test over the 3 seeds: t = 1.1, p = 0.38, not significant. **So 39 was kept because it is simpler** (one mixing method instead of two) and used for all models. Box-aware mixing and zoom gave nothing extra.

| Used for | Paper |
|---|---|
| MixUp | Zhang et al. (2018), *mixup: Beyond Empirical Risk Minimization* — https://arxiv.org/abs/1710.09412 |
| CutMix | Yun et al. (2019), *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features* — https://arxiv.org/abs/1905.04899 |
| ResizeMix | Qin et al. (2020), *ResizeMix: Mixing Data with Preserved Object Information and True Labels* — https://arxiv.org/abs/2012.11101 |
| RandAugment | Cubuk et al. (2020), *RandAugment* — https://arxiv.org/abs/1909.13719 |
| Zoom (scale jittering) | Simonyan & Zisserman (2015) — https://arxiv.org/abs/1409.1556 |
