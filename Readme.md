# Robustness Analysis of Standard CNN and Predictive Coding CNN Models for Image Classification under Gaussian and Salt-and-Pepper Noise

## Overview

**Problem.** Image classifiers are usually tested on clean images, but real images are often noisy (poor light, low-quality cameras, compression, sensor noise). Noise and other corruptions can make a network wrong, so clean accuracy alone does not show how reliable a model is. A Predictive-Coding CNN (PC-CNN) corrects its features with top-down predictions and bottom-up prediction errors over several steps (iterative inference), which may help with noisy images. But there is little direct comparison of a Standard CNN and a PC-CNN with the same structure, both trained on clean images and tested under controlled noise.

**Research question.** How do a Standard CNN and a PC-CNN differ in robustness when the Tiny ImageNet validation images get Gaussian or salt-and-pepper noise at test time?

**Hypothesis.** Both kinds of noise will lower the results of both models. The PC-CNN may behave differently from the Standard CNN under noise, because it uses prediction-error correction and iterative inference.

**What we do:**
1. Build two models with the same layers: **StandardCNN** (one feedforward pass) and **HPCCNNRaoV2** (the same CNN + 7 cycles of top-down predictions and bottom-up prediction errors).
2. Train both only on clean Tiny ImageNet training images (200 classes), with the same settings and 3 seeds each.
3. Test on the clean validation images (baseline).
4. Add Gaussian and salt-and-pepper noise to the validation images at 0.05, 0.1, 0.2, 0.3 and 0.5 (test only, never in training) and test again.
5. Compare the models: accuracy drop, loss increase, robustness retention, corruption error, calibration, statistical tests over the seeds, and compute cost (see Evaluation).

Extra experiment: **PCCNNEP**, the same CNN trained with a predictive-coding learning rule (Equilibrium Propagation) instead of backprop, one seed.

<img src="assets/divider.svg" width="100%" height="2" alt="">

## Installation

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

**4. Run an experiment.** Open a notebook in `experiments_in_seeds/` (see the runs table below) and run all cells. Checkpoints and `metrics.csv` are written to `checkpoints/<model>/<run>/`. To continue a stopped run, set `RESUME = True` in the first code cell.

Training time per seed on a MacBook Pro (Apple M3 Pro: 11-core CPU, 14-core GPU via MPS, 18 GB memory):
- StandardCNN: about 3.7 h
- HPCCNNRaoV2: about 76 h (about 3.2 days)
- PCCNNEP: about 4 days (estimated)

<img src="assets/divider.svg" width="100%" height="2" alt="">

## Data

- **Tiny ImageNet** (200 classes, 64×64, 500 train + 50 val images per class), in `data/tiny_imagenet/tiny-imagenet-200` (val images sorted into one folder per class).
- Normalisation: mean/std of the Tiny ImageNet training set. Validation: clean images. Training augmentation: see Augmentation below.
- Code: `src/dataset/`.

| Used for | Paper |
|---|---|
| Dataset | Le & Yang (2015), *Tiny ImageNet Visual Recognition Challenge* — http://cs231n.stanford.edu/reports/2015/pdfs/yle_project.pdf |

<img src="assets/divider.svg" width="100%" height="2" alt="">

## Experiments

- All 200 classes: notebooks in `experiments_in_seeds/`, checkpoints in `checkpoints/`.
- 10-class test runs: `archive/_03_experiments_10_classes/` (checkpoints in its `checkpoints/`).

| Model | Experiments | Seeds |
|---|---|---|
| StandardCNN | `experiments_in_seeds/cnn/` | 42, 21, 84 |
| HPCCNNRaoV2 (the PC-CNN) | `experiments_in_seeds/hpc_cnn_rao_v2/` | 42, 21, 84 |
| PCCNNEP (EP, extra experiment) | `experiments_in_seeds/pc_cnn_ep/` | 42 |

**Settings** (the same for every model; optimizer and learning rate: see StandardCNN):

| Setting | Value | Why |
|---|---|---|
| Seeds | 42, 21, 84 (EP: 42 only) | 3 seeds show how much a result changes by chance, and allow paired tests (CNN seed 42 vs PC-CNN seed 42, and so on). Fixed before training, not tuned. EP has one seed because one run takes about 4 days. |
| What a seed sets | start weights, order of the training images, augmentation and dropout randomness | With the same seed, every model starts from the same weights and sees the same batches, so the models differ only in the model itself. |
| Images | all 500 train + 50 val per class, 200 classes (100,000 + 10,000) | The full dataset; the seed does not change which images are used. |
| Batch size | 64 | The same for all models. |
| Epochs | 100 | Cosine learning-rate schedule down to 1e-5 over the 100 epochs. |
| Data-loader workers | 4 | Fastest in our test (`00_mac_worker_test.ipynb`, 11-core M3 Pro): 0 / 2 / 4 / 6 / 8 workers = 35 / 54 / 66 / 58 / 49 batches per second. |
| Clean-train check | epoch 1, then every 5 epochs | Accuracy on the training images without augmentation, to watch overfitting. Not every epoch because it costs time. |
| Checkpoints | `latest_checkpoint.pth` every epoch, `best_model.pth` at the best val accuracy | `latest` stores everything to resume exactly (weights, optimizer, schedule, random states). Evaluation uses `latest` (epoch 100). |

<img src="assets/divider.svg" width="100%" height="2" alt="">

## Augmentation

`src/dataset/resizemix.py`. Training only, the same for all models: random crop (pad 4) + horizontal flip → ResizeMix on 50% of images → RandAugment (2 ops, magnitude 5).

**How it was chosen.** The Standard CNN overfits: even with MixUp + CutMix (run 33) it gets 84% on the training images (without augmentation) but 64% on validation, a 20-point gap. So we compared mixing augmentations as regularisers. Most errors were on small or off-centre objects, so some runs target those. All runs: the Standard CNN, RandAugment (2 ops, magnitude 5), 100 epochs, seed 42. Val = mean of epochs 96–100; clean train = training images without augmentation, epoch 100. Notebooks and checkpoints: `archive/_01_z_cnn/_05_cnn/`.

| Run | Augmentation | Why we tried it | Val (%) | Clean train (%) |
|---|---|---|---|---|
| 33 | 50% clean, 25% MixUp, 25% CutMix | Standard mixing regularisers | 63.88 | 84.38 |
| 38 | ResizeMix on every image | Whole image shrunk and pasted: the object stays complete, label by area | 62.22 | 75.36 |
| **39** | **ResizeMix on 50% of images** | 38 underfit, so mix only half | **64.07** | 83.27 |
| 60 | Zoom 0.8–1.2 on 50% of images | Objects at different sizes (scale jittering) | 63.26 | 82.72 |
| 61 | 33 with box-aware CutMix | Cut around the object (training boxes), label by the visible object | 63.77 | 84.45 |
| 62 | 39 with box-aware ResizeMix | Label by the visible object | 63.94 | 83.67 |
| 63 | 39 with box-aware CutMix instead of ResizeMix | CutMix label by the visible object | 63.86 | 82.77 |

**Why ResizeMix (39).** All runs except 38 ended within one point, so the best two (33 and 39) were run with 3 seeds (42, 21, 84):

| Run | Seed 42 | Seed 21 | Seed 84 | Mean ± SD |
|---|---|---|---|---|
| 33 MixUp + CutMix | 63.88 | 63.62 | 63.36 | 63.62 ± 0.26 |
| **39 ResizeMix 50%** | 64.07 | 63.52 | 63.84 | **63.81 ± 0.28** |

39 has the higher 3-seed mean (+0.19, paired t-test p = 0.38, so a tie in accuracy) and is the simpler recipe (one mixing method instead of two), so it was kept for all models. Box-aware mixing and zoom gave nothing extra.

| Used for | Paper |
|---|---|
| MixUp | Zhang et al. (2018), *mixup: Beyond Empirical Risk Minimization* — https://arxiv.org/abs/1710.09412 |
| CutMix | Yun et al. (2019), *CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features* — https://arxiv.org/abs/1905.04899 |
| ResizeMix | Qin et al. (2020), *ResizeMix: Mixing Data with Preserved Object Information and True Labels* — https://arxiv.org/abs/2012.11101 |
| RandAugment | Cubuk et al. (2020), *RandAugment* — https://arxiv.org/abs/1909.13719 |
| Zoom (scale jittering) | Simonyan & Zisserman (2015) — https://arxiv.org/abs/1409.1556 |

<img src="assets/divider.svg" width="100%" height="2" alt="">

## StandardCNN

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

- Training: SGD (Nesterov, momentum 0.9, lr 0.05, weight decay 1e-4), cosine schedule, label smoothing 0.1, batch 64, 100 epochs.

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
| Cosine schedule | Loshchilov & Hutter (2017), SGDR — https://arxiv.org/abs/1608.03983 |

<img src="assets/divider.svg" width="100%" height="2" alt="">

## HPCCNNRaoV2 (the PC-CNN)

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
- **Top-down predictions:** no new weights (the transposed conv of the same conv).
- **Not learned (buffers):** λ per layer (top eigenvalue of WᵀW with the BatchNorm scale included, sets the step a/λ) and its 8 power-iteration vectors.
- **Compute:** 7 inference steps (cycles) after the feedforward pass, in training and at test. Each step = one top-down + one bottom-up sweep through all 8 convs.
- **Training:** the same loss, optimizer and schedule; backprop runs through all cycles. Hidden dropout uses one mask for all cycles.

| Used for | Paper |
|---|---|
| Top-down predictions, bottom-up errors, tied weights | Rao & Ballard (1999), *Predictive coding in the visual cortex*, Nature Neuroscience — https://doi.org/10.1038/4580 |
| Cycle schedule, update rates, training through the cycles | Wen et al. (2018), *Deep Predictive Coding Network for Object Recognition*, ICML — https://arxiv.org/abs/1802.04762 |
| Same dropout mask in every cycle | Gal & Ghahramani (2016), *A Theoretically Grounded Application of Dropout in Recurrent Neural Networks* — https://arxiv.org/abs/1512.05287 |

<img src="assets/divider.svg" width="100%" height="2" alt="">

## PCCNNEP (EP, extra experiment)

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

## Evaluation (to do, after training)

Every final checkpoint (epoch 100, `latest_checkpoint.pth`, not `best_model.pth`), all seeds, on the 10,000 validation images.

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

