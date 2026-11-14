# Standard CNN vs Predictive-Coding CNN

```commandline
pip install -r requirements.txt
```

## Data

- **Tiny ImageNet** (200 classes, 64×64, 500 train + 50 val images per class), in `data/tiny_imagenet/tiny-imagenet-200` (val images sorted into one folder per class).
- Normalisation: mean/std of the Tiny ImageNet training set. Training augmentation: random crop (pad 4) + horizontal flip → ResizeMix on 50% of images → RandAugment (2 ops, magnitude 5). Validation: clean images.
- Code: `src/dataset/`.
- Experiments: all 200 classes in `experiments_in_seeds/` (checkpoints in `checkpoints/`). 10-class test runs: `archive/_03_experiments_10_classes/`.

| Model | Experiments | Seeds |
|---|---|---|
| StandardCNN | `experiments_in_seeds/cnn/` | 42, 21, 84 |
| HPCCNNTopDownRaoPCNExactLambda (the PC-CNN) | `experiments_in_seeds/pc_cnn_top_down_rao_pcn_exact_lambda/` | 42, 21, 84 |
| PCCNN (EP, extra experiment) | `experiments_in_seeds/pc_cnn_ep/` | 42 |

| Used for | Paper |
|---|---|
| Dataset | Le & Yang (2015), *Tiny ImageNet Visual Recognition Challenge* — http://cs231n.stanford.edu/reports/2015/pdfs/yle_project.pdf |
| ResizeMix | Qin et al. (2020), *ResizeMix: Mixing Data with Preserved Object Information and True Labels* — https://arxiv.org/abs/2012.11101 |
| RandAugment | Cubuk et al. (2020), *RandAugment* — https://arxiv.org/abs/1909.13719 |

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

## HPCCNNTopDownRaoPCNExactLambda (the PC-CNN)

`src/model/pc_cnn_top_down_rao_pcn_exact_lambda.py` (class `HPCCNNTopDownRaoPCNExactLambda`). The Standard CNN (same layers, start weights, dropout and training) plus predictive coding at every conv layer.

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

## PCCNN (EP, extra experiment)

`src/model/pc_cnn_ep.py` (class `PCCNN`). The Standard CNN with its weights learned by centered Equilibrium Propagation: local updates, no backprop. One seed.

- Each batch: free phase (one feedforward pass), then the output is nudged toward the label (+β and −β, β = 0.01). 10 sweeps spread the prediction errors e = state − f(state below) down the layers, and each weight is updated from its local error × input.
- At test: one feedforward pass, like the Standard CNN.
- Difference from the PC-CNN: here predictive coding is the learning rule. In the PC-CNN it is the inference (7 cycles at test).

| Used for | Paper |
|---|---|
| Predictive coding as learning | Whittington & Bogacz (2017), *Neural Computation* — https://doi.org/10.1162/NECO_a_00949 |
| Equilibrium Propagation | Scellier & Bengio (2017), *Frontiers in Computational Neuroscience* — https://doi.org/10.3389/fncom.2017.00024 |
| Centered EP (+β and −β) | Laborieux et al. (2021), *Frontiers in Neuroscience* — https://doi.org/10.3389/fnins.2021.633674 |

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

