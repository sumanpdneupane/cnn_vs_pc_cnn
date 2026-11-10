# Standard CNN vs Predictive-Coding CNN

```commandline
pip install -r requirements.txt
```

## Data

- **Tiny ImageNet** (200 classes, 64×64, 500 train + 50 val images per class), in `data/tiny_imagenet/tiny-imagenet-200` (val images sorted into one folder per class).
- Experiments: all 200 classes in `experiments_in_seeds/` (checkpoints in `checkpoints/`). 10-class test runs: `archive/_03_experiments_10_classes/`.
- Normalisation: mean/std of the Tiny ImageNet training set. Training augmentation: random crop (pad 4) + horizontal flip → ResizeMix on 50% of images → RandAugment (2 ops, magnitude 5). Validation: clean images.
- Code: `src/dataset/`.

| Used for | Paper |
|---|---|
| Dataset | Le & Yang (2015), *Tiny ImageNet Visual Recognition Challenge* — http://cs231n.stanford.edu/reports/2015/pdfs/yle_project.pdf |
| ResizeMix | Qin et al. (2020), *ResizeMix: Mixing Data with Preserved Object Information and True Labels* — https://arxiv.org/abs/2012.11101 |
| RandAugment | Cubuk et al. (2020), *RandAugment* — https://arxiv.org/abs/1909.13719 |

## StandardCNN

`src/model/cnn.py`. The baseline, trained with backprop.

- 8 conv layers (3×3, 2 per block: 32-64 · 64-128 · 128-256 · 256-512), each conv → BatchNorm → LeakyReLU (0.05).
- Max-pool after blocks 1–3; conv8 dilation 2. Dropout2d 0.15 (block 3), 0.25 (block 4).
- Global average pooling → dense 512 (BN, LeakyReLU, dropout 0.35) → classifier. 2.71M parameters (200 classes).
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

## HPCCNNTopDownRaoPCN

`src/model/pc_cnn_top_down_rao_pcn.py` (class `HPCCNNTopDownRaoPCN`). The Standard CNN (same layers, start weights, dropout and training) plus predictive coding at every conv layer.

- Cycle 0 = one feedforward pass (= the Standard CNN). Then **7 cycles**, in training and at test:
  1. **Top-down:** each layer predicts the layer below, p = Wᵀr, and the layer moves toward the prediction.
  2. **Bottom-up:** the prediction error e = r − p goes up through W and corrects the layer above.
- Tied weights: the prediction uses the transposed conv of the same layer. Update rates learned per channel.
- **Per conv, not per block:** each of the 8 convs has its own top-down prediction (`conv_transpose2d` with that conv's weight). Unpool to the max-pool winners only after conv2, 4, 6.

| Block | Conv | Bottom-up (conv) | Top-down (transposed conv, same weight) |
|---|---|---|---|
| level1 | conv1 | image → r1 (32×64×64) | r1 → image |
| | conv2 + pool | r1 → r2 (64×32×32) | r2 → unpool → r1 |
| level2 | conv3 | r2 → r3 (64×32×32) | r3 → r2 |
| | conv4 + pool | r3 → r4 (128×16×16) | r4 → unpool → r3 |
| level3 | conv5 | r4 → r5 (128×16×16) | r5 → r4 |
| | conv6 + pool | r5 → r6 (256×8×8) | r6 → unpool → r5 |
| level4 | conv7 | r6 → r7 (256×8×8) | r7 → r6 |
| | conv8 (dilation 2) | r7 → r8 (512×8×8) | r8 → r7 |

- All weights learned by backprop through the cycles. Hidden dropout uses one mask for all cycles.
- 2,368 extra parameters (the rates).

| Used for | Paper |
|---|---|
| Top-down predictions, bottom-up errors, tied weights | Rao & Ballard (1999), *Predictive coding in the visual cortex*, Nature Neuroscience — https://doi.org/10.1038/4580 |
| Cycle schedule, update rates, training through the cycles | Wen et al. (2018), *Deep Predictive Coding Network for Object Recognition*, ICML — https://arxiv.org/abs/1802.04762 |
| Same dropout mask in every cycle | Gal & Ghahramani (2016), *A Theoretically Grounded Application of Dropout in Recurrent Neural Networks* — https://arxiv.org/abs/1512.05287 |
