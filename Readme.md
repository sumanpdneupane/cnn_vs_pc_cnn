```commandline
pip install --upgrade pip
pip install -r requirements.txt
```

# ImageNet-21K = 1,31,50,000 (1 crore 31 lakh 50 thousand), classes= 19,167, imbalanced
# ImageNet-21K-P = 1,15,82,723 (1 crore 15 lakh 82 thousand 7 hundred 23), classes= 10,450, balanced


# https://github.com/Alibaba-MIIL/ImageNet21K/blob/main/dataset_preprocessing/processing_instructions.md
# https://huggingface.co/datasets/timm/imagenet-w21-p
# https://huggingface.co/datasets/gmongaras/Imagenet21K

```
This:
The model reconstructs the input image through hierarchical predictive-coding inference and 
uses the higher-level latent representation (r2) for image classification.

Or a bit simpler:

The PC-CNN learns to reconstruct the input image and then uses the higher-level representation 
learned during predictive-coding inference to classify the image.

Unlike a standard CNN trained through end-to-end backpropagation, the proposed PC-CNN performs 
iterative predictive-coding inference, minimizes local hierarchical prediction errors, and 
updates its backbone weights using local learning rules. The resulting higher-level 
representation is then used for image classification.
```

SEEDS = [42, 123, 2024, 3407, 777]

https://www.leoniemonigatti.com/blog/pytorch-image-classification.html
1. CE-only StandardCNN cnn_1, cnn_4
   Best Val Acc = 78.00%
   Feature separation, 
    Mean Intra-Class Distance : 2.258936
    Mean Inter-Class Distance : 3.173573
    Inter / Intra Ratio       : 1.404898

2. CE + Center Loss cnn_2, cnn_5
   Best Val Acc = 76.80%
   Feature separation, 
   Mean Intra-Class Distance = 1.604844
   Mean Inter-Class Distance = 2.363687
   Inter / Intra Ratio       = 1.472845

3. CE + SupCon cnn_3
   Best Val Acc = 78.90%
   Not built yet
   Not trained yet


Spatial pooling 4×4 + Linear + bias=False
Spatial pooling 7×7 + Linear + bias=False
Spatial pooling 7×7 + Linear + bias=True
A hierarchical convolutional adaptation of Rao-style predictive coding with mini-batch local learning.

# Test
There are really two scientifically valid approaches:
| Design                                                | What it measures                                                   |
| ----------------------------------------------------- | ------------------------------------------------------------------ |
| Current Rao backbone + frozen linear readout          | Quality of **unsupervised Rao representations** for classification |
| Rao hierarchy + supervised top-level prediction error | Actual **Rao-style PC-CNN classification model**   .'.                |

### Experiment 1
### Test 1
r3 class separation mainly means two things:
1. Within-class variation should be SMALL
2. Between-class distance should be LARGE
make same-class r3 features closer together + make different-class r3 features farther apart
r3
├── predict r2 well          → low e2
└── separate classes well    → low within-class spread + high between-class distance
             weak r3 representation
                    ↓
both poor class separation and weak e2 behavior

1. Understand/improve e2 at r2 ↔ r3
2. Improve r3 class separation

After these diagnostics:
1. r2 ↔ r3 / e2   ✓ understood and reasonably healthy
2. r3 class separation   ← MAIN PROBLEM NOW

A. Class feedback itself does not separate r3
B. Class feedback separates r3, but learning fails to preserve it
Your result strongly supports B.

### Test 2
Reconstruction                 ✓
r1/r2                          ✓
r2 ↔ r3 predictive relation    ✓
e2                             ✓ reasonably healthy
Class feedback magnitude       ✓ sufficient
Class nudge                    ✓ improves r3 separation

Main remaining weakness:
NUDGED r3 structure → FREE r3 learning/preservation

50-nudge training did not improve the final FREE r3 classification representation
10 nudge steps:
classes are somewhat closer together,
but samples inside each class are more compact.
50 nudge steps:
class centers move slightly farther apart,
BUT each class also spreads out considerably.
This is why 50-step nudging looked excellent in the NUDGED r3, 
but did not improve the final FREE r3 representation we actually use at test time.

### Test 3
In our r3 representation analysis:
- Within-class spread = how far samples of the same class are spread around 
their own class centroid. Lower is better because same-class images are clustering more tightly.
- Between-class distance = distance between the centroids of different classes. 
Higher is better because different classes are farther apart.

those are the three core geometry measures we are using for FREE r3 class separation:
1. Within-class spread
2. Between-class distance
3. Separation ratio

Within-class spread: are samples from the same class close together? Lower is better.
Between-class distance: are different class centroids far apart? Higher is better.
Separation ratio: between-class distance / within-class spread

### Experiment 2: nonlinear_rao_pccnn_joint_supervised
FREE validation accuracy       ↑
FREE r3 within-class spread    ↓
FREE r3 between-class distance ↑
FREE r3 separation ratio       ↑
FREE nearest-centroid accuracy ↑


https://github.com/miladmozafari/predify/tree/master
https://github.com/dbersan/Predictive-Coding-Implementation/blob/main/README.md
https://medium.com/@oliviers.gaspard/training-brain-inspired-predictive-coding-models-in-python-5a7011e2779d
https://github.com/Bogacz-Group/PredictiveCoding/blob/main/README.md

CURRENT DIRECTION
Rao-style generative hierarchy
+
our manually attached supervised class feedback


MORE PRINCIPLED SUPERVISED DIRECTION
Rao-style local prediction/error hierarchy
+
Whittington-Bogacz idea:
class/output becomes the highest predictive level
+
manual local equations
+
NO autograd
+
NO backward()

# New comcept
Rao (inspired hierarchical predictive coding)
Karl Friston (Free Energy Principle (FEP) to minimize variational free energy for generative model)
Whittington–Bogacz (predictive-coding/free-energy formulation for supervised model)

        Rao & Ballard
            ↓
        Hierarchical predictive coding
        Top-down prediction
        Bottom-up prediction error
        
        Friston
            ↓
        Broader Free Energy Principle
        Variational inference / generative models
        Prediction-error minimization as part of a larger theory
        
        Whittington & Bogacz
            ↓
        Neural-network predictive coding
        Free-energy / error minimization
        Supervised target clamping
        Local learning rules


| Researcher / Framework                         | Main idea                                                                                                                                         | Role in predictive coding                                                                                             |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Rao & Ballard**                              | Hierarchical predictive coding for sensory processing: higher levels generate top-down predictions, lower levels send bottom-up prediction errors | Core hierarchical prediction/error architecture                                                                       |
| **Karl Friston — Free Energy Principle (FEP)** | Biological systems can be described as minimizing variational free energy under a generative model                                                | Broader theoretical framework linking perception, inference, prediction, and action                                   |
| **Whittington & Bogacz**                       | Predictive coding formulated as free-energy / prediction-error minimization, including supervised learning and local weight updates               | Practical neural-network formulation showing how PC can approximate learning normally associated with backpropagation |
