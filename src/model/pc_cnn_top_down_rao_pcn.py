import torch
import torch.nn.functional as F
from torch import nn

from src.model.cnn import StandardCNN

# HPCCNNTopDownRaoPCN: the Standard CNN + predictive coding at every conv layer.
# Rao & Ballard (1999), https://doi.org/10.1038/4580: tied weights, top-down predictions, bottom-up errors.
# Wen et al. (2018), https://arxiv.org/abs/1802.04762: cycle schedule, learned rates, backprop through the cycles.
# Gal & Ghahramani (2016), https://arxiv.org/abs/1512.05287: the same dropout mask in every cycle.
#
# HIERARCHY   r0 = image, r1..r8 = the 8 conv layers, r8 -> GAP -> dense 512 -> classifier
# W_l         conv_l x BatchNorm scale (gamma / sigma) x max-pool winners of cycle 0;  W_l^T = its exact transpose
#
# CYCLE       cycle 0 = one feedforward pass (= the Standard CNN), then 7 cycles of:
#   1. top-down    l = 8..2: p_l-1 = W_l^T r_l          r_l-1 <- phi((1 - b) r_l-1 + b p_l-1)   (Wen Eq. 11)
#   2. bottom-up   l = 1..8: e_l-1 = r_l-1 - p_l-1      r_l   <- phi(r_l + (a / lambda) W_l e_l-1) (Wen Eq. 10)
#   The bottom-up step is gradient descent on ||r_l-1 - W_l^T r_l||^2 (Rao & Ballard). Class read from r8 at the end.
#
# RATES       a, b learned per channel (start 1.0, 0.5 as in PCN). lambda = top eigenvalue of W W^T (power iteration),
#             so a = 1 is a stable step with BatchNorm.
# TRAINING    7 cycles in training and at test; all weights by backprop through the cycles. BatchNorm runs only in
#             cycle 0; the cycles use its scale gamma / running sigma.
# DROPOUT     As the Standard CNN (0.15 after r6, 0.25 after r8, dense 0.35), training only. One mask per batch, used in
#             all cycles: cycle 0 scales kept channels by 1/(1-p) once; the cycles re-zero dropped channels, no rescaling.
# VS PCN      BatchNorm, max-pool unpooling, LeakyReLU 0.05, dilation 2, dense layer, dropout, 7 cycles (PCN <= 6).

# One conv layer of the Standard CNN as W (bottom-up) and W^T (top-down). Holds references to the CNN's modules only.
class ConvLayer:
    def __init__(self, modules):
        self.conv = next(m for m in modules if isinstance(m, nn.Conv2d))
        self.batchnorm = next(m for m in modules if isinstance(m, nn.BatchNorm2d))
        self.pool = next((m for m in modules if isinstance(m, nn.MaxPool2d)), None)
        self.dropout = next((m.p for m in modules if isinstance(m, nn.Dropout2d)), 0.0)
        self.modules = [m for m in modules if not isinstance(m, nn.Dropout2d)]  # dropout is applied by the cycles

    # Cycle 0: conv -> BN -> LeakyReLU (-> max-pool), the Standard CNN's layer. Returns (output, max-pool winners).
    def forward(self, lower):
        output, winners = lower, None
        for module in self.modules:
            if isinstance(module, nn.MaxPool2d):
                output, winners = F.max_pool2d(output, module.kernel_size, module.stride, return_indices=True)
            else:
                output = module(output)
        return output, winners

    # BatchNorm scale gamma / running sigma
    def scale(self):
        bn = self.batchnorm
        return (bn.weight / torch.sqrt(bn.running_var + bn.eps)).view(1, -1, 1, 1)

    # W^T: top-down prediction of the layer below (unpool to the winners, scale, transposed conv)
    def predict_down(self, upper, winners, lower_size):
        if winners is not None:
            upper = F.max_unpool2d(upper, winners, 2, 2, output_size=lower_size)
        return self.transposed_conv(self.scale() * upper)

    # W: the error from below sent up (conv, scale, take the winners)
    def send_up(self, error):
        return self.scale() * self.conv(error)

    def transposed_conv(self, x):
        return F.conv_transpose2d(x, self.conv.weight, None, padding=self.conv.padding, dilation=self.conv.dilation)


class HPCCNNTopDownRaoPCN(StandardCNN):
    training_method = 'bp'
    top_down_inference = True  # False: one-pass ablation at test
    slope = 0.05  # LeakyReLU slope (phi), as in the Standard CNN

    def __init__(self, num_classes=200, block3_dropout=0.15, block4_dropout=0.25, classifier_dropout=0.35,
                 level4_dilation=2, inference_steps=7, bottom_up_rate=1.0, top_down_rate=0.5, image_size=64):
        # Same layers and init as StandardCNN: same start weights at seed 42
        super().__init__(num_classes, block3_dropout, block4_dropout, classifier_dropout, level4_dilation)
        self.inference_steps = inference_steps  # cycles in training and at test

        # conv_layers[l - 1] = layer l (r_l-1 -> r_l)
        self.conv_layers = [
            ConvLayer(group) for group in self.split_at_convs()
        ]
        channels = [
            layer.conv.out_channels for layer in self.conv_layers
        ]

        # Learned rates per channel: a_l for r1..r8 (bottom_up_rates[l - 1]), b_l for r1..r7 (top_down_rates[l - 1])
        self.bottom_up_rates = nn.ParameterList([
            nn.Parameter(torch.full((1, count, 1, 1), float(bottom_up_rate))) for count in channels
        ])
        self.top_down_rates = nn.ParameterList([
            nn.Parameter(torch.full((1, count, 1, 1), float(top_down_rate))) for count in channels[:-1]
        ])

        # Power-iteration vectors for lambda (own seed, global RNG unchanged) and lambda_l (0 = not measured yet)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            height = image_size
            for l, layer in enumerate(self.conv_layers, start=1):
                vector = torch.randn(1, layer.conv.in_channels, height, height)
                self.register_buffer(f'power_vector_{l}', vector / vector.norm())
                if layer.pool is not None:
                    height //= 2
        self.register_buffer('top_down_curvature', torch.zeros(len(self.conv_layers)))

    # The modules of level1..level4, split at every conv
    def split_at_convs(self):
        groups = []
        for level in [self.level1, self.level2, self.level3, self.level4]:
            for module in level:
                if isinstance(module, nn.Conv2d):
                    groups.append([])
                groups[-1].append(module)
        return groups

    def layer(self, l):
        return self.conv_layers[l - 1]

    # Train: 7 cycles with gradient. Eval: 7 cycles without gradient (or one pass if switched off).
    def forward(self, x, return_features=False):
        if self.training:
            self.update_top_down_curvature()
            return self.classify(self.cycles(x, self.inference_steps)[0][-1], return_features)
        if not self.top_down_inference:
            return super().forward(x, return_features)
        return self.classify(self.iterative_representations(x)[-1], return_features)

    # GAP + dense + classifier
    def classify(self, top, return_features=False):
        embedding = self.dense_transition(self.flatten(self.avgpool(top)))
        logits = self.classifier(embedding)
        return (logits, embedding) if return_features else logits

    # =============================== The cycles ===============================

    # One pass + `steps` cycles. Returns (r0..r8, [r8 after cycle 0..steps], predictions p0..p7, max-pool winners).
    def cycles(self, images, steps):
        masks = self.draw_dropout_masks(images)  # masks[l] for r_l (None in eval)
        r, winners = self.first_pass(images, masks)  # cycle 0
        top = len(self.conv_layers)
        top_per_cycle = [r[top]]
        p = [None] * top  # p[l - 1] = prediction of r_l-1 by layer l

        for _ in range(steps):
            # 1. Top-down: layer l predicts r_l-1, which moves toward the prediction
            for l in range(top, 1, -1):
                p[l - 1] = self.layer(l).predict_down(r[l], winners[l], r[l - 1].shape[-2:])
                r[l - 1] = self.top_down_update(l - 1, r[l - 1], p[l - 1], masks[l - 1])

            # 2. Bottom-up: layer 1 predicts the image, then each error from below corrects the layer above
            p[0] = self.layer(1).predict_down(r[1], winners[1], images.shape[-2:])
            for l in range(1, top + 1):
                error = self.bottom_up_error(l, r[l - 1] - p[l - 1], winners[l])
                r[l] = self.bottom_up_update(l, r[l], error, masks[l])

            top_per_cycle.append(r[top])
        return r, top_per_cycle, p, winners

    # Cycle 0: r0..r8 and the max-pool winners (winners[l] for layer l). Kept channels x 1/(1-p) once.
    def first_pass(self, images, masks=None):
        r, winners = [images], [None]
        for l, layer in enumerate(self.conv_layers, start=1):
            output, layer_winners = layer.forward(r[-1])
            if masks is not None and masks[l] is not None:
                output = output * masks[l] / (1.0 - layer.dropout)
            r.append(output)
            winners.append(layer_winners)
        return r, winners

    # Error from below sent up through layer l: W_l(r_l-1 - p_l-1)
    def bottom_up_error(self, l, difference, winners):
        error = self.layer(l).send_up(difference)
        if winners is None:
            return error
        return torch.gather(error.flatten(2), 2, winners.flatten(2)).view(winners.shape)

    # r_l moves toward the prediction from above: phi((1 - b) r + b p)   (Wen Eq. 11)
    def top_down_update(self, l, r, prediction, mask):
        b = F.relu(self.top_down_rates[l - 1])
        return self.drop(F.leaky_relu((1.0 - b) * r + b * prediction, self.slope), mask)

    # The error from below corrects r_l: phi(r + (a / lambda) e)   (Wen Eq. 10)
    def bottom_up_update(self, l, r, error, mask):
        step = F.relu(self.bottom_up_rates[l - 1]) / self.top_down_curvature[l - 1]
        return self.drop(F.leaky_relu(r + step * error, self.slope), mask)

    # =============================== Dropout (training only) ===============================

    # One 0/1 mask per image and channel for r_l with dropout, drawn once per batch; masks[0] (the image) is None
    def draw_dropout_masks(self, images):
        masks = [None]
        for layer in self.conv_layers:
            if not self.training or layer.dropout == 0.0:
                masks.append(None)
                continue
            keep = torch.full((images.shape[0], layer.conv.out_channels, 1, 1), 1.0 - layer.dropout,
                              device=images.device)
            masks.append(torch.bernoulli(keep))
        return masks

    # Re-zero the dropped channels after an update (no rescaling)
    @staticmethod
    def drop(r, mask):
        return r if mask is None else r * mask

    # =============================== lambda (step size) ===============================

    # lambda_l = top eigenvalue of W_l W_l^T by power iteration (5 per batch, 300 the first time)
    @torch.no_grad()
    def update_top_down_curvature(self, iterations=5):
        if not bool((self.top_down_curvature > 0).all()):
            iterations = 300
        for l, layer in enumerate(self.conv_layers, start=1):
            vector = getattr(self, f'power_vector_{l}')
            for _ in range(iterations):
                vector = layer.transposed_conv(layer.send_up(vector))
                eigenvalue = vector.norm()
                vector = vector / eigenvalue
            getattr(self, f'power_vector_{l}').copy_(vector)
            self.top_down_curvature[l - 1] = eigenvalue

    # Measure lambda once if the model was never trained
    def ensure_top_down_curvature(self):
        if not bool((self.top_down_curvature > 0).all()):
            self.update_top_down_curvature()

    # =============================== Test (eval mode, no gradient) ===============================

    def check_eval(self, name):
        if self.training:
            raise RuntimeError(f'{name} runs in eval mode: call model.eval() first.')
        self.ensure_top_down_curvature()

    # r0..r8 after one pass
    def feedforward_representations(self, images):
        return self.first_pass(images)[0]

    # r8 after every cycle 0..steps
    @torch.no_grad()
    def iterative_representations(self, images, steps=None):
        self.check_eval('Iterative inference')
        return self.cycles(images, self.inference_steps if steps is None else steps)[1]

    # Logits after every cycle 0..steps
    @torch.no_grad()
    def iterative_inference(self, images, steps=None):
        return [self.classify(top) for top in self.iterative_representations(images, steps)]

    # r0..r8 after `steps` cycles
    @torch.no_grad()
    def settled_representations(self, images, steps):
        self.check_eval('settled_representations')
        return self.cycles(images, steps)[0]

    # Per layer l: ||r_l-1 - W_l^T r_l||^2 and ||r_l-1||^2 after `steps` cycles (summed over the batch)
    @torch.no_grad()
    def prediction_errors(self, images, steps):
        self.check_eval('prediction_errors')
        r, _, _, winners = self.cycles(images, steps)
        errors, norms = [], []
        for l, layer in enumerate(self.conv_layers, start=1):
            prediction = layer.predict_down(r[l], winners[l], r[l - 1].shape[-2:])
            errors.append((r[l - 1] - prediction).pow(2).sum().item())
            norms.append(r[l - 1].pow(2).sum().item())
        return errors, norms
