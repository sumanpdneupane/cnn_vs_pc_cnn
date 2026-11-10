from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn

from src.model.cnn import StandardCNN

# PCCNN: the StandardCNN trained with centered Equilibrium Propagation (local updates, no backprop).
# Papers: https://doi.org/10.1162/NECO_a_00949 (Whittington & Bogacz 2017, predictive coding)
#         https://doi.org/10.3389/fncom.2017.00024 (Scellier & Bengio 2017, EP)
#         https://doi.org/10.3389/fnins.2021.633674 (Laborieux et al. 2021, centered EP)
#         https://arxiv.org/abs/2606.03584 (Kerjan et al. 2026, EP for predictive coding)
#
# states[k] = layer k (0..7 conv, 8 dense), e_k = states[k] - f_k(states[k-1]),  E = 1/2 sum ||e_k||^2 + beta CE
# One step: free phase (feedforward) -> nudged phases +beta / -beta (K sweeps) -> grad = (signal(-b) - signal(+b)) / 2b
# BatchNorm statistics per sweep and per phase.

# One layer's bottom-up pass, cached and reused during the sweeps
LayerPass = namedtuple('LayerPass', [
    'inputs',           # input of the layer (previous state, after its dropout)
    'pre_activation',   # pool(BN(weights_forward(inputs)))
    'normalized',       # BN-normalized weighted sum, before gamma / shift
    'inv_std',          # 1 / BN standard deviation
    'pool_indices',     # max-pool indices (None if the layer has no max-pool)
    'pool_input_size',  # height, width before max-pool (None if no max-pool)
    'batch_mean',       # BN batch mean
    'batch_var',        # BN batch variance (biased)
])


class PCCNN(StandardCNN):
    training_method = 'ep'  # default for TrainModel: centered EP (method='bp' still allowed, e.g. as a check)

    def __init__(self, num_classes=200, block3_dropout=0.15, block4_dropout=0.25, classifier_dropout=0.35,
                 level4_dilation=2, beta=0.01, steps=10, label_smoothing=0.1, seed=0):
        # Same layers, names and weight init as StandardCNN: seed 42 starts from the same weights as BP
        super().__init__(num_classes, block3_dropout, block4_dropout, classifier_dropout, level4_dilation)

        # Read the hidden states from the layers (plain lists / dicts, so the state_dict is unchanged).
        # Each BatchNorm ends one hidden state; a MaxPool / Dropout after it belongs to that state.
        self.convs, self.batchnorms, self.maxpool_layers, self.dropout = [], [], set(), {}
        for level in (self.level1, self.level2, self.level3, self.level4, self.dense_transition):
            for module in level:
                last_state = len(self.batchnorms) - 1
                if isinstance(module, nn.Conv2d):
                    self.convs.append(module)
                elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    self.batchnorms.append(module)
                elif isinstance(module, nn.LeakyReLU):
                    self.leaky_relu = module.negative_slope
                elif isinstance(module, nn.MaxPool2d):
                    self.maxpool_layers.add(last_state)
                elif isinstance(module, (nn.Dropout, nn.Dropout2d)):
                    self.dropout[last_state] = module.p
        self.num_layers = len(self.batchnorms)  # 9 hidden states: 8 conv + dense
        self.dense_index = self.num_layers - 1  # the last hidden state is the dense layer

        self.beta, self.steps, self.label_smoothing = beta, steps, label_smoothing
        self.num_classes = num_classes
        self.dropout_generator = torch.Generator().manual_seed(seed)  # random numbers for the EP dropout masks
        self.num_phases = 1  # 2 while the +beta and -beta phases run together in one batch
        self.dropout_masks = {}

    @torch.no_grad()
    def predict(self, images):
        return self(images)

    @property
    def dense_layer(self):
        return self.dense_transition[0]

    # =============================== 1. One EP training step ===============================

    # Writes the EP gradient to every param.grad, updates the BN running stats and returns the feedforward logits
    # (the optimizer, gradient clip and LR scheduler then run exactly as for BP)
    @torch.no_grad()
    def ep_gradients(self, images, labels_a, labels_b, lam):
        batch_size, device = images.size(0), images.device
        self.new_dropout_masks(batch_size, device)

        # Free phase
        self.num_phases = 1
        free_states = self.feedforward_states(images, update_running_stats=True)
        logits = free_states[-1].clone()

        # +beta and -beta phases together: rows 0..B-1 are +beta, rows B..2B-1 are -beta
        target = self.soft_target(labels_a, labels_b, lam)
        images_both_beta_phases = torch.cat((images, images))
        target_both_beta_phases = torch.cat((target, target))
        beta_per_row = images.new_full((2 * batch_size, 1), self.beta)
        beta_per_row[batch_size:] = -self.beta
        self.num_phases = 2
        start_states = [torch.cat((state, state)) for state in free_states]
        states, passes = self.nudged_states(images_both_beta_phases, start_states, target_both_beta_phases,
                                            beta_per_row)

        # Gradient = -(signal(+beta) - signal(-beta)) / (2 * beta * B)
        phase_sign = torch.ones(2 * batch_size, device=device)
        phase_sign[batch_size:] = -1
        signals = self.local_weight_signals(images_both_beta_phases, states, passes, phase_sign)
        self.num_phases = 1

        gradients = {param: signal * (-1.0 / (2 * self.beta * batch_size)) for param, signal in signals.items()}
        if not bool(torch.stack([torch.isfinite(gradient).all() for gradient in gradients.values()]).all()):
            raise RuntimeError('EP gradient is not finite; stopping before the weights are changed.')
        for param, gradient in gradients.items():
            param.grad = gradient.to(param.dtype)
        return logits

    # =============================== 2. Phases ===============================

    # Free phase: plain feedforward pass (optionally updating the BN running stats, as a BP training forward does)
    def feedforward_states(self, images, update_running_stats=False):
        states = []
        for layer_index in range(self.num_layers):
            layer_pass = self.bottom_up_predictions(images, states, layer_index)
            states.append(self.activation(layer_pass.pre_activation))
            if update_running_stats:
                self.update_batchnorm_running_stats(layer_index, layer_pass)
        states.append(self.classifier_prediction(states[-1]))
        return states

    # Nudged phase: K sweeps of output, then even layers, then odd layers (layers in one group do not depend on
    # each other). Each layer's bottom-up pass is cached in passes[k] and refreshed only when state k-1 changes.
    def nudged_states(self, images, states, target, beta_per_row):
        passes = [self.bottom_up_predictions(images, states, layer_index) for layer_index in range(self.num_layers)]
        even_layers, odd_layers = range(0, self.num_layers, 2), range(1, self.num_layers, 2)
        for _ in range(self.steps):
            states[-1] = (self.classifier_prediction(states[-2])
                          - beta_per_row * self.cross_entropy_gradient(states[-1], target))
            for layer_group in (even_layers, odd_layers):
                new_states = {layer_index: self.activation(passes[layer_index].pre_activation
                                                           + self.top_down_error(images, states, layer_index, passes))
                              for layer_index in layer_group}
                for layer_index, new_state in new_states.items():
                    states[layer_index] = new_state
                for layer_index in layer_group:
                    if layer_index + 1 < self.num_layers:
                        passes[layer_index + 1] = self.bottom_up_predictions(images, states, layer_index + 1)
        return states, passes

    # Local (error x input) signal of every parameter, summed over the batch with phase_sign +1 / -1 per row
    def local_weight_signals(self, images, states, passes, phase_sign):
        signals = {}
        for layer_index in range(self.num_layers):
            layer_pass = passes[layer_index]
            error = self.local_predictions_error_for_that_layer(states[layer_index], layer_pass)
            bn_input_error = self.batchnorm_transpose(layer_index, error, layer_pass.normalized, layer_pass.inv_std)
            if layer_index < self.dense_index:
                conv = self.convs[layer_index]
                row_sign = phase_sign.view(-1, 1, 1, 1)
                signals[conv.weight] = torch.nn.grad.conv2d_weight(
                    layer_pass.inputs, conv.weight.shape, bn_input_error * row_sign,
                    padding=conv.padding, dilation=conv.dilation)
                sum_dims = (0, 2, 3)
            else:
                row_sign = phase_sign.view(-1, 1)
                signals[self.dense_layer.weight] = (bn_input_error * row_sign).T @ layer_pass.inputs.mean((2, 3))
                sum_dims = (0,)
            signals[self.batchnorms[layer_index].weight] = (error * row_sign * layer_pass.normalized).sum(sum_dims)
            signals[self.batchnorms[layer_index].bias] = (error * row_sign).sum(sum_dims)
        output_error = (states[-1] - self.classifier_prediction(states[-2])) * phase_sign.view(-1, 1)
        dense_output = states[-2] * self.dropout_mask(self.dense_index, images.size(0))
        signals[self.classifier.weight] = output_error.T @ dense_output
        signals[self.classifier.bias] = output_error.sum(0)
        return signals

    # =============================== 3. One layer ===============================

    # Layer k's bottom-up pass from state k-1: inputs and pre_activation = pool(BN(weights_forward(inputs)))
    def bottom_up_predictions(self, images, states, layer_index):
        if layer_index == 0:
            inputs = images
        else:
            inputs = states[layer_index - 1] * self.dropout_mask(layer_index - 1, images.size(0))
        weighted_sum = self.weights_forward(layer_index, inputs)
        bn_output, normalized, inv_std, batch_mean, batch_var = self.batchnorm_forward(layer_index, weighted_sum)
        pool_indices = pool_input_size = None
        if layer_index in self.maxpool_layers:
            pool_input_size = bn_output.shape[-2:]
            bn_output, pool_indices = F.max_pool2d(bn_output, 2, return_indices=True)
        return LayerPass(inputs, bn_output, normalized, inv_std, pool_indices, pool_input_size, batch_mean, batch_var)

    # Prediction error of a layer times the activation slope, sent back through its max-pool
    def local_predictions_error_for_that_layer(self, state, layer_pass):
        pre_activation = layer_pass.pre_activation
        error = (state - self.activation(pre_activation)) * self.activation_slope(pre_activation)
        return self.maxpool_transpose(error, layer_pass.pool_indices, layer_pass.pool_input_size)

    # Error fed back to state k from the layer above (-dE/d states[k]), through the transpose of layer k+1
    def top_down_error(self, images, states, layer_index, passes):
        rows = images.size(0)
        if layer_index == self.dense_index:
            output_error = states[-1] - self.classifier_prediction(states[-2])
            return (output_error @ self.classifier.weight) * self.dropout_mask(self.dense_index, rows)
        above_pass = passes[layer_index + 1]
        error = self.local_predictions_error_for_that_layer(states[layer_index + 1], above_pass)
        bn_input_error = self.batchnorm_transpose(layer_index + 1, error, above_pass.normalized, above_pass.inv_std)
        error_to_state = self.weights_transpose(layer_index + 1, bn_input_error, above_pass.inputs)
        return error_to_state * self.dropout_mask(layer_index, rows)

    def classifier_prediction(self, dense_state):
        dense_output = dense_state * self.dropout_mask(self.dense_index, dense_state.size(0))
        return dense_output @ self.classifier.weight.T + self.classifier.bias

    # Weights x input of layer k (no bias): conv for layers 0-7, GAP + dense for layer 8
    def weights_forward(self, layer_index, inputs):
        if layer_index < self.dense_index:
            return self.conv_forward(layer_index, inputs)
        return self.dense_forward(inputs)

    # Transpose of weights_forward(): weights^T x error, back to the shape of the layer's `inputs`
    def weights_transpose(self, layer_index, error, inputs):
        if layer_index < self.dense_index:
            return self.conv_transpose(layer_index, error)
        return self.dense_transpose(error, inputs)

    def conv_forward(self, layer_index, inputs):
        conv = self.convs[layer_index]
        return F.conv2d(inputs, conv.weight, padding=conv.padding, dilation=conv.dilation)

    def conv_transpose(self, layer_index, error):
        conv = self.convs[layer_index]
        return F.conv_transpose2d(error, conv.weight, padding=conv.padding, dilation=conv.dilation)

    # Global average pool over height, width, then the dense weights
    def dense_forward(self, inputs):
        return inputs.mean((2, 3)) @ self.dense_layer.weight.T

    # Dense weights^T, then the average pool's transpose: each position gets 1 / (height * width) of the error
    def dense_transpose(self, error, inputs):
        height, width = inputs.shape[2], inputs.shape[3]
        return (error @ self.dense_layer.weight)[:, :, None, None].expand_as(inputs) / (height * width)

    # Transpose of max-pool: each error goes back to the position that was the max
    @staticmethod
    def maxpool_transpose(error, pool_indices, pool_input_size):
        if pool_indices is None:
            return error
        rows, channels = error.shape[:2]
        height, width = pool_input_size
        unpooled = error.new_zeros(rows, channels, height * width)
        return unpooled.scatter_(2, pool_indices.flatten(2), error.flatten(2)).view(rows, channels, height, width)

    # =============================== 4. Batch norm, per phase ===============================
    # Tensors are viewed as (phases, batch, channels, ...) so each phase gets its own statistics.
    # torch.var_mean is used for every mean: on MPS it is ~3x faster than .mean over these dims.

    def batchnorm_forward(self, layer_index, weighted_sum):
        batchnorm = self.batchnorms[layer_index]
        channel_shape, reduce_dims = ((-1, 1, 1), (1, 3, 4)) if weighted_sum.dim() == 4 else ((-1,), (1,))
        output_per_phase = weighted_sum.view(self.num_phases, -1, *weighted_sum.shape[1:])
        batch_var, batch_mean = torch.var_mean(output_per_phase, dim=reduce_dims, unbiased=False, keepdim=True)
        inv_std = torch.rsqrt(batch_var + batchnorm.eps)
        normalized = (output_per_phase - batch_mean) * inv_std
        gamma, shift = batchnorm.weight.view(*channel_shape), batchnorm.bias.view(*channel_shape)
        bn_output = torch.addcmul(shift, gamma, normalized)  # gamma * normalized + shift
        shape = weighted_sum.shape
        return bn_output.reshape(shape), normalized.reshape(shape), inv_std, batch_mean, batch_var

    # Transpose of the BN Jacobian (mean and variance terms included):
    # gamma / std * (error - mean(error) - normalized * mean(error * normalized))
    def batchnorm_transpose(self, layer_index, error, normalized, inv_std):
        channel_shape, reduce_dims = ((-1, 1, 1), (1, 3, 4)) if error.dim() == 4 else ((-1,), (1,))
        error_per_phase = error.view(self.num_phases, -1, *error.shape[1:])
        normalized_per_phase = normalized.view(self.num_phases, -1, *normalized.shape[1:])
        mean_error = torch.var_mean(error_per_phase, dim=reduce_dims, keepdim=True)[1]
        mean_error_times_normalized = torch.var_mean(error_per_phase * normalized_per_phase,
                                                     dim=reduce_dims, keepdim=True)[1]
        gamma = self.batchnorms[layer_index].weight.view(*channel_shape)
        centered_error = torch.addcmul(error_per_phase - mean_error, normalized_per_phase,
                                       mean_error_times_normalized, value=-1)
        bn_input_error = centered_error * (gamma * inv_std)
        return bn_input_error.reshape(error.shape)

    def update_batchnorm_running_stats(self, layer_index, layer_pass):
        batchnorm = self.batchnorms[layer_index]
        values_per_channel = layer_pass.normalized.numel() // layer_pass.normalized.shape[1]
        unbiased_var = layer_pass.batch_var.flatten() * values_per_channel / (values_per_channel - 1)
        momentum = batchnorm.momentum
        batchnorm.running_mean.mul_(1 - momentum).add_(layer_pass.batch_mean.flatten(), alpha=momentum)
        batchnorm.running_var.mul_(1 - momentum).add_(unbiased_var, alpha=momentum)
        batchnorm.num_batches_tracked.add_(1)

    # =============================== 5. Dropout, targets, activation ===============================

    # One fixed dropout mask per batch, shared by every sweep and both phases.
    # Dropout2d on conv states: one value per channel; Dropout on the dense state: one per unit.
    def new_dropout_masks(self, batch_size, device):
        def random_mask(layer_index, dropout_prob):
            channels = self.batchnorms[layer_index].num_features
            shape = (batch_size, channels) if layer_index == self.dense_index else (batch_size, channels, 1, 1)
            kept = torch.rand(*shape, generator=self.dropout_generator) > dropout_prob
            return (kept.float() / (1 - dropout_prob)).to(device)

        self.dropout_masks = {layer_index: random_mask(layer_index, dropout_prob)
                              for layer_index, dropout_prob in self.dropout.items()}

    # Dropout mask of a state for `rows` rows (2B in the nudged phases: the same mask repeated for both phases);
    # 1.0 for states without dropout
    def dropout_mask(self, layer_index, rows):
        if layer_index not in self.dropout_masks:
            return 1.0
        layer_mask = self.dropout_masks[layer_index]
        if layer_mask.size(0) == rows:
            return layer_mask
        return layer_mask.repeat(rows // layer_mask.size(0), *([1] * (layer_mask.dim() - 1)))

    # Soft target per image: lam * smoothed(label_a) + (1 - lam) * smoothed(label_b)
    def soft_target(self, labels_a, labels_b, lam):
        def smoothed(labels):
            one_hot = F.one_hot(labels, self.num_classes).float()
            return (1 - self.label_smoothing) * one_hot + self.label_smoothing / self.num_classes

        return lam[:, None] * smoothed(labels_a) + (1 - lam[:, None]) * smoothed(labels_b)

    # d CE / d logits
    def cross_entropy_gradient(self, logits, target):
        return logits.softmax(1) - target

    def activation(self, values):
        return F.leaky_relu(values, self.leaky_relu)

    def activation_slope(self, values):
        return torch.where(values > 0, 1.0, self.leaky_relu)
