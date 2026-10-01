"""Comprehensive, non-destructive diagnostics for a PyTorch image classifier.

Designed for StandardCNN with level1..level4, flatten and classifier, but works
with other PyTorch models. Runs full validation activation analysis and optional
train-batch gradient analysis on a deep copy of the model.
"""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


def _stats(t):
    t = t.detach().float().cpu().reshape(-1)
    if not t.numel():
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan,
                    abs_mean=np.nan, l2=np.nan, zero_pct=np.nan, finite_pct=np.nan)
    finite = torch.isfinite(t)
    v = t[finite]
    return dict(mean=v.mean().item() if v.numel() else np.nan,
                std=v.std(unbiased=False).item() if v.numel() else np.nan,
                min=v.min().item() if v.numel() else np.nan,
                max=v.max().item() if v.numel() else np.nan,
                abs_mean=v.abs().mean().item() if v.numel() else np.nan,
                l2=v.norm().item() if v.numel() else np.nan,
                zero_pct=100 * (t == 0).float().mean().item(),
                finite_pct=100 * finite.float().mean().item())


def analyze_cnn_full(model, val_loader, device, train_loader=None, criterion=None,
                     output_dir='full_cnn_diagnostics', max_val_batches=None,
                     max_grad_batches=3, class_names=None):
    """Return diagnostic tables. Does not modify the supplied model.

    Args:
      model: instantiated model with checkpoint weights already loaded.
      val_loader: clean validation loader; use shuffle=False and num_workers=0.
      train_loader: optional augmented training loader for representative gradients.
      criterion: training loss (default CrossEntropyLoss).
      max_val_batches: None = full validation set (recommended).
      max_grad_batches: independent batches used for gradient measurements.
    """
    device = torch.device(device)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = copy.deepcopy(model.module if isinstance(model, nn.DataParallel) else model).to(device)
    m.eval()
    criterion = criterion or nn.CrossEntropyLoss()

    # Weight and BN statistics from the checkpoint, without modifying it.
    weight_rows = []
    for name, p in m.named_parameters():
        weight_rows.append(dict(parameter=name, shape=str(tuple(p.shape)), count=p.numel(),
                                requires_grad=p.requires_grad, **_stats(p)))
    weights = pd.DataFrame(weight_rows)
    weights.to_csv(out / 'weights_biases.csv', index=False)

    bn_rows = []
    for name, layer in m.named_modules():
        if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            row = dict(layer=name, features=layer.num_features, eps=layer.eps,
                       momentum=layer.momentum, batches_tracked=int(layer.num_batches_tracked.item())
                       if layer.num_batches_tracked is not None else None)
            for label, t in [('running_mean', layer.running_mean),
                             ('running_var', layer.running_var),
                             ('gamma', layer.weight), ('beta', layer.bias)]:
                if t is not None:
                    for k, v in _stats(t).items():
                        row[f'{label}_{k}'] = v
            if layer.running_var is not None:
                row['near_zero_var_channels'] = int((layer.running_var < 1e-5).sum().item())
            bn_rows.append(row)
    bn = pd.DataFrame(bn_rows)
    bn.to_csv(out / 'batchnorm.csv', index=False)

    # ReLU statistics over ALL observed images and spatial locations.
    # Per-unit inactive counts require one boolean per channel/neuron, not a huge activation cache.
    relu_acc = {}
    handles = []
    def make_hook(name):
        def hook(_module, _inputs, output):
            t = output.detach()
            if not isinstance(t, torch.Tensor) or t.ndim < 2:
                return
            n = t.shape[0]
            c = t.shape[1]
            a = relu_acc.setdefault(name, dict(elements=0, zeros=0, abs_sum=0.,
                                                sum_sq=0., max_abs=0.,
                                                seen=torch.zeros(c, dtype=torch.bool),
                                                samples=0, active_per_image_sum=0.))
            # Evaluate per channel across batch and any spatial positions.
            channel_active = (t > 0).reshape(n, c, -1).any(dim=2)
            a['seen'] |= channel_active.any(dim=0).cpu()
            a['active_per_image_sum'] += channel_active.float().sum().item()
            a['samples'] += n
            a['elements'] += t.numel()
            a['zeros'] += (t == 0).sum().item()
            a['abs_sum'] += t.abs().sum().item()
            a['sum_sq'] += t.float().square().sum().item()
            a['max_abs'] = max(a['max_abs'], t.abs().max().item())
        return hook
    for name, layer in m.named_modules():
        if isinstance(layer, nn.ReLU):
            handles.append(layer.register_forward_hook(make_hook(name)))

    confusion = None
    correct = total = 0
    loss_sum = 0.
    prediction_counts = Counter()
    target_counts = Counter()
    try:
        with torch.inference_mode():
            for bi, (x, y) in enumerate(val_loader):
                if max_val_batches is not None and bi >= max_val_batches:
                    break
                x, y = x.to(device), y.to(device)
                logits = m(x)
                if isinstance(logits, tuple):
                    logits = logits[0]
                pred = logits.argmax(dim=1)
                total += y.numel()
                correct += (pred == y).sum().item()
                loss_sum += criterion(logits, y).item() * y.numel()
                yt, yp = y.cpu().tolist(), pred.cpu().tolist()
                target_counts.update(yt)
                prediction_counts.update(yp)
                if confusion is None:
                    confusion = torch.zeros((logits.shape[1], logits.shape[1]), dtype=torch.int64)
                yy = torch.tensor(yt)
                pp = torch.tensor(yp)
                confusion += torch.bincount(yy * logits.shape[1] + pp,
                                            minlength=logits.shape[1]**2).reshape(logits.shape[1], -1)
    finally:
        for h in handles:
            h.remove()

    relu_rows = []
    for name, a in relu_acc.items():
        dead = int((~a['seen']).sum().item())
        relu_rows.append(dict(layer=name, units=a['seen'].numel(),
                              dead_on_sample=dead,
                              dead_unit_pct=100 * dead / a['seen'].numel(),
                              zero_pct=100 * a['zeros'] / a['elements'],
                              mean_abs=a['abs_sum'] / a['elements'],
                              rms=(a['sum_sq'] / a['elements'])**0.5,
                              max_abs=a['max_abs'],
                              mean_active_units_per_image=a['active_per_image_sum']/a['samples'],
                              observed_images=a['samples']))
    relus = pd.DataFrame(relu_rows)
    relus.to_csv(out / 'relu_activity.csv', index=False)
    if confusion is not None:
        pd.DataFrame(confusion.numpy()).to_csv(out / 'confusion_matrix.csv', index=False, header=False)
    per_class = []
    if confusion is not None:
        for i in range(confusion.shape[0]):
            support = confusion[i].sum().item()
            predicted = confusion[:, i].sum().item()
            tp = confusion[i, i].item()
            per_class.append(dict(class_id=i,
                                  class_name=class_names[i] if class_names is not None else str(i),
                                  support=support, predicted=predicted,
                                  accuracy_pct=100*tp/support if support else np.nan,
                                  precision=tp/predicted if predicted else 0.))
    class_df = pd.DataFrame(per_class)
    class_df.to_csv(out / 'per_class.csv', index=False)

    # Independent gradients on a model copy: eval() ensures BN running stats remain unchanged.
    # Use train_loader for representative augmentation, otherwise use val_loader.
    # No optimizer.step() is ever called.
    grad_rows = []
    grad_loader = train_loader if train_loader is not None else val_loader
    grads = {}
    for bi, (x, y) in enumerate(grad_loader):
        if bi >= max_grad_batches:
            break
        m.zero_grad(set_to_none=True)
        x, y = x.to(device), y.to(device)
        with torch.enable_grad():
            logits = m(x)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, y)
            loss.backward()
        for name, p in m.named_parameters():
            if p.grad is None:
                continue
            g = p.grad.detach().float()
            d = grads.setdefault(name, dict(norms=[], zero_pcts=[], max_abs=[], finite_pcts=[]))
            d['norms'].append(g.norm().item())
            d['zero_pcts'].append(100 * (g == 0).float().mean().item())
            d['max_abs'].append(g.abs().max().item())
            d['finite_pcts'].append(100 * torch.isfinite(g).float().mean().item())
    for name, d in grads.items():
        p = dict(m.named_parameters())[name]
        weight_norm = p.detach().float().norm().item()
        mean_grad_norm = float(np.mean(d['norms']))
        grad_rows.append(dict(parameter=name, grad_norm_mean=mean_grad_norm,
                              grad_norm_min=min(d['norms']), grad_norm_max=max(d['norms']),
                              grad_zero_pct_mean=float(np.mean(d['zero_pcts'])),
                              grad_max_abs=max(d['max_abs']),
                              grad_finite_pct_min=min(d['finite_pcts']),
                              weight_norm=weight_norm,
                              grad_to_weight_ratio=mean_grad_norm/(weight_norm+1e-12),
                              batches=len(d['norms'])))
    gradients = pd.DataFrame(grad_rows)
    gradients.to_csv(out / 'gradients.csv', index=False)

    # Plots, deliberately one chart per figure.
    if not relus.empty:
        for column, filename, xlabel in [
            ('zero_pct', 'relu_zero_pct.png', 'Zero activations (%)'),
            ('dead_unit_pct', 'relu_dead_unit_pct.png', 'Inactive units in observed data (%)'),
            ('mean_abs', 'relu_mean_abs.png', 'Mean absolute activation')]:
            fig, ax = plt.subplots(figsize=(10, max(4, .48*len(relus))))
            ax.barh(relus['layer'], relus[column])
            ax.invert_yaxis()
            ax.set_xlabel(xlabel)
            ax.set_title(column.replace('_', ' ').title())
            fig.tight_layout()
            fig.savefig(out / filename, dpi=160)
            plt.close(fig)
    if not gradients.empty:
        g = gradients[gradients['parameter'].str.endswith('.weight')]
        fig, ax = plt.subplots(figsize=(10, max(5, .32*len(g))))
        ax.barh(g['parameter'], g['grad_norm_mean'])
        ax.invert_yaxis()
        ax.set_xlabel('Mean gradient L2 norm')
        ax.set_title('Gradient flow by weight layer')
        fig.tight_layout()
        fig.savefig(out / 'gradient_norms.png', dpi=160)
        plt.close(fig)

    summary = dict(observed_validation_images=total,
                   accuracy_pct=100*correct/total if total else None,
                   loss=loss_sum/total if total else None,
                   predicted_classes=len(prediction_counts),
                   target_classes=len(target_counts),
                   dead_relu_units_total=int(relus['dead_on_sample'].sum()) if not relus.empty else 0,
                   gradient_parameters_checked=len(gradients),
                   gradient_nonfinite_parameters=int((gradients['grad_finite_pct_min'] < 100).sum())
                   if not gradients.empty else None,
                   output_dir=str(out.resolve()))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print('\n=== SUMMARY ===')
    print(json.dumps(summary, indent=2))
    print('\n=== RELU ACTIVITY ===')
    print(relus.to_string(index=False, columns=['layer','zero_pct','dead_on_sample','units','mean_abs']))
    print('\n=== GRADIENT FLOW ===')
    if not gradients.empty:
        print(gradients[['parameter','grad_norm_mean','grad_to_weight_ratio','grad_zero_pct_mean']].to_string(index=False))
    print('\n=== BATCHNORM ===')
    if not bn.empty:
        print(bn[['layer','running_var_min','near_zero_var_channels','gamma_mean','beta_mean']].to_string(index=False))
    print('\nSaved CSV, JSON and PNG reports to:', out.resolve())
    return dict(summary=summary, relu=relus, gradients=gradients, weights=weights,
                batchnorm=bn, per_class=class_df, confusion=confusion)
