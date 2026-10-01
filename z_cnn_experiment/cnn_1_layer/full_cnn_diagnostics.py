"""Comprehensive diagnostics for ReLU / LeakyReLU PyTorch image classifiers.

Designed for StandardCNN with level1..level4, flatten and classifier.

Checks:
- Full validation accuracy and loss
- Predicted/target classes
- ReLU / LeakyReLU activation statistics
- Positive / negative / exact-zero activations
- Observed inactive channels/neurons
- Gradient flow
- Weight statistics
- BatchNorm statistics
- Per-class performance
- Confusion matrix

IMPORTANT:
For LeakyReLU, negative outputs are NOT considered inactive.
A unit is considered observed-inactive only if its output is exactly zero
for every observed validation example/spatial position.

The supplied model is never modified:
diagnostics run on a deep copy.
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


# ============================================================
# BASIC TENSOR STATISTICS
# ============================================================

def _stats(t):
    t = t.detach().float().cpu().reshape(-1)

    if not t.numel():
        return dict(
            mean=np.nan,
            std=np.nan,
            min=np.nan,
            max=np.nan,
            abs_mean=np.nan,
            l2=np.nan,
            zero_pct=np.nan,
            finite_pct=np.nan,
        )

    finite = torch.isfinite(t)
    v = t[finite]

    return dict(
        mean=v.mean().item() if v.numel() else np.nan,
        std=v.std(unbiased=False).item() if v.numel() else np.nan,
        min=v.min().item() if v.numel() else np.nan,
        max=v.max().item() if v.numel() else np.nan,
        abs_mean=v.abs().mean().item() if v.numel() else np.nan,
        l2=v.norm().item() if v.numel() else np.nan,
        zero_pct=100 * (t == 0).float().mean().item(),
        finite_pct=100 * finite.float().mean().item(),
    )


# ============================================================
# MAIN DIAGNOSTIC FUNCTION
# ============================================================

def analyze_cnn_full(
    model,
    val_loader,
    device,
    train_loader=None,
    criterion=None,
    output_dir="full_cnn_diagnostics_leakyrelu",
    max_val_batches=None,
    max_grad_batches=3,
    class_names=None,
):

    """
    Full diagnostic analysis for ReLU / LeakyReLU CNN.

    Args:
        model:
            Model with checkpoint weights already loaded.

        val_loader:
            Clean validation loader.
            Recommended:
                shuffle=False
                num_workers=0

        device:
            "cuda", "mps", or "cpu"

        train_loader:
            Optional augmented training loader used only
            for representative gradient measurements.

        criterion:
            Loss used for validation/gradient diagnostics.
            Defaults to CrossEntropyLoss().

        output_dir:
            Folder where CSV / JSON / PNG reports are saved.

        max_val_batches:
            None = use full validation dataset.

        max_grad_batches:
            Number of independent batches used for
            gradient measurements.

        class_names:
            Optional list of class names.
    """

    device = torch.device(device)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Deep copy so original model/checkpoint is not modified
    # --------------------------------------------------------

    base_model = (
        model.module
        if isinstance(model, nn.DataParallel)
        else model
    )

    m = copy.deepcopy(base_model).to(device)

    m.eval()

    criterion = criterion or nn.CrossEntropyLoss()


    # ========================================================
    # 1. WEIGHT / BIAS STATISTICS
    # ========================================================

    weight_rows = []

    for name, p in m.named_parameters():

        weight_rows.append(
            dict(
                parameter=name,
                shape=str(tuple(p.shape)),
                count=p.numel(),
                requires_grad=p.requires_grad,
                **_stats(p),
            )
        )

    weights = pd.DataFrame(weight_rows)

    weights.to_csv(
        out / "weights_biases.csv",
        index=False
    )


    # ========================================================
    # 2. BATCHNORM STATISTICS
    # ========================================================

    bn_rows = []

    for name, layer in m.named_modules():

        if isinstance(
            layer,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
            ),
        ):

            row = dict(
                layer=name,
                features=layer.num_features,
                eps=layer.eps,
                momentum=layer.momentum,
                batches_tracked=(
                    int(layer.num_batches_tracked.item())
                    if layer.num_batches_tracked is not None
                    else None
                ),
            )

            for label, t in [
                ("running_mean", layer.running_mean),
                ("running_var", layer.running_var),
                ("gamma", layer.weight),
                ("beta", layer.bias),
            ]:

                if t is not None:

                    for k, v in _stats(t).items():
                        row[f"{label}_{k}"] = v

            if layer.running_var is not None:

                row["near_zero_var_channels"] = int(
                    (layer.running_var < 1e-5)
                    .sum()
                    .item()
                )

            bn_rows.append(row)

    bn = pd.DataFrame(bn_rows)

    bn.to_csv(
        out / "batchnorm.csv",
        index=False
    )


    # ========================================================
    # 3. RELU / LEAKY RELU ACTIVATION ANALYSIS
    # ========================================================

    activation_acc = {}

    handles = []


    def make_hook(name, activation_type):

        def hook(_module, _inputs, output):

            if not isinstance(output, torch.Tensor):
                return

            t = output.detach()

            if t.ndim < 2:
                return

            n = t.shape[0]
            c = t.shape[1]

            a = activation_acc.setdefault(
                name,
                dict(
                    activation_type=activation_type,
                    elements=0,
                    zeros=0,
                    negatives=0,
                    positives=0,
                    abs_sum=0.0,
                    sum_sq=0.0,
                    max_abs=0.0,

                    # Tracks whether each channel/neuron
                    # produced any non-zero output.
                    seen_nonzero=torch.zeros(
                        c,
                        dtype=torch.bool
                    ),

                    samples=0,

                    active_per_image_sum=0.0,
                )
            )


            # ------------------------------------------------
            # IMPORTANT FOR LEAKY RELU
            #
            # Negative output is still ACTIVE.
            #
            # ReLU:
            #   negative preactivation -> 0
            #
            # LeakyReLU:
            #   negative preactivation -> negative_slope*x
            #
            # Therefore:
            #
            # t != 0
            #
            # is used for observed activity.
            # ------------------------------------------------

            channel_active = (
                (t != 0)
                .reshape(n, c, -1)
                .any(dim=2)
            )


            a["seen_nonzero"] |= (
                channel_active
                .any(dim=0)
                .cpu()
            )


            a["active_per_image_sum"] += (
                channel_active
                .float()
                .sum()
                .item()
            )

            a["samples"] += n


            # ------------------------------------------------
            # Activation statistics
            # ------------------------------------------------

            a["elements"] += t.numel()

            a["zeros"] += (
                (t == 0)
                .sum()
                .item()
            )

            a["negatives"] += (
                (t < 0)
                .sum()
                .item()
            )

            a["positives"] += (
                (t > 0)
                .sum()
                .item()
            )

            a["abs_sum"] += (
                t.abs()
                .sum()
                .item()
            )

            a["sum_sq"] += (
                t.float()
                .square()
                .sum()
                .item()
            )

            a["max_abs"] = max(
                a["max_abs"],
                t.abs().max().item()
            )

        return hook


    # --------------------------------------------------------
    # Register hooks for BOTH ReLU and LeakyReLU
    # --------------------------------------------------------

    for name, layer in m.named_modules():

        if isinstance(
            layer,
            (
                nn.ReLU,
                nn.LeakyReLU,
            ),
        ):

            activation_type = type(layer).__name__

            handles.append(
                layer.register_forward_hook(
                    make_hook(
                        name,
                        activation_type
                    )
                )
            )


    # ========================================================
    # 4. FULL VALIDATION ANALYSIS
    # ========================================================

    confusion = None

    correct = 0
    total = 0

    loss_sum = 0.0

    prediction_counts = Counter()
    target_counts = Counter()


    try:

        with torch.inference_mode():

            for bi, (x, y) in enumerate(val_loader):

                if (
                    max_val_batches is not None
                    and bi >= max_val_batches
                ):
                    break

                x = x.to(device)
                y = y.to(device)

                logits = m(x)

                if isinstance(logits, tuple):
                    logits = logits[0]

                pred = logits.argmax(dim=1)

                total += y.numel()

                correct += (
                    (pred == y)
                    .sum()
                    .item()
                )

                loss_sum += (
                    criterion(logits, y).item()
                    * y.numel()
                )


                yt = y.cpu().tolist()
                yp = pred.cpu().tolist()

                target_counts.update(yt)
                prediction_counts.update(yp)


                # --------------------------------------------
                # Confusion matrix
                # --------------------------------------------

                if confusion is None:

                    confusion = torch.zeros(
                        (
                            logits.shape[1],
                            logits.shape[1]
                        ),
                        dtype=torch.int64
                    )


                yy = torch.tensor(yt)
                pp = torch.tensor(yp)

                confusion += torch.bincount(
                    yy * logits.shape[1] + pp,
                    minlength=logits.shape[1] ** 2
                ).reshape(
                    logits.shape[1],
                    -1
                )


    finally:

        for h in handles:
            h.remove()


    # ========================================================
    # 5. ACTIVATION TABLE
    # ========================================================

    activation_rows = []

    for name, a in activation_acc.items():

        observed_inactive = int(
            (~a["seen_nonzero"])
            .sum()
            .item()
        )

        units = (
            a["seen_nonzero"]
            .numel()
        )

        elements = a["elements"]

        activation_rows.append(
            dict(
                layer=name,

                activation_type=(
                    a["activation_type"]
                ),

                units=units,

                observed_inactive_units=(
                    observed_inactive
                ),

                observed_inactive_pct=(
                    100
                    * observed_inactive
                    / units
                ),

                zero_pct=(
                    100
                    * a["zeros"]
                    / elements
                ),

                negative_pct=(
                    100
                    * a["negatives"]
                    / elements
                ),

                positive_pct=(
                    100
                    * a["positives"]
                    / elements
                ),

                mean_abs=(
                    a["abs_sum"]
                    / elements
                ),

                rms=(
                    a["sum_sq"]
                    / elements
                ) ** 0.5,

                max_abs=a["max_abs"],

                mean_active_units_per_image=(
                    a["active_per_image_sum"]
                    / a["samples"]
                ),

                observed_images=(
                    a["samples"]
                ),
            )
        )


    activations = pd.DataFrame(
        activation_rows
    )


    activations.to_csv(
        out / "activation_activity.csv",
        index=False
    )


    # ========================================================
    # 6. CONFUSION MATRIX
    # ========================================================

    if confusion is not None:

        pd.DataFrame(
            confusion.numpy()
        ).to_csv(
            out / "confusion_matrix.csv",
            index=False,
            header=False
        )


    # ========================================================
    # 7. PER-CLASS RESULTS
    # ========================================================

    per_class = []

    if confusion is not None:

        for i in range(
            confusion.shape[0]
        ):

            support = (
                confusion[i]
                .sum()
                .item()
            )

            predicted = (
                confusion[:, i]
                .sum()
                .item()
            )

            tp = (
                confusion[i, i]
                .item()
            )


            per_class.append(
                dict(
                    class_id=i,

                    class_name=(
                        class_names[i]
                        if class_names is not None
                        else str(i)
                    ),

                    support=support,

                    predicted=predicted,

                    accuracy_pct=(
                        100 * tp / support
                        if support
                        else np.nan
                    ),

                    precision=(
                        tp / predicted
                        if predicted
                        else 0
                    ),
                )
            )


    class_df = pd.DataFrame(
        per_class
    )

    class_df.to_csv(
        out / "per_class.csv",
        index=False
    )


    # ========================================================
    # 8. GRADIENT FLOW
    # ========================================================

    # Uses a COPY of the model.
    #
    # eval() keeps BatchNorm running statistics unchanged.
    #
    # No optimizer.step() is called.
    #
    # Therefore checkpoint is not modified.

    grad_rows = []

    grad_loader = (
        train_loader
        if train_loader is not None
        else val_loader
    )

    grads = {}


    for bi, (x, y) in enumerate(
        grad_loader
    ):

        if bi >= max_grad_batches:
            break


        m.zero_grad(
            set_to_none=True
        )


        x = x.to(device)
        y = y.to(device)


        with torch.enable_grad():

            logits = m(x)

            if isinstance(
                logits,
                tuple
            ):
                logits = logits[0]

            loss = criterion(
                logits,
                y
            )

            loss.backward()


        for name, p in m.named_parameters():

            if p.grad is None:
                continue


            g = (
                p.grad
                .detach()
                .float()
            )


            d = grads.setdefault(
                name,
                dict(
                    norms=[],
                    zero_pcts=[],
                    max_abs=[],
                    finite_pcts=[],
                )
            )


            d["norms"].append(
                g.norm().item()
            )


            d["zero_pcts"].append(
                100
                * (g == 0)
                .float()
                .mean()
                .item()
            )


            d["max_abs"].append(
                g.abs()
                .max()
                .item()
            )


            d["finite_pcts"].append(
                100
                * torch.isfinite(g)
                .float()
                .mean()
                .item()
            )


    # --------------------------------------------------------
    # Convert gradient measurements to table
    # --------------------------------------------------------

    named_parameters = dict(
        m.named_parameters()
    )


    for name, d in grads.items():

        p = named_parameters[name]

        weight_norm = (
            p.detach()
            .float()
            .norm()
            .item()
        )


        mean_grad_norm = float(
            np.mean(
                d["norms"]
            )
        )


        grad_rows.append(
            dict(
                parameter=name,

                grad_norm_mean=(
                    mean_grad_norm
                ),

                grad_norm_min=(
                    min(d["norms"])
                ),

                grad_norm_max=(
                    max(d["norms"])
                ),

                grad_zero_pct_mean=float(
                    np.mean(
                        d["zero_pcts"]
                    )
                ),

                grad_max_abs=(
                    max(d["max_abs"])
                ),

                grad_finite_pct_min=(
                    min(
                        d["finite_pcts"]
                    )
                ),

                weight_norm=(
                    weight_norm
                ),

                grad_to_weight_ratio=(
                    mean_grad_norm
                    / (
                        weight_norm
                        + 1e-12
                    )
                ),

                batches=len(
                    d["norms"]
                ),
            )
        )


    gradients = pd.DataFrame(
        grad_rows
    )


    gradients.to_csv(
        out / "gradients.csv",
        index=False
    )


    # ========================================================
    # 9. ACTIVATION PLOTS
    # ========================================================

    if not activations.empty:

        plot_columns = [

            (
                "zero_pct",
                "activation_zero_pct.png",
                "Exact-zero activations (%)"
            ),

            (
                "negative_pct",
                "activation_negative_pct.png",
                "Negative activations (%)"
            ),

            (
                "positive_pct",
                "activation_positive_pct.png",
                "Positive activations (%)"
            ),

            (
                "observed_inactive_pct",
                "activation_inactive_units.png",
                "Observed inactive units (%)"
            ),

            (
                "mean_abs",
                "activation_mean_abs.png",
                "Mean absolute activation"
            ),
        ]


        for (
            column,
            filename,
            xlabel
        ) in plot_columns:

            fig, ax = plt.subplots(
                figsize=(
                    10,
                    max(
                        4,
                        0.48
                        * len(activations)
                    )
                )
            )

            ax.barh(
                activations["layer"],
                activations[column]
            )

            ax.invert_yaxis()

            ax.set_xlabel(
                xlabel
            )

            ax.set_title(
                column
                .replace("_", " ")
                .title()
            )

            fig.tight_layout()

            fig.savefig(
                out / filename,
                dpi=160
            )

            plt.close(fig)


    # ========================================================
    # 10. GRADIENT PLOT
    # ========================================================

    if not gradients.empty:

        g = gradients[
            gradients["parameter"]
            .str.endswith(".weight")
        ]


        fig, ax = plt.subplots(
            figsize=(
                10,
                max(
                    5,
                    0.32 * len(g)
                )
            )
        )


        ax.barh(
            g["parameter"],
            g["grad_norm_mean"]
        )


        ax.invert_yaxis()

        ax.set_xlabel(
            "Mean gradient L2 norm"
        )

        ax.set_title(
            "Gradient Flow by Weight Layer"
        )


        fig.tight_layout()


        fig.savefig(
            out / "gradient_norms.png",
            dpi=160
        )


        plt.close(fig)


    # ========================================================
    # 11. SUMMARY
    # ========================================================

    if not activations.empty:

        total_inactive_units = int(
            activations[
                "observed_inactive_units"
            ].sum()
        )

    else:

        total_inactive_units = 0


    if not gradients.empty:

        gradient_nonfinite = int(
            (
                gradients[
                    "grad_finite_pct_min"
                ] < 100
            )
            .sum()
        )

    else:

        gradient_nonfinite = None


    summary = dict(

        observed_validation_images=(
            total
        ),

        accuracy_pct=(
            100 * correct / total
            if total
            else None
        ),

        loss=(
            loss_sum / total
            if total
            else None
        ),

        predicted_classes=(
            len(prediction_counts)
        ),

        target_classes=(
            len(target_counts)
        ),

        activation_layers_checked=(
            len(activations)
        ),

        observed_inactive_units_total=(
            total_inactive_units
        ),

        gradient_parameters_checked=(
            len(gradients)
        ),

        gradient_nonfinite_parameters=(
            gradient_nonfinite
        ),

        output_dir=str(
            out.resolve()
        ),
    )


    (
        out / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2
        )
    )


    # ========================================================
    # 12. PRINT RESULTS
    # ========================================================

    print(
        "\n=== SUMMARY ==="
    )

    print(
        json.dumps(
            summary,
            indent=2
        )
    )


    # --------------------------------------------------------
    # Activation table
    # --------------------------------------------------------

    print(
        "\n=== ACTIVATION ACTIVITY ==="
    )


    if not activations.empty:

        print(
            activations[
                [
                    "layer",
                    "activation_type",
                    "zero_pct",
                    "negative_pct",
                    "positive_pct",
                    "observed_inactive_units",
                    "units",
                    "mean_abs",
                ]
            ].to_string(
                index=False
            )
        )

    else:

        print(
            "No ReLU or LeakyReLU layers found."
        )


    # --------------------------------------------------------
    # Gradient table
    # --------------------------------------------------------

    print(
        "\n=== GRADIENT FLOW ==="
    )


    if not gradients.empty:

        print(
            gradients[
                [
                    "parameter",
                    "grad_norm_mean",
                    "grad_to_weight_ratio",
                    "grad_zero_pct_mean",
                ]
            ].to_string(
                index=False
            )
        )


    # --------------------------------------------------------
    # BatchNorm table
    # --------------------------------------------------------

    print(
        "\n=== BATCHNORM ==="
    )


    if not bn.empty:

        print(
            bn[
                [
                    "layer",
                    "running_var_min",
                    "near_zero_var_channels",
                    "gamma_mean",
                    "beta_mean",
                ]
            ].to_string(
                index=False
            )
        )


    print(
        "\nSaved CSV, JSON and PNG reports to:",
        out.resolve()
    )


    # ========================================================
    # RETURN EVERYTHING
    # ========================================================

    return dict(
        summary=summary,
        activations=activations,
        gradients=gradients,
        weights=weights,
        batchnorm=bn,
        per_class=class_df,
        confusion=confusion,
    )


# ============================================================
# EXAMPLE USAGE
# ============================================================

# IMPORTANT:
# First load your BEST LeakyReLU checkpoint into `model`.
#
# Example:
#
# checkpoint = torch.load(
#     "path/to/best_model.pt",
#     map_location=DEVICE
# )
#
# model.load_state_dict(checkpoint["model_state_dict"])
#
# If your checkpoint is directly a state_dict:
#
# model.load_state_dict(checkpoint)


# Run full diagnostic:
#
# results = analyze_cnn_full(
#     model=model,
#     val_loader=val_loader,
#     device=DEVICE,
#     train_loader=train_loader,
#     criterion=criterion,
#     output_dir="full_cnn_diagnostics_leakyrelu",
#     max_val_batches=None,      # Full 20,000 validation images
#     max_grad_batches=3,
#     class_names=None,
# )
#
#
# Access individual results:
#
# results["summary"]
# results["activations"]
# results["gradients"]
# results["batchnorm"]