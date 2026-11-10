import copy
import time

import numpy as np
import torch
import torch.nn as nn


# BP vs EP time per training step and GPU memory, measured on copies of the model with random data.
# The model, its EP dropout-mask generator and the global random state are left untouched.
def compare_step_time(model, batch_size, num_classes, label_smoothing, device, steps_per_epoch, epochs,
                      repeats=12, warmup=3):
    cpu_rng = torch.get_rng_state()
    mps_rng = torch.mps.get_rng_state() if device.type == 'mps' else None
    sync = torch.mps.synchronize if device.type == 'mps' else (lambda: None)

    generator = torch.Generator().manual_seed(0)
    images = torch.randn(batch_size, 3, 64, 64, generator=generator).to(device)
    labels_a = torch.randint(num_classes, (batch_size,), generator=generator).to(device)
    labels_b = torch.randint(num_classes, (batch_size,), generator=generator).to(device)
    lam = torch.rand(batch_size, generator=generator).to(device)
    per_image_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction='none')

    def median_seconds(step):
        times = []
        for i in range(repeats + warmup):
            sync()
            start = time.time()
            step()
            sync()
            if i >= warmup:
                times.append(time.time() - start)
        return float(np.median(times))

    bp_model = copy.deepcopy(model).train()

    def bp_step():
        bp_model.zero_grad(set_to_none=True)
        logits = bp_model(images)
        (lam * per_image_loss(logits, labels_a) + (1 - lam) * per_image_loss(logits, labels_b)).mean().backward()

    ep_model = copy.deepcopy(model).train()
    ep_model.dropout_generator = torch.Generator().manual_seed(0)  # own dropout masks, so training's generator is untouched

    bp_seconds = median_seconds(bp_step)
    ep_seconds = median_seconds(lambda: ep_model.ep_gradients(images, labels_a, labels_b, lam))
    memory = f'{torch.mps.driver_allocated_memory() / 2 ** 30:.2f} GB' if device.type == 'mps' else 'n/a'
    print(f'BP step {bp_seconds * 1000:.0f} ms | EP step {ep_seconds * 1000:.0f} ms ({ep_seconds / bp_seconds:.1f}x BP)'
          f' | MPS memory {memory}')
    print(f'EP training per epoch ~{ep_seconds * steps_per_epoch / 60:.0f} min, '
          f'{epochs} epochs ~{ep_seconds * steps_per_epoch * epochs / 3600:.0f} h (+ evaluation)')

    del bp_model, ep_model
    if device.type == 'mps':
        torch.mps.empty_cache()
    torch.set_rng_state(cpu_rng)
    if mps_rng is not None:
        torch.mps.set_rng_state(mps_rng)
    return dict(bp_seconds=bp_seconds, ep_seconds=ep_seconds, memory=memory)
