import numpy as np
import torch
from matplotlib import pyplot as plt
from src.dataset.tiny_imagenet_dataset_450train_100val import means, stds

def show_image(images, labels, num_show, class_name, name="", has_norm=True):
    MEAN_T = torch.tensor(means).view(3, 1, 1)
    STD_T = torch.tensor(stds).view(3, 1, 1)
    plt.figure(figsize=(16, 10))
    for i in range(min(num_show, len(images))):
        image = images[i].cpu()
        if has_norm:
            image = image * STD_T + MEAN_T
        image = image.permute(1, 2, 0).numpy()
        plt.subplot(4, 5, i + 1)
        plt.imshow(image)
        plt.title(class_name[int(labels[i])], fontsize=9)
        plt.axis("off")
    plt.suptitle(name, fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def show_mixed_images(images, labels_a, labels_b, lam, class_names, num_show, name="Mixed Images", has_norm=True):
    MEAN_T = torch.tensor(means).view(3, 1, 1)
    STD_T = torch.tensor(stds).view(3, 1, 1)
    images = images.detach().cpu()
    labels_a = labels_a.detach().cpu()
    labels_b = labels_b.detach().cpu()
    plt.figure(figsize=(15, 4))
    for i in range(min(num_show, len(images))):
        image = images[i]
        if has_norm:
            image = image * STD_T + MEAN_T
        image = image.clamp(0, 1).permute(1, 2, 0).numpy()
        plt.subplot(1, num_show, i + 1)
        plt.imshow(image)
        plt.title(f"{class_names[labels_a[i].item()]}\n+\n{class_names[labels_b[i].item()]}\nλ={lam:.2f}", fontsize=9)
        plt.axis("off")
    plt.suptitle(name, fontsize=14)
    plt.tight_layout()
    plt.show()


def apply_mixup(images, labels, alpha=0.2):
    batch_size = images.size(0)
    perm = torch.randperm(batch_size, device='cpu').to('cpu')
    lam = float(np.random.beta(alpha, alpha))
    mixed_images = lam * images + (1.0 - lam) * images[perm]
    labels_a = labels
    labels_b = labels[perm]
    return (mixed_images, labels_a, labels_b, lam, 'MixUp')

def apply_cutmix(images, labels, alpha=1.0):
    batch_size, _, height, width = images.shape
    perm = torch.randperm(batch_size, device='cpu').to('cpu')
    lam = float(np.random.beta(alpha, alpha))
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = int(np.random.randint(0, width))
    cy = int(np.random.randint(0, height))
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, height)
    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]
    patch_area = (x2 - x1) * (y2 - y1)
    lam_adjusted = 1.0 - patch_area / float(width * height)
    labels_a = labels
    labels_b = labels[perm]
    return (mixed_images, labels_a, labels_b, lam_adjusted, 'CutMix')

def apply_mixup_or_cutmix(images, labels, mixup_prob=0.5, mixup_alpha=1.0, cutmix_prob=0.5):
    if np.random.random() < mixup_prob:
        return apply_mixup(images, labels, alpha=mixup_alpha)
    return apply_cutmix(images, labels, alpha=cutmix_prob)