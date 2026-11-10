import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from dataset.mixup_cutmix import MixUpAugumentation, CutMixAugumentation

means = [0.4802, 0.4481, 0.3975]
stds = [0.2764, 0.2689, 0.2816]


# means = (0.485, 0.456, 0.406)
# stds = (0.229, 0.224, 0.225)

def get_device():
    """CUDA if available, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def print_device_info(device):
    """Prints how many devices of the chosen type are available (CUDA, Apple MPS or CPU)."""
    if device.type == 'cuda':
        print(f'Number of CUDA devices: {torch.cuda.device_count()} ({torch.cuda.get_device_name(device)})')
    elif device.type == 'mps':
        print(f'Number of MPS devices: {torch.mps.device_count()}')
    else:
        print(f'Running on CPU ({os.cpu_count()} cores)')


def convert_to_rgb(image):
    return image.convert("RGB")


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

def apply_mixup_or_cutmix(images, labels, mixup_prob=0.5, mixup_alpha=0.2, cutmix_alpha=1.0):
    if np.random.random() < mixup_prob:
        return MixUpAugumentation(images, labels, alpha=mixup_alpha)
    return CutMixAugumentation(images, labels, alpha=cutmix_alpha)


def show_mix_samples(train_dataset, train_subset, train_mix_dataset, num_images=8, num_versions=3):
    """Sample check before training: the top row shows the original images, the rows below show `num_versions`
    real training versions of each (crop, flip, mixing, RandAugment), titled with the pasted label share (1 - lam).
    The torch random state is saved and restored, so training is not affected."""
    from PIL import Image
    state = torch.get_rng_state()
    mean, std = torch.tensor(means).view(3, 1, 1), torch.tensor(stds).view(3, 1, 1)
    fig, axs = plt.subplots(num_versions + 1, num_images, figsize=(13, 1.9 * (num_versions + 1)))
    for c in range(num_images):
        path = train_dataset.samples[train_subset.samples[c][0]][0]
        axs[0, c].imshow(Image.open(path).convert('RGB'), interpolation='nearest')
        for r in range(1, num_versions + 1):
            x, _, _, lam = train_mix_dataset[c]
            axs[r, c].imshow((x * std + mean).clamp(0, 1).permute(1, 2, 0).numpy(), interpolation='nearest')
            axs[r, c].set_title(f'{1 - lam.item():.2f}', fontsize=8)
    for ax in axs.flat:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    torch.set_rng_state(state)


def plot_training_curves(history, train_label='Train'):
    """Accuracy, loss, learning rate and clean-train minus clean-val gap per epoch from a training `history`.
    Clean train (and so the gap) is measured only every few epochs; those series are plotted on the epochs that
    have a value, so their lines connect instead of vanishing between the empty epochs."""
    epoch = np.asarray(history['epoch'], dtype=float)

    def measured(key):
        y = np.array([np.nan if v is None else v for v in history[key]], dtype=float)
        keep = ~np.isnan(y)
        return epoch[keep], y[keep]

    fig, axs = plt.subplots(2, 2, figsize=(13, 9))
    axs[0, 0].plot(epoch, history['train_acc'], label=train_label)
    axs[0, 0].plot(epoch, history['val_acc'], label='Clean val')
    axs[0, 0].plot(*measured('clean_train_acc'), label='Clean train (measured epochs)', marker='o', markersize=3)
    axs[0, 0].set_title('Accuracy (%)')
    axs[0, 0].legend()

    axs[0, 1].plot(epoch, history['train_loss'], label='Train')
    axs[0, 1].plot(epoch, history['val_loss'], label='Val')
    axs[0, 1].plot(*measured('clean_train_loss'), label='Clean train (measured epochs)', marker='o', markersize=3)
    axs[0, 1].set_title('Loss')
    axs[0, 1].legend()

    axs[1, 0].plot(epoch, history['lr'])
    axs[1, 0].set_title('Learning rate')

    axs[1, 1].plot(*measured('generalization_gap'), marker='o', markersize=3)
    axs[1, 1].set_title('Clean-train minus clean-val accuracy (pp)')

    for ax in axs.flat:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
