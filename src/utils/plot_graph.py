import numpy as np
import torch
from matplotlib import pyplot as plt

from src.utils.utils import means, stds


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


# Original images (top row) and training versions with the pasted share
def show_mix_samples(train_dataset, train_subset, train_mix_dataset, num_images=8, num_versions=3):
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


# Accuracy, loss, LR and clean-train minus val gap per epoch
def plot_training_curves(history, train_label='Train'):
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
