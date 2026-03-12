import matplotlib.pyplot as plt
import torch

from tennisvision.core.data import IMAGENET_MEAN, IMAGENET_STD


def plot_image(image):  # plt needs HWC
    plt.imshow(image.permute(1, 2, 0))
    plt.axis("off")


def show_batch(images, idx_to_class, labels, n=12, ncols=4, denormalize=False):
    images = images[:n]
    labels = labels[:n]

    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()

    for i in range(n):
        img = images[i].permute(1, 2, 0)  #  # CHW -> HWC

        if denormalize:
            mean = torch.tensor(IMAGENET_MEAN).view(1, 1, 3)
            std = torch.tensor(IMAGENET_STD).view(1, 1, 3)
            img = (img * std + mean).clamp(0, 1)

        axes[i].imshow(img)
        axes[i].set_title(idx_to_class[int(labels[i])])
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()
