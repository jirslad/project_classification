# ------------------------------------------------------------------------------------------------------------
# https://pytorch.org/vision/main/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T


plt.rcParams["savefig.bbox"] = 'tight'
orig_img = Image.open('astronaut.jpg')
# if you change the seed, make sure that the randomly-applied transforms
# properly show that the image can be both transformed and *not* transformed!
torch.manual_seed(22)


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
# ------------------------------------------------------------------------------------------------------------

img_size = 224

transform = T.Compose([
            T.Resize(img_size+32, interpolation=T.InterpolationMode.BICUBIC),
            T.TrivialAugmentWide(num_magnitude_bins=31), # 31
            T.CenterCrop(img_size),
            # T.RandomHorizontalFlip(p=0.5),
            # T.ToTensor(),
            # T.Normalize(mean=[0.485, 0.456, 0.406],
            #             std=[0.229, 0.224, 0.225])
        ])
imgs = [[transform(orig_img) for _ in range(11)] for _ in range(5)]
plot(imgs)

transform = T.Compose([
            T.Resize(img_size+32, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomPerspective(distortion_scale=0.1, p=0.9, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.TrivialAugmentWide(num_magnitude_bins=31), # 31
            T.CenterCrop(img_size),
            T.RandomHorizontalFlip(p=0.5),
            # T.ToTensor(),
            # T.Normalize(mean=[0.485, 0.456, 0.406],
            #             std=[0.229, 0.224, 0.225])
        ])
imgs = [[transform(orig_img) for _ in range(11)] for _ in range(5)]
plot(imgs)


plt.show()

aa = 1