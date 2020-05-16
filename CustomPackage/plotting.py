import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects


def show_img(im, ax=None, figsize=(8, 8), title=None):
    if not ax: _, ax = plt.subplots(1, 1, figsize=figsize)
    if len(im.shape) == 2: im = np.tile(im[:, :, None], 3)
    ax.imshow(im);
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if title: ax.set_title(title)
    return ax


def show_imgs(ims, rows=1, figsize=(16, 8), title=[None]):
    title = title * len(ims) if len(title) == 1 else title
    _, ax = plt.subplots(rows, len(ims) // rows, figsize=figsize)
    [show_img(im, ax_, title=tit) for im, ax_, tit in zip(ims, ax.flatten(), title)]
    return ax


def draw_rect(ax, xy, w, h):
    patch = ax.add_patch(patches.Rectangle(xy, w, h, fill=False, edgecolor='yellow', lw=2))
    patch.set_path_effects([patheffects.Stroke(linewidth=4, foreground='black'), patheffects.Normal()])


def show_img_bbox(x, y, ax=None):
    h, w = x.shape[1:]
    ax = show_img(x.numpy().transpose(1, 2, 0), ax=ax, title=i2c[y[0]])
    draw_rect(ax, [y[1][0] * w, y[1][1] * h], y[1][2] * w, y[1][3] * h)
