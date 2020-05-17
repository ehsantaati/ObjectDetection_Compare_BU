import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from sys import platform as _platform


def one_epoch(net, loss, dl, opt=None, metric=None):
    if opt:
        net.train()  # only affects some layers
    else:
        net.eval()
        rq_stored = []
        for p in net.parameters():
            rq_stored.append(p.requires_grad)
            p.requires_grad = False

    L, M = [], []
    dl_it = iter(dl)
    for xb, yb in tqdm(dl_it, leave=False):
        xb = xb.cuda()
        if not isinstance(yb, list): yb = [yb]  # this is new(!)
        yb = [yb_.cuda() for yb_ in yb]
        y_ = net(xb)
        l = loss(y_, yb)
        if opt:
            opt.zero_grad()
            l.backward()
            opt.step()
        L.append(l.detach().cpu().numpy())
        if metric: M.append(metric(y_, yb).cpu().numpy())

    if not opt:
        for p, rq in zip(net.parameters(), rq_stored): p.requires_grad = rq

    return L, M


def fit(net, tr_dl, val_dl, loss=nn.CrossEntropyLoss(), epochs=3, lr=3e-3, wd=1e-3, plot=True):
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    Ltr_hist, Lval_hist = [], []
    for epoch in trange(epochs):
        Ltr, _ = one_epoch(net, loss, tr_dl, opt)
        Lval, Aval = one_epoch(net, loss, val_dl, None, accuracy)
        Ltr_hist.append(np.mean(Ltr))
        Lval_hist.append(np.mean(Lval))
        print(
            f'epoch: {epoch}\ttraining loss: {np.mean(Ltr):0.4f}\tvalidation loss: {np.mean(Lval):0.4f}\tvalidation accuracy: {np.mean(Aval):0.2f}')

    # plot the losses
    if plot:
        _, ax = plt.subplots(1, 1, figsize=(16, 4))
        ax.plot(1 + np.arange(len(Ltr_hist)), Ltr_hist)
        ax.plot(1 + np.arange(len(Lval_hist)), Lval_hist)
        ax.grid('on')
        ax.set_xlim(left=1, right=len(Ltr_hist))
        ax.legend(['training loss', 'validation loss']);

    return Ltr_hist, Lval_hist


def _freeze(md, fr=True):
    ch = list(md.children())
    for c in ch: _freeze(c, fr)
    if not ch and not isinstance(md, torch.nn.modules.batchnorm.BatchNorm2d):
        for p in md.parameters():
            p.requires_grad = not fr


def freeze_to(md, ix=-1):
    ch_all = list(md.children())
    for ch in ch_all[:ix]: _freeze(ch, True)


def unfreeze_to(md, ix=-1):
    ch_all = list(md.children())
    for ch in ch_all[:ix]: _freeze(ch, False)


def parallel(func, inp, workers=4):
    pool = ProcessPoolExecutor if _platform.startswith('linux') else ThreadPoolExecutor
    with pool(max_workers=workers) as ex:
        futures = [ex.submit(func, i) for i in inp]
        results = [r for r in tqdm(as_completed(futures), total=len(inp), leave=True)]  # results in 'random order'
    res2ix = {v: k for k, v in enumerate(results)}
    out = [results[res2ix[f]].result() for f in futures]
    return out


def accuracy(inp, tar):
    inp_cls = inp[:, :len(c2i)]
    tar_cls = tar[0].squeeze()
    return (inp_cls.max(dim=1)[1] == tar_cls).float().mean()


def myloss(inp, tar, reduction='mean'):
    inp_cls = inp[:, :len(c2i)]
    inp_reg = inp[:, len(c2i):]
    tar_cls, tar_reg = tar

    loss_cls = F.cross_entropy(inp_cls, tar_cls.squeeze(), reduction=reduction)
    loss_reg = F.mse_loss(torch.sigmoid(inp_reg), tar_reg, reduction=reduction)
    if reduction == 'none': loss_reg = loss_reg.mean(dim=-1)

    return loss_cls + 10 * loss_reg
