import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt
import proplot as pplt
from . import utils


def plot_profiles(ax, image, log=False, scale=0.15, **plot_kws):
    plot_kws.setdefault('color', 'white')
    xx = np.arange(image.shape[1])
    yy = np.arange(image.shape[0])
    profs = [np.sum(image, axis=i) for i in (0, 1)]
    for i, prof in enumerate(profs):
        prof = prof / np.sum(prof)
        if log:
            prof = np.log10(prof)
        prof = prof / np.max(prof)
        profs[i] = prof      
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot(xx, image.shape[1] * scale * profs[0], **plot_kws)
    ax.plot(image.shape[0] * scale * profs[1], yy, **plot_kws)
    return ax


def plot_image(image, ax=None, log=False, prof=False, prof_kws=None, **plot_kws):
    """2D density plot with overlayed profiles."""
    if ax is None:
        fig, ax = pplt.subplots()
    if prof_kws is None:
        prof_kws = dict()
        prof_kws.setdefault('color', 'white')
        prof_kws.setdefault('scale', 0.15)
    xx = np.arange(image.shape[1])
    yy = np.arange(image.shape[0])
    norm = None
    if log:
        norm = 'log'
        image = np.ma.masked_less_equal(image, 0)
    ax.pcolormesh(xx, yy, image, norm=norm, **plot_kws)
    if prof:
        plot_profiles(ax, image, log=log, **prof_kws)
    return ax


def linear_fit(x, y):
    def fit(x, slope, intercept):
        return slope * x + intercept
    
    popt, pcov = opt.curve_fit(fit, x, y)
    slope, intercept = popt
    yfit = fit(x, *popt)
    return yfit, slope, intercept


def corner(
    image, 
    labels=None, 
    diag_kind='line', 
    log=False, 
    fig_kws=None, 
    diag_kws=None, 
    **plot_kws
):
    n = image.ndim
    if labels is None:
        labels = n * ['']
    if fig_kws is None:
        fig_kws = dict()
    fig_kws.setdefault('figwidth', 1.5 * n)
    fig_kws.setdefault('aligny', True)
    if diag_kws is None:
        diag_kws = dict()
    diag_kws.setdefault('color', 'black')
    plot_kws.setdefault('ec', 'None')
    if log:
        plot_kws['norm'] = 'log'
        
    fig, axes = pplt.subplots(
        nrows=n, ncols=n, sharex=1, sharey=1, 
        spanx=False, spany=False, **fig_kws
    )
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
            elif i == j:
                h = utils.project(image, j)
                if diag_kind == 'line':
                    ax.plot(h, **diag_kws)
                elif diag_kind == 'bar':
                    ax.bar(h, **diag_kws)
                elif diag_kind == 'step':
                    ax.step(h, **diag_kws)
            else:
                H = utils.project(image, (j, i))
                if log:
                    H = np.ma.masked_less_equal(H, 0)
                ax.pcolormesh(H.T, **plot_kws)
    for ax, label in zip(axes[-1, :], labels):
        ax.format(xlabel=label)
    for ax, label in zip(axes[1:, 0], labels[1:]):
        ax.format(ylabel=label)
    for i in range(n):
        axes[:-1, i].format(xticklabels=[])
        if i > 0:
            axes[i, 1:].format(yticklabels=[])
    return axes