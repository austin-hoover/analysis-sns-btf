"""Utility plotting functions.

Note: In this module, `image` is assumed to have ij indexing as opposed
to xy indexing. So `image[0]` corresponds to the x-axis and `image[1]` 
corresponds to the y axis. Thus, when `image` is passed to a plotting 
routine, we call `ax.pcolormesh(image.T)`.
"""
import numpy as np
from scipy import optimize as opt
from matplotlib import pyplot as plt
import proplot as pplt
from . import utils


def linear_fit(x, y):
    def fit(x, slope, intercept):
        return slope * x + intercept
    
    popt, pcov = opt.curve_fit(fit, x, y)
    slope, intercept = popt
    yfit = fit(x, *popt)
    return yfit, slope, intercept


def plot_image(image, ax=None, x=None, y=None, prof=False, prof_kws=None, **plot_kws):
    """Plot image with profiles overlayed."""
    if 'norm' in plot_kws and plot_kws['norm'] == 'log':
        image += np.min(image[image > 0])
        if 'colorbar' in plot_kws and plot_kws['colorbar']:
            if 'colorbar_kw' not in plot_kws:
                plot_kws['colorbar_kw'] = dict()
            plot_kws['colorbar_kw']['formatter'] = 'log'
    if x is None:
        x = np.arange(image.shape[0])
    if y is None:
        y = np.arange(image.shape[1])
    ax.pcolormesh(x, y, image.T, **plot_kws)
    if prof:
        if prof_kws is None:
            prof_kws = dict()
        prof_kws.setdefault('color', 'white')
        prof_kws.setdefault('scale', 0.2)
        prof_kws.setdefault('kind', 'line')
        scale = prof_kws.pop('scale')
        kind = prof_kws.pop('kind')
        fx = np.sum(image, axis=1)
        fy = np.sum(image, axis=0)
        fx = fx / fx.max()
        fy = fy / fy.max()
        x1 = x
        y1 = scale * image.shape[1] * fx / fx.max()
        x2 = image.shape[0] * scale * fy
        y2 = y
        for i, (x, y) in enumerate(zip([x1, x2], [y1, y2])):
            if kind == 'line':
                ax.plot(x, y, **prof_kws)
            elif kind == 'bar':
                if i == 0:
                    ax.bar(x, y, **prof_kws)
                else:
                    ax.barh(y, x, **prof_kws)
            elif kind == 'step':
                # Align steps with pcolormesh bins.
                x -= 0.5 * (x[1] - x[0])
                y -= 0.5 * (y[1] - y[0])
                ax.step(x, y, **prof_kws)
    return ax


def corner(
    image, 
    labels=None, 
    diag_kind='line', 
    fig_kws=None, 
    diag_kws=None, 
    prof=False,
    prof_kws=None,
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
                plot_image(H, ax=ax, prof=prof, prof_kws=prof_kws, **plot_kws)
    for ax, label in zip(axes[-1, :], labels):
        ax.format(xlabel=label)
    for ax, label in zip(axes[1:, 0], labels[1:]):
        ax.format(ylabel=label)
    for i in range(n):
        axes[:-1, i].format(xticklabels=[])
        if i > 0:
            axes[i, 1:].format(yticklabels=[])
    return axes