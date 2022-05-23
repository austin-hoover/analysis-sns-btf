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


def plot_image(image, x=None, y=None, ax=None, profx=False, profy=False, 
               prof_kws=None, frac_thresh=None, **plot_kws):
    """Plot image with profiles overlayed."""
    log = 'norm' in plot_kws and plot_kws['norm'] == 'log'
    if log:
        if 'colorbar' in plot_kws and plot_kws['colorbar']:
            if 'colorbar_kw' not in plot_kws:
                plot_kws['colorbar_kw'] = dict()
            plot_kws['colorbar_kw']['formatter'] = 'log'
    if x is None:
        x = np.arange(image.shape[0])
    if y is None:
        y = np.arange(image.shape[1])
    if x.ndim == 2:
        x = x.T
    if y.ndim == 2:
        y = y.T
    if frac_thresh is not None:
        image[image < frac_thresh * np.max(image)] = 0
    if log:
        image = image + np.min(image[image > 0])
    ax.pcolormesh(x, y, image.T, **plot_kws)
    if profx or profy:
        if prof_kws is None:
            prof_kws = dict()
        prof_kws.setdefault('color', 'white')
        prof_kws.setdefault('lw', 1.0)
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
            if i == 0 and not profx:
                continue
            if i == 1 and not profy:
                continue
            if kind == 'line':
                ax.plot(x, y, **prof_kws)
            elif kind == 'bar':
                if i == 0:
                    ax.bar(x, y, **prof_kws)
                else:
                    ax.barh(y, x, **prof_kws)
            elif kind == 'step':
                ax.step(x, y, where='mid', **prof_kws)
    return ax


def corner(
    image, 
    coords=None,
    labels=None, 
    diag_kind='line',
    frac_thresh=None,
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
    
    if coords is None:
        coords = [np.arange(s) for s in image.shape]
    
    if diag_kind is None or diag_kind.lower() == 'none':
        axes = _corner_nodiag(
            image, 
            coords=coords,
            labels=labels, 
            frac_thresh=frac_thresh,
            fig_kws=fig_kws, 
            prof=prof,
            prof_kws=prof_kws,
            **plot_kws
        )
        return axes
    
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
                    ax.step(h, where='mid', **diag_kws)
            else:
                if prof == 'edges':
                    profx = i == n - 1
                    profy = j == 0
                else:
                    profx = profy = prof
                H = utils.project(image, (j, i))
                plot_image(H, ax=ax, x=coords[j], y=coords[i],
                           profx=profx, profy=profy, prof_kws=prof_kws, 
                           frac_thresh=frac_thresh, **plot_kws)
    for ax, label in zip(axes[-1, :], labels):
        ax.format(xlabel=label)
    for ax, label in zip(axes[1:, 0], labels[1:]):
        ax.format(ylabel=label)
    for i in range(n):
        axes[:-1, i].format(xticklabels=[])
        if i > 0:
            axes[i, 1:].format(yticklabels=[])
    return axes


def _corner_nodiag(
    image, 
    coords=None,
    labels=None, 
    frac_thresh=None,
    fig_kws=None, 
    prof=False,
    prof_kws=None,
    **plot_kws
):
    n = image.ndim
    if labels is None:
        labels = n * ['']
    if fig_kws is None:
        fig_kws = dict()
    fig_kws.setdefault('figwidth', 1.5 * (n - 1))
    fig_kws.setdefault('aligny', True)
    plot_kws.setdefault('ec', 'None')
    
    fig, axes = pplt.subplots(
        nrows=n-1, ncols=n-1, 
        spanx=False, spany=False, **fig_kws
    )
    for i in range(n - 1):
        for j in range(n - 1):
            ax = axes[i, j]
            if j > i:
                ax.axis('off')
                continue
            if prof == 'edges':
                profy = j == 0
                profx = i == n - 2
            else:
                profx = profy = prof
            H = utils.project(image, (j, i + 1))
            
            x = coords[j]
            y = coords[i + 1]
            if x.ndim > 1:
                axis = [k for k in range(x.ndim) if k not in (j, i + 1)]
                ind = len(axis) * [0]
                idx = utils.make_slice(x.ndim, axis, ind)
                x = x[idx]
                y = y[idx]
                
            plot_image(H, ax=ax, x=x, y=y,
                       profx=profx, profy=profy, prof_kws=prof_kws, 
                       frac_thresh=frac_thresh, **plot_kws)
    for ax, label in zip(axes[-1, :], labels):
        ax.format(xlabel=label)
    for ax, label in zip(axes[:, 0], labels[1:]):
        ax.format(ylabel=label)
    return axes