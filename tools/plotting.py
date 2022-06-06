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
               prof_kws=None, frac_thresh=None, contour=False, contour_kws=None,
               return_mesh=False,
               **plot_kws):
    """Plot image with profiles overlayed.
    
    To do: clean up, add documentation.
    """
    log = 'norm' in plot_kws and plot_kws['norm'] == 'log'
    if log:
        if 'colorbar' in plot_kws and plot_kws['colorbar']:
            if 'colorbar_kw' not in plot_kws:
                plot_kws['colorbar_kw'] = dict()
            plot_kws['colorbar_kw']['formatter'] = 'log'
    if contour and contour_kws is None:
        contour_kws = dict()
        contour_kws.setdefault('color', 'white')
        contour_kws.setdefault('lw', 1.0)
        contour_kws.setdefault('alpha', 0.5)
    if x is None:
        x = np.arange(image.shape[0])
    if y is None:
        y = np.arange(image.shape[1])
    if x.ndim == 2:
        x = x.T
    if y.ndim == 2:
        y = y.T
    image_max = np.max(image)
    if frac_thresh is not None:
        floor = max(1e-12, frac_thresh * image_max)
        image[image < floor] = 0
    if log:
        floor = 1e-12
        if image_max > 0:
            floor = np.min(image[image > 0])
        image = image + floor
    mesh = ax.pcolormesh(x, y, image.T, **plot_kws)
    if contour:
        ax.contour(x, y, image.T, **contour_kws)
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
        fx_max = np.max(fx)
        fy_max = np.max(fy)
        if fx_max > 0:
            fx = fx / fx_max
        if fy_max > 0:
            fy = fy / fy.max()
        x1 = x
        y1 = scale * np.abs(y[-1] - y[0]) * fx
        x2 = np.abs(x[-1] - x[0]) * scale * fy
        y1 = y1 + y[0]
        x2 = x2 + x[0]
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
    if return_mesh:
        return ax, mesh
    else:
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
    return_fig=False,
    **plot_kws
):
    """Plot all 1D/2D projections in a matrix of subplots.
    
    To do: 
    
    Clean this up and merge with `scdist.tools.plotting.corner`, 
    which performs binning first. I believe in scdist I also found
    a nicer way to handle diag on/off. (One function that plots
    the off-diagonals, recieving axes as an argument.
    """
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
            return_fig=return_fig,
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
    if return_fig:
        return fig, axes
    return axes


def _corner_nodiag(
    image, 
    coords=None,
    labels=None, 
    frac_thresh=None,
    fig_kws=None, 
    prof=False,
    prof_kws=None,
    return_fig=False,
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
    if return_fig:
        return fig, axes
    return axes