import numpy as np
from matplotlib import pyplot as plt
import proplot as pplt


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


def plot_compare_images(im1, im2, **plot_kws):
    """Plot images side by side, and a second row in log scale."""
    fig, axes = pplt.subplots(ncols=2, nrows=2, figwidth=None, sharex=False, sharey=False)
    for col, _im in enumerate([im1, im2]):
        for row, log in enumerate([False, True]):
            if log:
                _im = np.log10(_im + 1e-6)
            axes[row, col].pcolormesh(_im, **plot_kws)
    axes.format(xticks=[], yticks=[], leftlabels=['Normal scale', 'log scale'])
    return axes


# labeller = lambda x: '{:.1f}'.format(np.log10(x))
# contour_levels = tuple([10**i for i in (-4, -3, -2, -1)])
# default_norm = colors.LogNorm(vmin=1e-4, vmax=1, clip=False)

# def thisplot(array2plot,labels,norm=default_norm,ax1=None,ax2=None):
#     try:
#         plt.pcolor(ax1,ax2,array2plot)
#     except:
#         plt.pcolor(array2plot)
#     plt.xlabel(labels[0]); plt.ylabel(labels[1])
#     plt.colorbar()
#     # -- contour
#     try:
#         CS = plt.contour(ax1,ax2,array2plot/array2plot.max(),levels=contour_levels,norm=norm,cmap='Set1')
#     except:
#         CS = plt.contour(array2plot/array2plot.max(),levels=contour_levels,norm=norm,cmap='Set1')
#     plt.clabel(CS,fmt = labeller)