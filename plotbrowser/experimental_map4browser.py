from operator import inv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from scipy import linalg

from cmcrameri import cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy

import sys
import numpy.fft as fft
import scipy.fft as sfft
import os
import errno
import h5py
from scipy import stats
from scipy.stats import norm, skew
from scipy import signal
import copy

import argparse
import re
from tqdm import tqdm
import warnings
import time as tm

from mapeditor.mapeditor import Atlas

import accept_mod.stats_list as stats_list

stats_list = stats_list.stats_list

from spikes import spike_data, spike_list, get_spike_list
from mapdata import Map

warnings.filterwarnings(
    "ignore", category=RuntimeWarning
)  # Ignore warnings caused by mask nan/inf weights
warnings.filterwarnings(
    "ignore", category=UserWarning
)  # Ignore warning when producing log plot of empty array


def gaussian_kernelXY(sigma_x, sigma_y, n_sigma):
    """
    Function returning a 2D Gaussian kernal for pixel smoothing.
    """
    size_x = int(n_sigma * sigma_x)  # Grid boundaries for kernal
    size_y = int(n_sigma * sigma_y)
    x, y = np.mgrid[
        -size_x : size_x + 1, -size_y : size_y + 1
    ]  # Seting up the kernal's grid
    g = np.exp(
        -(x**2 / (2.0 * sigma_x**2) + y**2 / (2.0 * sigma_y**2))
    )  # Computing the Gaussian Kernal
    return g / g.sum()


def smoothXY3D(map, sigma_x, sigma_y, n_sigma):
    kernel = gaussian_kernelXY(
        sigma_x, sigma_y, n_sigma
    )  # Computing 2D Gaussian Kernal

    map = signal.fftconvolve(
        map, kernel[np.newaxis, :, :], mode="same", axes=(1, 2)
    ).astype(np.float32)
    return map


def plot_sb_avg_maps(map, color_lim=[None, None], feed=None):

    x_lim, y_lim = [None, None], [None, None]

    x, y = map.x, map.y
    dx = (x[1] - x[0]) / np.cos(np.radians(np.mean(y)))
    x_lim[0] = x[0] - 0.5 * dx
    x_lim[1] = x[-1] + 0.5 * dx
    dy = y[1] - y[0]
    y_lim[0] = y[1] - 0.5 * dy
    y_lim[1] = y[-1] + 0.5 * dy

    if feed == None:
        mapdata = map.map_coadd[...].astype(np.float64)
        nhitdata = map.nhit_coadd[...].astype(np.float64)
        rmsdata = map.rms_coadd[...].astype(np.float64)
    else:
        mapdata = map.map[feed, ...].astype(np.float64)
        nhitdata = map.nhit[feed, ...].astype(np.float64)
        rmsdata = map.rms[feed, ...].astype(np.float64)

    nhitdata = np.where(nhitdata > 1, nhitdata, np.nan)
    rmsdata = np.where(nhitdata > 1, rmsdata, np.nan)

    # inv_var = 1 / rmsdata ** 2

    # mapdata *= inv_var
    where = np.nanprod(nhitdata, axis=1) > 1

    mapdata = np.nansum(
        np.abs(mapdata), axis=1
    )  # Coadding over freq channels per sideband
    # mapdata[where] /= np.nansum(inv_var, axis = 1)[where]

    mapdata = np.where(where == True, mapdata, np.nan)
    # mapdata[where == False] = np.nan * mapdata[where == False]
    mapdata *= 1e6  # K to muK

    fontsize = 16
    fonts = {
        "font.family": "sans-serif",
        "axes.labelsize": fontsize,
        "font.size": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
    }
    plt.rcParams.update(fonts)

    cmap = cm.roma.reversed()

    # cmap.set_bad("0", 1) # Set color of masked elements to black.
    cmap.set_bad("0.8", 1)  # Set color of masked elements to gray.
    # cmap.set_bad("1", 1) # Set color of masked elements to white.

    fig, ax = plt.subplots(2, 2, figsize=(16, 14), sharex=True, sharey=True)

    sb_name = ["A:LSB", "A:USB", "B:LSB", "B:USB"]

    ax[0, 0].set_ylabel("Declination [deg]")
    ax[1, 0].set_ylabel("Declination [deg]")
    ax[1, 0].set_xlabel("Right Ascension [deg]")
    ax[1, 1].set_xlabel("Right Ascension [deg]")

    if map.scanmap:
        fig.suptitle(f"Sideband avg. Scanid: {map.scanid}")
    elif map.obsmap:
        fig.suptitle(f"Sideband avg. Obsid: {map.obsid}")

    for i in range(2):
        for j in range(2):
            sb = i * 2 + j

            # if color_lim[0] is None or color_lim[1] is None:
            # min = np.abs(np.nanmin(mapdata[sb, ...]))
            # max = np.nanmax((np.nanmax(mapdata[sb, ...]), min))
            # color_lim[1] = 0.01 * max #0.4 * np.nanstd(mapdata[sb, ...])
            # color_lim[1] = 0.4 * np.nanstd(mapdata[sb, ...])
            # color_lim[0] = - color_lim[1]

            aspect = dx / dy
            img = ax[i, j].imshow(
                mapdata[sb, ...],
                extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]),
                interpolation="nearest",
                aspect=aspect,
                cmap="CMRmap",
                origin="lower",
                vmin=color_lim[0],
                vmax=color_lim[1],
                rasterized=True,
            )
            ax[i, j].set_title(f"{sb_name[sb]}")

            divider = make_axes_locatable(ax[i, j])
            cax = divider.append_axes("right", size="5%", pad=0.05)

            cbar = fig.colorbar(img, ax=ax[i, j], cax=cax)

            cbar.set_label("$\mu K$")

    fig.tight_layout()
    fig.savefig(
        outpath + "sb_avg/" + map.outname + "_sb_avg" + ".png", bbox_inches="tight"
    )


def plot_quarter_sb_avg_maps(map, color_lim=[None, None], feed=None):
    x_lim, y_lim = [None, None], [None, None]

    if map.scanmap:
        px = 3
    elif map.obsmap:
        px = 20

    x, y = map.x[px:-px], map.y[px:-px]
    dx = (x[1] - x[0]) / np.cos(np.radians(np.mean(y)))
    x_lim[0] = x[0] - 0.5 * dx
    x_lim[1] = x[-1] + 0.5 * dx
    dy = y[1] - y[0]
    y_lim[0] = y[1] - 0.5 * dy
    y_lim[1] = y[-1] + 0.5 * dy

    if feed == None:
        mapdata = map.map_coadd[..., px:-px].astype(np.float64)
        nhitdata = map.nhit_coadd[..., px:-px].astype(np.float64)
        rmsdata = map.rms_coadd[..., px:-px].astype(np.float64)
    else:
        mapdata = map.map[feed, :, :, px:-px].astype(np.float64)
        nhitdata = map.nhit[feed, :, :, px:-px].astype(np.float64)
        rmsdata = map.rms[feed, :, :, px:-px].astype(np.float64)

    mapdata = np.where(nhitdata > 1, mapdata, np.nan)
    nhitdata = np.where(nhitdata > 1, nhitdata, np.nan)
    rmsdata = np.where(nhitdata > 1, rmsdata, np.nan)

    # inv_var = 1 / rmsdata ** 2

    # mapdata *= inv_var
    mapdata *= 1e6
    freq = mapobj.freq
    freq = freq.reshape(freq.shape[0], int(freq.shape[1] / 16), 16)
    freq = np.mean(freq, axis=2)  # Finding new frequency channel center by averaging

    dfreq = 1e3 * (freq[0, 1] - freq[0, 0])

    fontsize = 16
    fonts = {
        "font.family": "sans-serif",
        "axes.labelsize": fontsize - 4,
        "font.size": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize - 4,
        "ytick.labelsize": fontsize - 4,
    }
    plt.rcParams.update(fonts)

    cmap = cm.roma.reversed()
    # cmap.set_bad("0", 1) # Set color of masked elements to black.
    # cmap.set_bad("0.8", 1) # Set color of masked elements to gray.
    cmap.set_bad("1", 1)  # Set color of masked elements to white.

    sb_name = ["A:LSB", "A:USB", "B:LSB", "B:USB"]

    fig = plt.figure(figsize=(10, 9.5))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.05, hspace=0.05)

    grids = [gs[i].subgridspec(2, 2, wspace=0.0, hspace=0.0) for i in range(4)]

    # if map.scanmap:
    #    fig.suptitle(f"Quarter sideband avg. Scanid: {map.scanid}")
    # elif map.obsmap:
    #    fig.suptitle(f"Quarter sideband avg. Obsid: {map.obsid}")

    if map.scanmap:
        fig.suptitle(
            f"Scanid: {map.scanid} | Feed: {feed} | Bandwidth: {dfreq:.2f} MHz\n"
            + rf"(RA,Dec)=({np.mean(x):.2f}$^\circ$,{np.mean(y):.2f}$^\circ$)"
            + " | "
            + rf"RA$\times$Dec={x[-1] - x[0]:.2f}$^\circ\times${y[-1] - y[0]:.2f}$^\circ$"
        )
    elif map.obsmap:
        fig.suptitle(
            f"Obsid: {map.obsid} | Feed:{feed} | Bandwidth: {dfreq:.2f} MHz\n"
            + rf"(RA,Dec)=({np.mean(x):.2f}$^\circ$,{np.mean(y):.2f}$^\circ$)"
            + " | "
            + rf"RA$\times$Dec={x[-1] - x[0]:.2f}$^\circ\times${y[-1] - y[0]:.2f}$^\circ$"
        )
    # axes = {}
    aspect = dx / dy
    for i in range(4):
        # for j in range(2):
        #    for k in range(2):
        ax_dummy = fig.add_subplot(grids[i][:, :])
        # ax_dummy.axis("off")

        # ax_dummy.xaxis.set_visible(False)
        # ax_dummy.yaxis.set_visible(False)
        for key, spine in ax_dummy.spines.items():
            spine.set_visible(False)

        plt.setp(ax_dummy.get_xticklabels(), visible=False)
        plt.setp(ax_dummy.get_yticklabels(), visible=False)
        ax_dummy.tick_params(
            left=False, bottom=False, labelleft=False, labelbottom=False
        )
        # ax_dummy.patch.set_visible(False)
        # ax_dummy.set_title(" " * 6 + sb_name[i] + "\n")

        # where = np.nanprod(nhitdata[i, 32:48, ...], axis = 0) > 1

        # data = np.nansum(mapdata[i, 32:48, ...], axis = 0)     # Coadding over freq channels per sideband
        # data[where] /= np.nansum(inv_var[i, 32:48, ...], axis = 0)[where]

        # data  = np.where(where == True, data, np.nan)
        # data[where == False] = np.nan * data[where == False]
        # data *= 1e6                          # K to muK
        ax2 = fig.add_subplot(grids[i][1, 0])
        f_mid = np.mean(freq[i, 2])
        ax2.text(0.02, 0.05, f"{f_mid:.2f} GHz", fontsize=12, transform=ax2.transAxes)
        # ax2.set_title(f"Ch: 33-48 avg. \n freq: {map.freq[i, 32]:.2f}-{map.freq[i, 47]:.2f} GHz", fontsize = 14)
        # axes[f"{i}{0}{0}"] = ax

        # color_lim[1] = 0.6 * np.nanstd(data)
        # color_lim[0] = - color_lim[1]

        img = ax2.imshow(
            mapdata[i, 2],
            extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]),
            interpolation="nearest",
            aspect=aspect,
            cmap=cmap,
            origin="lower",
            vmin=color_lim[0],
            vmax=color_lim[1],
            rasterized=True,
        )

        # divider = make_axes_locatable(ax2)
        # cax = divider.append_axes("right", size="5%", pad=0.05)

        # cbar = fig.colorbar(img, ax = ax2, cax = cax)

        # cbar.set_label("$\mu K$")

        # cbar.ax.tick_params(rotation = 90)

        # where = np.nanprod(nhitdata[i, 0:16, ...], axis = 0) > 1

        # data = np.nansum(mapdata[i, 0:16, ...], axis = 0)     # Coadding over freq channels per sideband
        # data[where] /= np.nansum(inv_var[i, 0:16, ...], axis = 0)[where]

        # data  = np.where(where == True, data, np.nan)
        # data[where == False] = np.nan * data[where == False]
        # data *= 1e6                          # K to muK

        ax1 = fig.add_subplot(grids[i][0, 0], sharex=ax2)
        f_mid = np.mean(freq[i, 0])
        ax1.text(0.02, 0.05, f"{f_mid:.2f} GHz", fontsize=12, transform=ax1.transAxes)
        ax1.text(0.02, 0.9, f"{sb_name[i]}", fontsize=12, transform=ax1.transAxes)
        # ax1.set_title(f"C:. 1-16 avg. \n freq: {map.freq[i, 0]:.2f}-{map.freq[i, 15]:.2f} GHz", fontsize = 14)
        plt.setp(ax1.get_xticklabels(), visible=False)

        # axes[f"{i}{0}{0}"] = ax

        img = ax1.imshow(
            mapdata[i, 0],
            extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]),
            interpolation="nearest",
            aspect=aspect,
            cmap=cmap,
            origin="lower",
            vmin=color_lim[0],
            vmax=color_lim[1],
            rasterized=True,
        )

        # divider = make_axes_locatable(ax1)
        # cax = divider.append_axes("right", size="5%", pad=0.05)

        # cbar = fig.colorbar(img, ax = ax1, cax = cax)

        # cbar.set_label("$\mu K$")

        # cbar.ax.tick_params(rotation = 90)

        # where = np.nanprod(nhitdata[i, 48:64, ...], axis = 0) > 1

        # data = np.nansum(mapdata[i, 48:64, ...], axis = 0)     # Coadding over freq channels per sideband
        # data[where] /= np.nansum(inv_var[i, 48:64, ...], axis = 0)[where]

        # data  = np.where(where == True, data, np.nan)
        # data[where == False] = np.nan * data[where == False]
        # data *= 1e6                          # K to muK

        ax3 = fig.add_subplot(grids[i][1, 1], sharey=ax2)
        # ax3.set_title(f"Ch: 49-64 avg. \n freq: {map.freq[i, 48]:.2f}-{map.freq[i, 63]:.2f} GHz", fontsize = 14)
        # axes[f"{i}{0}{0}"] = ax
        f_mid = np.mean(freq[i, 3])
        ax3.text(0.02, 0.05, f"{f_mid:.2f} GHz", fontsize=12, transform=ax3.transAxes)
        plt.setp(ax3.get_yticklabels(), visible=False)

        img = ax3.imshow(
            mapdata[i, 3],
            extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]),
            interpolation="nearest",
            aspect=aspect,
            cmap=cmap,
            origin="lower",
            vmin=color_lim[0],
            vmax=color_lim[1],
            rasterized=True,
        )

        # divider = make_axes_locatable(ax3)
        # cax = divider.append_axes("right", size="5%", pad=0.05)

        # cbar = fig.colorbar(img, ax = ax3, cax = cax)

        # cbar.set_label("$\mu K$")

        # cbar.ax.tick_params(rotation = 90)

        # where = np.nanprod(nhitdata[i, 16:32, ...], axis = 0) > 1

        # data = np.nansum(mapdata[i, 16:32, ...], axis = 0)     # Coadding over freq channels per sideband
        # data[where] /= np.nansum(inv_var[i, 16:32, ...], axis = 0)[where]

        # data  = np.where(where == True, data, np.nan)
        # data[where == False] = np.nan * data[where == False]
        # data *= 1e6                          # K to muK

        ax4 = fig.add_subplot(grids[i][0, 1], sharex=ax3, sharey=ax1)
        # ax4.set_title(f"Ch: 17-32 avg. \n freq: {map.freq[i, 16]:.2f}-{map.freq[i, 31]:.2f} GHz", fontsize = 14)
        f_mid = np.mean(freq[i, 1])
        ax4.text(0.02, 0.05, f"{f_mid:.2f} GHz", fontsize=12, transform=ax4.transAxes)
        plt.setp(ax4.get_yticklabels(), visible=False)
        plt.setp(ax4.get_xticklabels(), visible=False)

        img = ax4.imshow(
            mapdata[i, 1],
            extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]),
            interpolation="nearest",
            aspect=aspect,
            cmap=cmap,
            origin="lower",
            vmin=color_lim[0],
            vmax=color_lim[1],
            rasterized=True,
        )

        if i == 2 or i == 3:
            ax_dummy.set_xlabel("Right Ascension [deg]", labelpad=20)
            # ax2.set_xlabel('Right Ascension [deg]')
            # ax3.set_xlabel('Right Ascension [deg]')
        else:
            plt.setp(ax3.get_xticklabels(), visible=False)
            plt.setp(ax2.get_xticklabels(), visible=False)

        if i == 0 or i == 2:
            ax_dummy.set_ylabel("Right Ascension [deg]", labelpad=20)
            # ax1.set_ylabel('Right Ascension [deg]')
            # ax2.set_ylabel('Right Ascension [deg]')
        else:
            plt.setp(ax1.get_yticklabels(), visible=False)
            plt.setp(ax2.get_yticklabels(), visible=False)

    ax_dummy2 = fig.add_subplot(gs[:])
    ax_dummy2.axis("off")

    cax = inset_axes(
        ax_dummy2,
        width="2%",  # width = 50% of parent_bbox width
        height="100%",  # height : 5%
        loc="center right",
        borderpad=-1.05,
    )
    cbar = fig.colorbar(img, ax=ax_dummy2, cax=cax)

    cbar.set_label("$\mu K$")

    cbar.ax.tick_params(rotation=90)

    # fig.tight_layout()
    fig.savefig(
        outpath + "quarter_sb_avg/" + map.outname + "_quater_sb_avg" + ".png",
        bbox_inches="tight",
    )


def plot_quarter_sb_avg_column_maps(map, color_lim=[None, None], feed=None):

    x_lim, y_lim = [None, None], [None, None]

    x, y = map.x, map.y
    dx = (x[1] - x[0]) / np.cos(np.radians(np.mean(y)))
    x_lim[0] = x[0] - 0.5 * dx
    x_lim[1] = x[-1] + 0.5 * dx
    dy = y[1] - y[0]
    y_lim[0] = y[1] - 0.5 * dy
    y_lim[1] = y[-1] + 0.5 * dy

    if feed == None:
        mapdata = map.map_coadd[...].astype(np.float64)
        nhitdata = map.nhit_coadd[...].astype(np.float64)
        rmsdata = map.rms_coadd[...].astype(np.float64)
    else:
        mapdata = map.map[feed, ...].astype(np.float64)
        nhitdata = map.nhit[feed, ...].astype(np.float64)
        rmsdata = map.rms[feed, ...].astype(np.float64)

    mapdata = np.where(nhitdata > 1, mapdata, np.nan)
    nhitdata = np.where(nhitdata > 1, nhitdata, np.nan)
    rmsdata = np.where(nhitdata > 1, rmsdata, np.nan)

    mapdata = mapdata.reshape(map.n_sb * map.n_freq, map.n_x, map.n_y, order="F")
    nhitdata = nhitdata.reshape(map.n_sb * map.n_freq, map.n_x, map.n_y, order="F")
    rmsdata = rmsdata.reshape(map.n_sb * map.n_freq, map.n_x, map.n_y, order="F")

    mapdata *= 1e6

    freq = map.freq
    dfreq = 1e3 * (freq[0, 1] - freq[0, 0])
    freq = freq.reshape(map.n_sb * map.n_freq, order="F")

    fontsize = 12
    fonts = {
        "font.family": "sans-serif",
        "axes.labelsize": fontsize - 4,
        "font.size": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize - 4,
        "ytick.labelsize": fontsize - 4,
    }
    plt.rcParams.update(fonts)

    cmap = cm.roma.reversed()
    # cmap.set_bad("0", 1) # Set color of masked elements to black.
    # cmap.set_bad("0.8", 1) # Set color of masked elements to gray.
    cmap.set_bad("1", 1)  # Set color of masked elements to white.

    sb_name = ["A:LSB", "A:USB", "B:LSB", "B:USB"]

    fig = plt.figure(figsize=(map.n_sb, map.n_freq))
    gs = gridspec.GridSpec(map.n_freq, map.n_sb, figure=fig, wspace=0.0, hspace=0.0)

    if feed == None:
        feedname = "All"
    if map.scanmap:
        fig.suptitle(
            f"Scanid: {map.scanid} | Feed: All\nBandwidth: {dfreq:.2f} MHz"
            + rf"(RA,Dec)=({np.mean(x[10:-20]):.2f}$^\circ$,{np.mean(y[10:-20]):.2f}$^\circ$)"
            + "\n"
            + rf"RA$\times$Dec={x[-20] - x[10]:.2f}$^\circ\times${y[-20] - y[10]:.2f}$^\circ$"
        )
    elif map.obsmap:
        fig.suptitle(
            f"Obsid: {map.obsid} | Feed: All\nBandwidth: {dfreq:.2f} MHz\n"
            + rf"(RA,Dec)=({np.mean(x[10:-20]):.2f}$^\circ$,{np.mean(y[10:-20]):.2f}$^\circ$)"
            + "\n"
            + rf"RA$\times$Dec={x[-20] - x[10]:.2f}$^\circ\times${y[-20] - y[10]:.2f}$^\circ$"
        )

    axes = []
    aspect = dx / dy

    color_lim[1] = 0.6 * np.nanstd(mapdata)
    color_lim[0] = -color_lim[1]

    ax_dummy = fig.add_subplot(gs[-1, :])
    ax_dummy.axis("off")

    for i in range(16):
        for j in range(4):
            idx = i * 4 + j
            ax = fig.add_subplot(gs[idx])
            axes.append(ax)

            img = ax.imshow(
                mapdata[idx, 10:-20, 10:-20],
                extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]),
                interpolation="nearest",
                aspect=aspect,
                cmap=cmap,
                origin="lower",
                vmin=color_lim[0],
                vmax=color_lim[1],
                rasterized=True,
            )
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.text(
                0.02, 0.05, f"{freq[idx]:.2f} GHz", fontsize=6, transform=ax.transAxes
            )

    axes[0].set_title(sb_name[1])
    axes[1].set_title(sb_name[1])
    axes[2].set_title(sb_name[2])
    axes[3].set_title(sb_name[3])

    # divider = make_axes_locatable(ax_dummy)
    # cax = divider.append_axes("bottom", size = "2%", pad = 10)

    cax = inset_axes(
        ax_dummy,
        width="100%",  # width = 50% of parent_bbox width
        height="10%",  # height : 5%
        loc="lower center",
        borderpad=-1,
    )  # ,
    # bbox_to_anchor = (0, -0.01, 1, 1),
    # bbox_transform = ax.transAxes)

    cbar = fig.colorbar(img, ax=ax_dummy, cax=cax, orientation="horizontal")

    cbar.set_label("$\mu K$")

    cax.xaxis.set_label_position("bottom")
    # cbar.ax.tick_params(rotation = 90)
    # fig.tight_layout()
    fig.savefig(
        outpath + "quarter_sb_avg_column/" + map.outname + "_quater_sb_avg" + ".png",
        bbox_inches="tight",
        dpi=300,
    )
    print("hei")
    plt.show()


def plot_skewness_maps(map, color_lim=[None, None], feed=None):

    x_lim, y_lim = [None, None], [None, None]

    x, y = map.x, map.y
    dx = (x[1] - x[0]) / np.cos(np.radians(np.mean(y)))
    x_lim[0] = x[0] - 0.5 * dx
    x_lim[1] = x[-1] + 0.5 * dx
    dy = y[1] - y[0]
    y_lim[0] = y[1] - 0.5 * dy
    y_lim[1] = y[-1] + 0.5 * dy

    if feed == None:
        mapdata = map.map_coadd[...].astype(np.float64)
        nhitdata = map.nhit_coadd[...].astype(np.float64)
        rmsdata = map.rms_coadd[...].astype(np.float64)
    else:
        mapdata = map.map[feed, ...].astype(np.float64)
        nhitdata = map.nhit[feed, ...].astype(np.float64)
        rmsdata = map.rms[feed, ...].astype(np.float64)

    # mapdata *= 1e6

    mapdata = np.where(nhitdata > 1, mapdata, np.nan)
    nhitdata = np.where(nhitdata > 1, nhitdata, np.nan)
    rmsdata = np.where(nhitdata > 1, rmsdata, np.nan)

    mapdata = mapdata / rmsdata

    n_ch = np.nansum(nhitdata, axis=(0, 1))
    print(
        mapdata.reshape(map.n_sb * map.n_freq, map.n_x, map.n_y).shape,
        np.nansum(nhitdata, axis=(0, 1)).shape,
    )

    skewness = (
        np.sqrt(n_ch * (n_ch - 1))
        / (n_ch - 2)
        * skew(
            mapdata.reshape(map.n_sb * map.n_freq, map.n_x, map.n_y),
            axis=0,
            nan_policy="omit",
        )
    )

    skewness = np.where(np.nansum(nhitdata, axis=(0, 1)) > 1, skewness, np.nan)

    var_expect = 6 * n_ch * (n_ch - 1) / ((n_ch - 2) * (n_ch + 1) * (n_ch + 3))
    std_expect = np.sqrt(var_expect)

    print(
        np.any(np.isnan(var_expect)),
        np.any(np.isnan(n_ch)),
        np.any(np.isnan(skewness)),
        np.max(n_ch),
        np.min(n_ch),
    )

    # mapdata = mapdata.reshape(map.n_sb * map.n_freq, map.n_x, map.n_y, order = "F")
    # nhitdata = nhitdata.reshape(map.n_sb * map.n_freq, map.n_x, map.n_y, order = "F")
    # rmsdata  = rmsdata.reshape(map.n_sb * map.n_freq, map.n_x, map.n_y, order = "F")

    fontsize = 16
    fonts = {
        "font.family": "sans-serif",
        "axes.labelsize": fontsize - 4,
        "font.size": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize - 4,
        "ytick.labelsize": fontsize - 4,
    }
    plt.rcParams.update(fonts)

    cmap = cm.roma.reversed()
    # cmap.set_bad("0", 1) # Set color of masked elements to black.
    # cmap.set_bad("0.8", 1) # Set color of masked elements to gray.
    cmap.set_bad("1", 1)  # Set color of masked elements to white.

    sb_name = ["A:LSB", "A:USB", "B:LSB", "B:USB"]

    color_lim[1] = 1.2 * np.nanstd(skewness)
    color_lim[0] = -color_lim[1]

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    aspect = dx / dy

    img = ax[0].imshow(
        skewness,
        extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]),
        interpolation="nearest",
        aspect=aspect,
        cmap=cmap,
        origin="lower",
        vmin=color_lim[0],
        vmax=color_lim[1],
        rasterized=True,
    )

    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = fig.colorbar(img, ax=ax[0], cax=cax)

    cbar.set_label("$G_1$")

    color_lim[1] = 1.2 * np.nanstd(skewness / std_expect)
    color_lim[0] = -color_lim[1]
    img = ax[1].imshow(
        skewness / std_expect,
        extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]),
        interpolation="nearest",
        aspect=aspect,
        cmap=cmap,
        origin="lower",
        vmin=color_lim[0],
        vmax=color_lim[1],
        rasterized=True,
    )

    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    cbar = fig.colorbar(img, ax=ax[1], cax=cax)

    cbar.set_label("$G_1 / \sqrt{var(G_1)}$")

    fig.tight_layout()
    fig.savefig(
        outpath + "skewness/" + map.outname + "_skewness" + ".png",
        bbox_inches="tight",
        dpi=300,
    )


def get_svd_basis(data):  # , x, y, freq):

    nsb, nfreq, nra, ndec = data.shape

    freq_vec, singular_values, ang_vec = linalg.svd(
        data.reshape(nsb * nfreq, nra * ndec), full_matrices=False
    )

    ncomp = singular_values.size

    return (
        freq_vec.T.reshape(ncomp, nsb, nfreq),
        ang_vec.reshape(ncomp, nra, ndec),
        singular_values,
    )


def plot_pca_maps(map, outpath, color_lim=[-3e-2, 3e-2], feeds=range(20)):

    allcomps = []
    allampls = []
    allampls_fft = []
    ncomp = 10

    nfeeds = len(feeds)

    l2path = "/mn/stornext/d22/cmbco/comap/protodir/level2/Ka/co2/co2_001900006.h5"
    with h5py.File(l2path, "r") as l2file:
        tsys = l2file["Tsys_lowres"][()]
        l2freqs = l2file["nu"][0, ...]
        pixels = l2file["pixels"][()]

    tsys = np.where(tsys != 0, tsys, np.nan)

    temp = np.zeros((nfeeds, *tsys.shape[1:]))
    temp = tsys[pixels - 1, ...]
    tsys = temp
    tsys -= np.nanmean(tsys, axis=(1, 2))[:, None, None]

    tsys[:, [0, 2], :] = tsys[:, [0, 2], ::-1]
    l2freqs[[0, 2], :] = l2freqs[[0, 2], ::-1]

    sb_names = ["A:LSB", "A:USB", "B:LSB", "B:USB"]

    for feed in feeds:
        print(f"Feed: {feed + 1}")
        x_lim, y_lim = [None, None], [None, None]

        x, y = map.x, map.y

        if feed == None:
            mapdata = map.map_coadd[...].astype(np.float64)
            nhitdata = map.nhit_coadd[...].astype(np.float64)
            rmsdata = map.rms_coadd[...].astype(np.float64)
        else:
            mapdata = map.map[feed, ...].astype(np.float64)
            nhitdata = map.nhit[feed, ...].astype(np.float64)
            rmsdata = map.rms[feed, ...].astype(np.float64)

        """mapdata_kelvin = np.where(nhitdata > 0, mapdata, 0)
        mapdata = np.where(nhitdata > 0, mapdata / rmsdata, 0)
        # mapdata = np.where(np.isinf(mapdata) == False, mapdata, 0)

        pca = PCA()
        pca_data = pca.fit(mapdata.reshape(map.n_sb * map.n_freq, map.n_x * map.n_y))
        comps = pca_data.components_
        eigs = comps.copy()

        comps = comps.reshape(map.n_sb, map.n_freq, map.n_x, map.n_y)
        # comps    = comps.reshape(map.n_sb * map.n_freq, map.n_x, map.n_y)
        comps = np.ascontiguousarray(comps, np.float32)

        # mapdata = np.where(nhitdata, mapdata, 0)

        mapvec = mapdata.reshape(map.n_sb * map.n_freq, map.n_x * map.n_y)
        mapvec_kelvin = mapdata_kelvin.reshape(map.n_sb * map.n_freq, map.n_x * map.n_y)

        _ampl = np.sum(eigs[None, :10, ...] * mapvec[:, None, :], axis=-1)
        ampl = np.sum(eigs[None, :10, ...] * mapvec_kelvin[:, None, :], axis=-1)
        # comps = smoothXY3D(comps, 0.5, 0.5, 1)
        allrms = np.nansum(1 / rmsdata**2, axis=(0, 1))
        allrms = np.sqrt(1 / allrms)
        allrms = np.broadcast_to(allrms, mapdata.shape)
        allrms = np.ascontiguousarray(allrms, np.float32)
        """

        allhits = np.nansum(nhitdata, axis=(0, 1))
        allhits = np.broadcast_to(allhits, nhitdata.shape)
        allhits = np.ascontiguousarray(allhits, np.int32)

        rmsdata = rmsdata
        rmsdata = np.where(np.isfinite(rmsdata), rmsdata, 0)

        rms_ampl, rms_comps, rms_singular_values = get_svd_basis(
            rmsdata.astype("float64")
        )

        rms_hat = (
            rms_ampl[0, :, :, None, None]
            * rms_singular_values[0]
            * rms_comps[0, None, None, :, :]
        )

        # mapdata = mapdata / rmsdata
        mapdata = mapdata / rms_hat
        mapdata = np.where(np.isfinite(mapdata), mapdata, 0)

        ampl, comps, singular_values = get_svd_basis(mapdata)
        ncomp, nsb, nfreq = ampl.shape

        # n_sb, n_freq, n_x, n_y = comps.shape
        # comps = np.where(allhits > 1, comps, np.nan)

        # comps = comps.reshape(n_sb * n_freq, n_x, n_y)

        # Frequency basis vector
        freqvec = rms_ampl * ampl

        # Std along freuqnecy axis
        freqstd = np.nanstd(freqvec, axis=(1, 2))

        # Multiply std
        compdata = comps * rms_comps[0, None, :, :] * freqstd[:, None, None]

        # To muK
        compdata *= (
            1e6
            * rms_singular_values[0, None, None, None]
            * singular_values[:, None, None]
        )
        compdata = np.where(np.abs(compdata) >= 1e-16, compdata, np.nan)

        compvec = comps * rms_comps[0, None, :, :]
        np.where(np.abs(compvec) >= 1e-16, compvec, np.nan)

        # Std over pixel direction
        ampldata = freqvec * np.nanstd(compvec, axis=(1, 2))[:, None, None]

        # To muK
        ampldata *= (1e6 * rms_singular_values[0] * singular_values)[:, None, None]
        # ampldata = ampl  # * singular_values[:, None, None]
        ampldata = ampldata.reshape(ncomp, nsb * nfreq)

        freq = map.freq
        dfreq = 1e3 * (freq[0, 1] - freq[0, 0])

        fft_all = sfft.rfft(ampldata, ampldata.shape[-1], axis=1)
        psd_all = np.abs(fft_all) ** 2

        fftfreq_all = sfft.rfftfreq(ampldata.shape[-1], 1e-3 * dfreq)

        fft_sb = sfft.rfft(ampl, ampl.shape[-1], axis=2)
        psd_sb = np.abs(fft_sb) ** 2

        fftfreq_sb = sfft.rfftfreq(ampl.shape[-1], 1e-3 * dfreq)

        allcomps.append(compdata[:, :])
        allampls.append(ampldata[:, :])
        # allampls_fft.append(ampldata_fft)

        ampldata = ampl  # * singular_values[:, None, None]

        dx = (x[1] - x[0]) / np.cos(np.radians(np.mean(y)))
        x_lim[0] = x[0] - 0.5 * dx
        x_lim[1] = x[-1] + 0.5 * dx
        dy = y[1] - y[0]
        y_lim[0] = y[1] - 0.5 * dy
        y_lim[1] = y[-1] + 0.5 * dy
        aspect = dx / dy

        fontsize = 10
        fonts = {
            "font.family": "sans-serif",
            "axes.labelsize": fontsize - 4,
            "font.size": fontsize,
            "legend.fontsize": fontsize,
            "xtick.labelsize": fontsize - 4,
            "ytick.labelsize": fontsize - 4,
        }
        plt.rcParams.update(fonts)

        # cmap = mpl.cm.RdBu.reversed()
        # cmap = mpl.cm.PiYG.reversed()
        # cmap = mpl.cm.bwr.reversed()
        cmap = cm.roma.reversed()
        # cmap.set_bad("0", 1) # Set color of masked elements to black.
        cmap.set_bad("0.8", 1)  # Set color of masked elements to gray.
        # cmap.set_bad("1", 1) # Set color of masked elements to white.

        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(5, 10, figure=fig, wspace=0.05, hspace=0.1)

        xtickslabel = [f"{i:.1f}" for i in x[20:-20:10]]
        ytickslabel = [f"{i:.1f}" for i in y[20:-20:10]]

        # for i in range(5 * 8):
        #     ax = fig.add_subplot(gs[i])
        #     ax.text(0.5, 0.5, f"{i}", transform=ax.transAxes)

        for i in range(5):
            ax = fig.add_subplot(gs[i * 10 : i * 10 + 3])

            # ax = fig.add_subplot(gs[i * 5 : i * 5 + 2])
            # amplim = np.std(ampldata[i, :])

            amplim = np.max(np.abs(ampldata[i, :]))

            ax.plot(
                freq.flatten(),
                ampldata[i, :].flatten(),
                label=rf"PCA$_{i}$",
                lw=0.1,
                c="gray",
                # alpha=0.2,
            )

            ax.plot(
                l2freqs.flatten(),
                tsys[feed, :].flatten() / np.nanmax(tsys[feed, ...]) * amplim,
                lw=1,
                c="r",
                alpha=0.5,
            )

            ax.scatter(freq.flatten(), ampldata[i, :], label=rf"PCA$_{i}$", s=1, c="b")

            # ax.legend(loc="upper right", fontsize=7)
            # ax.set_ylabel(r"$\mathrm{\mu K}$")
            ax.set_ylim(-1.1 * amplim, 1.1 * amplim)
            ax.set_xlim(np.min(freq) - 0.5, np.max(freq))

            if i == 4:
                ax.set_xlabel(r"Frequency [GHz]")
                # ax.set_xticklabels(ax.get_xticks(), rotation = 45)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

            if i == 0:
                ax.set_title("Frequency unit eigen-vector, with Tsys")
            # ax.tick_params(axis="y", direction="in", pad=-22)

            ##############################################################################

            ax = fig.add_subplot(gs[i * 10 + 3 : i * 10 + 6])

            ax.plot(fftfreq_all, psd_all[i, :], label="All SB", alpha=0.5)

            for sb in range(4):
                if not np.all(psd_sb[i, sb, :] == 0):
                    ax.plot(
                        fftfreq_sb,
                        psd_sb[i, sb, :],
                        label=f"{sb_names[sb]}",
                        alpha=0.5,
                    )

            if i == 4:
                ax.set_xlabel(r"[1/GHz]")
                # ax.set_xticklabels(ax.get_xticks(), rotation = 45)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

            if i == 0:
                ax.legend(loc="upper right", fontsize=6, ncol=5)
                ax.set_title("PSD of freq. eigen-vector [arbitrary units]")
            # ax.set_yscale("log")
            ax.tick_params(axis="y", direction="in", pad=-15, rotation=0)
            ax.set_xlim(np.min(fftfreq_all) - 1, np.max(fftfreq_all))

            ##############################################################################

            # color_lim[0] = color_lim[1] = np.nanstd(comps[i, ...])
            # color_lim[1] = - color_lim[0]
            # clim = 0.8 * np.nanstd(compdata[i, 40:-40, 40:-40])
            clim = (0.6 ** (i + 1)) * np.nanstd(compdata[i])
            # clim = np.nanmax(np.abs(compdata[i, 40:-40, 40:-40]))
            # compdata[i, 45:-45, 45:-45] = np.nan
            ax = fig.add_subplot(gs[i * 10 + 6 : i * 10 + 8])
            # ax = fig.add_subplot(gs[i * 5 + 2 : i * 5 + 4])

            img = ax.imshow(
                compdata[i, 20:-20, 20:-20],
                extent=(
                    x[20:-20].min(),
                    x[20:-20].max(),
                    y[20:-20].min(),
                    y[20:-20].max(),
                ),
                interpolation="none",
                cmap=cmap,
                origin="lower",
                vmin=-clim,
                vmax=clim,
                # vmin = 10 * color_lim[0],
                # vmax = 10 * color_lim[1],
                aspect="auto",
                rasterized=True,
            )

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.5)
            cbar = plt.colorbar(img, cax=cax, orientation="vertical", extend="both")
            # cax.yaxis.set_label_position("right")
            # cax.yaxis.set_ticks_position("right")
            cbar.set_label(r"$\mathrm{\mu K}$")

            ax.set_xticks(x[20:-20:10], minor=False)
            ax.set_xticklabels(xtickslabel, minor=False, rotation=90)
            ax.set_yticks(y[20:-20:10], minor=False)
            ax.set_yticklabels(ytickslabel, minor=False)
            ax.set_ylabel(r"Dec [deg]")
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()

            if i == 4:
                ax.set_xlabel(r"RA [deg]")
                # ax.set_xticklabels(ax.get_xticks(), rotation = 45)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)

        fig.tight_layout()
        fig.savefig(
            outpath + map.outname + "_pca_ampl_feed" + f"{feed + 1}" + ".pdf",
            bbox_inches="tight",
            dpi=300,
        )

    fontsize = 10
    fonts = {
        "font.family": "sans-serif",
        "axes.labelsize": fontsize - 4,
        "font.size": fontsize,
        "legend.fontsize": fontsize,
        "xtick.labelsize": fontsize - 4,
        "ytick.labelsize": fontsize - 4,
    }
    plt.rcParams.update(fonts)

    allampls = np.array(allampls)
    allcomps = np.array(allcomps)

    for comp in range(5):
        print(f"PCA mode # {comp}")
        comps = allcomps[:, comp, ...]
        ampls = allampls[:, comp, ...]

        fig = plt.figure(figsize=(14, 6.2))
        gs = gridspec.GridSpec(7, 3, figure=fig, wspace=0.05, hspace=0.25)
        grids = [gs[i].subgridspec(1, 5, wspace=1.5, hspace=0.05) for i in range(21)]

        xtickslabel = [
            f"{i:.1f}" for i in np.linspace(np.min(x[20:-20]), np.max(x[20:-20]), 3)
        ]
        ytickslabel = [
            f"{i:.1f}" for i in np.linspace(np.min(y[20:-20]), np.max(y[20:-20]), 3)
        ]

        clim = (0.7 ** (comp + 1)) * np.nanstd(comps[comp])

        amplim = 5 * clim
        # amplim = np.max(np.abs(ampls))

        for i in range(7):
            for j in range(3):
                idx = i * 3 + j

                if idx > 18:
                    break
                elif idx == 18:
                    ax = fig.add_subplot(grids[6 * 3 + 1][:3])
                elif i > 6:
                    continue
                else:
                    ax = fig.add_subplot(grids[idx][:3])

                if idx == 1:
                    ax.set_title(f"PCA mode # {comp}")
                # ax.scatter(freq.flatten(), ampls[idx, :], label=rf"Feed {idx + 1}", s=1)
                ax.plot(
                    freq.flatten(),
                    ampls[idx, :],
                    lw=0.2,
                    c="gray",
                    # alpha=0.2,
                )
                ax.scatter(
                    freq.flatten(), ampls[idx, :], label=rf"Feed {idx + 1}", s=1, c="b"
                )

                ax.legend(loc="upper left", fontsize=6)

                ax.set_ylim(-1.05 * amplim, 1.05 * amplim)

                if j == 0:
                    ax.set_ylabel(r"$\mathrm{\mu K}$")
                else:
                    plt.setp(ax.get_yticklabels(), visible=False)
                ax.set_xlim(np.min(freq), np.max(freq))

                if i >= 5:
                    if j != 1:
                        ax.set_xlabel(r"Frequency [GHz]")
                    else:
                        plt.setp(ax.get_xticklabels(), visible=False)
                else:
                    plt.setp(ax.get_xticklabels(), visible=False)

                if idx > 18:
                    break

                elif idx == 18:
                    ax = fig.add_subplot(grids[6 * 3 + 1][3:])
                elif i > 6 and idx != 18:
                    continue
                else:
                    ax = fig.add_subplot(grids[idx][3:])

                # color_lim[0] = np.nanstd(comps[idx, ...])
                # color_lim[1] = - color_lim[0]

                img = ax.imshow(
                    comps[idx, 20:-20, 20:-20],
                    extent=(
                        np.min(x[20:-20]),
                        np.max(x[20:-20]),
                        np.min(y[20:-20]),
                        np.max(y[20:-20]),
                    ),
                    interpolation="none",
                    cmap=cmap,
                    origin="lower",
                    aspect="auto",
                    vmin=-clim,
                    vmax=clim,
                    rasterized=True,
                )

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("left", size="5%", pad=0.05)
                cbar = plt.colorbar(img, cax=cax, orientation="vertical", extend="both")
                cax.yaxis.set_label_position("left")
                cax.yaxis.set_ticks_position("left")
                cbar.set_label(r"$\mathrm{\mu K}$", labelpad=-10)

                ax.set_xticks(
                    np.linspace(np.min(x[20:-20]), np.max(x[20:-20]), 3),
                    minor=False,
                )
                ax.set_xticklabels(xtickslabel, minor=False)
                ax.set_yticks(
                    np.linspace(np.min(y[20:-20]), np.max(y[20:-20]), 3),
                    minor=False,
                )
                ax.set_yticklabels(ytickslabel, minor=False)
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()

                if i == 5:
                    if j == 0 or j == 2:
                        ax.set_xlabel(r"RA [deg]")
                    else:
                        plt.setp(ax.get_xticklabels(), visible=False, rotation=90)

                elif i == 6:
                    if j == 0:
                        ax.set_xlabel(r"RA [deg]")
                    else:
                        plt.setp(ax.get_xticklabels(), visible=False, rotation=90)

                else:
                    plt.setp(ax.get_xticklabels(), visible=False, rotation=90)

                if j == 2:
                    ax.set_ylabel(r"Dec [deg]")
                elif j == 0:
                    if i == 6:
                        ax.set_ylabel(r"Dec [deg]")
                    else:
                        plt.setp(ax.get_yticklabels(), visible=False)

                else:
                    plt.setp(ax.get_yticklabels(), visible=False)

        fig.tight_layout()
        fig.savefig(
            outpath + map.outname + f"_pca_comp{comp}" + ".pdf",
            bbox_inches="tight",
            dpi=300,
        )
    # plt.show()


if __name__ == "__main__":
    # plt.show()

    # mappath = "/mn/stornext/d22/cmbco/comap/d16/protodir/maps/"
    # mappath = "/mn/stornext/d22/cmbco/comap/nils/COMAP_general/data/maps/new_pca/new/"
    mappath = "/mn/stornext/d22/cmbco/comap/nils/COMAP_general/data/maps/new_pca/"
    # mappath = "/mn/stornext/d22/cmbco/comap/nils/COMAP_general/data/maps/new_pca/old/"
    outpath = "/mn/stornext/d22/cmbco/comap/nils/comap_aux/plotbrowser/test_figs/mapplots/good_subset/"
    # outpath = "/mn/stornext/d16/cmbco/comap/nils/plotbrowser/test_figs/mapplots/"

    # obsid   = "012338"
    # obsid   = "012115"
    # obsid    = "015021"
    # obsid    = "008344"
    # obsid    = "013615"
    # obsid    = "007243"
    # obsid    = "006879"
    # obsid    = "007680"

    # field   = "co2"
    # name    = "map_test_high_ps_chi2_smoothed_downsampled"
    # name    = "map_test_high_ps_chi2_downsampled"
    # name    = "map_test_high_ps_chi2_smoothed"
    # name    = "map_test_high_ps_chi2"
    # name    = "co6_map_old_pca.h5"

    mapeditor = Atlas(no_init=True)

    """
    allmaps = os.listdir(mappath)

    #allmaps = [f for f in allmaps if re.match(r"co\d_\d{6}_\d{6}_map.h5", f)]
    #allmaps = [f for f in allmaps if re.match(r"co\d_\d{8}_map_old_pca.h5", f)]
    allmaps = [f for f in allmaps if re.match(r"co\d_\d{8}_map_old_pca_wo_lowelev.h5", f)]
    #allmaps = [f for f in allmaps if re.match(r"co\d_011837\d{2}_map_old_pca_wo_lowelev.h5", f)]

    #allmaps = [f for f in allmaps if re.match(r"co\d_map_old_pca.h5", f)]
    #allmaps = ["co7_016493_201229_map.h5"]
    print(allmaps)
    feeds = list(range(19))
    for name in tqdm(allmaps):
        obsid = name.split("_")[1]

        try:
            #feed    = 12
            print(name)
            scanmaps    = {}
            #obsname = mappath + f"{field}_{obsid}_{name}.h5"
            #obsname = mappath + f"co2_map_summer.h5"
            #obsname = f"co2_map_summer_liss.h5"
            #obsname = f"co6_map_signal_new_data_liss.h5"
            #obsmap  = Map(name)
            #scans   = range(2, 16)

            mapobj = Map(mappath + name)
            mapobj.read_map()
            #mapobj.outname = f"{field}_{obsid}_{name}_feed{feed}"

            #mapobj.outname = f"co2_map_summer_liss"
            #print(name.split(".h5")[0])
            mapobj.outname = name.split(".h5")[0]
            mapobj.scanmap = False
            mapobj.obsmap  = True
            mapobj.obsid   = int(obsid)
            #plot_skewness_maps(mapobj, feed = feed)
            if np.all(mapobj.nhit == 0):
                print("Map:", name, "is empty!")
                continue

            print(mapobj.map.shape)

            mapeditor.merge_numXY = 2
            mapeditor.C_dgradeXY5D(mapobj.map, mapobj.nhit, mapobj.rms)
            mapobj.map, mapobj.nhit, mapobj.rms = mapeditor.map, mapeditor.nhit, mapeditor.rms
            x = mapobj.x.reshape(int(len(mapobj.x) / mapeditor.merge_numXY), mapeditor.merge_numXY) 
            y = mapobj.y.reshape(int(len(mapobj.y) / mapeditor.merge_numXY), mapeditor.merge_numXY)
            mapobj.x      = np.mean(x, axis = 1)   # Finding new pixel center by averaging neighboring pixel x coordinates
            mapobj.y      = np.mean(y, axis = 1)   # Finding new pixel center by averaging neighboring pixel x coordinates

            mapobj.n_x,  mapobj.n_y = mapobj.map.shape[3:]

            plot_pca_maps(mapobj, feeds = feeds)
        except:
            continue
    sys.exit()
    """

    feed = 5
    feeds = list(range(19))
    field = "co6"
    scanmaps = {}
    # obsname = mappath + f"{field}_{obsid}_{name}.h5"
    # obsname = mappath + f"co6_map_old_pca.h5"
    # obsname = mappath + f"co6_map_new_pca.h5"
    # obsname = mappath + f"co6_map_new_pca_wo_highelev.h5"
    # obsname = mappath + f"co6_map_old_pca_wo_lowelev.h5"
    # obsname = mappath + f"co6_map_new_pca_with_downsamp.h5"

    # obsname = mappath + f"co6_map_new_pca_wo_downsamp.h5"
    # obsname = mappath + f"co6_map_new_pca_downsamp_corrclean.h5"
    # obsname = mappath + f"co2_map_good_az_edge_cut.h5"
    # obsname = mappath + f"co2_map_good_diff_pca_params.h5"
    obsname = mappath + f"co2_map_good_wo_highpass.h5"
    # obsname = mappath + f"co2_map_summer.h5"
    # obsname = f"co2_map_summer_liss.h5"
    # obsname = f"co6_map_signal_new_data_liss.h5"
    obsmap = Map(obsname)
    scans = range(2, 16)

    mapobj = Map(obsname)
    mapobj.read_map()
    # mapobj.outname = f"co6_map_old_pca_v3"
    # mapobj.outname = f"co6_map_new_pca_wo_highelev"
    # mapobj.outname = f"co6_map_new_pca_with_downsamp"
    # mapobj.outname = f"co6_map_new_pca_wo_downsamp"
    # mapobj.outname = f"co2_map_good_diff_pca_params"
    mapobj.outname = f"co2_map_good_wo_highpass"

    # mapobj.outname = f"co2_map_summer_liss"
    # mapobj.outname = f"co2_map_good_diff_pca_params"
    # mapobj.outname = f"co6_map_signal_new_data_liss"
    mapobj.scanmap = False
    mapobj.obsmap = True
    # mapobj.obsid   = int(obsid)
    mapobj.obsid = None

    # plot_skewness_maps(mapobj, feed = feed)
    plot_pca_maps(mapobj, outpath, feeds=feeds)

    sys.exit()

    mapeditor.merge_numZ = 16
    mapeditor.C_dgradeZ5D(mapobj.map, mapobj.nhit.astype(np.int32), mapobj.rms)
    mapobj.map, mapobj.nhit, mapobj.rms = mapeditor.map, mapeditor.nhit, mapeditor.rms

    climabs = 1e6 * 0.2 * np.max(np.abs(mapobj.map / mapobj.rms))
    plot_sb_avg_maps(mapobj, color_lim=[0, climabs], feed=feed)

    clim = 1e6 * np.nanstd(np.where(mapobj.nhit > 1, mapobj.map, np.nan))
    plot_quarter_sb_avg_maps(mapobj, color_lim=[-clim, clim], feed=feed)

    plot_quarter_sb_avg_column_maps(mapobj)

    # sys.exit()

    for i in tqdm(scans):
        scanid = f"{obsid}{i:02}"
        scanname = mappath + f"{field}_{scanid}_{name}.h5"
        try:
            mapobj = Map(scanname)
            mapobj.read_map()
            mapobj.outname = f"{field}_{scanid}_{name}"
            mapobj.scanmap = True
            mapobj.obsmap = False
            mapobj.obsid = int(obsid)
            mapobj.scanid = int(scanid)

            scanmaps[scanid] = mapobj

        except:
            print(f"No map found for scan {scanid}!")
            continue

        if np.all(mapobj.map[feed, ...] == 0):
            print("All map elements are zero!")
            continue

        if np.all(mapobj.nhit[feed, ...] == 0):
            print("All nhit elements are zero!")
            continue

        if np.all(mapobj.rms[feed, ...] == 0):
            print("All rms elements are zero!")
            continue

        # plot_pca_maps(mapobj, feed = feed)

        mapeditor.merge_numXY = 4
        mapeditor.merge_numZ = 16
        mapeditor.C_dgradeXYZ5D(mapobj.map, mapobj.nhit, mapobj.rms)
        mapobj.map, mapobj.nhit, mapobj.rms = (
            mapeditor.map,
            mapeditor.nhit,
            mapeditor.rms,
        )

        mapeditor.n_sigma = 1
        mapeditor.sigmaX = 8e-1
        mapeditor.sigmaY = 8e-1

        mapeditor.gaussian_smoothXY(mapobj.map, mapobj.nhit, mapobj.rms)
        mapobj.map, mapobj.nhit, mapobj.rms = (
            mapeditor.map,
            mapeditor.nhit,
            mapeditor.rms,
        )

        (
            mapobj.n_det,
            mapobj.n_sb,
            mapobj.n_freq,
            mapobj.n_x,
            mapobj.n_y,
        ) = mapobj.map.shape

        x = mapobj.x.reshape(
            int(len(mapobj.x) / mapeditor.merge_numXY), mapeditor.merge_numXY
        )
        y = mapobj.y.reshape(
            int(len(mapobj.y) / mapeditor.merge_numXY), mapeditor.merge_numXY
        )
        mapobj.x = np.mean(
            x, axis=1
        )  # Finding new pixel center by averaging neighboring pixel x coordinates
        mapobj.y = np.mean(
            y, axis=1
        )  # Finding new pixel center by averaging neighboring pixel x coordinates

        plot_sb_avg_maps(mapobj, color_lim=[0, climabs], feed=feed)
        plot_quarter_sb_avg_maps(mapobj, color_lim=[-clim, clim], feed=feed)
