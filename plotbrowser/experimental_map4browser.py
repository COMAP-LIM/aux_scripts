from operator import inv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA

from cmcrameri import cm
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy

import sys
import numpy.fft as fft
import os
import errno
import h5py
from scipy import stats
from scipy.stats import norm, skew
import copy

import argparse
import re 
from tqdm import tqdm
import warnings
import time as tm



import accept_mod.stats_list as stats_list
stats_list = stats_list.stats_list

from spikes import spike_data, spike_list, get_spike_list
from mapdata import Map

warnings.filterwarnings("ignore", category = RuntimeWarning) # Ignore warnings caused by mask nan/inf weights
warnings.filterwarnings("ignore", category = UserWarning)    # Ignore warning when producing log plot of empty array

index = stats_list.index('ps_s_chi2')

with h5py.File("../accept_mod/scan_data_complete_co2.h5", "r") as infile:
    ps_chi2 = infile["scan_data"][:, 0, 0, index]
    scan_list = infile["scan_list"][()]


idx_max = np.where(ps_chi2 == np.nanmax(ps_chi2))[0]
max_val = np.nanmax(ps_chi2)

idx_min = np.where(ps_chi2 == np.nanmin(ps_chi2))[0]
min_val = np.nanmin(ps_chi2)

idx_mean = np.where(ps_chi2 == ps_chi2[np.argmin(np.abs(ps_chi2 - np.mean(ps_chi2)))])
mean_val = np.mean(ps_chi2)


print(ps_chi2)
print(scan_list[idx_max], idx_max, max_val)
print(scan_list[idx_min], idx_min, min_val)
print(scan_list[idx_mean], idx_mean, mean_val)

fig, ax = plt.subplots()

ax.scatter(scan_list, ps_chi2, s = 1)
ax.scatter(scan_list[idx_max], max_val)
ax.scatter(scan_list[idx_min], min_val)
ax.scatter(scan_list[idx_mean], mean_val)
ax.scatter(1211506, ps_chi2[scan_list == 1211506])


def plot_sb_avg_maps(map, color_lim = [None, None], feed = None):
    
    x_lim, y_lim = [None,None], [None,None]
    
    x, y     = map.x, map.y
    dx       = (x[1] - x[0]) / np.cos(np.radians(np.mean(y)))
    x_lim[0] = x[0] - 0.5 * dx
    x_lim[1] = x[-1] + 0.5 * dx
    dy       = y[1] - y[0]
    y_lim[0] = y[1] - 0.5 * dy
    y_lim[1] = y[-1] + 0.5 * dy

    if feed == None:
        mapdata  = map.map_coadd[...].astype(np.float64)
        nhitdata = map.nhit_coadd[...].astype(np.float64)
        rmsdata  = map.rms_coadd[...].astype(np.float64)
    else:
        mapdata  = map.map[feed, ...].astype(np.float64)
        nhitdata = map.nhit[feed, ...].astype(np.float64)
        rmsdata  = map.rms[feed, ...].astype(np.float64)

    nhitdata = np.where(nhitdata > 1, nhitdata, np.nan)
    rmsdata  = np.where(nhitdata > 1, rmsdata, np.nan)

    inv_var = 1 / rmsdata ** 2

    mapdata *= inv_var
    where = np.nanprod(nhitdata, axis = 1) > 1
    
    mapdata = np.nansum(mapdata, axis = 1)     # Coadding over freq channels per sideband
    mapdata[where] /= np.nansum(inv_var, axis = 1)[where]  
    
    mapdata  = np.where(where == True, mapdata, np.nan)
    #mapdata[where == False] = np.nan * mapdata[where == False]
    mapdata *= 1e6                          # K to muK

    fontsize = 16
    fonts = {
    "font.family": "sans-serif",
    "axes.labelsize": fontsize,
    "font.size": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize
    }
    plt.rcParams.update(fonts)

    cmap = cm.roma.reversed()
    #cmap.set_bad("0", 1) # Set color of masked elements to black.
    cmap.set_bad("0.8", 1) # Set color of masked elements to gray.
    #cmap.set_bad("1", 1) # Set color of masked elements to white.
    
    fig, ax = plt.subplots(2, 2, figsize=(16, 14), sharex = True, sharey = True)

    sb_name = ["A:LSB", "A:USB", "B:LSB", "B:USB"]

    ax[0, 0].set_ylabel('Declination [deg]')
    ax[1, 0].set_ylabel('Declination [deg]')
    ax[1, 0].set_xlabel('Right Ascension [deg]')
    ax[1, 1].set_xlabel('Right Ascension [deg]')

    if map.scanmap:
        fig.suptitle(f"Sideband avg. Scanid: {map.scanid}")
    elif map.obsmap: 
        fig.suptitle(f"Sideband avg. Obsid: {map.obsid}")

    for i in range(2):
        for j in range(2):
            sb = i * 2 + j 

            #if color_lim[0] is None or color_lim[1] is None:
            min = np.abs(np.nanmin(mapdata[sb, ...]))
            max = np.nanmax((np.nanmax(mapdata[sb, ...]), min))
            #color_lim[1] = 0.01 * max #0.4 * np.nanstd(mapdata[sb, ...])
            color_lim[1] = 0.4 * np.nanstd(mapdata[sb, ...])
            color_lim[0] = - color_lim[1]
        
            aspect = dx / dy
            img = ax[i, j].imshow(mapdata[sb, ...], extent = (x_lim[0], x_lim[1], y_lim[0], y_lim[1]), interpolation = 'nearest',
                                aspect = aspect, cmap = cmap, origin = 'lower',
                                vmin = color_lim[0], vmax = color_lim[1], rasterized = True)
            ax[i, j].set_title(f"{sb_name[sb]}")

            divider = make_axes_locatable(ax[i, j])
            cax = divider.append_axes("right", size="5%", pad=0.05)
 
            cbar = fig.colorbar(img, ax = ax[i, j], cax = cax)
            
            cbar.set_label("$\mu K$")

    fig.tight_layout()
    fig.savefig(outpath + "sb_avg/" + map.outname + "_sb_avg" + ".png", bbox_inches='tight')

def plot_quarter_sb_avg_maps(map, color_lim = [None, None], feed = None):
    
    x_lim, y_lim = [None,None], [None,None]
    
    x, y     = map.x, map.y
    dx       = (x[1] - x[0]) / np.cos(np.radians(np.mean(y)))
    x_lim[0] = x[0] - 0.5 * dx
    x_lim[1] = x[-1] + 0.5 * dx
    dy       = y[1] - y[0]
    y_lim[0] = y[1] - 0.5 * dy
    y_lim[1] = y[-1] + 0.5 * dy

    
    if feed == None:
        mapdata  = map.map_coadd[...].astype(np.float64)
        nhitdata = map.nhit_coadd[...].astype(np.float64)
        rmsdata  = map.rms_coadd[...].astype(np.float64)
    else:
        mapdata  = map.map[feed, ...].astype(np.float64)
        nhitdata = map.nhit[feed, ...].astype(np.float64)
        rmsdata  = map.rms[feed, ...].astype(np.float64)

    nhitdata = np.where(nhitdata > 1, nhitdata, np.nan)
    rmsdata  = np.where(nhitdata > 1, rmsdata, np.nan)

    inv_var = 1 / rmsdata ** 2

    mapdata *= inv_var

    fontsize = 16
    fonts = {
    "font.family": "sans-serif",
    "axes.labelsize": fontsize - 4,
    "font.size": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize - 4,
    "ytick.labelsize": fontsize - 4
    }
    plt.rcParams.update(fonts)

    cmap = cm.roma.reversed()
    #cmap.set_bad("0", 1) # Set color of masked elements to black.
    cmap.set_bad("0.8", 1) # Set color of masked elements to gray.
    #cmap.set_bad("1", 1) # Set color of masked elements to white.
    
    sb_name = ["A:LSB", "A:USB", "B:LSB", "B:USB"]

    fig = plt.figure(figsize = (16, 14))
    gs = gridspec.GridSpec(2, 2, figure = fig, wspace = 0.15, hspace = 0.25)

    grids = [gs[i].subgridspec(2, 2, wspace = 0.0, hspace = 0.25) for i in range(4)]

    if map.scanmap:
        fig.suptitle(f"Quarter sideband avg. Scanid: {map.scanid}")
    elif map.obsmap: 
        fig.suptitle(f"Quarter sideband avg. Obsid: {map.obsid}")

    #axes = {}
    aspect = dx / dy
    for i in range(4):
        #for j in range(2):
        #    for k in range(2):
        ax_dummy = fig.add_subplot(grids[i][:, :])
        ax_dummy.axis("off")
        ax_dummy.set_title(" " * 6 + sb_name[i] + "\n")
            
        where = np.nanprod(nhitdata[i, 32:48, ...], axis = 0) > 1

        data = np.nansum(mapdata[i, 32:48, ...], axis = 0)     # Coadding over freq channels per sideband
        data[where] /= np.nansum(inv_var[i, 32:48, ...], axis = 0)[where]  
        
        data  = np.where(where == True, data, np.nan)
        #data[where == False] = np.nan * data[where == False]
        data *= 1e6                          # K to muK

        ax2 = fig.add_subplot(grids[i][1, 0])
        ax2.set_title(f"Ch: 33-48 avg. \n freq: {map.freq[i, 32]:.2f}-{map.freq[i, 47]:.2f} GHz", fontsize = 14)
        #axes[f"{i}{0}{0}"] = ax

        color_lim[1] = 0.6 * np.nanstd(data)
        color_lim[0] = - color_lim[1]

        img = ax2.imshow(data, extent = (x_lim[0], x_lim[1], y_lim[0], y_lim[1]), interpolation = 'nearest',
                            aspect = aspect, cmap = cmap, origin = 'lower',
                            vmin = color_lim[0], vmax = color_lim[1], rasterized = True)

        #divider = make_axes_locatable(ax2)
        #cax = divider.append_axes("right", size="5%", pad=0.05)

        #cbar = fig.colorbar(img, ax = ax2, cax = cax)
        
        #cbar.set_label("$\mu K$")

        #cbar.ax.tick_params(rotation = 90)
        
        where = np.nanprod(nhitdata[i, 0:16, ...], axis = 0) > 1

        data = np.nansum(mapdata[i, 0:16, ...], axis = 0)     # Coadding over freq channels per sideband
        data[where] /= np.nansum(inv_var[i, 0:16, ...], axis = 0)[where]  
        
        data  = np.where(where == True, data, np.nan)
        #data[where == False] = np.nan * data[where == False]
        data *= 1e6                          # K to muK

        ax1 = fig.add_subplot(grids[i][0, 0], sharex = ax2)
        ax1.set_title(f"C:. 1-16 avg. \n freq: {map.freq[i, 0]:.2f}-{map.freq[i, 15]:.2f} GHz", fontsize = 14)
        plt.setp(ax1.get_xticklabels(), visible=False)
        #axes[f"{i}{0}{0}"] = ax



        img = ax1.imshow(data, extent = (x_lim[0], x_lim[1], y_lim[0], y_lim[1]), interpolation = 'nearest',
                            aspect = aspect, cmap = cmap, origin = 'lower',
                            vmin = color_lim[0], vmax = color_lim[1], rasterized = True)

        #divider = make_axes_locatable(ax1)
        #cax = divider.append_axes("right", size="5%", pad=0.05)

        #cbar = fig.colorbar(img, ax = ax1, cax = cax)
        
        #cbar.set_label("$\mu K$")

        #cbar.ax.tick_params(rotation = 90)

        where = np.nanprod(nhitdata[i, 48:64, ...], axis = 0) > 1

        data = np.nansum(mapdata[i, 48:64, ...], axis = 0)     # Coadding over freq channels per sideband
        data[where] /= np.nansum(inv_var[i, 48:64, ...], axis = 0)[where]  
        
        data  = np.where(where == True, data, np.nan)
        #data[where == False] = np.nan * data[where == False]
        data *= 1e6                          # K to muK


        ax3 = fig.add_subplot(grids[i][1, 1], sharey = ax2)
        ax3.set_title(f"Ch: 49-64 avg. \n freq: {map.freq[i, 48]:.2f}-{map.freq[i, 63]:.2f} GHz", fontsize = 14)
        #axes[f"{i}{0}{0}"] = ax
        plt.setp(ax3.get_yticklabels(), visible=False)


        img = ax3.imshow(data, extent = (x_lim[0], x_lim[1], y_lim[0], y_lim[1]), interpolation = 'nearest',
                            aspect = aspect, cmap = cmap, origin = 'lower',
                            vmin = color_lim[0], vmax = color_lim[1], rasterized = True)

        #divider = make_axes_locatable(ax3)
        #cax = divider.append_axes("right", size="5%", pad=0.05)

        #cbar = fig.colorbar(img, ax = ax3, cax = cax)
        
        #cbar.set_label("$\mu K$")

        #cbar.ax.tick_params(rotation = 90)

        where = np.nanprod(nhitdata[i, 16:32, ...], axis = 0) > 1

        data = np.nansum(mapdata[i, 16:32, ...], axis = 0)     # Coadding over freq channels per sideband
        data[where] /= np.nansum(inv_var[i, 16:32, ...], axis = 0)[where]  
        
        data  = np.where(where == True, data, np.nan)
        #data[where == False] = np.nan * data[where == False]
        data *= 1e6                          # K to muK

        ax4 = fig.add_subplot(grids[i][0, 1], sharex = ax3, sharey = ax1)
        ax4.set_title(f"Ch: 17-32 avg. \n freq: {map.freq[i, 16]:.2f}-{map.freq[i, 31]:.2f} GHz", fontsize = 14)
        plt.setp(ax4.get_yticklabels(), visible=False)
        plt.setp(ax4.get_xticklabels(), visible=False)
    
        img = ax4.imshow(data, extent = (x_lim[0], x_lim[1], y_lim[0], y_lim[1]), interpolation = 'nearest',
                            aspect = aspect, cmap = cmap, origin = 'lower',
                            vmin = color_lim[0], vmax = color_lim[1], rasterized = True)

        divider = make_axes_locatable(ax_dummy)
        cax = divider.append_axes("right", size="3%", pad=0.4)

        cbar = fig.colorbar(img, ax = ax_dummy, cax = cax)
        
        cbar.set_label("$\mu K$")

        cbar.ax.tick_params(rotation = 90)

        ax1.set_ylabel('Declination [deg]')
        ax2.set_xlabel('Right Ascension [deg]')
        ax2.set_ylabel('Declination [deg]')
        ax3.set_xlabel('Right Ascension [deg]')

    #fig.tight_layout()
    fig.savefig(outpath + "quarter_sb_avg/" + map.outname + "_quater_sb_avg" + ".png", bbox_inches='tight')
    
def plot_quarter_sb_avg_column_maps(map, color_lim = [None, None], feed = None):
    
    x_lim, y_lim = [None,None], [None,None]
    
    x, y     = map.x, map.y
    dx       = (x[1] - x[0]) / np.cos(np.radians(np.mean(y)))
    x_lim[0] = x[0] - 0.5 * dx
    x_lim[1] = x[-1] + 0.5 * dx
    dy       = y[1] - y[0]
    y_lim[0] = y[1] - 0.5 * dy
    y_lim[1] = y[-1] + 0.5 * dy

    
    if feed == None:
        mapdata  = map.map_coadd[...].astype(np.float64)
        nhitdata = map.nhit_coadd[...].astype(np.float64)
        rmsdata  = map.rms_coadd[...].astype(np.float64)
    else:
        mapdata  = map.map[feed, ...].astype(np.float64)
        nhitdata = map.nhit[feed, ...].astype(np.float64)
        rmsdata  = map.rms[feed, ...].astype(np.float64)

    mapdata = np.where(nhitdata > 1, mapdata, np.nan)
    nhitdata = np.where(nhitdata > 1, nhitdata, np.nan)
    rmsdata  = np.where(nhitdata > 1, rmsdata, np.nan)

    mapdata = mapdata.reshape(map.n_sb * map.n_freq, map.n_x, map.n_y, order = "F")
    nhitdata = nhitdata.reshape(map.n_sb * map.n_freq, map.n_x, map.n_y, order = "F")
    rmsdata  = rmsdata.reshape(map.n_sb * map.n_freq, map.n_x, map.n_y, order = "F")

    mapdata *= 1e6

    freq = map.freq
    dfreq = 1e3 * (freq[0, 1] - freq[0, 0])
    freq = freq.reshape(map.n_sb * map.n_freq, order = "F")

    fontsize = 12
    fonts = {
    "font.family": "sans-serif",
    "axes.labelsize": fontsize - 4,
    "font.size": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize - 4,
    "ytick.labelsize": fontsize - 4
    }
    plt.rcParams.update(fonts)

    cmap = cm.roma.reversed()
    #cmap.set_bad("0", 1) # Set color of masked elements to black.
    #cmap.set_bad("0.8", 1) # Set color of masked elements to gray.
    cmap.set_bad("1", 1) # Set color of masked elements to white.
    
    sb_name = ["A:LSB", "A:USB", "B:LSB", "B:USB"]

    fig = plt.figure(figsize = (map.n_sb, map.n_freq))
    gs = gridspec.GridSpec(map.n_freq, map.n_sb, figure = fig, wspace = 0.0, hspace = 0.0)

    if feed == None:
        feedname = "All"
    if map.scanmap:
        fig.suptitle(f"Scanid: {map.scanid} | Feed: All\nBandwidth: {dfreq:.2f} MHz"
                      + fr"(RA,Dec)=({np.mean(x[10:-20]):.2f}$^\circ$,{np.mean(y[10:-20]):.2f}$^\circ$)" + "\n"
                      + fr"RA$\times$Dec={x[-20] - x[10]:.2f}$^\circ\times${y[-20] - y[10]:.2f}$^\circ$")
    elif map.obsmap: 
        fig.suptitle(f"Obsid: {map.obsid} | Feed: All\nBandwidth: {dfreq:.2f} MHz\n"
                      + fr"(RA,Dec)=({np.mean(x[10:-20]):.2f}$^\circ$,{np.mean(y[10:-20]):.2f}$^\circ$)" + "\n"
                      + fr"RA$\times$Dec={x[-20] - x[10]:.2f}$^\circ\times${y[-20] - y[10]:.2f}$^\circ$")

    axes = []
    aspect = dx / dy

    color_lim[1] = 0.6 * np.nanstd(mapdata)
    color_lim[0] = - color_lim[1]
    
    ax_dummy = fig.add_subplot(gs[-1, :])
    ax_dummy.axis("off")

    for i in range(16):
        for j in range(4):
            idx = i * 4 + j
            ax = fig.add_subplot(gs[idx])
            axes.append(ax)
            
            img = ax.imshow(mapdata[idx, 10:-20, 10:-20], extent = (x_lim[0], x_lim[1], y_lim[0], y_lim[1]), interpolation = 'nearest',
                            aspect = aspect, cmap = cmap, origin = 'lower',
                            vmin = color_lim[0], vmax = color_lim[1], rasterized = True)
            plt.setp(ax.get_yticklabels(), visible=False)
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.text(0.02, 0.05, f"{freq[idx]:.2f} GHz", fontsize = 6, transform=ax.transAxes)
    
    axes[0].set_title(sb_name[1])
    axes[1].set_title(sb_name[1])
    axes[2].set_title(sb_name[2])
    axes[3].set_title(sb_name[3])

    #divider = make_axes_locatable(ax_dummy)
    #cax = divider.append_axes("bottom", size = "2%", pad = 10)

    cax = inset_axes(ax_dummy,
                    width="100%",  # width = 50% of parent_bbox width
                    height="10%",  # height : 5%
                    loc='lower center',
                    borderpad = - 1)#,
                    #bbox_to_anchor = (0, -0.01, 1, 1),
                    #bbox_transform = ax.transAxes)
    
    cbar = fig.colorbar(img, ax = ax_dummy, cax = cax, orientation = "horizontal")
    
    cbar.set_label("$\mu K$")


    cax.xaxis.set_label_position('bottom')
    #cbar.ax.tick_params(rotation = 90)   
    fig.tight_layout()
    fig.savefig(outpath + "quarter_sb_avg_column/" + map.outname + "_quater_sb_avg" + ".png", bbox_inches='tight', dpi = 300)
    plt.show()

def plot_pca_maps(map, color_lim = [None, None], feed = None):
    
    x_lim, y_lim = [None,None], [None,None]
    
    x, y     = map.x, map.y
    dx       = (x[1] - x[0]) / np.cos(np.radians(np.mean(y)))
    x_lim[0] = x[0] - 0.5 * dx
    x_lim[1] = x[-1] + 0.5 * dx
    dy       = y[1] - y[0]
    y_lim[0] = y[1] - 0.5 * dy
    y_lim[1] = y[-1] + 0.5 * dy

    
    if feed == None:
        mapdata  = map.map_coadd[...].astype(np.float64)
        nhitdata = map.nhit_coadd[...].astype(np.float64)
        rmsdata  = map.rms_coadd[...].astype(np.float64)
    else:
        mapdata  = map.map[feed, ...].astype(np.float64)
        nhitdata = map.nhit[feed, ...].astype(np.float64)
        rmsdata  = map.rms[feed, ...].astype(np.float64)
    mapdata *= 1e6

    pca      = PCA()
    pca_data = pca.fit(mapdata.reshape(map.n_sb * map.n_freq, map.n_x * map.n_y))
    comps    = pca_data.components_
    comps    = comps.reshape(comps.shape[0], map.n_x, map.n_y)
    
    comps = np.where(np.nansum(nhitdata, axis = (0, 1)) > 1, comps, np.nan)
    
    print(comps.shape)
    #mapdata = np.where(nhitdata > 1, mapdata, np.nan)
    #nhitdata = np.where(nhitdata > 1, nhitdata, np.nan)
    #rmsdata  = np.where(nhitdata > 1, rmsdata, np.nan)

    #mapdata = mapdata.reshape(map.n_sb * map.n_freq, map.n_x, map.n_y, order = "F")
    #nhitdata = nhitdata.reshape(map.n_sb * map.n_freq, map.n_x, map.n_y, order = "F")
    #rmsdata  = rmsdata.reshape(map.n_sb * map.n_freq, map.n_x, map.n_y, order = "F")


    freq = map.freq
    dfreq = 1e3 * (freq[0, 1] - freq[0, 0])
    #print("dfreq", dfreq)
    #freq = freq.reshape(map.n_sb * map.n_freq, order = "F")

    fontsize = 12
    fonts = {
    "font.family": "sans-serif",
    "axes.labelsize": fontsize - 4,
    "font.size": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize - 4,
    "ytick.labelsize": fontsize - 4
    }
    plt.rcParams.update(fonts)

    cmap = cm.roma.reversed()
    #cmap.set_bad("0", 1) # Set color of masked elements to black.
    #cmap.set_bad("0.8", 1) # Set color of masked elements to gray.
    cmap.set_bad("1", 1) # Set color of masked elements to white.
    
    sb_name = ["A:LSB", "A:USB", "B:LSB", "B:USB"]
    
    color_lim[1] = 0.5 * np.nanstd(comps[0, ...])
    color_lim[0] = - color_lim[1]

    fig, ax = plt.subplots(5, 5, figsize = (16, 14))
    aspect = dx / dy
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            img = ax[i, j].imshow(comps[idx, ...], extent = (x_lim[0], x_lim[1], y_lim[0], y_lim[1]), interpolation = 'nearest',
                            aspect = aspect, cmap = cmap, origin = 'lower',
                            vmin = color_lim[0], vmax = color_lim[1], rasterized = True)
    fig.tight_layout()
    #fig.savefig(outpath + "quarter_sb_avg_column/" + map.outname + "_hist" + ".png", bbox_inches='tight', dpi = 300)
    plt.show()

def plot_skewness_maps(map, color_lim = [None, None], feed = None):
    
    x_lim, y_lim = [None,None], [None,None]
    
    x, y     = map.x, map.y
    dx       = (x[1] - x[0]) / np.cos(np.radians(np.mean(y)))
    x_lim[0] = x[0] - 0.5 * dx
    x_lim[1] = x[-1] + 0.5 * dx
    dy       = y[1] - y[0]
    y_lim[0] = y[1] - 0.5 * dy
    y_lim[1] = y[-1] + 0.5 * dy

    
    if feed == None:
        mapdata  = map.map_coadd[...].astype(np.float64)
        nhitdata = map.nhit_coadd[...].astype(np.float64)
        rmsdata  = map.rms_coadd[...].astype(np.float64)
    else:
        mapdata  = map.map[feed, ...].astype(np.float64)
        nhitdata = map.nhit[feed, ...].astype(np.float64)
        rmsdata  = map.rms[feed, ...].astype(np.float64)
    mapdata *= 1e6

    mapdata = np.where(nhitdata > 1, mapdata, np.nan)
    nhitdata = np.where(nhitdata > 1, nhitdata, np.nan)
    rmsdata  = np.where(nhitdata > 1, rmsdata, np.nan)

    skewness = skew(mapdata, axis = 1, nan_policy = "omit")

    skewness = np.where(np.nansum(nhitdata, axis = 1) > 1, skewness, np.nan)

    #mapdata = mapdata.reshape(map.n_sb * map.n_freq, map.n_x, map.n_y, order = "F")
    #nhitdata = nhitdata.reshape(map.n_sb * map.n_freq, map.n_x, map.n_y, order = "F")
    #rmsdata  = rmsdata.reshape(map.n_sb * map.n_freq, map.n_x, map.n_y, order = "F")


    freq = map.freq
    dfreq = 1e3 * (freq[0, 1] - freq[0, 0])
    #print("dfreq", dfreq)
    #freq = freq.reshape(map.n_sb * map.n_freq, order = "F")

    fontsize = 12
    fonts = {
    "font.family": "sans-serif",
    "axes.labelsize": fontsize - 4,
    "font.size": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize - 4,
    "ytick.labelsize": fontsize - 4
    }
    plt.rcParams.update(fonts)

    cmap = cm.roma.reversed()
    #cmap.set_bad("0", 1) # Set color of masked elements to black.
    #cmap.set_bad("0.8", 1) # Set color of masked elements to gray.
    cmap.set_bad("1", 1) # Set color of masked elements to white.
    
    sb_name = ["A:LSB", "A:USB", "B:LSB", "B:USB"]
    
    color_lim[1] = 0.5 * np.nanstd(comps[0, ...])
    color_lim[0] = - color_lim[1]

    fig, ax = plt.subplots(5, 5, figsize = (16, 14))
    aspect = dx / dy
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            img = ax[i, j].imshow(comps[idx, ...], extent = (x_lim[0], x_lim[1], y_lim[0], y_lim[1]), interpolation = 'nearest',
                            aspect = aspect, cmap = cmap, origin = 'lower',
                            vmin = color_lim[0], vmax = color_lim[1], rasterized = True)
    fig.tight_layout()
    #fig.savefig(outpath + "quarter_sb_avg_column/" + map.outname + "_hist" + ".png", bbox_inches='tight', dpi = 300)
    plt.show()

mappath = "/mn/stornext/d16/cmbco/comap/protodir/maps/"
outpath = "/mn/stornext/d16/cmbco/comap/nils/plotbrowser/test_figs/mapplots/"

#obsid   = "012338"
obsid   = "012115"
#obsid    = "015021"

field   = "co2"
#name    = "map_test_high_ps_chi2_smoothed_downsampled"
#name    = "map_test_high_ps_chi2_downsampled"
#name    = "map_test_high_ps_chi2_smoothed"
name    = "map_test_high_ps_chi2"

scanmaps    = {}
obsname = mappath + f"{field}_{obsid}_{name}.h5"
obsmap  = Map(obsname)
scans   = range(2, 16)

mapobj = Map(obsname)
mapobj.read_map()
mapobj.outname = f"{field}_{obsid}_{name}"
mapobj.scanmap = False
mapobj.obsmap  = True
mapobj.obsid   = int(obsid)

#plot_sb_avg_maps(mapobj)
#plot_quarter_sb_avg_maps(mapobj)

print(mapobj.freq)

#plot_quarter_sb_avg_column_maps(mapobj)
plot_pca_maps(mapobj)
sys.exit()

for i in tqdm(scans):
    scanid = f"{obsid}{i:02}"
    scanname = mappath + f"{field}_{scanid}_{name}.h5"
    try:
        mapobj = Map(scanname)
        mapobj.read_map()
        mapobj.outname = f"{field}_{scanid}_{name}"
        mapobj.scanmap = True
        mapobj.obsmap  = False
        mapobj.obsid   = int(obsid)
        mapobj.scanid  = int(scanid)

        scanmaps[scanid] = mapobj

    except:
        print(f"No map found for scan {scanid}!")
        continue

    """if np.all(mapobj.map[15, ...] == 0):
        print("All map elements are zero!")
        continue

    if np.all(mapobj.nhit[15, ...] == 0):
        print("All nhit elements are zero!")
        continue

    if np.all(mapobj.rms[15, ...] == 0):
        print("All rms elements are zero!")
        continue
    """
    plot_sb_avg_maps(mapobj)
    plot_quarter_sb_avg_maps(mapobj)


