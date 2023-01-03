import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy.fft as fft
import time
import glob
import os
import h5py
import multiprocessing
from scipy.io import FortranFile
from scipy import stats
import astropy.coordinates as coord
from astropy.time import Time
from scipy import interpolate
import astropy.units as u
from astropy.coordinates import SkyCoord
import subprocess
import scipy as sp
import scipy.ndimage
from astropy.io import fits
from scipy.stats import norm
from scipy.optimize import curve_fit
import copy
import math
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body, get_moon
import healpy as hp
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from chainconsumer import ChainConsumer

import accept_mod.stats_list as stats_list

filename = 'scan_data_test_co7.h5'
# filename = 'scan_data_test_co2.h5'
# filename = 'scan_data_good_new_co6.h5'
# filename = 'scan_data_NCP.h5'
# filename2 = 'scan_data_NCP_VANE.h5'
with h5py.File(filename, mode="r") as my_file:
    data = np.array(my_file['scan_data'][()])
    scanids = np.array(my_file['scan_list'][()])
print(data.shape)

n_scans, n_feed, n_sb, n_data = data.shape

#data = data.reshape(n_scans * n_feed * n_sb, n_data)
stats_list = stats_list.stats_list
print(len(stats_list))
index = stats_list.index('ps_s_sb_chi2')
ps_chi2 = data[:,:,:, index]
index = stats_list.index('ps_s_feed_chi2')
ps_s_feed = data[:,:,:, index]
index = stats_list.index('ps_s_chi2')
ps_s_chi2 = data[:,:,:, index]

index = stats_list.index('ps_o_sb_chi2')
ps_o_sb = data[0,:,:, index]
index = stats_list.index('ps_o_feed_chi2')
ps_o_feed = data[0,:,:, index]
index = stats_list.index('ps_o_chi2')
ps_o_chi2 = data[0,:,:, index]

n_scans = len(ps_chi2[:, 0, 0])

fig5 = plt.figure(figsize=(12, 10))  # constrained_layout=True, 

n_cols = 2 * n_scans
n_rows = 5

widths = np.zeros(n_cols)
heights = np.zeros(n_rows)

heights = [10, 1, 3, 1, 1]

for i in range(n_scans):
    widths[2 * i] = 3
    widths[2 * i + 1] = 1

spec5 = fig5.add_gridspec(ncols=n_cols, nrows=n_rows, width_ratios=widths,
                          height_ratios=heights)
# for col in range(n_cols):
#     row = 0
#     ax = fig5.add_subplot(spec5[row, col])
    #label = 'Width: {}\nHeight: {}'.format(widths[col], heights[row])
    #ax.annotate(label, (0.1, 0.5), xycoords='axes fraction', va='center')




vmax = 10
n_det = 20
row = 0
ax1 = fig5.add_subplot(spec5[row, 0])
im = ax1.imshow(ps_chi2[0], interpolation='none', aspect='auto', vmin=-vmax, vmax=vmax, extent=(0.5, 4.5, n_det + 0.5, 0.5))
new_tick_locations = np.array(range(n_det))+1
ax1.set_yticks(new_tick_locations)
x_tick_loc = [1, 2, 3, 4]
x_tick_labels = ['LA', 'UA', 'LB', 'UB']
ax1.set_xticks(x_tick_loc)
# ax1.set_xticklabels(x_tick_labels, rotation=90)
ax1.title.set_text(str(scanids[0]))


# cbar = fig5.colorbar(im)
# cbar.set_label('ps_chi2')
# cbaxes = fig5.add_axes([0.8, 0.1, 0.03, 0.8]) 
# cb = plt.colorbar(ax1, cax = cbaxes)  
# fig5.colorbar(im, orientation="horizontal", pad=0.2)
# m = cm.ScalarMappable()
# m.set_array(ps_chi2[0])

# cbar = fig5.colorbar(m)
# cbar.set_label('ps_chi2')
# fig5.subplots_adjust(top=0.87)
fig5.subplots_adjust(right=0.85)
fig5.suptitle('Obsid:' + str(scanids[0])[:-2], fontsize=20)
# cbar_ax = fig5.add_axes([0.15, 0.90, 0.70, 0.02])
cbar_ax = fig5.add_axes([0.87, 0.1, 0.02, 0.8]) 
cbar = fig5.colorbar(im, cax=cbar_ax)#, orientation='horizontal')
cbar.set_label('ps_chi2')
# cbar_ax.xaxis.set_label_position('top')
# cbar_ax.xaxis.set_ticks_position('top')
# axins1 = inset_axes(ax1,
#                     width="50%",  # width = 50% of parent_bbox width
#                     height="5%",  # height : 5%
#                     loc='upper right')
# fig5.colorbar(im, cax=axins1, orientation="horizontal", ticks=[1, 2, 3])
ax2 = fig5.add_subplot(spec5[row, 1], sharey=ax1)
ax2.imshow(ps_s_feed[0, :], interpolation='none', aspect='auto', vmin=-vmax, vmax=vmax, extent=(0.5, 1.5, n_det + 0.5, 0.5))
new_tick_locations = np.array(range(n_det))+1
ax2.set_yticks(new_tick_locations)
x_tick_loc = []
ax2.set_xticks(x_tick_loc)
#ax2.set_xlabel(['feed'])

plt.setp(ax2.get_yticklabels(), visible=False)
row = 1
ax = fig5.add_subplot(spec5[row, 0: 2])
# print(ps_s_chi2[6])
ax.imshow(ps_s_chi2[0], interpolation='none', aspect='auto', vmin=-vmax, vmax=vmax)
plt.setp(ax.get_yticklabels(), visible=False)
plt.setp(ax.get_xticklabels(), visible=False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('scan')
for scan in range(1,n_scans):
    row = 0
    ax3 = fig5.add_subplot(spec5[row, 2 * scan], sharey=ax1)
    ax3.imshow(ps_chi2[scan], interpolation='none', aspect='auto', vmin=-vmax, vmax=vmax, extent=(0.5, 4.5, n_det + 0.5, 0.5))
    new_tick_locations = np.array(range(n_det))+1
    ax3.set_yticks(new_tick_locations)
    ax3.title.set_text(str(scanids[scan]))
    x_tick_loc = [1, 2, 3, 4]
    x_tick_labels = ['LA', 'UA', 'LB', 'UB']
    ax3.set_xticks(x_tick_loc)
    # ax1.set_xticklabels(x_tick_labels, rotation=90)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax4 = fig5.add_subplot(spec5[row, 2 * scan+1], sharey=ax1)
    ax4.imshow(ps_s_feed[scan], interpolation='none', aspect='auto', vmin=-vmax, vmax=vmax, extent=(0.5, 1.5, n_det + 0.5, 0.5))
    new_tick_locations = np.array(range(n_det))+1
    ax4.set_yticks(new_tick_locations)
    x_tick_loc = []
    ax4.set_xticks(x_tick_loc)
    plt.setp(ax4.get_yticklabels(), visible=False)
    row = 1
    ax = fig5.add_subplot(spec5[row, 2 * scan:2 * scan + 2])
    ax.imshow(ps_s_chi2[scan], interpolation='none', aspect='auto', vmin=-vmax, vmax=vmax)
    plt.setp(ax.get_yticklabels(), visible=False)
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_xticks([])
    ax.set_yticks([])
    # label = 'Width: {}\nHeight: {}'.format(1, heights[row])
    # ax.annotate(label, (0.1, 0.5), xycoords='axes fraction', va='center')
row = 2 
ax6 = fig5.add_subplot(spec5[row, :])
ax6.title.set_text('Data from full obsid:')
# print(ps_o_sb.T)
im = ax6.imshow(ps_o_sb.T, interpolation='none', aspect='auto', vmin=-vmax, vmax=vmax, extent=(0.5, n_det + 0.5, 4.5, 0.5))
new_tick_locations = np.array(range(n_det))+1
ax6.set_xticks(new_tick_locations)
y_tick_loc = [1, 2, 3, 4]
y_tick_labels = ['LA', 'UA', 'LB', 'UB']
ax6.set_yticks(y_tick_loc)
# ax1.set_xticklabels(x_tick_labels, rotation=90)
# ax1.title.set_text(str(scanids[0]))
row = 3 
ax7 = fig5.add_subplot(spec5[row, :])
im = ax7.imshow(ps_o_feed.T, interpolation='none', aspect='auto', vmin=-vmax, vmax=vmax, extent=(0.5, n_det + 0.5, 1.5, 0.5))
new_tick_locations = np.array(range(n_det))+1
ax7.set_xticks(new_tick_locations)
ax7.set_xticklabels([])
y_tick_loc = []
y_tick_labels = ['LA', 'UA', 'LB', 'UB']
ax7.set_yticks(y_tick_loc)
row = 4 
ax8 = fig5.add_subplot(spec5[row, :])
ax8.imshow(ps_o_chi2, interpolation='none', aspect='auto', vmin=-vmax, vmax=vmax)
plt.setp(ax8.get_yticklabels(), visible=False)
plt.setp(ax8.get_xticklabels(), visible=False)
ax8.set_xticks([])
ax8.set_yticks([])
#plt.imshow(ps_chi2[0], interpolation='none')
# plt.tight_layout()
plt.savefig('test.png', bbox_inches='tight', dpi=100)
plt.show()
