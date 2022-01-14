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
import argparse
import re 
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning) #ignore warnings caused by weights cut-off

import accept_mod.stats_list as stats_list
stats_list = stats_list.stats_list

from spikes import spike_data, spike_list, get_spike_list

class L2plots():
    def __init__(self):
        self.outpath = "/mn/stornext/d16/cmbco/comap/nils/plotbrowser/test_figs/"
        self.input()
        self.read_paramfile()

    def input(self):
        """
        Function parsing the command line input.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("-p", "--param", type = str,
                            help = """Full path and name to parameter file.""")
       
        args = parser.parse_args()
       
        if args.param == None:
            message = """No input parameterfile given, please provide an input parameterfile"""
            raise NameError(message)
        else:
            self.param_file     = args.param

    def read_paramfile(self):
        """
        Function reading the parameter file provided by the command line
        argument, and defining class parameters.
        """
        param_file  = open(self.param_file, "r")
        params      = param_file.read()

        scan_data_path = re.search(r"\nACCEPT_DATA_FOLDER\s*=\s*'(\/.*?)'", params)  # Defining regex pattern to search for scan data path in parameter file.
        self.scan_data_path = str(scan_data_path.group(1))                          # Extracting path

        id_string = re.search(r"\nACCEPT_DATA_ID_STRING\s*=\s*'(.*?)'", params)  # Defining regex pattern to search for scan data id string in parameter file.
        self.id_string = str(id_string.group(1))                                  # Extracting path
        
        l2_path = re.search(r"\nLEVEL2_DIR\s*=\s*'(\/.*?)'", params)  # Defining regex pattern to search for runlist path in parameter file.
        self.l2_path = str(l2_path.group(1))                  # Extracting path
        
        runlist_path = re.search(r"\nRUNLIST\s*=\s*'(\/.*?)'", params)  # Defining regex pattern to search for runlist path in parameter file.
        self.runlist_path = str(runlist_path.group(1))                  # Extracting path
        
        runlist_file = open(self.runlist_path, "r")         # Opening 
        runlist = runlist_file.read()
        
        tod_in_list = re.findall(r"\/.*?\.\w+", runlist)
        self.tod_in_list = tod_in_list

        patch_name = re.search(r"\s([a-zA-Z0-9]+)\s", runlist)
        self.patch_name = str(patch_name.group(1))

        obsIDs = re.findall(r"\s\d{6}\s", runlist)         # Regex pattern to find all obsIDs in runlist
        self.obsIDs = [num.strip() for num in obsIDs]
        self.nobsIDs = len(self.obsIDs)                    # Number of obsIDs in runlist
        
        scans_per_obsid = re.findall(r"\d\s+(\d+)\s+\/", runlist)
        self.scans_per_obsid = [int(num.strip()) - 2 for num in scans_per_obsid]

        param_file.close()
        runlist_file.close()

    def run(self):
        ps_chi2, ps_s_feed, ps_s_chi2, ps_o_sb, ps_o_feed, ps_o_chi2 = self.open_scan_data()
        
        for i in tqdm(range(self.nobsIDs)):
            first_scanid = self.obsIDs[i] + "02"
            idx = np.argmin(np.abs(self.allscanids - int(first_scanid)))
            start = idx
            stop  = idx + self.scans_per_obsid[i]

            self.scanids = self.allscanids[start:stop]
            self.n_scans = len(self.scanids)
            self.ps_chi2 = ps_chi2[start:stop, ...]
            self.ps_s_feed = ps_s_feed[start:stop, ...]
            self.ps_s_chi2 = ps_s_chi2[start:stop, ...]
            self.ps_o_sb   = ps_o_sb
            self.ps_o_feed = ps_o_feed
            self.ps_o_chi2 = ps_o_chi2
            self.plot_ps_chi_data()
            
            l2data = []
            for i in range(self.n_scans):
                self.l2name = f"{self.patch_name}_{self.scanids[i]:09}.h5"
                print(l2name)
            
            break

    def open_scan_data(self):
        print("Loading scan data:")
        filename = "scan_data_" + self.id_string + "_" + self.patch_name + ".h5"
        filename = self.scan_data_path + filename

        with h5py.File(filename, "r") as infile:
            data = np.array(infile['scan_data'][()])
            self.allscanids = np.array(infile['scan_list'][()])

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
        print(ps_chi2.shape, ps_s_feed.shape, ps_s_chi2.shape, ps_o_sb.shape, ps_o_feed.shape, ps_o_chi2.shape)
        return ps_chi2, ps_s_feed, ps_s_chi2, ps_o_sb, ps_o_feed, ps_o_chi2

    def get_corr(self, scan):
        with h5py.File(scan, mode="r") as my_file:
            scan_id     = self.l2name[-12:-3]# + filename[-5:-3]
            tod_ind     = np.array(my_file['tod'][:])
            n_det_ind, n_sb, n_freq, n_samp = tod_ind.shape
            sb_mean_ind = np.array(my_file['sb_mean'][:])
            mask_ind    = my_file['freqmask'][:]
            mask_full_ind = my_file['freqmask_full'][:]
            reason_ind    = my_file['freqmask_reason'][:]
            pixels        = np.array(my_file['pixels'][:]) - 1 
            pix2ind       = my_file['pix2ind'][:]
            try: 
                sd_ind = np.array(my_file['spike_data'])
            except KeyError:
                sd_ind = np.zeros((3, n_det_ind, n_sb, 4, 1000))
            try: 
                chi2_ind = np.array(my_file['chi2'])
            except KeyError:
                chi2_ind = np.zeros_like(tod_ind[:,:,:,0])
            try:
                acc_ind = np.array(my_file['acceptrate'])
            except KeyError:
                acc_ind = np.zeros_like(tod_ind[:,:,0,0])
                print("Found no acceptrate")
            time = np.array(my_file['time'])
            mjd = time
            try:
                pca      = np.array(my_file['pca_comp'])
                eigv     = np.array(my_file['pca_eigv'])
                ampl_ind = np.array(my_file['pca_ampl'])
            except KeyError:
                pca = np.zeros((4, 10000))
                eigv = np.zeros(0)
                ampl_ind = np.zeros((4, *mask_full_ind.shape))
                print('Found no pca comps')
            try:
                tsys_ind = np.array(my_file['Tsys_lowres'])
            except KeyError:
                tsys_ind = np.zeros_like(tod_ind[:,:,:,0]) + 40
                print("Found no tsys")
        t0   = time[0]
        time = (time - time[0]) * (24 * 60)  # minutes

        n_freq_hr = len(mask_full_ind[0,0])
        n_det     = np.max(pixels) + 1 
        # print(n_det)

        ## transform to full arrays with all pixels
        tod       = np.zeros((n_det, n_sb, n_freq, n_samp))
        mask      = np.zeros((n_det, n_sb, n_freq))
        mask_full = np.zeros((n_det, n_sb, n_freq_hr))
        acc       = np.zeros((n_det, n_sb))
        ampl      = np.zeros((4, n_det, n_sb, n_freq_hr))
        tsys      = np.zeros((n_det, n_sb, n_freq))
        chi2      = np.zeros((n_det, n_sb, n_freq))
        sd        = np.zeros((3, n_det, n_sb, 4, 1000))
        sb_mean   = np.zeros((n_det, n_sb, n_samp))
        reason    = np.zeros((n_det, n_sb, n_freq_hr))
        # print(ampl_ind.shape)
        # print(ampl[:, pixels, :, :].shape)

        tod[pixels]       = tod_ind
        mask[pixels]      = mask_ind
        mask_full[pixels] = mask_full_ind
        reason[pixels]    = reason_ind
        acc[pixels]       = acc_ind
        ampl[:, pixels, :, :]  = ampl_ind
        tsys[pixels]           = tsys_ind
        chi2[pixels]           = chi2_ind
        sd[:, pixels, :, :, :] = sd_ind
        sb_mean[pixels]        = sb_mean_ind 

        acc = acc.flatten()
        # n_det, n_sb, n_freq, n_samp = tod.shape

        tod            = tod[:, :, :, :] * mask[:, :, :, None]
        tod[:, (0, 2)] = tod[:, (0, 2), ::-1]

        tod_flat = tod.reshape((n_det * n_sb * n_freq, n_samp))
        corr     = np.corrcoef(tod_flat)
        
        chi2 = chi2.flatten()
        chi2[(chi2 == 0.0)] = np.nan
        tsys = tsys.flatten()
        tsys[(tsys == 0.0)] = np.nan
        
        my_spikes = get_spike_list(sb_mean, sd, scan_id, mjd)

        reasonarr = np.zeros(6)
        r = reason[pixels[:-1]].flatten()
        reasonarr[0] = len(r[(r == 1)])
        reasonarr[1] = len(r[(r == 2)])
        reasonarr[2] = len(np.where(np.logical_and(r>=3, r<=13))[0])
        reasonarr[3] = len(r[(r == 15)])
        reasonarr[4] = len(np.where(np.logical_and(r>=16, r<=27))[0])
        reasonarr[5] = len(np.where(np.logical_and(r>=40, r<=41))[0])
        

        reasonarr /= n_det_ind * n_sb * n_freq
        
        return corr, 1.0 / n_samp, n_det, acc, chi2, tsys, t0, my_spikes, reasonarr

    def plot_ps_chi_data(self):
        ps_chi2 = self.ps_chi2 
        ps_s_feed = self.ps_s_feed 
        ps_s_chi2 = self.ps_s_chi2 
        ps_o_sb = self.ps_o_sb   
        ps_o_feed = self.ps_o_feed 
        ps_o_chi2 = self.ps_o_chi2 

        
        fig5 = plt.figure(figsize=(12, 10))  # constrained_layout=True, 

        n_cols = 2 * self.n_scans
        n_rows = 5

        widths = np.zeros(n_cols)
        heights = np.zeros(n_rows)

        heights = [10, 1, 3, 1, 1]

        for i in range(self.n_scans):
            widths[2 * i] = 3
            widths[2 * i + 1] = 1

        print("Generating ps chi2 plot:")

        spec5 = fig5.add_gridspec(ncols=n_cols, nrows=n_rows, width_ratios=widths,
                                height_ratios=heights)
        vmax = 10
        n_det = 20
        row = 0
        ax1 = fig5.add_subplot(spec5[row, 0])
        im = ax1.imshow(ps_chi2[0], interpolation = 'none', aspect = 'auto', vmin=-vmax, vmax=vmax, extent=(0.5, 4.5, n_det + 0.5, 0.5))
        new_tick_locations = np.array(range(n_det))+1
        ax1.set_yticks(new_tick_locations)
        x_tick_loc = [1, 2, 3, 4]
        x_tick_labels = ['LA', 'UA', 'LB', 'UB']
        ax1.set_xticks(x_tick_loc)
        # ax1.set_xticklabels(x_tick_labels, rotation=90)
        #ax1.title.set_text(str(self.scanids[0]))
        ax1.set_title(str(self.scanids[0]), rotation = 45)

        fig5.subplots_adjust(right=0.85)
        fig5.suptitle('Obsid:' + str(self.scanids[0])[:-2], fontsize=20)
        # cbar_ax = fig5.add_axes([0.15, 0.90, 0.70, 0.02])
        cbar_ax = fig5.add_axes([0.87, 0.1, 0.02, 0.8]) 
        cbar = fig5.colorbar(im, cax=cbar_ax)#, orientation='horizontal')
        cbar.set_label('ps_chi2')

        ax2 = fig5.add_subplot(spec5[row, 1], sharey=ax1)
        ax2.imshow(ps_s_feed[0, :], interpolation = 'none', aspect = 'auto', vmin=-vmax, vmax=vmax, extent=(0.5, 1.5, n_det + 0.5, 0.5))
        new_tick_locations = np.array(range(n_det))+1
        ax2.set_yticks(new_tick_locations)
        x_tick_loc = []
        ax2.set_xticks(x_tick_loc)
        #ax2.set_xlabel(['feed'])

        plt.setp(ax2.get_yticklabels(), visible=False)
        row = 1
        ax = fig5.add_subplot(spec5[row, 0: 2])
        # print(ps_s_chi2[6])
        ax.imshow(ps_s_chi2[0], interpolation = 'none', aspect = 'auto', vmin=-vmax, vmax=vmax)
        plt.setp(ax.get_yticklabels(), visible=False)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel('scan')
        for scan in range(1, self.n_scans):
            row = 0
            ax3 = fig5.add_subplot(spec5[row, 2 * scan], sharey=ax1)
            ax3.imshow(ps_chi2[scan], interpolation = 'none', aspect = 'auto', vmin=-vmax, vmax=vmax, extent=(0.5, 4.5, n_det + 0.5, 0.5))
            new_tick_locations = np.array(range(n_det))+1
            ax3.set_yticks(new_tick_locations)
            #ax3.title.set_text(str(self.scanids[scan]), rotation = 45)
            ax3.set_title(str(self.scanids[scan]), rotation = 45)
            x_tick_loc = [1, 2, 3, 4]
            x_tick_labels = ['LA', 'UA', 'LB', 'UB']
            ax3.set_xticks(x_tick_loc)
            # ax1.set_xticklabels(x_tick_labels, rotation=90)
            plt.setp(ax3.get_yticklabels(), visible=False)
            ax4 = fig5.add_subplot(spec5[row, 2 * scan+1], sharey=ax1)
            ax4.imshow(ps_s_feed[scan], interpolation = 'none', aspect = 'auto', vmin=-vmax, vmax=vmax, extent=(0.5, 1.5, n_det + 0.5, 0.5))
            new_tick_locations = np.array(range(n_det))+1
            ax4.set_yticks(new_tick_locations)
            x_tick_loc = []
            ax4.set_xticks(x_tick_loc)
            plt.setp(ax4.get_yticklabels(), visible=False)
            row = 1
            ax = fig5.add_subplot(spec5[row, 2 * scan:2 * scan + 2])
            ax.imshow(ps_s_chi2[scan], interpolation = 'none', aspect = 'auto', vmin=-vmax, vmax=vmax)
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
        im = ax6.imshow(ps_o_sb.T, interpolation = 'none', aspect = 'auto', vmin=-vmax, vmax=vmax, extent=(0.5, n_det + 0.5, 4.5, 0.5))
        new_tick_locations = np.array(range(n_det))+1
        ax6.set_xticks(new_tick_locations)
        y_tick_loc = [1, 2, 3, 4]
        y_tick_labels = ['LA', 'UA', 'LB', 'UB']
        ax6.set_yticks(y_tick_loc)
        # ax1.set_xticklabels(x_tick_labels, rotation=90)
        # ax1.title.set_text(str(self.scanids[0]))
        row = 3 
        ax7 = fig5.add_subplot(spec5[row, :])
        im = ax7.imshow(ps_o_feed.T, interpolation = 'none', aspect = 'auto', vmin=-vmax, vmax=vmax, extent=(0.5, n_det + 0.5, 1.5, 0.5))
        new_tick_locations = np.array(range(n_det))+1
        ax7.set_xticks(new_tick_locations)
        ax7.set_xticklabels([])
        y_tick_loc = []
        y_tick_labels = ['LA', 'UA', 'LB', 'UB']
        ax7.set_yticks(y_tick_loc)
        row = 4 
        ax8 = fig5.add_subplot(spec5[row, :])
        ax8.imshow(ps_o_chi2, interpolation = 'none', aspect = 'auto', vmin=-vmax, vmax=vmax)
        plt.setp(ax8.get_yticklabels(), visible=False)
        plt.setp(ax8.get_xticklabels(), visible=False)
        ax8.set_xticks([])
        ax8.set_yticks([])
        #plt.imshow(ps_chi2[0], interpolation='none')
        # plt.tight_layout()
        outname = f"ps_chi_plot_{str(self.scanids[0])[:-2]}.png"
        plt.savefig(self.outpath + outname, bbox_inches = 'tight', dpi=100)
        #plt.show()
        

if __name__ == "__main__":
    l2plotter = L2plots()
    l2plotter.run()
