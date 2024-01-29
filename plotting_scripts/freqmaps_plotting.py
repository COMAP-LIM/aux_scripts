import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import scipy.signal
from tqdm import trange, tqdm
from scipy.fft import fft, ifft, fftfreq, rfft, irfft
import time
from scipy.ndimage import gaussian_filter
from multiprocessing import Pool, Process
matplotlib.rcParams['figure.figsize'] = (16.0, 6.0)
plt.rcParams.update({'font.size': 10})
plt.style.use('seaborn-v0_8-darkgrid')
import warnings
warnings.filterwarnings('ignore')

sb_names = ["A:LSB", "A:USB", "B:LSB", "B:USB"]

def smooth_map(data, gauss_sigma=1.0):
    """
        Smooths the input map and returns the new map. NB: Also turns signal map into signal/rms map.
    """
    _map = data.copy()
    _map[~np.isfinite(_map)] = 0.0
    _map = gaussian_filter(_map, 1.0)
    _map[~np.isfinite(data)] = 0.0
    return _map

def make_maps(map1_path, map1_dataname, map2_path="", map2_dataname="", name1="", name2="", mode="sigma", compare=False, with_diff=False, out_filepath="", title="", smooth=True, res_factor=1.0, sidebands=[0,1,2,4]):
    """
        map1_path: Absolute filepath to first map.
        map1_dataname: Path to the dataset inside the hd5 file. Should be the path to the map-datafile, even if you want to plot rms or sens.
        map2_path: Absolute filepath to second map. Only required if "compare" is True.
        map2_dataname: Path to the dataset inside the hd5 file.
        name1: Title for the map1 maps.
        name2: Title for the map2 maps.
        mode:
            "sigma": Maps of the signal data in units of sigma (divided by the rms), with contours of the rms.
            "rms": Maps of the rms.
            "sens": Maps of the inverse rms.
        compare: If True, plots map1 and map2 next to each other for every frequency.
        with_diff: Requires compare to be True. Also plots the difference between map1 and map2.
        out_filepath: Full path and, optionally, beginning of filename (more will be appended).
        smooth: Whether to smooth the maps. Only applies to "sigma" mode.
        res_factor: Resolution factor for the images. Use 0.5-1.0 for faster plotting and slightly pixelated images. Don't go much over 1.0. 
    """

    with h5py.File(map1_path, "r") as f:
        map1 = f[map1_dataname][()]
        rms1 = f[map1_dataname.replace("map", "sigma_wn")][()]

    map1[map1 == 0] = np.nan
    rms1[~np.isfinite(rms1)] = np.inf
    rms1[rms1 == 0] = np.inf
    if mode == "sigma":
        map1 = map1/rms1
        if smooth:
            for feed in range(19):
                for sb in range(4):
                    for freq in range(64):
                        map1[feed,sb,freq] = smooth_map(map1[feed,sb,freq])
    sens1 = 1/rms1
    hitmap = np.array(sens1 > 0.1*np.nanmax(sens1, axis=(-1,-2))[:,:,:,None,None], dtype=float)

    if compare:
        if map2_path == "" or map2_dataname == "":
            raise ValueError("When compare is set to True, a map2_path and map2_dataname must be provided.")
        with h5py.File(map2_path, "r") as f:
            map2 = f[map2_dataname][()]
            rms2 = f[map2_dataname.replace("map", "sigma_wn")][()]
        map2[map2 == 0] = np.nan
        rms2[~np.isfinite(rms2)] = np.inf
        rms2[rms2 == 0] = np.inf
        sens2 = 1/rms2

        if mode == "sigma":
            map2 = map2/rms2
            if smooth:
                for feed in range(19):
                    for sb in range(4):
                        for freq in range(64):
                            map2[feed,sb,freq] = smooth_map(map2[feed,sb,freq])

        if with_diff:
            if mode == "sens":
                diff_map = sens2 - sens1
            else:
                ValueError("Currently only mode=sens supports with_diff=True.")

        hitmap = np.array((sens1 + sens2) > 0.1*np.nanmax(sens1 + sens2, axis=(-1,-2))[:,:,:,None,None], dtype=float)

    if not compare:
        for sb in tqdm(sidebands):
            fig, ax = plt.subplots(64, 19, figsize=(40*res_factor, 160*res_factor))
            fig.suptitle(f"{title} {sb_names[sb]}", fontsize=80*res_factor, y=1.005)
            for freq in range(64):
                for feed in range(19):
                    center_x, center_y = 60, 60
                    if np.sum(hitmap[feed,sb,freq]) > 0:
                        center_x = int(np.mean(np.argwhere(np.sum(hitmap[feed,sb,freq], axis=0))))
                        center_y = int(np.mean(np.argwhere(np.sum(hitmap[feed,sb,freq], axis=1))))
                    if mode == "sigma":
                        ax[freq,feed].imshow(map1[feed,sb,freq], vmin=-3, vmax=3, cmap="bwr", interpolation="nearest")
                        ax[freq,feed].contour(rms1[feed,sb,freq]*1e6, levels=[0, 200, 500, 1e10], cmap="gray", linewidths=0.3, alpha=0.75)
                    elif mode == "rms":
                        ax[freq,feed].imshow(rms1[feed,sb,freq], vmin=0, vmax=5*np.nanmin(rms1), cmap="gray", interpolation="nearest")
                    elif mode == "sens":
                        ax[freq,feed].imshow(sens1[feed,sb,freq], vmin=0, vmax=np.nanmax(sens1), cmap="gray_r", interpolation="nearest")
                    ax[freq,feed].set_xlim(center_x-32,center_x+32)
                    ax[freq,feed].set_ylim(center_y+32,center_y-32)
                    ax[freq,feed].grid(False)
                    ax[freq,feed].axis(False)
                    ax[freq,feed].set_title(f"feed {feed+1}\n {sb_names[sb]} - ch {freq}", fontsize=10.0*res_factor)
            plt.tight_layout()
            _filename = f"{out_filepath}_sb{sb}"
            _filename += ".png"
            plt.savefig(_filename, bbox_inches="tight")
            plt.clf()

    else: # If compare
        for sb in tqdm(sidebands):
            fig, ax = plt.subplots(64, 19*3, figsize=(40*3*res_factor, 160*res_factor))
            fig.suptitle(f"{title}. {sb_names[sb]}.", fontsize=180*res_factor, y=1.01)
            for freq in range(64):
                for feed in range(19):
                    center_x, center_y = 60, 60
                    if np.sum(hitmap[feed,sb,freq]) > 0:
                        center_x = int(np.mean(np.argwhere(np.sum(hitmap[feed,sb,freq], axis=0))))
                        center_y = int(np.mean(np.argwhere(np.sum(hitmap[feed,sb,freq], axis=1))))
                    for l in range(3):
                        ax[freq,feed*3+l].grid(False)
                        ax[freq,feed*3+l].axis(False)
                        ax[freq,feed*3+l].set_xlim(center_x-32,center_x+32)
                        ax[freq,feed*3+l].set_ylim(center_y+32,center_y-32)

                    if mode == "sigma":
                        ax[freq,feed*3].imshow(map1[feed,sb,freq], vmin=-3, vmax=3, cmap="bwr", interpolation="nearest")
                        ax[freq,feed*3+1].imshow(map2[feed,sb,freq], vmin=-3, vmax=3, cmap="bwr", interpolation="nearest")
                        ax[freq,feed*3].contour(rms1[feed,sb,freq]*1e6, levels=[0, 200, 500, 1e10], cmap="gray", linewidths=0.3, alpha=0.75)
                        ax[freq,feed*3+1].contour(rms2[feed,sb,freq]*1e6, levels=[0, 200, 500, 1e10], cmap="gray", linewidths=0.3, alpha=0.75)
                    elif mode == "rms":
                        ax[freq,feed*3].imshow(rms1[feed,sb,freq], vmin=0, vmax=5*np.nanmin(rms1), cmap="gray", interpolation="nearest")
                        ax[freq,feed*3+1].imshow(rms2[feed,sb,freq], vmin=0, vmax=5*np.nanmin(rms1), cmap="gray", interpolation="nearest")
                    elif mode == "sens":
                        ax[freq,feed*3].imshow(sens1[feed,sb,freq], vmin=0, vmax=np.nanmax(sens1), cmap="gray_r", interpolation="nearest")
                        ax[freq,feed*3+1].imshow(sens2[feed,sb,freq], vmin=0, vmax=np.nanmax(sens1), cmap="gray_r", interpolation="nearest")

                    ax[freq,feed*3].set_title(f"feed {feed+1}\n {sb_names[sb]} - ch {freq}\n {name1}", fontsize=10.0*res_factor)
                    ax[freq,feed*3+1].set_title(f"{name2}", fontsize=10.0*res_factor)

                    if with_diff:
                        if mode == "sens":
                            ax[freq,feed*3+2].imshow(sens2[feed,sb,freq] - sens1[feed,sb,freq], vmin=-max(np.nanmax(sens1), np.nanmax(sens2)), vmax=max(np.nanmax(sens1), np.nanmax(sens2)), cmap="bwr", interpolation="nearest")
                            ax[freq,feed*3+2].set_title(f"diff", fontsize=10.0*res_factor)

            plt.tight_layout()
            _filename = f"{out_filepath}_sb{sb}"
            _filename += ".png"
            plt.savefig(_filename, bbox_inches="tight")
            plt.clf()



if __name__ == "__main__":
    # make_maps(
    #     "/mn/stornext/d16/cmbco/comap/data/maps/co7_sep23_v3_n5_subtr_sigma_wn.h5",
    #     "/multisplits/rise/map_rise0",
    #     out_filepath="/mn/stornext/d16/www_cmb/jonas/frequency_maps/tests/"
    # )

    # make_maps(
    #     "/mn/stornext/d16/cmbco/comap/data/maps/co7_sep23_v3_n5_subtr_sigma_wn.h5",
    #     "/multisplits/rise/map_rise0",
    #     "/mn/stornext/d16/cmbco/comap/data/maps/co7_sep23_v3_n5_subtr_sigma_wn.h5",
    #     "/multisplits/rise/map_rise1",
    #     mode="sens",
    #     out_filepath="/mn/stornext/d16/www_cmb/jonas/frequency_maps/tests/comp",
    #     compare=True,
    # )

    mappath = "/mn/stornext/d16/cmbco/comap/data/power_spectrum/rnderrors_v4.2/average_spectra/co6_apr22_v4.2_null_no_rain_no_day_even_splits_take3_rnd1194182_n5_subtr_sigma_wn/co6_apr22_v4.2_null_no_rain_no_day_even_splits_take3_rnd1194182_n5_subtr_sigma_wn_average_fpxs.h5"
    for split in ["ambt", "wind", "pres", "s01f", "sudi"]:
        make_maps(
            f"{mappath}",
            f"/multisplits/{split}/map_{split}0elev0",
            f"{mappath}",
            f"/multisplits/{split}/map_{split}1elev0",
            mode="sens",
            out_filepath=f"/mn/stornext/d16/www_cmb/jonas/frequency_maps/null_splits/co6_comp_evenrise_new_{split}",
            title=f"{split}0elev0 vs {split}1elev0",
            compare=True,
            with_diff=True,
            res_factor=0.7,
            sidebands=[0],
        )

        