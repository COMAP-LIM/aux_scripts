import numpy as np
import matplotlib.pyplot as plt
import h5py
from os import listdir
from os.path import isfile, join
from tqdm import tqdm, trange
import matplotlib
import time
from scipy.optimize import curve_fit
from scipy.fft import fft, ifft, next_fast_len
from tqdm import trange
import multiprocessing as mp
import sys

def Wiener_filter(signal, fknee=0.01, alpha=4.0, samprate=50):
    """ Applies a lowpass (Wiener) filter to a signal, and returns the low-frequency result.
        Uses mirrored padding of signal to deal with edge effects.
        Assumes signal is of shape [freq, time].
    """
    N = signal.shape[-1]
    if N > 0:
        fastlen = next_fast_len(2*N)
        signal_padded = np.zeros((fastlen))
        signal_padded[fastlen//2-N:fastlen//2] = signal
        signal_padded[fastlen//2:fastlen//2+N] = signal[::-1]
        signal_padded = np.pad(signal_padded[fastlen//2-N:fastlen//2+N], (fastlen-2*N)//2, mode="reflect")

        freq_full = np.fft.fftfreq(fastlen)*samprate
        W = 1.0/(1 + (freq_full/fknee)**alpha)
        return ifft(fft(signal_padded)*W).real[fastlen//2-N:fastlen//2]
    else:
        return np.zeros(0)

def Wiener_filter_safe(signal, fknee=0.01, alpha=4.0, samprate=50):
    """ Applies a lowpass (Wiener) filter to a signal, and returns the low-frequency result.
        Uses mirrored padding of signal to deal with edge effects.
        Assumes signal is of shape [freq, time].
    """
    N = signal.shape[-1]
    if N > 0:
        signal_padded = np.zeros((2*N))
        signal_padded[:N] = signal
        signal_padded[N:] = signal[::-1]

        freq_full = np.fft.fftfreq(2*N)*samprate
        W = 1.0/(1 + (freq_full/fknee)**alpha)
        return ifft(fft(signal_padded)*W).real[:N]
    else:
        return np.zeros(0)


def az_el_func(x, g, d, c):
    return g*x[0] + d*x[1] + c



if __name__ == "__main__":
    inpath = sys.argv[1]  # Path to the obsid hdf5 files.
    outfilename = sys.argv[2]  # Path and name of the database to be written.
    
    filenames = []
    obsids = []
    for f in listdir(inpath):
        if isfile(join(inpath, f)):
            if f[-4:] == ".hd5" or f[-3:] == ".h5":
                filenames.append(join(inpath, f))
                obsids.append(f.split(".")[0])                

    Nfiles = len(filenames)
    print(f"Found {Nfiles} hdf5 files.")

    with h5py.File(outfilename, "w") as outfile:
        for i in trange(Nfiles):
            try:
                filename = filenames[i]
                obsid = obsids[i]
                with h5py.File(filename, "r") as infile:
                    Thot = infile["Thot"][()]
                    Phot = infile["Phot"][()]
                    points_used_Thot = infile["points_used_Thot"][()]
                    points_used_Phot = infile["points_used_Phot"][()]
                    tsys = infile["Tsys_obsidmean"][()]
                    successful = infile["successful"][()]
                    calib_times = infile["calib_times"][()]
                    # tod_sbmean = infile["tod_sbmean"][()]
                    # az = infile["az"][()]
                    # el = infile["el"][()]
                
                # N = tod_sbmean.shape[-1]
                # if N > 0:
                
                #     # Normalization
                #     tod_sbmean_norm = np.zeros_like(tod_sbmean)
                #     with mp.Pool() as p:
                #         tod_sbmean_norm = p.map(Wiener_filter_safe, tod_sbmean.reshape(tod_sbmean.shape[0]*4, tod_sbmean.shape[-1]))
                #     tod_sbmean_norm = np.array(tod_sbmean_norm).reshape(tod_sbmean.shape[0], 4, tod_sbmean.shape[-1])
                #     tod_sbmean_norm = tod_sbmean/tod_sbmean_norm - 1

                #     #Pointing template subtraction
                #     tod_sbmean_point_subtr = np.zeros_like(tod_sbmean_norm)
                #     g, d, c = 0,0,0
                #     for feed in range(tod_sbmean.shape[0]):
                #         for sb in range(4):
                #             if np.isfinite(tod_sbmean_norm[feed, sb]).all():
                #                 (g, d, c), _ = curve_fit(az_el_func, (1.0/np.sin(el[feed]*np.pi/180.0), az[feed]), tod_sbmean_norm[feed, sb], (g, d, c))
                #                 tod_sbmean_point_subtr[feed,sb] = tod_sbmean_norm[feed,sb] - az_el_func((1.0/np.sin(el[feed]*np.pi/180.0), az[feed]), g, d, c)

                    try:
                        calib_indices_tod = infile["calib_indices_tod"][()]
                    except:
                        calib_indices_tod = np.zeros((2,2)) + np.nan
                    datagroup = "obsid/" + obsid + "/"
                    outfile.create_dataset(datagroup + "Thot", data=Thot)
                    outfile.create_dataset(datagroup + "Phot", data=Phot)
                    outfile.create_dataset(datagroup + "points_used_Thot", data=points_used_Thot)
                    outfile.create_dataset(datagroup + "points_used_Phot", data=points_used_Phot)
                    outfile.create_dataset(datagroup + "Tsys_obsidmean", data=tsys)
                    outfile.create_dataset(datagroup + "successful", data=successful)
                    outfile.create_dataset(datagroup + "calib_times", data=calib_times)
                    # outfile.create_dataset(datagroup + "tod_sbmean", data=tod_sbmean)
                    # outfile.create_dataset(datagroup + "az", data=az)
                    # outfile.create_dataset(datagroup + "el", data=el)
                    # outfile.create_dataset(datagroup + "tod_sbmean_norm", data=tod_sbmean_norm)
                    # outfile.create_dataset(datagroup + "tod_sbmean_point_subtr", data=tod_sbmean_point_subtr)
                    outfile.create_dataset(datagroup + "calib_indices_tod", data=calib_indices_tod)
            except:
                print(f"Failed obsid {obsid}")