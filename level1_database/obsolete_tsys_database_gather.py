import numpy as np
import h5py
from os import listdir
from os.path import isfile, join
from tqdm import tqdm, trange
import multiprocessing as mp
import sys


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
                    feeds = infile["feeds"][()]
                    Thot = infile["Thot"][()]
                    Phot = infile["Phot"][()]
                    points_used_Thot = infile["points_used_Thot"][()]
                    points_used_Phot = infile["points_used_Phot"][()]
                    successful = infile["successful"][()]
                    calib_times = infile["calib_times"][()]
                    calib_startstop_times = infile["calib_startstop_times"][()]
                    calib_indices_tod = infile["calib_indices_tod"][()]
                    Tsys_obsidmean = infile["Tsys_obsidmean"][()]
                    G_obsidmean = infile["G_obsidmean"][()]

                    try:
                        calib_indices_tod = infile["calib_indices_tod"][()]
                    except:
                        calib_indices_tod = np.zeros((2,2)) + np.nan
                    datagroup = "obsid/" + obsid + "/"
                    outfile.create_dataset(datagroup + "feeds", data=feeds)
                    outfile.create_dataset(datagroup + "Thot", data=Thot)
                    outfile.create_dataset(datagroup + "Phot", data=Phot)
                    outfile.create_dataset(datagroup + "points_used_Thot", data=points_used_Thot)
                    outfile.create_dataset(datagroup + "points_used_Phot", data=points_used_Phot)
                    outfile.create_dataset(datagroup + "successful", data=successful)
                    outfile.create_dataset(datagroup + "calib_times", data=calib_times)
                    outfile.create_dataset(datagroup + "calib_startstop_times", data=calib_startstop_times)
                    outfile.create_dataset(datagroup + "calib_indices_tod", data=calib_indices_tod)
                    outfile.create_dataset(datagroup + "Tsys_obsidmean", data=Tsys_obsidmean)
                    outfile.create_dataset(datagroup + "G_obsidmean", data=G_obsidmean)
            except:
                print(f"Failed obsid {obsid}")