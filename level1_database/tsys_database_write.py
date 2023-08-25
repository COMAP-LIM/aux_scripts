"""Example usage:
python3 -W ignore tsys_database_write_new.py -o -n 28 -p level1_database_files -m 2019-10 2019-09 2019-08
"""
import numpy as np
import h5py
from os import listdir, makedirs
from os.path import isfile, join, exists
import time
from tqdm import trange, tqdm
from tsysmeasure import TsysMeasure
import multiprocessing as mp
import argparse


def worker(fileidx):
    if fileidx < Nthreads:
        print(f"Worker {fileidx} waiting {fileidx*30} seconds.")
        time.sleep(fileidx*30)

    t0 = time.time()
    filename = filenames[fileidx]
    obsid = filename.split("/")[-1].split("-")[1]
    print(f"Obsid {obsid} started ({fileidx}/{Nfiles}).")
    try:
        Tsys = TsysMeasure()
        Tsys.load_data_from_file(filename)
        Tsys.solve()
        feeds = Tsys.feeds
        Thot = np.zeros((20, 2))
        Thot[feeds-1] = Tsys.Thot
        Phot = np.zeros((20, 4, 1024, 2))
        Phot[feeds-1] = Tsys.Phot
        points_used_Thot = np.zeros((20, 2), dtype=int)
        points_used_Thot[feeds-1] = Tsys.points_used_Thot
        points_used_Phot = np.zeros((20, 2), dtype=int)
        points_used_Phot[feeds-1] = Tsys.points_used_Phot
        successful = np.zeros((20, 2), dtype=int)
        successful[feeds-1] = Tsys.successful
        calib_times = np.zeros((20, 2))
        calib_times[feeds-1] = Tsys.calib_times
        calib_startstop_times = np.zeros((20, 2, 2))
        calib_startstop_times[feeds-1] = Tsys.calib_startstop_times
        tsys = np.zeros((20, 4, 1024))
        tsys[feeds-1] = Tsys.Tsys
        G = np.zeros((20, 4, 1024))
        G[feeds-1] = Tsys.G
        calib_indices_tod = Tsys.calib_startstopindices_tod
 
    except:
        print(f"Tsys failed for {obsid}.")
        feeds = np.zeros(0)
        Thot = np.zeros((20, 2)) + np.nan
        Phot = np.zeros((20, 4, 1024, 2)) + np.nan
        points_used_Thot = np.zeros((20, 2), dtype=int) 
        points_used_Phot = np.zeros((20, 2), dtype=int)
        successful = np.zeros((20, 2), dtype=int) - 10
        calib_times = np.zeros((20, 2)) + np.nan
        calib_startstop_times = np.zeros((20, 2, 2)) + np.nan
        tsys = np.zeros((20, 4, 1024)) + np.nan
        G = np.zeros((20, 4, 1024)) + np.nan
        calib_indices_tod = np.zeros_like((2, 2)) + np.nan

    
    with h5py.File(f"{args.path}/{obsid}.h5", "w") as outfile:
        outfile.create_dataset("feeds", data=feeds)
        outfile.create_dataset("Thot", data=Thot)
        outfile.create_dataset("Phot", data=Phot)
        outfile.create_dataset("points_used_Thot", data=points_used_Thot)
        outfile.create_dataset("points_used_Phot", data=points_used_Phot)
        outfile.create_dataset("successful", data=successful)
        outfile.create_dataset("calib_times", data=calib_times)
        outfile.create_dataset("calib_startstop_times", data=calib_startstop_times)
        outfile.create_dataset("calib_indices_tod", data=calib_indices_tod)
        outfile.create_dataset("Tsys_obsidmean", data=tsys)
        outfile.create_dataset("G_obsidmean", data=G)
    del(Tsys)
    
    print(f"Obsid {obsid} finished in {time.time()-t0:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nthreads", type=int, default=15, help="Number of threads to use. Memory requirement: ca 60GB/thread.")
    parser.add_argument("-m", "--months", type=str, nargs="+", required=True, help="Months to include in database. Eg. -m 2021-02 2021-03 2021-04.")
    parser.add_argument("-p", "--path", type=str, default="level1_database_files", help="Path where the database files will be written.")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite database files, if they already exist.")
    args = parser.parse_args()
    Nthreads = args.nthreads
    if not exists(args.path):
        makedirs(args.path)
        print(f"Path {args.path} does not already exist. Creating dir.")
    months = []
    for month in args.months:
        months.append(month)
    blacklist = []
    if not args.overwrite:
        print(f"Creating list of already existing files...")
        for f in tqdm(listdir(args.path)):
            if isfile(join(args.path, f)):
                if f[-4:] == ".hd5" or f[-3:] == ".h5":
                    with h5py.File(join(args.path, f), "r") as infile:
                        if "Thot" in infile and "Phot" in infile and "points_used_Thot" in infile and "points_used_Phot" in infile and "Tsys_obsidmean" in infile and "Tsys_obsidmean" in infile and "successful" in infile and "calib_times" in infile and "calib_startstop_times" in infile:
                            blacklist.append(f.split(".")[0])

        print(f"Writing in append mode. Ignoring {len(blacklist)} already existing files.")
    filenames = []
    for month in months:
        path = f"/mn/stornext/d22/cmbco/comap/protodir/level1/{month}/"
        for f in listdir(path):
            if isfile(join(path, f)):
                if f[-4:] == ".hd5" or f[-3:] == ".h5":
                    if len(f.split("-")) > 1:
                        if not f.split("-")[1] in blacklist:
                            filenames.append(join(path, f))
    Nfiles = len(filenames)
    print(f"Found {Nfiles} level 1 files.")
    t1 = time.time()
    with mp.Pool(Nthreads) as p:
        p.map(worker, range(Nfiles), chunksize=1)
    print(f"Finished database run in {(time.time()-t1):.2f} seconds ({(time.time()-t1)/3600:.2f} hours).")