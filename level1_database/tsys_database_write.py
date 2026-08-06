"""Example usage:
python3 -W ignore tsys_database_write.py -o -n 28 -p level1_database_files -m 2019-10 2019-09 2019-08
"""
import sys
import numpy as np
import h5py
from os import listdir, makedirs
from os.path import isfile, isdir, join, exists
import re
import time
from tqdm import trange, tqdm
from tsysmeasure import TsysMeasure
import multiprocessing as mp
import argparse

L1_PATH = "/mn/stornext/d16/cmbco/comap/data/level1"
MONTH_DIR = re.compile(r"\d{4}-\d{2}")


def resolve_months(months, l1_path):
    """Expand the --months argument, turning "all" into every month present in the level1 tree.

    Args:
        months: The months given on the command line, or ["all"].
        l1_path: Root of the level1 data tree to scan when "all" is given.

    Returns:
        Sorted list of YYYY-MM month names. Passing "all" (in any position) scans l1_path for
        directories matching the YYYY-MM pattern, so a full rebuild needs no explicit list; the
        per-field directories (wide/, ncp/, l1_temp/) are skipped by the pattern.
    """
    if any(month.lower() == "all" for month in months):
        months = sorted(d for d in listdir(l1_path) if MONTH_DIR.fullmatch(d) and isdir(join(l1_path, d)))
        print(f"Month 'all' given: found {len(months)} month directories in {l1_path}.")
        return months
    return sorted(months)


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
        # Per-dip diagnostics, so a later cut can be made without re-reading the level1 data.
        tsys_vane = np.zeros((20, 2)) + np.nan          # Tsys implied by each individual dip.
        tsys_vane[feeds-1] = Tsys.Tsys_vane
        phot_finite_frac = np.zeros((20, 2)) + np.nan   # Finite fraction of the Phot spectrum.
        phot_finite_frac[feeds-1] = Tsys.Phot_finite_frac
        feed_usable = np.zeros(20, dtype=bool)          # Feed has at least one surviving dip.
        feed_usable[feeds-1] = Tsys.feed_usable
        n_vane_runs = Tsys.n_vane_runs                  # Vane flag activations found (expect 2).
 
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
        tsys_vane = np.zeros((20, 2)) + np.nan
        phot_finite_frac = np.zeros((20, 2)) + np.nan
        feed_usable = np.zeros(20, dtype=bool)
        n_vane_runs = 0

    
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
        outfile.create_dataset("Tsys_vane", data=tsys_vane)
        outfile.create_dataset("Phot_finite_frac", data=phot_finite_frac)
        outfile.create_dataset("feed_usable", data=feed_usable)
        outfile.create_dataset("n_vane_runs", data=n_vane_runs)
    del(Tsys)
    
    print(f"Obsid {obsid} finished in {time.time()-t0:.2f} seconds.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--nthreads", type=int, default=15, help="Number of threads to use. Memory requirement: ca 60GB/thread.")
    parser.add_argument("-m", "--months", type=str, nargs="+", required=True, help="Months to include in database. Eg. -m 2021-02 2021-03 2021-04. Pass -m all to use every YYYY-MM month directory found in the level1 tree.")
    parser.add_argument("-p", "--path", type=str, default="/mn/stornext/d16/cmbco/comap/data/aux_data/level1_database_files/", help="Path where the database files will be written.")
    parser.add_argument("-l", "--level1-path", type=str, default=L1_PATH, help=f"Root of the level1 data tree to read observations from. Default: {L1_PATH}.")
    parser.add_argument("-o", "--overwrite", action="store_true", help="Overwrite database files, if they already exist.")
    args = parser.parse_args()
    Nthreads = args.nthreads
    if not exists(args.path):
        makedirs(args.path)
        print(f"Path {args.path} does not already exist. Creating dir.")
    months = resolve_months(args.months, args.level1_path)
    blacklist = []
    if not args.overwrite:
        print(f"Creating list of already existing files...")
        for f in tqdm(listdir(args.path), file=sys.stdout):
            if isfile(join(args.path, f)):
                if f[-4:] == ".hd5" or f[-3:] == ".h5":
                    with h5py.File(join(args.path, f), "r") as infile:
                        if "Thot" in infile and "Phot" in infile and "points_used_Thot" in infile and "points_used_Phot" in infile and "Tsys_obsidmean" in infile and "Tsys_obsidmean" in infile and "successful" in infile and "calib_times" in infile and "calib_startstop_times" in infile:
                            blacklist.append(f.split(".")[0])

        print(f"Writing in append mode. Ignoring {len(blacklist)} already existing files.")
    filenames = []
    for month in months:
        path = join(args.level1_path, month)
        if exists(path):
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