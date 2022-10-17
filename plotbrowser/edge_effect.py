import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from tqdm import tqdm
import multiprocessing as mp
import warnings

warnings.filterwarnings(
    "ignore", category=RuntimeWarning
)  # ignore warnings caused by weights cut-off

field = "co7"
path = f"/mn/stornext/d22/cmbco/comap/protodir/level2/Ka/{field}/"
allfiles = os.listdir(path)
allfiles = [file for file in allfiles if f"{field}" in file]
allids = np.array([int(file[4:-3]) for file in allfiles])
sort = np.argsort(allids)
allids = allids[sort]
allfiles = np.array(allfiles)[sort]
Nfiles = len(allfiles)


# def ddaz_binned(allnames):
def ddaz_binned(idx):
    pid = mp.current_process()._identity[0]
    ids = []
    with ddaz_binned.lock:
        print(
            "Progress: ~",
            ddaz_binned.iterr.value,
            "/",
            Nfiles,
            "|",
            round(ddaz_binned.iterr.value / Nfiles * 100, 4),
            "%",
        )
        ddaz_binned.iterr.value += 1

    ncomps = 10
    nbins = 100
    nsb = 4
    nchannels = 64
    nfreqs = nsb * nchannels
    nfeeds = 20

    compsdata = np.zeros((nfeeds, ncomps, nbins))
    ampldata = np.zeros((nfeeds, ncomps, nfreqs))

    compsdata_daz = np.zeros((nfeeds, ncomps, nbins))
    ampldata_daz = np.zeros((nfeeds, ncomps, nfreqs))

    compsdata_ddaz = np.zeros((nfeeds, ncomps, nbins))
    ampldata_ddaz = np.zeros((nfeeds, ncomps, nfreqs))

    # bindata_az = np.zeros((nfeeds, nfreqs, nbins))
    # bindata_daz = np.zeros((nfeeds, nfreqs, nbins))
    # bindata_ddaz = np.zeros((nfeeds, nfreqs, nbins))

    # hitdata_az = np.zeros((nfeeds, nfreqs, nbins))
    # hitdata_daz = np.zeros((nfeeds, nfreqs, nbins))
    # hitdata_ddaz = np.zeros((nfeeds, nfreqs, nbins))

    # allids    = []
    # for num, name in enumerate(tqdm(allnames, desc= "Scans", position = 0)):
    name = allfiles[idx]
    try:
        with h5py.File(path + name, "r") as infile:
            id = int(name[4:-3])
            # allids.append(id)
            tod = infile["tod"][()]
            freq = infile["nu"][0, ...]
            rms = infile["sigma0"][()]
            pixels = infile["pixels"][()]
            point_tel = infile["point_tel"][()]
            point_cel = infile["point_cel"][()]
            time = infile["time"][()]
    except:
        print("Skipping:", name)
        return None

    freq[0, :] = freq[0, ::-1]
    freq[2, :] = freq[2, ::-1]

    tod[:, 0, :, :] = tod[:, 0, ::-1, :]
    tod[:, 2, :, :] = tod[:, 2, ::-1, :]

    _temp = tod.copy()
    tod = np.zeros((nfeeds, nsb, nchannels, tod.shape[-1]))
    tod[pixels - 1, ...] = _temp
    tod = np.where(np.isnan(tod) == False, tod, 0)

    rms[:, 0, :] = rms[:, 0, ::-1]
    rms[:, 2, :] = rms[:, 2, ::-1]
    _temp = rms.copy()
    rms = np.zeros((nfeeds, *rms.shape[1:]))
    rms[pixels - 1, ...] = _temp

    _temp = point_tel.copy()
    point_tel = np.zeros((nfeeds, *point_tel.shape[1:]))
    point_tel[pixels - 1, ...] = _temp

    _temp = point_cel.copy()
    point_cel = np.zeros((nfeeds, *point_cel.shape[1:]))
    point_cel[pixels - 1, ...] = _temp

    az = point_tel[:, :, 0]
    el = point_tel[:, :, 1]

    ra = point_cel[:, :, 0]
    dec = point_cel[:, :, 1]

    # for feed in tqdm(range(nfeeds), desc = f"Rank: {pid}; Index: {idx}/{Nfiles}", position = pid, leave = False):
    for feed in range(nfeeds):
        allaz = []
        alldaz = []
        allddaz = []

        # all_hits_az = []
        # all_hits_daz = []
        # all_hits_ddaz = []

        for i in range(4):
            for j in range(64):
                histsum, bins = np.histogram(
                    az[feed], bins=nbins, weights=(tod[feed, i, j, :] / rms[feed, i, j])
                )
                nhit = np.histogram(az[feed], bins=nbins)[0]
                normhist = histsum / nhit * np.sqrt(nhit)
                allaz.append(normhist)
                # all_hits_az.append(nhit)

                daz = np.gradient(az[feed], (time[1] - time[0]) * 3600 * 24)
                # daz = (az[feed, 1:] - az[feed, :-1]) / (time[1] - time[0]) * 3600 * 24
                bins = np.linspace(-1.0, 1.0, nbins + 1)
                bins[0] = -1e6  # Infinite bin edges
                bins[-1] = 1e6
                histsum, dazbins = np.histogram(
                    daz, bins=bins, weights=(tod[feed, i, j, :] / rms[feed, i, j])
                )
                nhit = np.histogram(daz, bins=bins)[0]
                normhist = histsum / nhit * np.sqrt(nhit)
                alldaz.append(normhist)
                # all_hits_daz.append(nhit)

                ddaz = np.gradient(daz, (time[1] - time[0]) * 3600 * 24)
                # ddaz = (az[feed, 2:] - 2 * az[feed, 1:-1] + az[feed, 0:-2]) / ((time[1] - time[0]) * 3600 * 24) ** 2
                bins = np.linspace(-10, 10, nbins + 1)
                bins[0] = -1e6
                bins[-1] = 1e6
                histsum, ddazbins = np.histogram(
                    ddaz, bins=bins, weights=(tod[feed, i, j, :] / rms[feed, i, j])
                )
                nhit = np.histogram(ddaz, bins=bins)[0]
                normhist = histsum / nhit * np.sqrt(nhit)
                allddaz.append(normhist)
                # all_hits_ddaz.append(nhit)

        allaz = np.array(allaz)
        alldaz = np.array(alldaz)
        allddaz = np.array(allddaz)

        # all_hits_az = np.array(all_hits_az)
        # all_hits_daz = np.array(all_hits_daz)
        # all_hits_ddaz = np.array(all_hits_ddaz)

        pcadata = np.where(np.isnan(allaz) == False, allaz, 0)

        pca = PCA(n_components=ncomps)
        pca.fit(pcadata)
        comps = pca.components_
        ampl = np.sum(pcadata[:, None, :] * comps[None, :, :], axis=-1).T

        pcadata_daz = np.where(np.isnan(alldaz) == False, alldaz, 0)

        pca_daz = PCA(n_components=ncomps)
        pca_daz.fit(pcadata_daz)
        comps_daz = pca_daz.components_
        ampl_daz = np.sum(pcadata_daz[:, None, :] * comps_daz[None, :, :], axis=-1).T

        pcadata_ddaz = np.where(np.isnan(alldaz) == False, alldaz, 0)

        pca_ddaz = PCA(n_components=ncomps)
        pca_ddaz.fit(pcadata_ddaz)
        comps_ddaz = pca_ddaz.components_
        ampl_ddaz = np.sum(pcadata_ddaz[:, None, :] * comps_ddaz[None, :, :], axis=-1).T

        # compsdata[num, feed, ...] = comps
        # ampldata[num, feed, ...] = ampl

        # compsdata_daz[num, feed, ...] = comps_daz
        # ampldata_daz[num, feed, ...] = ampl_daz

        # compsdata_ddaz[num, feed, ...] = comps_ddaz
        # ampldata_ddaz[num, feed, ...] = ampl_ddaz

        # compsdata = np.concatenate((compsdata, comps[:, None, ...]), axis = 1)
        # ampldata  = np.concatenate((ampldata,  ampl[:, None, ...]), axis = 1)
        # compsdata_daz = np.concatenate((compsdata_daz, comps_daz[:, None, ...]), axis = 1)
        # ampldata_daz  = np.concatenate((ampldata_daz,  ampl_daz[:, None, ...]), axis = 1)

        compsdata[feed, ...] = comps
        ampldata[feed, ...] = ampl

        compsdata_daz[feed, ...] = comps_daz
        ampldata_daz[feed, ...] = ampl_daz

        compsdata_ddaz[feed, ...] = comps_ddaz
        ampldata_ddaz[feed, ...] = ampl_ddaz

        # bindata_az[feed, ...] = allaz
        # bindata_daz[feed, ...] = alldaz
        # bindata_ddaz[feed, ...] = allddaz

        # hitdata_az[feed, ...] = all_hits_az
        # hitdata_daz[feed, ...] = all_hits_daz
        # hitdata_ddaz[feed, ...] = all_hits_ddaz
        # ids.append(id)

    result = {
        "comps_az": compsdata,
        "ampl_az": ampldata,
        "comps_daz": compsdata_daz,
        "ampl_daz": ampldata_daz,
        "comps_ddaz": compsdata_ddaz,
        "ampl_ddaz": ampldata_ddaz,
        # "bindata_az": bindata_az,
        # "bindata_daz": bindata_daz,
        # "bindata_ddaz": bindata_ddaz,
        # "hitdata_az": hitdata_az,
        # "hitdata_daz": hitdata_daz,
        # "hitdata_ddaz": hitdata_ddaz,
        "azbins": bins,
        "dazbins": dazbins,
        "ddazbins": ddazbins,
        "freq": freq,
        # "scanid": np.array(ids)
    }

    return result


def ddaz_binned_init(q, lock, iterr):

    ddaz_binned.q = q
    ddaz_binned.lock = lock
    ddaz_binned.iterr = iterr


m = mp.Manager()  # Multiprocess manager used to manage Queue and Lock.
q = m.Queue()
lock = m.Lock()
iterr = mp.Value("i", 0)  # Initializing shared iterator value

NP = 50

with mp.Pool(NP, ddaz_binned_init, [q, lock, iterr]) as pool:
    result = pool.map(ddaz_binned, range(len(allfiles)))

allresults = {}
print("Saving to file:")
with h5py.File(f"edge_effect_data_S2_{field}_v2.h5", "w") as outfile:
    for i, element in enumerate(tqdm(result)):
        # for i, element in enumerate(result):
        if result:
            for key, value in element.items():
                if i == 0:
                    if key not in ["freq"]:
                        allresults[key] = value[None, ...]

                else:
                    if key not in ["freq"]:
                        allresults[key] = np.concatenate(
                            (allresults[key], value[None, ...])
                        )
        else:
            continue
    outfile.create_dataset("scanid", data=allids)
    for key, value in allresults.items():
        if key not in ["freq", "scanid"]:
            if key in "scanid":
                continue
            else:
                outfile.create_dataset(key, data=value)
        else:
            outfile.create_dataset(key, data=value[0, ...])


# with h5py.File("edge_effect_data_v3.h5", "w") as outfile:
#     outfile.create_dataset("comps_az", data = compsdata)
#     outfile.create_dataset("ampl_az", data = ampldata)
#     outfile.create_dataset("comps_daz", data = compsdata_daz)
#     outfile.create_dataset("ampl_daz", data = ampldata_daz)
#     outfile.create_dataset("comps_ddaz", data = compsdata_ddaz)
#     outfile.create_dataset("ampl_ddaz", data = ampldata_ddaz)
#     outfile.create_dataset("azbins", data = bins)
#     outfile.create_dataset("dazbins", data = dazbins)
#     outfile.create_dataset("ddazbins", data = ddazbins)
#     outfile.create_dataset("freq", data = freq)
#     outfile.create_dataset("scanid", data = allids)
