import numpy as np
import matplotlib.pyplot as plt
import h5py


filename = "/mn/stornext/d22/cmbco/comap/nils/COMAP_general/data/level3/co7/edge_effect_data_v3_co7.h5"  #'edge_effect_data_S2_co2_v2.h5'

obsid_start = 6800
obsid_end = 25000

feed = 16
n_comp = 1
n_data = 100
n_freq = 256

scans = []
data = []
with h5py.File(filename, mode="r") as my_file:
    scans = list(my_file.keys())
    print(len(scans))
    # for i in range(obsid_start, obsid_end):
    #    obsid = i
    #    for j in range(20):
    #        scanid = obsid * 100 + j
    #        scanid_str = "%09i" % scanid
    for i, scan in enumerate(scans[:-1]):

        print(scan)
        full_arr = np.zeros((n_data + 256))
        # try:
        ampl_daz = np.array(my_file[scan + "/ampl_daz"][feed - 1, :n_comp, :])
        comp_daz = np.array(my_file[scan + "/comps_daz"][feed - 1, :n_comp, :])
        full_arr[:n_data] = comp_daz  # .sum(0)
        full_arr[n_data:] = ampl_daz  # .sum(0)
        data.append(full_arr)
        # scans.append(scanid)
        # except KeyError:
        #    pass

scans = np.array(scans[:-1]).astype(int)
data = np.array(data)

np.savetxt("scan_list_co7.txt", scans)
np.save("az_data_single_comp_co7", data)
