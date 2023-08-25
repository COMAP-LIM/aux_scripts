import numpy as np 
import h5py
import matplotlib.pyplot as plt
import warnings
import sys

warnings.filterwarnings("ignore", category = RuntimeWarning) # Ignore warnings caused by mask nan/inf weights
warnings.filterwarnings("ignore", category = UserWarning)    # Ignore warning when producing log plot of emty array

mappath = "/mn/stornext/d16/cmbco/comap/protodir/maps/"

outfile = h5py.File("co6_map_signal_new_data_liss.h5", "w")
infile = h5py.File(mappath + "co6_map_signal_new_data.h5", "r")


for key in infile.keys():

    if "multisplits" in key:
        mapdata = infile["multisplits/map_elev"][()]        
        nhitdata = infile["multisplits/nhit_elev"][()]
        rmsdata = infile["multisplits/rms_elev"][()]
        
        feed = -1
        sb = 0
        freq = 20
        """
        # CES
        map_ces  = mapdata[1::2, ...] 
        nhit_ces = nhitdata[1::2, ...] 
        rms_ces  = rmsdata[1::2, ...]       


        condition = nhit_ces > 0
        inv_var_ces = np.zeros_like(map_ces)
        inv_var_ces[condition] = 1 / rms_ces[condition] ** 2
        map_ces[condition] = map_ces[condition] * inv_var_ces[condition]
        
        print(np.all(map_ces == 0), np.all(nhit_ces == 0), np.all(rms_ces == 0))
        print(np.any(np.isnan(map_ces)), np.any(np.isnan(nhit_ces)), np.any(np.isnan(rms_ces)), np.any(np.isinf(inv_var_ces)))
        
        map = np.zeros_like(map_ces[0, ...])
        nhit = np.zeros_like(nhit_ces[0, ...])
        rms = np.zeros_like(rms_ces[0, ...])

        nhit = np.nansum(nhit_ces, axis = 0)        
        map = np.nansum(map_ces, axis = 0) / np.nansum(inv_var_ces, axis = 0)      
        rms = np.sqrt(1 / np.nansum(inv_var_ces, axis = 0))
        rms[nhit < 1] = 0
        map[nhit < 1] = 0

        #print(map.shape, rms.shape, nhit.shape, condition.shape)
        print(np.any(np.isnan(map)), np.any(np.isnan(nhit)), np.any(np.isnan(rms)))
        #print(np.any(np.isinf(map)), np.any(np.isinf(nhit)), np.any(np.isinf(rms)))
        print(np.all(map == 0), np.all(nhit == 0), np.all(rms == 0))

        outfile.create_dataset("map", data = map)
        outfile.create_dataset("nhit", data = nhit)
        outfile.create_dataset("rms", data = rms)

        fig, ax = plt.subplots(1, 2)
        fig.suptitle("ces")
        plotdata = np.where(nhit[feed, sb, freq, ...] > 1, 1e6 * map[feed, sb, freq, ...], np.nan)
        ax[0].imshow(plotdata, vmin = -2000, vmax = 2000)
        plotdata = np.where(nhit[feed, sb, freq, ...] > 1, nhit[feed, sb, freq, ...], np.nan)
        ax[1].imshow(plotdata)
        
        condition = nhit > 0
        inv_var = np.zeros_like(map)
        inv_var[condition] = 1 / rms[condition] ** 2
        map[condition] = map[condition] * inv_var[condition]

        print("hei", np.all(map == 0), np.all(nhit == 0), np.all(rms == 0))

        fig, ax = plt.subplots(1, 2)
        fig.suptitle("hei")
        ax[0].imshow(np.nansum(map[:, sb, freq, ...], axis = 0))
        ax[1].imshow(np.nansum(inv_var[:, sb, freq, ...], axis = 0))

        map_coadd = np.zeros_like(map[0, ...])
        nhit_coadd = np.zeros_like(nhit[0, ...])
        rms_coadd = np.zeros_like(rms[0, ...])

        condition = nhit > 0
        map_coadd = np.nansum(map, axis = 0) / np.nansum(inv_var, axis = 0)        
        print("hei", np.all(map_coadd == 0), np.all(nhit_coadd == 0), np.all(rms_coadd == 0), np.all(np.nansum(inv_var, axis = 0) == 0))
        rms_coadd = np.sqrt(1 / np.nansum(inv_var, axis = 0))
        print("hei", np.all(map_coadd == 0), np.all(nhit_coadd == 0), np.all(rms_coadd == 0))
        nhit_coadd = np.nansum(nhit, axis = 0)        
        print("hei", np.all(map_coadd == 0), np.all(nhit_coadd == 0), np.all(rms_coadd == 0))
        rms_coadd[nhit_coadd < 1] = 0
        map_coadd[nhit_coadd < 1] = 0
        #print(map.shape, rms.shape, nhit.shape, condition.shape)
        #print(np.any(np.isnan(map_coadd)), np.any(np.isnan(nhit_coadd)), np.any(np.isnan(rms_coadd)))
        #print(np.any(np.isinf(map_coadd)), np.any(np.isinf(nhit_coadd)), np.any(np.isinf(rms_coadd)))
        print("hei", np.all(map_coadd == 0), np.all(nhit_coadd == 0), np.all(rms_coadd == 0))

        fig, ax = plt.subplots(1, 2)
        fig.suptitle("ces coadd")
        plotdata = np.where(nhit_coadd[sb, freq, ...] > 1, 1e6 * map_coadd[sb, freq, ...], np.nan)
        ax[0].imshow(plotdata, vmin = -2000, vmax = 2000)
        plotdata = np.where(nhit_coadd[sb, freq, ...] > 1, nhit_coadd[sb, freq, ...], np.nan)
        ax[1].imshow(plotdata)
        
        outfile.create_dataset("map_coadd", data = map_coadd)
        outfile.create_dataset("nhit_coadd", data = nhit_coadd)
        outfile.create_dataset("rms_coadd", data = rms_coadd)

        """
        # Lissajous
        map_liss  = mapdata[0::2, ...] 
        nhit_liss = nhitdata[0::2, ...] 
        rms_liss  = rmsdata[0::2, ...] 

        condition = nhit_liss > 0
        inv_var_liss = np.zeros_like(map_liss)
        inv_var_liss[condition] = 1 / rms_liss[condition] ** 2
        map_liss[condition] = map_liss[condition] * inv_var_liss[condition]

        map = np.zeros_like(map_liss[0, ...])
        nhit = np.zeros_like(nhit_liss[0, ...])
        rms = np.zeros_like(rms_liss[0, ...])

        print(map.shape, rms.shape, nhit.shape, condition.shape)
        condition = nhit_liss > 1
        map = np.nansum(map_liss, axis = 0) / np.nansum(inv_var_liss, axis = 0)        
        rms = np.sqrt(1 / np.nansum(inv_var_liss, axis = 0))
        nhit = np.nansum(nhit_liss, axis = 0)        
        rms[nhit < 1] = 0
        map[nhit < 1] = 0
    
        print(np.any(np.isnan(map)), np.any(np.isnan(nhit)), np.any(np.isnan(rms)))
        print(np.any(np.isinf(map)), np.any(np.isinf(nhit)), np.any(np.isinf(rms)))

        outfile.create_dataset("map", data = map)
        outfile.create_dataset("nhit", data = nhit)
        outfile.create_dataset("rms", data = rms)

        fig, ax = plt.subplots(1, 2)
        fig.suptitle("liss")
        plotdata = np.where(nhit[feed, sb, freq, ...] > 1, 1e6 * map[feed, sb, freq, ...], np.nan)
        ax[0].imshow(plotdata, vmin = -2000, vmax = 2000)
        plotdata = np.where(nhit[feed, sb, freq, ...] > 1, nhit[feed, sb, freq, ...], np.nan)
        ax[1].imshow(plotdata)
        


        condition = nhit > 0
        inv_var = np.zeros_like(map)
        inv_var[condition] = 1 / rms[condition] ** 2
        map[condition] = map[condition] * inv_var[condition]

        map_coadd = np.zeros_like(map[0, ...])
        nhit_coadd = np.zeros_like(nhit[0, ...])
        rms_coadd = np.zeros_like(rms[0, ...])

        condition = nhit > 1
        map_coadd = np.nansum(map, axis = 0) / np.nansum(inv_var, axis = 0)        
        rms_coadd = np.sqrt(1 / np.nansum(inv_var, axis = 0))
        nhit_coadd = np.nansum(nhit, axis = 0)        
        rms_coadd[nhit_coadd < 1] = 0
        map_coadd[nhit_coadd < 1] = 0

        print(map.shape, rms.shape, nhit.shape, condition.shape)
        print(np.any(np.isnan(map_coadd)), np.any(np.isnan(nhit_coadd)), np.any(np.isnan(rms_coadd)))
        print(np.any(np.isinf(map_coadd)), np.any(np.isinf(nhit_coadd)), np.any(np.isinf(rms_coadd)))
     
        outfile.create_dataset("map_coadd", data = map_coadd)
        outfile.create_dataset("nhit_coadd", data = nhit_coadd)
        outfile.create_dataset("rms_coadd", data = rms_coadd)

        
        fig, ax = plt.subplots(1, 2)
        fig.suptitle("ces coadd")
        plotdata = np.where(nhit_coadd[sb, freq, ...] > 1, 1e6 * map_coadd[sb, freq, ...], np.nan)
        ax[0].imshow(plotdata, vmin = -2000, vmax = 2000)
        plotdata = np.where(nhit_coadd[sb, freq, ...] > 1, nhit_coadd[sb, freq, ...], np.nan)
        ax[1].imshow(plotdata)
        plt.show()
        
    elif "map" in key:
        continue
    elif "nhit" in key:
        continue
    elif "rms" in key:
        continue
    else:
        data = infile[key][()]
        outfile.create_dataset(key, data = data)
     
