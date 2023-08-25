import numpy as np 
import h5py 
import sys
import matplotlib.pyplot as plt

class Map():
    def __init__(self, mapname):
        self.mapname = mapname
        self.field   = self.mapname.split("/")[-1]
        self.field   = self.field.split("_")[0]
        
    def read_map(self):
        with h5py.File(self.mapname, "r") as infile:
            self.feeds = infile["feeds"][()]
            self.freq  = infile["freq"][()]
            
            if np.all(self.freq == 0):
                self.freq = np.linspace(26, 34, 256)

            self.map   = infile["map"][()]
            self.nhit  = infile["nhit"][()]
            self.rms   = infile["rms"][()]

            # self.map   = infile["multisplits/map_elev"][()]
            # self.nhit  = infile["multisplits/nhit_elev"][()]
            # self.rms   = infile["multisplits/rms_elev"][()]
           

            # #idx = (0, 2) # Lissajous
            # idx = (1, 3)  # CES
            # self.map = self.map[idx, ...]
            # self.nhit = self.nhit[idx, ...]
            # self.rms = self.rms[idx, ...]


            # var_inv = np.where(self.nhit > 0, 1 / (self.rms ** 2), 0)

            # map = np.where(self.nhit > 0, self.map * var_inv, 0)

            # map = np.sum(map, axis = 0)

            # inv_var = np.sum(var_inv, axis = 0)

            # map = np.where(np.isnan(map / inv_var) == False, map / inv_var, 0)

            # self.map = map.copy()
            # # print(np.all(self.rms == 0))
            # # print(np.all(self.rms == 0))
            # self.nhit = np.sum(self.nhit, axis = 0).astype(np.int32)
            # self.rms = np.where(self.nhit > 0, 1 / np.sqrt(inv_var), 0)

            # #print(np.where(self.nhit != 0))
            # #fig, ax = plt.subplots(2, 2)
            # fig, ax = plt.subplots()
            # # ax[0, 0].imshow(self.nhit[0, 2, 2, 6, ...])
            # # ax[0, 1].imshow(self.nhit[1, 2, 2, 6, ...])
            # # ax[1, 0].imshow(self.nhit[2, 2, 2, 6, ...])
            # # ax[1, 1].imshow(self.nhit[3, 2, 2, 6, ...])
            # ax.imshow(self.rms[2, 2, 6, ...])
            # plt.show()
            # sys.exit()

            try:
                self.map_coadd   = infile["map_coadd"][()]
                self.nhit_coadd  = infile["nhit_coadd"][()]
                self.rms_coadd   = infile["rms_coadd"][()]
            except:
                self.map_coadd   = infile["map_beam"][()]
                self.nhit_coadd  = infile["nhit_beam"][()]
                self.rms_coadd   = infile["rms_beam"][()]

            self.mean_az = infile["mean_az"][()]
            self.mean_el = infile["mean_el"][()]
            self.n_x     = infile["n_x"][()]
            self.n_y     = infile["n_y"][()]

            try: 
                self.nside   = infile["nside"][()]
                self.nsim    = infile["nsim"][()]
                self.nsplit  = infile["nsplit"][()]
                self.runID   = infile["runID"][()]
                self.patch_center = infile["patch_center"][()]
            except:
                self.nside   = 0
                self.nsim    = 0
                self.nsplit  = 0
                self.runID   = None
                self.patch_center = [None, None]
            
            self.time    = infile["time"][()]
            self.x       = infile["x"][()]
            self.y       = infile["y"][()]

        self.n_det, self.n_sb, self.n_freq = self.map.shape[:3]
        
