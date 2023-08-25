import numpy as np
import h5py
import matplotlib.pyplot as plt
import time

class TsysMeasure:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def load_data_from_arrays(self, vane_angles, vane_times, array_features, T_hot, airtemp, tod, tod_times, feeds):
        if self.verbose:
            print("Starting arrayload")
            t0 = time.time()
        self.vane_angles = vane_angles
        self.vane_times = vane_times
        self.array_features = array_features
        self.Thot_cont = T_hot/100.0 + 273.15
        self.airtemp = airtemp
        self.tod_times = tod_times
        self.nr_vane_times = len(vane_times)

        vane_active = array_features&(2**13) != 0
        self.vane_time1 = vane_times[:self.nr_vane_times//2]
        self.vane_time2 = vane_times[self.nr_vane_times//2:]
        self.vane_active1 = vane_active[:self.nr_vane_times//2]
        self.vane_active2 = vane_active[self.nr_vane_times//2:]

        self.nfeeds, self.nbands, self.nfreqs, self.ntod = tod.shape
        self.feeds = feeds

        self.tod = tod

        self.Thot = np.zeros((self.nfeeds, 2))
        self.Pcold_obsidmean = np.zeros((self.nfeeds, self.nbands, self.nfreqs))
        self.Phot = np.zeros((self.nfeeds, self.nbands, self.nfreqs, 2))  # P_hot measurements from beginning and end of obsid.
        self.Phot_unc = np.zeros((self.nfeeds, self.nbands, self.nfreqs, 2))
        self.Phot[:] = np.nan  # All failed calcuations of Tsys should result in a nan, not a zero.
        self.Phot_unc[:] = np.nan

        self.calib_times = np.zeros((self.nfeeds, 2))
        self.points_used_Phot = np.zeros((self.nfeeds, 2))
        self.points_used_Thot = np.zeros((self.nfeeds, 2))
        self.calib_startstopindices_tod = np.zeros((2, 2), dtype=int)  # Start and end indices, in tod_time format, for "calibration phase".
        self.calib_startstop_times = np.zeros((self.nfeeds, 2, 2))  # MJD times for calibration interval (actually used datapoints).
        self.successful = np.zeros((self.nfeeds, 2), dtype=int)

        self.ERROR = 0

        self.TCMB = 2.725
        if self.verbose:
            print("Finished arrayload in %.2f s" % (time.time()-t0))


    def load_data_from_file(self, filename):
        if self.verbose:
            print("Starting fileread")
            t0 = time.time()
        f = h5py.File(filename, "r")
        vane_angles    = f["/hk/antenna0/vane/angle"][()]/100.0  # Degrees
        vane_times     = f["/hk/antenna0/vane/utc"][()]
        array_features = f["/hk/array/frame/features"][()]
        tod            = f["/spectrometer/tod"][()]
        tod_times      = f["/spectrometer/MJD"][()]
        feeds          = f["/spectrometer/feeds"][()]
        airtemp        = f["/hk/array/weather/airTemperature"][()] + 273.15   # Warning: We assume that /hk/array/weather/utc == /hk/antenna0/vane/utc,
        if tod_times[0] > 58712.03706:                                        # which is not strictly true, but off by a second or two at most (I think),
            T_hot      = f["/hk/antenna0/vane/Tvane"][()]                     # but fine for our purposes of comparing Thot to ambient air temp.
        else:
            T_hot      = f["/hk/antenna0/env/ambientLoadTemp"][()]
        if self.verbose:
            print("Finished fileread in %.2f s" % (time.time()-t0))
        self.load_data_from_arrays(vane_angles, vane_times, array_features, T_hot, airtemp, tod, tod_times, feeds)


    def solve(self):
        if self.verbose:
            print("Starting solve")
            t0 = time.time()
        ### Step 1: Calculate P_hot at the start and end Tsys measurement points. ###
        vane_time1, vane_time2, vane_active1, vane_active2, tod, tod_times = self.vane_time1, self.vane_time2, self.vane_active1, self.vane_active2, self.tod, self.tod_times
        nfeeds, nbands, nfreqs, ntod = self.nfeeds, self.nbands, self.nfreqs, self.ntod
        for i, vane_timei, vane_activei in [[0, vane_time1, vane_active1], [1, vane_time2, vane_active2]]:
            if np.sum(vane_activei) > 1: # Require at least 2 timestamps of active vane.
                vane_timei = vane_timei[vane_activei]
                tod_start_idx = np.argmin(np.abs(vane_timei[0]-tod_times))  # Find closest TOD timestamps to the vane start and stop timestamps.
                tod_stop_idx = np.argmin(np.abs(vane_timei[-1]-tod_times))
                self.calib_startstopindices_tod[i, :] = tod_start_idx, tod_stop_idx
                for feed_idx in range(nfeeds):
                    if self.feeds[feed_idx] != 20:  # Ignore feed 20.
                        todi = tod[feed_idx, :, :, tod_start_idx:tod_stop_idx]
                        tod_timesi = tod_times[tod_start_idx:tod_stop_idx]
                        tod_freq_mean = np.nanmean(todi, axis=(0,1))
                        if np.sum(tod_freq_mean > 0) > 10:  # Check number of valid points. Also catches NaNs.
                            threshold = 0.95*np.nanmax(tod_freq_mean)
                            threshold_idxs = np.argwhere(tod_freq_mean > threshold)  # Points where tod is at least 95% of max. (We assume this is only true during Tsys measurement).
                            if threshold > 2*np.median(tod_freq_mean):  # The calib should be substantially higher in power than the median. If not, there is likely no successful calib.
                                if threshold_idxs[-1][0] - threshold_idxs[0][0] > 90:
                                    margin = 40
                                elif threshold_idxs[-1][0] - threshold_idxs[0][0] < 40:
                                    self.successful[feed_idx, i] = -4  # Fewer than 40 TOD timestamps in calibration.
                                    continue
                                else:
                                    margin = (threshold_idxs[-1][0] - threshold_idxs[0][0])//2 - 5

                                min_idxi = threshold_idxs[0][0] + margin  # Take the first and last of points fulfilling the above condition, assume they represent start and end of 
                                max_idxi = threshold_idxs[-1][0] - margin  # Tsys measurement, and add a 40-idx safety margin (40*20ms = 0.8 seconds.)
                                min_idx_vane = np.argmin(np.abs(self.vane_times - tod_timesi[min_idxi]))  # Find closest vane timestamps to the selected TOD timestamps.
                                max_idx_vane = np.argmin(np.abs(self.vane_times - tod_timesi[max_idxi]))
                                if max_idxi > min_idxi + 1 and max_idx_vane > min_idx_vane:  # Require at least 2 TOD timestamps and 1 vane timestamp to be in calibration region.
                                    Thot_cont_calib = self.Thot_cont[min_idx_vane:max_idx_vane]
                                    airtemp_calib = self.airtemp[min_idx_vane:max_idx_vane]
                                    tempdiff = np.abs(Thot_cont_calib - airtemp_calib)
                                    # Cut if outside -30C to 60C degrees range, and if difference to air temperature exceeds 38 degrees.
                                    Thot_cont_calib = Thot_cont_calib[(tempdiff < 38) * (Thot_cont_calib > -20 + 273.15) * (Thot_cont_calib < 50 + 273.15)]
                                    if np.size(Thot_cont_calib) > 0:  # Check if at least one Thot measurements made the cut.
                                        self.points_used_Thot[feed_idx, i] = np.size(Thot_cont_calib)
                                    else:
                                        if np.size(airtemp_calib) > 0:
                                            Thot_cont_calib = airtemp_calib
                                            self.points_used_Thot[feed_idx, i] = -1  # points_used_Thot = -1 indicates that ambient air temp was used instead.
                                        else:
                                            self.successful[feed_idx, i] = -6  # No Thot or ambient 
                                            continue

                                    self.Thot[feed_idx, i] = np.nanmean(Thot_cont_calib)
                                    self.Phot[feed_idx, :, :, i] = np.nanmean(todi[:,:,min_idxi:max_idxi], axis=(2))
                                    self.Phot_unc[feed_idx, :, :, i] = np.nanstd(todi[:,:,min_idxi:max_idxi], axis=(2))/np.sqrt(max_idxi-min_idxi)
                                    self.points_used_Phot[feed_idx, i] = max_idxi - min_idxi
                                    self.calib_startstop_times[feed_idx, i] = (tod_timesi[min_idxi], tod_timesi[max_idxi])
                                    self.calib_times[feed_idx, i] = (tod_timesi[max_idxi] + tod_timesi[min_idxi])/2
                                    if np.isfinite(self.Thot[feed_idx, i]):
                                        if np.isfinite(self.Phot[feed_idx, :, :, i]).all():
                                            self.successful[feed_idx, i] = 1
                                        else:
                                            self.successful[feed_idx, i] = -8  # Nans in Phot output.
                                    else:
                                        self.successful[feed_idx, i] = -7  # Nans in Thot output.
                                else:
                                    self.successful[feed_idx, i] = -5  # Fewer than 2 timestamps within 95% power interval.
                            else:
                                self.successful[feed_idx, i] = -3  # Too low power for calib. Probably failed vane.
                        else:
                            self.successful[feed_idx, i] = -2  # Too few valid TOD points within vane active flag.
                    else:
                        self.successful[feed_idx, i] = -20  # Feed 20.
                    
            else:
                self.successful[:, i] = -1  # Less than 5 timestamps within vane active flag.

        self.Pcold_obsidmean = np.zeros((nfeeds, nbands, nfreqs))
        for feed in range(nfeeds):
            self.Pcold_obsidmean[feed,:,:] = np.nanmean(self.tod[feed,:,:,self.calib_startstopindices_tod[0,1]+1000:self.calib_startstopindices_tod[1,0]-1000], axis=-1)
        self.Phot_obsidmean = np.nanmean(self.Phot, axis=-1)
        self.Thot_obsidmean = np.nanmean(self.Thot, axis=-1)
        self.G = (self.Phot_obsidmean - self.Pcold_obsidmean)/(self.Thot_obsidmean[:, None, None] - self.TCMB)
        self.Tsys = self.Pcold_obsidmean/self.G

        if self.verbose:
            print("Finished solve in %.2f s" % (time.time()-t0))