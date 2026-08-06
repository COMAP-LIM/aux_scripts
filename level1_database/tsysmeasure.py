import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
from scipy.ndimage import median_filter

class TsysMeasure:
    """Solve for the system temperature from the calibration vane dips of an observation.

    A room-temperature vane is flipped in front of the focal plane at the start and end of each
    ~1 hour observation. Comparing the power with the vane in (P_hot, at the known vane
    temperature T_hot) against the power on sky (P_cold) gives the gain, and from it Tsys.

    The status of every (feed, vane dip) pair is recorded in `successful`: 1 on success, and a
    negative code identifying the first check that rejected it otherwise. See FAILURE_CODES in
    level1_database/tsys_failure_stats.py for the full legend.
    """

    MEDFILT = 5                 # Median filter width applied to the band-averaged power.
    MIN_FINITE_FRAC = 0.25      # Minimum fraction of finite Phot channels to accept a dip.
    TSYS_RANGE = (10.0, 100.0)  # Physically plausible Tsys [K]; outside this the dip is rejected.
    SKY_GUARD = 500             # Samples skipped either side of a dip before sampling sky power.
    SKY_LENGTH = 2000           # Length of the sky reference window, in TOD samples.

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

        # The vane dips are located from this flag directly (see vane_runs), rather than by
        # splitting the array in half and assuming one dip per half.
        self.vane_active = array_features&(2**13) != 0

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

        # Diagnostics written alongside the calibration, so that a later cut can be made without
        # re-reading the level1 data.
        self.Tsys_vane = np.zeros((self.nfeeds, 2)) + np.nan      # Tsys per individual vane dip.
        self.Phot_finite_frac = np.zeros((self.nfeeds, 2)) + np.nan  # Finite fraction of Phot.
        self.feed_usable = np.zeros(self.nfeeds, dtype=bool)      # Feed has >= 1 good dip.
        self.n_vane_runs = 0                                      # Vane flag activations found.

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


    def vane_runs(self):
        """Find the contiguous stretches of vane-active housekeeping samples.

        Returns:
            List of (start, stop) index pairs into the housekeeping arrays, stop exclusive,
            for every run of at least 2 active samples.
        """
        edges = np.diff(np.concatenate([[0], self.vane_active.astype(int), [0]]))
        starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
        return [(a, b) for a, b in zip(starts, stops) if b - a > 1]


    def sky_window(self, tod_start_idx, tod_stop_idx):
        """Pick a stretch of plain sky next to a vane dip, for the Tsys sanity check.

        Takes whichever side of the dip has more room, skipping SKY_GUARD samples first so the
        vane transition itself is never included.
        """
        before = slice(max(0, tod_start_idx - self.SKY_GUARD - self.SKY_LENGTH),
                       max(0, tod_start_idx - self.SKY_GUARD))
        after = slice(min(self.ntod, tod_stop_idx + self.SKY_GUARD),
                      min(self.ntod, tod_stop_idx + self.SKY_GUARD + self.SKY_LENGTH))
        return after if (after.stop - after.start) >= (before.stop - before.start) else before


    def solve(self):
        if self.verbose:
            print("Starting solve")
            t0 = time.time()
        tod, tod_times = self.tod, self.tod_times
        nfeeds, nbands, nfreqs, ntod = self.nfeeds, self.nbands, self.nfreqs, self.ntod

        ### Step 1: Locate the two vane dips. ###
        # The dips are taken as contiguous runs of the vane-active flag. Splitting the
        # housekeeping arrays in half and treating each half as one dip fails whenever both dips
        # land in the same half: the calibration window then runs from the first sample of the
        # first dip to the last of the second, averaging P_hot over the minutes of sky data in
        # between, which silently produces a badly inflated Tsys that still passes every check.
        runs = self.vane_runs()
        self.n_vane_runs = len(runs)
        dips = [None, None]
        if len(runs) >= 2:
            # First run is the opening dip, last is the closing one. Extra runs in between are
            # spurious flag activations and are ignored rather than merged into the window.
            dips[0], dips[1] = runs[0], runs[-1]
        elif len(runs) == 1:
            # Only one dip was recorded. Assign it to the end of the observation it sits in, so
            # that the surviving calibration is attributed to the right half.
            mid_time = 0.5*(tod_times[0] + tod_times[-1])
            dips[0 if self.vane_times[runs[0][0]] < mid_time else 1] = runs[0]

        ### Step 2: Calculate P_hot at the start and end Tsys measurement points. ###
        for i, dip in enumerate(dips):
            if dip is None:
                self.successful[:, i] = -1  # No vane dip recorded at this end of the observation.
                continue
            first, last = dip[0], dip[1] - 1
            tod_start_idx = np.argmin(np.abs(self.vane_times[first] - tod_times))  # Find closest TOD timestamps to the vane start and stop timestamps.
            tod_stop_idx = np.argmin(np.abs(self.vane_times[last] - tod_times))
            self.calib_startstopindices_tod[i, :] = tod_start_idx, tod_stop_idx
            sky_slice = self.sky_window(tod_start_idx, tod_stop_idx)

            for feed_idx in range(nfeeds):
                if self.feeds[feed_idx] == 20:
                    self.successful[feed_idx, i] = -20  # Feed 20.
                    continue
                todi = tod[feed_idx, :, :, tod_start_idx:tod_stop_idx]
                tod_timesi = tod_times[tod_start_idx:tod_stop_idx]

                # Median filter the band-averaged power before anything measures its maximum.
                # Isolated RFI or readout spikes otherwise set the maximum, which both hides the
                # real power level and, on a dead feed, makes a couple of glitches look like a
                # calibration. The filter removes single-sample excursions while leaving the
                # plateau untouched; the wider settling overshoot is handled by the plateau
                # reference below.
                tod_freq_mean = median_filter(np.nanmean(todi, axis=(0, 1)), size=self.MEDFILT)

                if np.sum(tod_freq_mean > 0) <= 10:  # Check number of valid points. Also catches NaNs.
                    self.successful[feed_idx, i] = -2  # Too few valid TOD points within vane active flag.
                    continue

                # Reference the threshold to the vane plateau rather than to the peak sample.
                # The dip frequently overshoots by 10-20% for a few tenths of a second as the
                # vane settles, which is too wide for the median filter to remove; keying off
                # the maximum then puts 0.95*max above the plateau itself, so only the overshoot
                # counts as "in calibration" and the dip is discarded as too short (-4).
                # Everything above the midpoint between peak and sky is vane, and the plateau
                # outlasts the overshoot, so the median of that region is the plateau level.
                cut = 0.5*(np.nanmax(tod_freq_mean) + np.nanpercentile(tod_freq_mean, 5))
                high = tod_freq_mean > cut
                plateau = np.nanmedian(tod_freq_mean[high]) if np.any(high) else np.nanmax(tod_freq_mean)
                threshold = 0.95*plateau  # Points where tod is at least 95% of the vane plateau.
                # nanpercentile, not percentile: percentile propagates NaNs, so a single NaN
                # sample anywhere in the window made this comparison NaN > x, i.e. False, and
                # rejected the feed no matter how good the calibration was.
                # The check itself is very permissive. With the 5th percentile sitting at the
                # sky level, requiring 0.95*max > 2*p5 is requiring P_hot/P_cold > 2.1, i.e.
                # Tsys < ~250 K, where a healthy feed is 30-60 K. It only fires when the vane
                # never actually appeared in front of the feed.
                if threshold <= 2*np.nanpercentile(tod_freq_mean, 5):
                    self.successful[feed_idx, i] = -3  # Too low power for calib. Probably failed vane.
                    continue

                threshold_idxs = np.argwhere(tod_freq_mean > threshold)
                span = threshold_idxs[-1][0] - threshold_idxs[0][0]
                if span < 40:
                    self.successful[feed_idx, i] = -4  # Fewer than 40 TOD timestamps in calibration.
                    continue
                # Take the first and last of the points fulfilling the above condition, assume
                # they represent start and end of the Tsys measurement, and trim a safety margin
                # off each side (40*20ms = 0.8 seconds).
                margin = 40 if span > 90 else span//2 - 5
                min_idxi = threshold_idxs[0][0] + margin
                max_idxi = threshold_idxs[-1][0] - margin
                if max_idxi <= min_idxi + 1:  # Require at least 2 TOD timestamps in the calibration region.
                    self.successful[feed_idx, i] = -5  # Fewer than 2 timestamps within 95% power interval.
                    continue

                # Find the closest vane timestamps to the selected TOD timestamps. The vane
                # thermometer is sampled at ~2 Hz while the trimmed window is often only a
                # fraction of a second, so it regularly contains no housekeeping sample at all.
                # Requiring one strictly inside the window therefore threw away good dips;
                # T_vane is constant across a dip, so the nearest samples are equally valid, and
                # the inclusive slice below can never come out empty.
                min_idx_vane = np.argmin(np.abs(self.vane_times - tod_timesi[min_idxi]))
                max_idx_vane = np.argmin(np.abs(self.vane_times - tod_timesi[max_idxi]))
                Thot_cont_calib = self.Thot_cont[min_idx_vane:max_idx_vane+1]
                airtemp_calib = self.airtemp[min_idx_vane:max_idx_vane+1]
                tempdiff = np.abs(Thot_cont_calib - airtemp_calib)
                # Cut if outside -30C to 60C degrees range, and if difference to air temperature exceeds 38 degrees.
                Thot_cont_calib = Thot_cont_calib[(tempdiff < 38) * (Thot_cont_calib > -20 + 273.15) * (Thot_cont_calib < 50 + 273.15)]
                if np.size(Thot_cont_calib) > 0:  # Check if at least one Thot measurements made the cut.
                    self.points_used_Thot[feed_idx, i] = np.size(Thot_cont_calib)
                elif np.size(airtemp_calib) > 0:
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

                if not np.isfinite(self.Thot[feed_idx, i]):
                    self.successful[feed_idx, i] = -7  # Nans in Thot output.
                    continue

                # Judge P_hot on the fraction of channels that came out finite, rather than
                # demanding every one of them. Tsys is stored per (feed, band, frequency), so a
                # dead channel already turns into a NaN Tsys for that channel alone; failing the
                # whole dip for it discarded good data, and a feed with one dead sideband still
                # has half its band usable.
                finite_frac = np.isfinite(self.Phot[feed_idx, :, :, i]).mean()
                self.Phot_finite_frac[feed_idx, i] = finite_frac
                if finite_frac < self.MIN_FINITE_FRAC:
                    self.successful[feed_idx, i] = -8  # Too much of the Phot spectrum is NaN.
                    continue

                # Sanity-check the dip against physics before accepting it. For a vane
                # calibration Tsys = (Thot - TCMB)/(Phot/Pcold - 1), so comparing P_hot against
                # nearby sky power gives the system temperature this dip implies. Anything far
                # outside the plausible range means the window did not capture a clean vane dip,
                # which is the one failure that is otherwise silent.
                Pcold_ref = np.nanmean(tod[feed_idx, :, :, sky_slice])
                Phot_ref = np.nanmean(self.Phot[feed_idx, :, :, i])
                with np.errstate(invalid="ignore", divide="ignore"):
                    tsys_vane = (self.Thot[feed_idx, i] - self.TCMB)/(Phot_ref/Pcold_ref - 1)
                self.Tsys_vane[feed_idx, i] = tsys_vane
                if not self.TSYS_RANGE[0] < tsys_vane < self.TSYS_RANGE[1]:
                    self.successful[feed_idx, i] = -9  # Implied Tsys is unphysical.
                    continue

                self.successful[feed_idx, i] = 1

        # A rejected dip must not leak into the obsID averages below. Codes that fail late
        # (-7, -8, -9) have already written Phot and Thot, so clear them here.
        for i in range(2):
            rejected = self.successful[:, i] != 1
            self.Phot[rejected, :, :, i] = np.nan
            self.Phot_unc[rejected, :, :, i] = np.nan
            self.Thot[rejected, i] = np.nan

        # A feed is usable as long as one of its two dips survived: a single dip still
        # calibrates the observation, only without the start-to-end interpolation. This is the
        # flag downstream should cut on, rather than discarding a feed (or a whole observation)
        # because any one dip failed.
        self.feed_usable = np.any(self.successful == 1, axis=1)

        ### Step 3: Calculate P_cold, the gain, and Tsys. ###
        # P_cold is the sky power between the two dips. Where a dip is missing, extend the
        # window to that end of the observation instead of leaving it undefined.
        cold_start = self.calib_startstopindices_tod[0, 1] + 1000 if dips[0] is not None else 0
        cold_stop = self.calib_startstopindices_tod[1, 0] - 1000 if dips[1] is not None else ntod
        cold_start, cold_stop = max(0, cold_start), min(ntod, cold_stop)
        if cold_stop - cold_start < 100:  # Degenerate window (very short observation): use it all.
            cold_start, cold_stop = 0, ntod

        self.Pcold_obsidmean = np.zeros((nfeeds, nbands, nfreqs))
        for feed in range(nfeeds):
            self.Pcold_obsidmean[feed,:,:] = np.nanmean(self.tod[feed,:,:,cold_start:cold_stop], axis=-1)
        self.Phot_obsidmean = np.nanmean(self.Phot, axis=-1)
        self.Thot_obsidmean = np.nanmean(self.Thot, axis=-1)
        self.G = (self.Phot_obsidmean - self.Pcold_obsidmean)/(self.Thot_obsidmean[:, None, None] - self.TCMB)
        self.Tsys = self.Pcold_obsidmean/self.G

        if self.verbose:
            print("Finished solve in %.2f s" % (time.time()-t0))
