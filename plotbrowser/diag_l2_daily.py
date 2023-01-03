import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy.fft as fft
import time
import glob
import os
import h5py
import math

class spike_data():
    def __init__(self):
        self.spike_types = ['spike', 'jump', 'anomaly', 'edge spike']
        pass

class spike_list():
    def __init__(self):
        self.spikes = []
        self.spike_types = ['spike', 'jump', 'anomaly', 'edge spike']
    
    def add(self, spike):
        self.spikes.append(spike)
    
    def addlist(self, sp_list):
        for sp in sp_list:
            self.add(sp)

    def sorted(self):
        lists = [[], [], []]
        for spike in self.spikes:
            lists[spike.type].append(spike)
        for typelist in lists:
            typelist.sort(key=lambda x: np.abs(x.amp), reverse=True)  # hat tip: https://stackoverflow.com/a/403426/5238625
        return lists

def get_spike_list(sb_mean, sd, scan_id, mjd):
    my_spikes = spike_list()
    for spike_type in range(3):
        for spike in range(1000):
            sbs = sd[0, :, :, spike_type, spike]
            if np.all(sbs == 0):
                break
            max_sb = np.unravel_index(np.argmax(np.abs(sbs), axis=None), sbs.shape)
            max_ind = int(sd[1, max_sb[0], max_sb[1], spike_type, spike]) - 1
            s = spike_data()
            s.amp = sd[0, max_sb[0], max_sb[1], spike_type, spike]
            s.ind = np.array((max_sb[0], max_sb[1], max_ind))  # feed, sb, ind
            s.mjd = mjd[max_ind]
            s.data = sb_mean[max_sb[0], max_sb[1], max_ind - 200:max_ind + 200]
            s.type = spike_type
            s.scanid = scan_id
            my_spikes.add(s)
    return my_spikes

def get_corr(scan):
    with h5py.File(scan, mode="r") as my_file:
        scan_id = filename[-12:-3]# + filename[-5:-3]
        tod_ind = np.array(my_file['tod'][:])
        n_det_ind, n_sb, n_freq, n_samp = tod_ind.shape
        sb_mean_ind = np.array(my_file['sb_mean'][:])
        mask_ind = my_file['freqmask'][:]
        mask_full_ind = my_file['freqmask_full'][:]
        reason_ind = my_file['freqmask_reason'][:]
        pixels = np.array(my_file['pixels'][:]) - 1 
        pix2ind = my_file['pix2ind'][:]
        try: 
            sd_ind = np.array(my_file['spike_data'])
        except KeyError:
            sd_ind = np.zeros((3, n_det_ind, n_sb, 4, 1000))
        try: 
            chi2_ind = np.array(my_file['chi2'])
        except KeyError:
            chi2_ind = np.zeros_like(tod_ind[:,:,:,0])
        try:
            acc_ind = np.array(my_file['acceptrate'])
        except KeyError:
            acc_ind = np.zeros_like(tod_ind[:,:,0,0])
            print("Found no acceptrate")
        time = np.array(my_file['time'])
        mjd = time
        try:
            pca = np.array(my_file['pca_comp'])
            eigv = np.array(my_file['pca_eigv'])
            ampl_ind = np.array(my_file['pca_ampl'])
        except KeyError:
            pca = np.zeros((4, 10000))
            eigv = np.zeros(0)
            ampl_ind = np.zeros((4, *mask_full_ind.shape))
            print('Found no pca comps')
        try:
            tsys_ind = np.array(my_file['Tsys_lowres'])
        except KeyError:
            tsys_ind = np.zeros_like(tod_ind[:,:,:,0]) + 40
            print("Found no tsys")
    t0 = time[0]
    time = (time - time[0]) * (24 * 60)  # minutes

    n_freq_hr = len(mask_full_ind[0,0])
    n_det = np.max(pixels) + 1 
    # print(n_det)

    ## transform to full arrays with all pixels
    tod = np.zeros((n_det, n_sb, n_freq, n_samp))
    mask = np.zeros((n_det, n_sb, n_freq))
    mask_full = np.zeros((n_det, n_sb, n_freq_hr))
    acc = np.zeros((n_det, n_sb))
    ampl = np.zeros((4, n_det, n_sb, n_freq_hr))
    tsys = np.zeros((n_det, n_sb, n_freq))
    chi2 = np.zeros((n_det, n_sb, n_freq))
    sd = np.zeros((3, n_det, n_sb, 4, 1000))
    sb_mean = np.zeros((n_det, n_sb, n_samp))
    reason = np.zeros((n_det, n_sb, n_freq_hr))
    # print(ampl_ind.shape)
    # print(ampl[:, pixels, :, :].shape)

    tod[pixels] = tod_ind
    mask[pixels] = mask_ind
    mask_full[pixels] = mask_full_ind
    reason[pixels] = reason_ind
    acc[pixels] = acc_ind
    ampl[:, pixels, :, :] = ampl_ind
    tsys[pixels] = tsys_ind
    chi2[pixels] = chi2_ind
    sd[:, pixels, :, :, :] = sd_ind
    sb_mean[pixels] = sb_mean_ind 

    acc = acc.flatten()
    # n_det, n_sb, n_freq, n_samp = tod.shape

    tod = tod[:, :, :, :] * mask[:, :, :, None]
    tod[:, (0, 2)] = tod[:, (0, 2), ::-1]

    tod_flat = tod.reshape((n_det * n_sb * n_freq, n_samp))
    corr = np.corrcoef(tod_flat)
    
    chi2 = chi2.flatten()
    chi2[(chi2 == 0.0)] = np.nan
    tsys = tsys.flatten()
    tsys[(tsys == 0.0)] = np.nan
    
    my_spikes = get_spike_list(sb_mean, sd, scan_id, mjd)

    reasonarr = np.zeros(6)
    r = reason[pixels[:-1]].flatten()
    reasonarr[0] = len(r[(r == 1)])
    reasonarr[1] = len(r[(r == 2)])
    reasonarr[2] = len(np.where(np.logical_and(r>=3, r<=13))[0])
    reasonarr[3] = len(r[(r == 15)])
    reasonarr[4] = len(np.where(np.logical_and(r>=16, r<=27))[0])
    reasonarr[5] = len(np.where(np.logical_and(r>=40, r<=41))[0])
    

    reasonarr /= n_det_ind * n_sb * n_freq
    return corr, 1.0 / n_samp, n_det, acc, chi2, tsys, t0, my_spikes, reasonarr

# param_file = sys.argv[1]
try:
    param_file = sys.argv[1]
except IndexError:
    print('You need to provide param file as command-line argument')
    sys.exit()


params = {}
with open(param_file) as f:
    fr = f.readlines()

    fr = [f[:-1] for f in fr]

    frs = [f.split(" = ") for f in fr]

    for stuff in frs:
        try:
            i, j = stuff
            params[str(i).strip()] = eval(j)
        except (ValueError, SyntaxError) as ex:
            pass


filename = params['RUNLIST']

l2_path = params['LEVEL2_DIR']

with open(filename) as my_file:
    lines = [line.split() for line in my_file]
i = 0

obsids = []

print(filename)

n_fields = int(lines[i][0])
i = i + 1
for i_field in range(n_fields):
    fieldname = lines[i][0]
    n_obsids = int(lines[i][1])
    print(fieldname)
    i = i + 1
    for j in range(n_obsids):
        obsids.append(lines[i][0])
        n_scans = int(lines[i][3])
        #print(n_scans)
        i = i + n_scans + 1 
    #print(n_scans)

    for obsid in obsids:
        if int(obsid) < 7321:
            continue
        print(l2_path + '/' + fieldname + '/' + fieldname + '_0' + obsid)
        scan_list = glob.glob(l2_path + '/' + fieldname + '/' + fieldname + '_0' + obsid + '*.h5')
        n_scans = len(scan_list)
        #print(scan_list)
        if (len(scan_list) == 0):
            print('No scans in obsid: ' + obsid)
            continue
        corrs = []
        variances = []
        accs = []
        chi2s = []
        tsyss = []
        t0s = []
        rs = []
        my_spikes = spike_list()
        for scan in scan_list:
            try:
                corr, var, n_det, acc, chi2, tsys, t0, spike_dat, r = get_corr(scan)
            except:
                continue
            #corr, var, n_det, acc, chi2, tsys, t0, spike_dat, r = get_corr(scan)
            corrs.append(corr)
            variances.append(var)
            accs.append(acc)
            chi2s.append(chi2)
            tsyss.append(tsys)
            t0s.append(t0)
            rs.append(r)
            my_spikes.addlist(spike_dat.spikes)
        if len(corrs) == 0:
            print('No working scans in obsid')
            continue
        t0 = (np.mean(np.array(t0s)) * 24 - 7) % 24 
        tstart = np.min(np.array(t0s))
        corrs = np.array(corrs)
        variances = np.array(variances)
        reason = np.nanmean(np.array(rs), 0)
        weights = 1 / variances[:, None, None] * (np.isfinite(corrs)) #(corrs != 0)
        
        W = np.sum(weights, 0)
        
        
        corr[(W > 0)] = 1 / W[(W > 0)] * np.nansum(corrs[:, (W > 0)] * weights[:, (W > 0)], 0)

        fig = plt.figure()
        ax = fig.add_subplot(111)

        vmax = 0.05
        im = ax.imshow(corr, vmin=-vmax, vmax=vmax,  #vmin=-1, vmax=1,#vmin=-0.1, vmax=0.1,
                        extent=(0.5, n_det + 0.5, n_det + 0.5, 0.5))
        cbar = fig.colorbar(im)
        cbar.set_label('Correlation')
        new_tick_locations = np.array(range(n_det)) + 1
        ax.set_xticks(new_tick_locations)
        ax.set_yticks(new_tick_locations)
        # ax.vlines(np.linspace(0.5, n_det + 0.5, n_det + 1), ymin=0, ymax=1, linestyle='--', color='k', lw=0.6, alpha=0.2)
        xl = np.linspace(0.5, n_det + 0.5, n_det * 1 + 1)
        ax.vlines(xl, ymin=0.5, ymax=n_det + 0.5, linestyle='-', color='k', lw=0.05, alpha=1.0)
        ax.hlines(xl, xmin=0.5, xmax=n_det + 0.5, linestyle='-', color='k', lw=0.05, alpha=1.0)
        # ax.hlines(np.linspace(0.5, n_det + 0.5, n_det * 4 + 1), xmin=0.5, xmax=n_det + 0.5, linestyle='--', color='k', lw=0.01, alpha=0.1)
        for j in range(n_det):
            plt.text(xl[j] + 0.7, xl[j] + 0.2, str(j+1), rotation=0, verticalalignment='center', fontsize=2)
        ax.set_xlabel('Feed')
        ax.set_ylabel('Feed')
        ax.set_title('Obsid: ' + str(obsid))
        #save_folder = '/mn/stornext/u3/haavarti/www_docs/diag/'
        save_folder = 'test_figs/'
        save_string = '_%06i.png' % (int(obsid))
        print(save_folder + 'corr_hr/' + 'corr_highres' + save_string)
        plt.savefig(save_folder + 'corr_hr/' + 'corr_highres' + save_string, bbox_inches='tight', dpi=800) 
        plt.savefig(save_folder + 'corr_lr/' + 'corr_lowres' + save_string, bbox_inches='tight', dpi=150) 
        
        acc = np.mean(np.array(accs), 0)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(311)#211)
        
        ax.set_title('Obsid: ' + str(obsid) + ', utc - 7: %02i [h]' % round(t0))

        n = len(acc)
        # for j in range(n):
        #     print(np.array(file_list[i]['freqmask_full']).flatten()[j])
        x = np.linspace(0.5, n_det + 0.5, len(acc) + 1)
        plotter = np.zeros((len(acc) + 1))
        plotter[:-1] = acc
        eff_feeds = (acc.sum()/4.0)
        ax.step(x, plotter, where='post', label='Effective feeds: ' + '%.2f' % eff_feeds)##############, width=0.3)

        ax.vlines(np.linspace(0.5, n_det + 0.5, n_det + 1), ymin=0, ymax=1, linestyle='--', color='k', lw=0.6, alpha=0.2)
        ax.vlines(np.linspace(0.5, n_det + 0.5, n_det * 4 + 1), ymin=0, ymax=1, linestyle='--', color='k', lw=0.2, alpha=0.2)
        new_tick_locations = np.array(range(n_det)) + 1
        ax.set_xticks(new_tick_locations)
        ax.set_xlim(0.5, n_det + 0.48)
        ax.set_ylim(0.0, 1)
        # ax.text(1.0, 0.81, 'Mean acceptrate' + str(acc.mean()), fontsize=8)
        # ax.set_xlabel('Detector')
        ax.set_ylabel('Acceptance rate')
        #save_string = 'acc_chi2_%08i.pdf' % (int(obsid))
        #plt.savefig(save_string, bbox_inches='tight')
        plt.legend()
        chi2s = np.array(chi2s)
        n_samp = 1 / np.array(variances)
        chi2 = (np.nansum(chi2s * np.sqrt(2 * n_samp[:, None]), 0)) / np.sqrt(2 * np.sum(n_samp[:,None] * (np.isfinite(chi2s)), 0))
        # for i in range(5):
        #     print(i)
        #     print(chi2[1000 * i])
        #     print(chi2s[:,1000 * i])
        #     print(n_samp)
        #     print(np.sum(n_samp[:,None] * (np.isfinite(chi2s)), 0)[1000*i])
        ax = fig.add_subplot(312)#211)
        ymin = -5
        ymax = 5
        n = len(chi2)
        # for j in range(n):
        #     print(np.array(file_list[i]['freqmask_full']).flatten()[j])
        x = np.linspace(0.5, n_det + 0.5, len(chi2))
        plotter = np.zeros((len(chi2)))
        plotter = chi2
        chi2mean = np.nanmean(chi2)
        #ax.step(x, plotter, where='post', label=r'Mean $\chi^2$: ' + str(np.nanmean(chi2)))##############, width=0.3)
        ax.plot(x, plotter, label=r'$\langle \chi^2\rangle$: ' + '%.2f' % chi2mean)##############, width=0.3)

        ax.vlines(np.linspace(0.5, n_det + 0.5, n_det + 1), ymin=ymin, ymax=ymax, linestyle='--', color='k', lw=0.6, alpha=0.2)
        ax.vlines(np.linspace(0.5, n_det + 0.5, n_det * 4 + 1), ymin=ymin, ymax=ymax, linestyle='--', color='k', lw=0.2, alpha=0.2)
        new_tick_locations = np.array(range(n_det)) + 1
        ax.set_xticks(new_tick_locations)
        ax.set_xlim(0.5, n_det + 0.48)
        ax.set_ylim(ymin, ymax)
        # ax.text(1.0, 0.81, 'Mean acceptrate' + str(acc.mean()), fontsize=8)
        # ax.set_xlabel('Detector')
        ax.set_ylabel(r'$\chi^2$')
        plt.legend()
        tsys = np.nanmean(np.array(tsyss), 0)
        ax = fig.add_subplot(313)#211)
        ymin = 20
        ymax = 100
        n = len(tsys)
        # for j in range(n):
        #     print(np.array(file_list[i]['freqmask_full']).flatten()[j])
        x = np.linspace(0.5, n_det + 0.5, len(tsys) + 1)
        plotter = np.zeros((len(tsys) + 1))
        plotter[:-1] = tsys
        tsysmean = np.nanmean(tsys)
        #ax.step(x, plotter, where='post', label=r'Mean $T_{sys}$: ' + str(np.nanmean(tsys)))##############, width=0.3)
        ax.plot(x, plotter, '.', markersize=1.5, label=r'$\langle T_{sys}\rangle$: ' + '%.2f' % tsysmean)##############, width=0.3)

        ax.vlines(np.linspace(0.5, n_det + 0.5, n_det + 1), ymin=ymin, ymax=ymax, linestyle='--', color='k', lw=0.6, alpha=0.2)
        ax.vlines(np.linspace(0.5, n_det + 0.5, n_det * 4 + 1), ymin=ymin, ymax=ymax, linestyle='--', color='k', lw=0.2, alpha=0.2)
        new_tick_locations = np.array(range(n_det)) + 1
        ax.set_xticks(new_tick_locations)
        ax.set_xlim(0.5, n_det + 0.48)
        ax.set_ylim(ymin, ymax)
        # ax.text(1.0, 0.81, 'Mean acceptrate' + str(acc.mean()), fontsize=8)
        # ax.set_xlabel('Detector')
        ax.set_ylabel(r'$T_{sys}$')
        plt.legend()

        str_id = ['fixed mask', 'NaN', 'outl. freq', 'bad sb', 'corr. struct', 'edge corrs']
        
        inds = np.argsort(reason)[::-1]
        
        figstrings = [str_id[k] + ': %.2f \n' % reason[k] for k in inds]
        figstring = "".join(figstrings)

        plt.figtext(0.92, 0.7, figstring[:-2])
        save_string = 'acc_chi2_%06i.pdf' % (int(obsid))
        plt.savefig(save_folder + 'acc_chi2/' + save_string, bbox_inches='tight')
        

        sortedlists = my_spikes.sorted()
        n_spikes = len(sortedlists[0])
        n_jumps = len(sortedlists[1])
        n_anom = len(sortedlists[2]) 

        n_spike_types = 3
        n_plots = 3
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle('Obsid: %06i, %i spikes, %i jumps and %i anomalies. Top: 3 largest spikes.\
 Middle: 3 largest jumps. Bottom: 3 largest anomalies' % (
        int(obsid), n_spikes, n_jumps, n_anom))
        #fig.suptitle('Top: 3 largest spikes. Middle: 3 largest jumps. Bottom: 3 largest anomalies')
        for i in range(n_spike_types):
            for j in range(n_plots):
                try:
                    spike = sortedlists[i][j]
                    ax = fig.add_subplot(n_spike_types, n_plots, n_plots * i + j + 1)
                    ind = spike.ind
                    lab = 'Feed %i, sb %i, Ampl: %.3f' % (ind[0] + 1, ind[1] + 1, spike.amp)
                    dt = 0.02  # sampling rate
                    mjd = (spike.mjd - tstart) * 24 * 3600
                    time = np.linspace(mjd - 200 * dt, mjd + 199 * dt, len(spike.data))
                    ax.plot(time, spike.data, label=lab)#sb_mean[max_sb[0], max_sb[1], max_ind - 200:max_ind + 200])
                    ax.set_xlabel('time [s]')
                    plt.legend()
                    if j == 0:
                        ax.set_ylabel('sb avg tod')
                except IndexError:
                    ax = fig.add_subplot(n_spike_types, n_plots, n_plots * i + j + 1)
                    pass
        save_string = 'spikes_%06i.pdf' % (int(obsid))  # , i+1)
        plt.savefig(save_folder + 'spikes/' + save_string, bbox_inches='tight')
        

        plt.close('all')
        

        #statsdir = '/mn/stornext/d16/cmbco/comap/protodir/auxiliary/l2stats/'
        statsdir = '/mn/stornext/d16/cmbco/comap/nils/plotbrowser/test_figs/'
        fields = ['co2', 'co6', 'co7']
        stats = np.zeros(50)
        stats[0] = int(obsid)
        stats[1] = n_scans
        stats[2] = fields.index(fieldname)
        stats[3] = 0.0  # Scanning strategy 
        stats[4] = tstart
        stats[5] = eff_feeds
        stats[6] = chi2mean
        stats[7] = tsysmean
        stats[8] = n_spikes
        stats[9] = n_jumps
        stats[10] = n_anom
        
        save_string = 'l2_stats_%06i.txt' % (int(obsid))
        np.savetxt(statsdir + save_string, stats)
