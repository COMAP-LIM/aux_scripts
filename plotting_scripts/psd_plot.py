import numpy as np
import matplotlib.pyplot as plt
import sys
import numpy.fft as fft
import time
import glob
import os
import h5py
import scipy.signal

def decimate_array(arr, factor=16):
    n = len(arr)
    n_after = n - n % factor
    print(n, n_after)

    arr = arr[:n_after]
    arr = arr.reshape(n_after // factor, factor).mean(1)
    return arr

filename = '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2020-01/comp_comap-0010488-2020-01-14-050940.hd5'

sbnames = ['A:LSB', 'A:USB', 'B:LSB', 'B:USB']

feed = 2
sb = 2
freq = 250
n_freq = 4

ncut = 3500
nuse = 50 * 60 * 4 # 5 minutes
samprate = 50

fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

with h5py.File(filename, mode="r") as my_file:
    tod = my_file['spectrometer/tod'][feed-1, :, :, ncut:ncut+nuse] / 1e6
    tod_sb = my_file['spectrometer/band_average'][feed-1, :, ncut:ncut+nuse] / 1e6
    t = my_file['spectrometer/MJD'][ncut:ncut+nuse]
    fr = my_file['spectrometer/frequency'][()] 
    t = (t - t[0]) * 24 * 60  # minutes
    ts = t * 60  # seconds
# for i in range(n_freq):
lab = r'feed %i, %s, $\nu$ = %0.3f GHz' % (feed, sbnames[sb-1], fr[sb-1, freq])
# lab = r'ch %i' % (freq + i)
ax1.plot(ts, tod[sb-1, freq-1, :], label=lab, lw=0.1, c='k', rasterized=True)
ax1.set_ylim(np.mean(tod[sb-1, freq-1, :])-0.0035, np.mean(tod[sb-1, freq-1, :]+0.0035))
ax1.text(5, 0.1036, 'constant elevation scan', rotation=0, verticalalignment='center', fontsize=11, rasterized=True)
n = len(t)
print(n)
factor = 1

psd = (np.abs(fft.rfft(tod[sb-1, freq-1])) ** 2 / (0.5 * n * samprate))[1:]
psd = decimate_array(psd, factor)
f = fft.rfftfreq(n, d=1.0/samprate)[1:]
f = decimate_array(f, factor)
psd_sav = scipy.signal.savgol_filter(psd, 101, 1)
N = len(f)
ax3.loglog(f[0:2], psd[0:2], c='k', lw=1, label=lab, rasterized=True)
for i in range(N//10):
    ax3.loglog(f[i*10:(i+2)*10], psd[i*10:(i+2)*10], c='k', lw=1.0/(0.1*i+1.0), alpha=0.7/(0.1*i+1.0)+0.3, rasterized=True)
    if i > 6:
        ax3.plot(f[i*10:(i+2)*10], psd_sav[i*10:(i+2)*10], c="crimson", lw=2, alpha=(0.8 + 10.0)/(N//10-i + 10.0)+0.2, rasterized=True)
ax3.set_ylim(1e-11, 5e-5)
ax3.set_xlim(f[0], f[-1])
ax3.set_xlabel('Frequency [Hz]')
ax1.set_xlabel('Time [s]')
leg = ax1.legend(loc='lower right', frameon=False)
for line in leg.get_lines():
    line.set_linewidth(1.0)
ax1.set_xlim(ts[0], ts[-1])

filename = '/mn/stornext/d16/cmbco/comap/pathfinder/ovro/2020-01/comp_comap-0010515-2020-01-15-050844.hd5'

with h5py.File(filename, mode="r") as my_file:
    tod = my_file['spectrometer/tod'][feed-1, :, :, ncut:ncut+nuse] / 1e6
    tod_sb = my_file['spectrometer/band_average'][feed-1, :, ncut:ncut+nuse] / 1e6
    t = my_file['spectrometer/MJD'][ncut:ncut+nuse] 
    t = (t - t[0]) * 24 * 60  # minutes
    ts = t * 60  # seconds

lab = r'feed %i, %s, %0.2f' % (feed, sbnames[sb-1], freq)
# lab = r'ch %i' % (freq + i)
ax2.plot(ts, tod[sb-1, freq-1, :], label=lab, lw=0.1, c='k', rasterized=True)
ax2.set_ylim(np.mean(tod[sb-1, freq-1, :])-0.0035, np.mean(tod[sb-1, freq-1, :]+0.0035))
ax2.text(5, 0.1022, 'lissajous scan', rotation=0, verticalalignment='center', fontsize=11, rasterized=True)
n = len(t)
print(n)
psd = (np.abs(fft.rfft(tod[sb-1, freq-1])) ** 2 / (0.5 * n * samprate))[1:]
psd = decimate_array(psd, factor)
f = np.abs(fft.rfftfreq(n, d=1.0/samprate))[1:]
f = decimate_array(f, factor)
psd_sav = scipy.signal.savgol_filter(psd, 101, 1)
ax4.plot(f[-2:-1], psd_sav[-2:-1], c="crimson", lw=2, label="running mean", rasterized=True)
for i in range(N//10):
    ax4.loglog(f[i*10:(i+2)*10], psd[i*10:(i+2)*10], c='k', lw=1.0/(0.1*i+1.0), alpha=0.7/(0.1*i+1.0)+0.3, rasterized=True)
    if i > 6:
        ax4.plot(f[i*10:(i+2)*10], psd_sav[i*10:(i+2)*10], c="crimson", lw=2, alpha=(0.8 + 10.0)/(N//10-i + 10.0)+0.2, rasterized=True)
ax4.set_ylim(1e-11, 5e-5)
ax4.set_xlim(f[0], f[-1])
ax4.set_xlabel('Frequency [Hz]')
ax2.set_xlabel('Time [s]')
# leg = ax2.legend(loc='lower right')
# for line in leg.get_lines():
#     line.set_linewidth(1.0)
ax2.set_xlim(ts[0], ts[-1])


ax1.set_ylabel(r'power [MW Hz${}^{-1}$]')
ax2.set_ylabel(r'power [MW Hz${}^{-1}$]')
ax3.set_ylabel(r'PSD [MW${}^2$ Hz${}^{-3}$]')
ax4.set_ylabel(r'PSD [MW${}^2$ Hz${}^{-3}$]')

use_log = False
if use_log:
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    ax4.set_yscale('log')

ax3.legend(loc='lower left', frameon=False)
ax4.legend(loc='upper right', frameon=False)
# ax4.legend(loc='lower left')
# plt.setp(ax1.get_xticklabels(), visible=False)
# plt.setp(ax2.get_xticklabels(), visible=False)
# plt.setp(ax3.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=0.4)
plt.subplots_adjust(wspace=0.23)
plt.savefig('tod_ps.pdf', bbox_inches='tight', dpi=200)
plt.show()
