from os import listdir
from os.path import isfile, join, getatime, getsize
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

archivepath = "/mn/stornext/d16/cmbco/archive/comap/data/pathfinder/ovro/"
months = ["2021-01", "2021-02", "2021-03"]

files = []
for month in months:
    path = join(archivepath, month)
    for file in listdir(path):
        filepath = join(path, file)
        if isfile(filepath):
            if file[-4:] == ".hd5":
                files.append(filepath)

times = []
sizes = []
for file in files:
    size = getsize(file)
    if size > 10e9:
        times.append(getatime(file))
        sizes.append(size)

times = np.array(times)
sizes = np.array(sizes)

order = np.argsort(times)
times = times[order]
sizes = times[order]

N = len(times)
speeds = np.zeros(N-1)
for i in range(N-1):
    speeds[i] = sizes[i+1]/(times[i+1] - times[i])

times = times[1:]
times -= times[0]
times /= 3600
speeds /= 1e6

uni_times = np.linspace(times[0], times[-1], 10000)

gauss_speed = gaussian_filter1d(speeds, 20)

plt.figure(figsize=(16,8))
plt.scatter(times, speeds, s=10)
plt.plot(times, gauss_speed, c="r")
plt.ylim(0, 40)
plt.xlabel("Time [hours]")
plt.ylabel("Speed [MB/s]")
plt.savefig("speed.png")