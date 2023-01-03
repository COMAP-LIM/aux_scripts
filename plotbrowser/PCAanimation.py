import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

path = "/mn/stornext/d22/cmbco/comap/nils/comap_aux/plotbrowser/"
filename = path + "PCA_frequency_data.h5"

with h5py.File(filename, "r") as infile:
    comps = []
    amps  = []
    obsids  = []
    dset = infile["co7"]
    freq = infile["freq"][()]
    for key in dset.keys():
        if np.all(dset[key + "/amplitudes"] == 0):
            continue
        obsids.append(key)
        comps.append(dset[key + "/modes"][()])
        amps.append(dset[key + "/amplitudes"][()])

comps = np.array(comps)
amps = np.array(amps)
print(amps.shape)
fig, ax = plt.subplots()
xdata, ydata = [], []
ln, = plt.plot([], [], "bo", markersize = 1)

title = ax.text(0.05, 0.95, "", transform=ax.transAxes)

def init():
    ax.set_xlim(0.99 * np.min(freq), 1.01 * np.max(freq))
    ax.set_ylim(-50, 50)
    return ln,

def update(frame):
    xdata = freq
    ydata = amps[frame, 0, 10, :]
    ln.set_data(xdata, ydata)
    title.set_text("obID: " + obsids[frame] + f" {frame}")
    print(obsids[frame], frame)
    return ln, title

ani = FuncAnimation(fig, update, frames = range(len(obsids)),
                    init_func=init, blit=True, interval = 1000)
plt.show()