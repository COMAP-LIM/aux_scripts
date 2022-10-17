import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


field = "co7"
scans = np.loadtxt(f"scan_list_{field}.txt")

az_data = np.load(f"az_data_{field}.npy")

# n_cut = 5000

# scans = scans[:n_cut]
# az_data = az_data[:n_cut]

print(az_data.shape)

# signs = np.sign(np.sum(az_data[:, 45:50], 1))
# print(signs.shape)

# az_data[:, :] = az_data[:, :] * signs[:, None]

s2_index = find_nearest(scans, 1537700)
print(s2_index)
print(az_data.shape)
s1_index = 0

selected_index = find_nearest(scans, 872605)

# weird_idx_1 = min(16760, n_cut-1)
# weird_idx_2 = min(25250, n_cut-1)
weird_idx_1 = 16760
weird_idx_2 = 25250
vmax = 4
# az_data[:, :100] *= 10
my_cmap = "RdBu_r"
smooth_sig = 15
smooth_y = 0.5
# az_data_plot = gaussian_filter1d(gaussian_filter1d(az_data, smooth_sig, axis=0), smooth_y, axis=1)
az_data_plot = az_data
rast = False
interp = "antialiased"  #'none'
x_label_list = [
    "Season 1",
    "Season 2",
    str(int(scans[weird_idx_1] // 100)),
    str(int(scans[weird_idx_2] // 100)),
]
y_label_list = [1, 0, -1, 50, 100, 150, 200, 250]
# fig, (ax1, ax2) = plt.subplots(2,1)
fig, ax1 = plt.subplots(1, 1)
img = ax1.imshow(
    az_data_plot[:, :].T,
    aspect="auto",
    vmin=-vmax,
    vmax=vmax,
    cmap=my_cmap,
    rasterized=rast,
    interpolation=interp,
)
ax1.set_yticks([0, 50, 100, 150, 200, 250, 300, 350])
ax1.set_yticklabels(y_label_list, rotation=45)
ax1.set_xticks([s1_index, s2_index, weird_idx_1, weird_idx_2])
# ax1.set_xticklabels([], rotation=45)
ax1.set_xticklabels(x_label_list, rotation=45)
ax1.hlines(
    y=100, xmin=0, xmax=len(az_data[:, 0]) - 1, linewidth=1, color="k", alpha=0.5
)
ax1.vlines(x=s2_index, ymin=0, ymax=len(az_data[0, :]) - 1, linewidth=1, color="k")
ax1.set_ylabel("    Frequencies     daz/dt")
plt.savefig(f"daz_{field}_feed16.pdf", bbox_inches="tight", dpi=500)

U, s, Vh = linalg.svd(az_data)
S = np.diag(s)
plt.figure()
plt.plot(s)

n_modes = 20
fig, ax2 = plt.subplots(1, 1)
approx = np.dot(U[:, :n_modes], np.dot(S[:n_modes, :n_modes], Vh[:n_modes, :]))
print("hei", approx.shape)
np.save(f"az_pca_fit_872605.npy", approx[selected_index, :])
# approx_plot = gaussian_filter1d(gaussian_filter1d(approx, smooth_sig, axis=0), smooth_y, axis=1)
approx_plot = approx
ax2.imshow(
    (az_data_plot[:, :] - approx_plot).T,
    aspect="auto",
    vmin=-vmax,
    vmax=vmax,
    cmap=my_cmap,
    rasterized=rast,
    interpolation=interp,
)
ax2.set_yticks([0, 50, 100, 150, 200, 250, 300, 350])
ax2.set_yticklabels(y_label_list, rotation=45)
ax2.set_xticks([s1_index, s2_index, weird_idx_1, weird_idx_2])
ax2.set_xticklabels(x_label_list, rotation=45)
ax2.hlines(
    y=100, xmin=0, xmax=len(az_data[:, 0]) - 1, linewidth=1, color="k", alpha=0.5
)
ax2.vlines(x=s2_index, ymin=0, ymax=len(az_data[0, :]) - 1, linewidth=1, color="k")
ax2.set_ylabel("    Frequencies     daz/dt")
plt.savefig(f"daz_{field}_feed16_cleaned.pdf", bbox_inches="tight", dpi=500)
# ax2.set_ylabel('                Frequencies                         daz/dt bins')

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(img, cax=cbar_ax)

# plt.figure()
# plt.plot(s)

plt.show()

# fig, ax = plt.subplots(1,1)

# img = ax.imshow(z,extent=[-1,1,-1,1])

# x_label_list = ['A2', 'B2', 'C2', 'D2']

# ax.set_xticks([-0.75,-0.25,0.25,0.75])

# ax.set_xticklabels(x_label_list)

# fig.colorbar(img)
