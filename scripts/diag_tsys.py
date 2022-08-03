import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
import math
from glob import glob


def find_feat(time, time_feat, feat):  # hat tip: https://stackoverflow.com/a/26026189/5238625
    idx = np.searchsorted(time_feat, time, side="left")
    if idx > 0 and (idx == len(time_feat) or math.fabs(time - time_feat[idx-1]) < math.fabs(time - time_feat[idx])):
        return feat[idx-1]
    else:
        return feat[idx]

try:
    obsid = sys.argv[1]
except IndexError:
    print('Missing obsid!')
    print('Usage: python diag_tsys.py obsid startsample endsample')
    sys.exit(1)

l1_dir = '/mn/stornext/d22/cmbco/comap/protodir/level1'

if len(obsid) == 4:
    filename = glob(l1_dir + '/**/comp_comap-000' + obsid + '*.hd5')[0]
else:
    filename = glob(l1_dir + '/**/comp_comap-00' + obsid + '*.hd5')[0]


print(filename)

try:
    t_start = int(sys.argv[2])
except:
    t_start = 0.0
try: 
    t_end = int(sys.argv[3])
except:
    t_end = 200 # s
det = 5   # 19
sb = 1     # 2
freq_hr = 10

samprate = 50 # Hz

#filename = 'comap-0006262-2019-05-31-001116.hd5'
n_start = int(t_start * samprate)
n_cut = int(t_end * samprate)
#print(n_cut)

# "hk/antenna0/vane/state",          data%amb_state)                                                                        
#     call read_hdf(file, "hk/antenna0/vane/utc",

with h5py.File(filename, mode="r") as my_file:
    obs_id = filename[-29:-22]# + filename[-5:-3]
    tod = np.array(my_file['spectrometer/tod'][det,sb,freq_hr, n_start:n_cut])
    t0 = np.array(my_file['spectrometer/MJD'])[0]
    time = np.array(my_file['spectrometer/MJD'])[n_start:n_cut]
    feat_arr = np.array(my_file[u'/hk/array/frame/features'])
    t_feat = np.array(my_file[u'/hk/array/frame/utc'])
    t_vane = np.array(my_file[u'/hk/antenna0/vane/utc'])
    vane = np.array(my_file[u'/hk/antenna0/vane/state'])#position']) #state'])

print('MJD: ', t0)

time = (time - t0) * 24 * 60 * 60
t_feat = (t_feat - t_feat[0]) * 24 * 60 * 60
t_vane = (t_vane - t_vane[0]) * 24 * 60 * 60
# print(vane == 0)
#print(obs_id)
v_state = np.zeros_like(time)
for i in range(len(time)):
    v_state[i] = find_feat(time[i], t_vane, vane)

plt.title(obs_id)

# 0 - vane not covering feeds(cold)
# 1 - vane covering feeds(hot)
# 2 - vane moving from cold to hot
# 3 - vane moving from hot to cold
# 4 - vane stuck

d_red = tod[(v_state == 0)]
time_red = time[(v_state == 0)]
d_b = tod[(v_state == 1)]
time_b = time[(v_state == 1)]
d_g = tod[(v_state == 2)]
time_g = time[(v_state == 2)]
d_c = tod[(v_state == 3)]
time_c = time[(v_state == 3)]
d_y = tod[(v_state == 4)]
time_y = time[(v_state == 4)]
# plt.plot(time, tod, '--')
plt.plot(time_red, d_red, '.r', label='vane.state = 0')
plt.plot(time_b, d_b, '.b', label='vane.state = 1')
plt.plot(time_g, d_g, '.g', label='vane.state = 2')
plt.plot(time_c, d_c, '.c', label='vane.state = 3')
plt.plot(time_y, d_y, '.y', label='vane.state = 4')
plt.plot(t_feat, feat_arr / np.max(feat_arr) * np.max(tod))

print(np.max(feat_arr))

# plt.yscale('log')
plt.legend()
plt.xlim(time[0], time[-1])
save_string = 'diag_tsys_%08i.pdf' % (int(obs_id))
plt.savefig(save_string, bbox_inches='tight')
plt.show()
