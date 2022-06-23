import numpy as np

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

