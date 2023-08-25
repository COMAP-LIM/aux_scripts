# Script which parses a l2gen_mon.txt file and outputs a summary. Example usage:
# python3 parse_l2gen_mon.py l2gen_mon.txt

import numpy as np
import matplotlib.pyplot as plt
import sys

filename = sys.argv[1]

last_timestamp = np.zeros(1000)
MPI_threads = 0
times = {}

with open(filename, "r") as infile:
    for line in infile:
        words = line.split()
        if len(words) == 6:
            time = float(words[0])
            mpi = int(words[1])
            if mpi > MPI_threads:
                MPI_threads = mpi
            event = words[-1]
            if not event in times:
                times[event] = []
            times[event].append(time - last_timestamp[mpi])
            last_timestamp[mpi] = time
MPI_threads += 1  # 0-indexed
mean_times = []
events = []
for event in times:
    events.append(event)
    mean_times.append(np.sum(times[event]))

numscans = np.max([len(times[event]) for event in times])
tot_time = np.sum([np.sum(times[event]) for event in times])
print(f"L2GEN run with {MPI_threads} MPI threads. Times shown in total CPU time across threads.")
print(f"Total runtime = {tot_time:.1f}(s) = {tot_time/3600.0:.1f}(h) for {len(times['write_l2'])} scans at {tot_time/len(times['write_l2']):.1f} s/scan")
print(f"{'Event name':27s}{'Num events':>16s}{'Time (s)':>16s}{'Time/event (s)':>16s}{'Time (%)':>16s}")
for event in times:
    print(f"{event:27s}{len(times[event]):16d}{np.sum(times[event]):16.1f}{np.sum(times[event])/len(times[event]):16.1f}{100*np.sum(times[event])/tot_time:16.1f}")