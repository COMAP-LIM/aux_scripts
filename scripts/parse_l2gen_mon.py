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
        time = float(words[0])
        mpi = int(words[1])
        if mpi > MPI_threads:
            MPI_threads = mpi
        event = words[-1]
        if not event in times:
            times[event] = []
        times[event].append(time - last_timestamp[mpi])
        last_timestamp[mpi] = time

mean_times = []
events = []
for event in times:
    events.append(event)
    mean_times.append(np.sum(times[event]))

numscans = np.max([len(times[event]) for event in times])
tot_time = np.sum([np.sum(times[event]) for event in times])
print(f"L2GEN run with {MPI_threads} MPI threads. Times shown in total CPU time across threads.")
print(f"Total runtime (s) = {tot_time:.1f}")
print(f"{'Event name':27s}{'Num events':>12s}{'Time (s)':>12s}{'Time (%)':>12s}")
for event in times:
    print(f"{event:27s}{len(times[event]):12d}{np.sum(times[event]):12.1f}{100*np.sum(times[event])/tot_time:12.1f}")