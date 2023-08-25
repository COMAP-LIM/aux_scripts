#!/bin/bash -l
echo "Running daily scan-detect at $(date)."
python3 /mn/stornext/d22/cmbco/comap/protodir/pipeline/scan_detect/scan_detect.py --runlist /mn/stornext/d22/cmbco/comap/protodir/runlists/runlist_allco.txt --fields co2 co6 co7
python3 /mn/stornext/d22/cmbco/comap/protodir/pipeline/scan_detect/scan_detect.py --runlist /mn/stornext/d22/cmbco/comap/protodir/runlists/runlist_everything.txt --fields all