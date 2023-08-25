#!/bin/bash -l
echo "Running daily tsys database write at $(date)."
MONTH="$1"
python3 -W ignore /mn/stornext/d22/cmbco/comap/protodir/comap_aux/level1_database/tsys_database_write.py -n 12 -p /mn/stornext/d22/cmbco/comap/protodir/aux_data/level1_database_files -m $MONTH