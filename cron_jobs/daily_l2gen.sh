#!/bin/bash -l
echo "Running daily l2gen at $(date)."
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 mpirun -n 12 python3 -u -m mpi4py /mn/stornext/d22/cmbco/comap/protodir/pipeline/l2gen.py -p /mn/stornext/d22/cmbco/comap/protodir/params/param_allco_default.txt