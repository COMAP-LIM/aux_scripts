#!/bin/bash
export PATH=/net/alruba2.uio.no/mn/alruba2/astro/local/opt/intel/oneapi/compiler/2022.2.0/linux/bin/intel64:/net/alruba2.uio.no/mn/alruba2/astro/local/opt/intel/oneapi/mpi/2021.7.0/libfabric/bin:/net/alruba2.uio.no/mn/alruba2/astro/local/opt/intel/oneapi/mpi/2021.7.0/bin:/mn/stornext/u3/sigurdkn/local/moduledata/python3.10/bin:/mn/stornext/u3/jonassl/local/bin:/mn/stornext/u3/jonassl/.local/bin:/astro/local/opt/Intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/bin:/astro/local/opt/Intel/compilers_and_libraries_2020.4.304/linux/bin/intel64:/astro/local/opt/Intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/libfabric/bin:/astro/local/opt/Intel/debugger_2020/gdb/intel64/bin:/usr/lib64/qt-3.3/bin:/astro/local/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/usr/lib64/openmpi/bin:/opt/dell/srvadmin/bin
export PYTHONPATH=/mn/stornext/u3/jonassl/local/moduledata/mpi4py/lib/python3.10/site-packages:/mn/stornext/u3/sigurdkn/local/moduledata/python3.10/lib/python3.10:/mn/stornext/u3/jonassl/local/lib/python3.10/site-packages
export LD_LIBRARY_PATH=/net/alruba2.uio.no/mn/alruba2/astro/local/opt/intel/oneapi/mkl/2022.2.0/lib/intel64:/net/alruba2.uio.no/mn/alruba2/astro/local/opt/intel/oneapi/tbb/2021.7.0/lib/intel64/gcc4.8:/net/alruba2.uio.no/mn/alruba2/astro/local/opt/intel/oneapi/compiler/2022.2.0/linux/lib:/net/alruba2.uio.no/mn/alruba2/astro/local/opt/intel/oneapi/compiler/2022.2.0/linux/lib/x64:/net/alruba2.uio.no/mn/alruba2/astro/local/opt/intel/oneapi/compiler/2022.2.0/linux/compiler/lib/intel64_lin:/net/alruba2.uio.no/mn/alruba2/astro/local/opt/intel/oneapi/mpi/2021.7.0/libfabric/lib:/net/alruba2.uio.no/mn/alruba2/astro/local/opt/intel/oneapi/mpi/2021.7.0/lib/release:/net/alruba2.uio.no/mn/alruba2/astro/local/opt/intel/oneapi/mpi/2021.7.0/lib:/mn/stornext/u3/sigurdkn/local/moduledata/python3.10/lib:/mn/stornext/u3/jonassl/local/lib:/astro/local/opt/Intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/lib/release:/astro/local/opt/Intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/lib:/astro/local/opt/Intel/compilers_and_libraries_2020.4.304/linux/compiler/lib/intel64_lin:/astro/local/opt/Intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/libfabric/lib:/astro/local/opt/Intel/compilers_and_libraries_2020.4.304/linux/ipp/lib/intel64:/astro/local/opt/Intel/compilers_and_libraries_2020.4.304/linux/mkl/lib/intel64_lin:/astro/local/opt/Intel/compilers_and_libraries_2020.4.304/linux/tbb/lib/intel64/gcc4.8:/astro/local/opt/Intel/debugger_2020/libipt/intel64/lib:/astro/local/opt/Intel/compilers_and_libraries_2020.4.304/linux/daal/lib/inte64_lin:/astro/local/opt/Intel/compilers_and_libraries_2020.4.304/linux/daal/../tbb/lib/intel64_lin/gcc4.4:/astro/local/opt/Intel/compilers_and_libraries_2020.4.304/linux/daal/../tbb/lib/intel64_lin/gcc4.8:/astro/local/opt/Intel/lib/intel64
echo "Running daily tsys database write at $(date)."
MONTH="$1"
python3 -W ignore /mn/stornext/d22/cmbco/comap/protodir/comap_aux/level1_database/tsys_database_write.py -n 12 -p /mn/stornext/d22/cmbco/comap/protodir/aux_data/level1_database_files -m $MONTH