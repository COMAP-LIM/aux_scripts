#!/bin/bash -l
#export PATH=/astro/local/anaconda/envs/py38/bin/:/astro/local/anaconda/bin:/astro/local/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/usr/lib64/openmpi/bin:/opt/dell/srvadmin/bin
echo "Running daily weather-nn at $(date)."
python3 /mn/stornext/d22/cmbco/comap/protodir/COMAP_weather_nn/weathernet/load_net.py