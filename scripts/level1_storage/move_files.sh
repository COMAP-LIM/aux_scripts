#!/bin/bash
# This script moves level1 files from one directory to another, and creates a symlink in a third directory (can be set to be the first directory).

set -e  # Exit on error

# SET THESE VARIABLES
from_dir="/mn/stornext/d23/cmbco_nobackup/comap/data/level1/"
to_dir="/mn/stornext/d22/cmbco_nobackup/comap/data/level1/"
link_dir="/mn/stornext/d16/cmbco/comap/data/level1/"

# Get the list of directories in from_dir that start with "20"
months=$(ls -d "${from_dir}"20*/ 2>/dev/null | xargs -n 1 basename)

if [ -z "$months" ]; then
    echo "No matching directories found"
    exit 1
fi

echo "$months"

for month in $months; do
    echo "Starting ${from_dir}${month} to ${to_dir}${month}"
    
    if ! chmod -R u+w "${from_dir}${month}"; then
        echo "Failed to set write permissions on source directory"
        exit 1
    fi
    
    if ! /usr/cvfs/bin/cvcp -ad "${from_dir}${month}/" "${to_dir}/"; then
        echo "Failed to copy directory"
        exit 1
    fi
    
    chmod -R a-w "${to_dir}${month}" && \
    du "${from_dir}${month}" && \
    du "${to_dir}${month}" && \
    ls "${from_dir}${month}" | wc && \
    ls "${to_dir}${month}" | wc && \
    rm "${from_dir}${month}"/* && \
    rmdir "${from_dir}${month}" && \
    chmod u+w "${link_dir}${month}" && \
    rm "${link_dir}/${month}" && \
    ln -sf "${to_dir}${month}" "${link_dir}/" && \
    echo "Finished ${from_dir}${month} to ${to_dir}${month}"
done

echo "Done"