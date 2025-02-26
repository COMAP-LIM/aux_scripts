#!/bin/bash
export PATH=/net/alruba2.uio.no/mn/alruba2/astro/local/opt/intel/oneapi/compiler/2022.2.0/linux/bin/intel64:/net/alruba2.uio.no/mn/alruba2/astro/local/opt/intel/oneapi/mpi/2021.7.0/libfabric/bin:/net/alruba2.uio.no/mn/alruba2/astro/local/opt/intel/oneapi/mpi/2021.7.0/bin:/mn/stornext/u3/sigurdkn/local/moduledata/python3.10/bin:/mn/stornext/u3/jonassl/local/bin:/mn/stornext/u3/jonassl/.local/bin:/astro/local/opt/Intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/bin:/astro/local/opt/Intel/compilers_and_libraries_2020.4.304/linux/bin/intel64:/astro/local/opt/Intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/libfabric/bin:/astro/local/opt/Intel/debugger_2020/gdb/intel64/bin:/usr/lib64/qt-3.3/bin:/astro/local/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/usr/lib64/openmpi/bin:/opt/dell/srvadmin/bin

echo "Running parallel daily compression at $(date)."

sourcepath="/mn/stornext/d22/cmbco_nobackup/comap/data/level1/l1_temp/from_ovro"
temppath="/mn/stornext/d22/cmbco_nobackup/comap/data/level1/l1_temp/compressed"
destpath="/mn/stornext/d22/cmbco_nobackup/comap/data/level1"

process_file() {
    filepath="$1"
    temppath="$2"
    destpath="$3"

    filename=$(basename "$filepath")
    
    # Check if the filename starts with "comap-" and ends with ".hd5"
    if [[ $filename == comap-*.hd5 ]]; then
        if h5dump -a comap/source "$filepath" | grep -qE "co[267]"; then
            echo "File $filename contains a CO field. Continuing."

            reldir=$(dirname "$filepath" | sed "s|$sourcepath/||")

            # Extract the directory and filename parts
            directory=$(dirname "$filepath")
            base_filename=$(basename "$filepath")
            working_filename="working_$base_filename"
            working_filepath="$directory/$working_filename"

            # Rename the file by appending "working_" to its name
            mv "$filepath" "$working_filepath"

            echo "Repacking $working_filename to $temppath/$reldir"
            h5repack -f /spectrometer/tod:SHUF \
                    -f /spectrometer/tod:GZIP=3 \
                    -l /spectrometer/tod:CHUNK=1x4x1024x4000 \
                    "$working_filepath" "$temppath/$reldir/$working_filename"

            # Check if h5repack succeeded
            if [ $? -ne 0 ]; then
                echo "h5repack failed for $working_filename. Skipping further processing for this file."
                mv "$working_filepath" "$filepath" # Revert file renaming before exiting
                return
            fi

            # Revert the renaming for final copying
            final_dest_filename=$base_filename

            echo "Moving repacked $working_filename to $destpath/$reldir as $final_dest_filename"
            /usr/cvfs/bin/cvcp -ad "$temppath/$reldir/$working_filename" "$destpath/$reldir/$final_dest_filename"

            echo "Removing write permission for $destpath/$reldir"
            chgrp astcomap "$destpath/$reldir/$final_dest_filename"
            chmod a-w "$destpath/$reldir/$final_dest_filename"

            echo "Deleting original file $working_filename"
            rm "$working_filepath"
            echo "Deleting temp file $temppath/$reldir/$working_filename"
            rm "$temppath/$reldir/$working_filename"
        else
            echo "File" $filename "is not a co2, co6, or co7 file. Deleting."
            rm "$filepath"
        fi
    fi
}

export -f process_file
export sourcepath
export temppath
export destpath

for dir in $sourcepath/*
do
    if [ -d "$dir" ]; then  # Ensure it's a directory
        echo "Processing directory: $dir"

        # Create the corresponding directories in the temppath and destpath
        reldir="${dir#$sourcepath/}"
        mkdir -p "$temppath/$reldir"
        mkdir -p "$destpath/$reldir"

        # Find all files in $dir and process them in parallel using GNU parallel
        find "$dir" -type f | /astro/local/bin/parallel -j 24 process_file {} "$temppath" "$destpath"
    fi
done
