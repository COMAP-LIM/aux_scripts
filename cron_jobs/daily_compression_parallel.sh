#!/bin/bash

sourcepath="/mn/stornext/d22/cmbco/comap/protodir/level1/l1_temp/from_ovro"
temppath="/mn/stornext/d22/cmbco/comap/protodir/level1/l1_temp/compressed"
destpath="/mn/stornext/d22/cmbco/comap/protodir/level1"

process_file() {
    filepath="$1"
    temppath="$2"
    destpath="$3"

    filename=$(basename "$filepath")
    
    # Check if the filename starts with "comap-" and ends with ".hd5"
    if [[ $filename == comap-*.hd5 ]]; then
        reldir=$(dirname "$filepath" | sed "s|$sourcepath/||")

        echo "Repacking $filename to $temppath/$reldir"
        h5repack -f /spectrometer/tod:SHUF \
                 -f /spectrometer/tod:GZIP=3 \
                 -l /spectrometer/tod:CHUNK=1x4x1024x4000 \
                 "$filepath" "$temppath/$reldir/$filename"

        echo "Moving repacked $filename to $destpath/$reldir"
        mv "$temppath/$reldir/$filename" "$destpath/$reldir/"
        echo "Removing write permission for $destpath/$reldir"
        chmod a-w $destpath/$reldir/$filename

        echo "Deleting original $filename"
        rm "$filepath"
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
