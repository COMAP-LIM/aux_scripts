sourcepath="/mn/stornext/d22/cmbco/comap/protodir/level1/l1_temp/from_ovro"
temppath="/mn/stornext/d22/cmbco/comap/protodir/level1/l1_temp/compressed"
destpath="/mn/stornext/d22/cmbco/comap/protodir/level1_comptest"

for dir in $sourcepath/*
do
    if [ -d "$dir" ]; then  # Ensure it's a directory
        echo "Processing directory: $dir"

        # Extract the relative directory path
        reldir="${dir#$sourcepath/}"

        # Create the corresponding directories in the temppath and destpath
        mkdir -p "$temppath/$reldir"
        mkdir -p "$destpath/$reldir"

        for filepath in $dir/*
        do
            filename=$(basename "$filepath")
            
            # Check if the filename starts with "comap-" and ends with ".hd5"
            if [[ $filename == comap-*.hd5 ]]; then
                echo "Repacking $filename to $temppath/$reldir"
                
                # Using h5repack to transform the file and save it to the temporary path
                h5repack \
                         -f /spectrometer/tod:GZIP=3 \
                         -l /spectrometer/tod:CHUNK=1x4x1024x4000 \
                         "$filepath" "$temppath/$reldir/$filename"

                # Copy the repacked file to the destpath
                echo "Copying repacked $filename to $destpath/$reldir"
                cp --archive "$temppath/$reldir/$filename" "$destpath/$reldir/"

                # Delete the original file from sourcepath and the repacked file from temppath
                echo "Deleting original and repacked $filename"
                rm "$filepath"
                rm "$temppath/$reldir/$filename"
            fi
        done
    fi
done
