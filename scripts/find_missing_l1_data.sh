#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <YYYY>"
    exit 1
fi

YEAR=$1
REMOTE="comap_analysis@presto.caltech.edu"
BASE_LOCAL_DEST="/mn/stornext/d16/cmbco/comap/data/level1"
LIST_DIR="${BASE_LOCAL_DEST}/transfer_lists/yearly_${YEAR}"

# The working Conda h5dump binary
H5DUMP_BIN="/scr/comap_analysis/etc/anaconda3/envs/reduce/bin/h5dump"

mkdir -p "$LIST_DIR"
echo "Starting compilation of missing files for $YEAR..."

for MONTH in {01..12}; do
    YM="${YEAR}-${MONTH}"
    echo "----------------------------------------"
    echo "Processing Month: $YM"
    
    LOCAL_DEST="${BASE_LOCAL_DEST}/${YM}"
    LIST_FILE="${LIST_DIR}/l1_transfer_${YM}.log"
    
    # 1. Fetch remote files 
    REMOTE_FILES=$(ssh "$REMOTE" "find /comapdata*/pathfinder/level1/${YM}* -maxdepth 1 -name 'comap-*${YM}*.hd5' 2>/dev/null")
    
    if [ -z "$REMOTE_FILES" ]; then
        echo "Total files found on remote: 0"
        continue
    fi
    
    # 2. Check local existence
    MISSING_LIST_TMP=$(mktemp)
    TOTAL_FILES=0
    MISSING_COUNT=0
    
    for remotepath in $REMOTE_FILES; do
        ((TOTAL_FILES++))
        filename=$(basename "$remotepath")
        
        if [ ! -f "${LOCAL_DEST}/${filename}" ]; then
            echo "$remotepath" >> "$MISSING_LIST_TMP"
            ((MISSING_COUNT++))
        fi
    done
    
    echo "Total files found on remote: $TOTAL_FILES"
    echo "Total missing files locally: $MISSING_COUNT"
    
    if [ "$MISSING_COUNT" -eq 0 ]; then
        echo "No missing files to process for $YM. Skipping."
        rm -f "$MISSING_LIST_TMP"
        continue
    fi
    
    # 3. Validate via remote h5dump
    echo "Validating HDF5 headers via remote h5dump..."
    ssh "$REMOTE" "
        while read -r filepath; do
            [ -z \"\$filepath\" ] && continue
            
            if $H5DUMP_BIN -a comap/source \"\$filepath\" < /dev/null 2>/dev/null | grep -iqE \"co[267]\"; then
                echo \"\${filepath#/}\"
            fi
        done
    " < "$MISSING_LIST_TMP" > "$LIST_FILE"
    
    MATCHED_COUNT=$(wc -l < "$LIST_FILE" 2>/dev/null || echo 0)
    echo "Missing files matching pattern 'co[267]': $MATCHED_COUNT"
    
    rm -f "$MISSING_LIST_TMP"
    
    if [ ! -s "$LIST_FILE" ]; then
        echo "No missing files met the criteria for $YM."
        # Clean up empty log files
        rm -f "$LIST_FILE" 
        continue
    fi
    
    echo "List compiled successfully: $LIST_FILE"
done
