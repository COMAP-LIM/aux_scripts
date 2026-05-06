#!/bin/bash
### USAGE: Run as `bash transfer_l1_from_list.sh /mn/stornext/d16/cmbco/comap/data/level1/transfer_lists/yearly_2026/l1_transfer_2026-01.log /mn/stornext/d16/cmbco/comap/data/level1` ###

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <list_file> <output_base_directory>"
    echo "Example: $0 /path/to/l1_transfer_2026-04.log /mn/stornext/d16/cmbco/comap/data/level1"
    exit 1
fi

LIST_FILE=$1
OUT_DIR=$2

REMOTE="comap_analysis@presto.caltech.edu"
PARALLEL_BIN="/astro/local/bin/parallel"

if [ ! -f "$LIST_FILE" ]; then
    echo "Error: List file '$LIST_FILE' not found."
    exit 1
fi

echo "Preparing transfer for files in: $LIST_FILE"
echo "Base destination directory: $OUT_DIR"

# 1. Pre-create all unique YYYY-MM directories based on the contents of the list
# This prevents race conditions where multiple parallel rsyncs try to mkdir simultaneously
awk -v out="$OUT_DIR" '{
    match($0, /[0-9]{4}-[0-9]{2}/); 
    print out "/" substr($0, RSTART, RLENGTH)
}' "$LIST_FILE" | sort -u | xargs mkdir -p

echo "Executing 12-thread parallel rsync..."

# 2. Build the individual rsync commands dynamically and feed them to parallel
# This guarantees each file is routed into its specific YYYY-MM subfolder
awk -v remote="$REMOTE" -v out="$OUT_DIR" '{
    match($0, /[0-9]{4}-[0-9]{2}/);
    ym = substr($0, RSTART, RLENGTH);
    print "rsync -Pavt --no-links " remote ":/" $0 " " out "/" ym "/"
}' "$LIST_FILE" | "$PARALLEL_BIN" --will-cite -j 12

echo "Transfer complete."
