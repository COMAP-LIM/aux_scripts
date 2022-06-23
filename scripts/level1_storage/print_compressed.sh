# Script which checks all comap-*.hd5 files in /mn/stornext/d22/cmbco/comap/protodir/level1/*,
# and checks if there exist compressed versions (comp_comap-*.hd5) in the same directory, and
# if uncompressed files can be found on tape, at /mn/stornext/d16/cmbco/archive/comap/data/pathfinder/ovro.

fmt="%-10s%-10s%-10s%-10s%-10s%-10s\n"

printf "__Number of uncompressed and compressed files in all subdirs__\n"
printf "$fmt" "month" "presto" "uncomp" "comp" "d16tape" "d22tape"
for d in /mn/stornext/d22/cmbco/comap/protodir/level1/20*/ ; do
    month=${d:45:7}
    
    presto=$(ssh comap_analysis@presto.caltech.edu ls /comapdata*/pathfinder/level1/$month*/comap-*.hd5 2> /dev/null | wc -l)
    uncomp=$(ls $d\comap-*.hd5 2> /dev/null | wc -l)
    comp=$(ls $d\comp*.hd5 2> /dev/null | wc -l)
    d16tape=$(ls /mn/stornext/d16/cmbco/archive/comap/data/pathfinder/ovro/$month/comap-*.hd5 2> /dev/null | wc -l)
    d22tape=$(ls /mn/stornext/d22/cmbco_archive/comap/protodir/level1/$month/comap-*.hd5 2> /dev/null | wc -l)
    printf "$fmt" "$month" "$presto" "$uncomp" "$comp" "$d16tape" "$d22tape"
done
