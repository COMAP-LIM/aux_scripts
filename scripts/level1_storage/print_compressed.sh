# Script which checks all comap-*.hd5 files in /mn/stornext/d22/cmbco/comap/protodir/level1/*,
# and checks if there exist compressed versions (comp_comap-*.hd5) in the same directory, and
# if uncompressed files can be found on tape, at /mn/stornext/d16/cmbco/archive/comap/data/pathfinder/ovro.

echo "__Number of uncompressed and compressed files in all subdirs__"
echo "dir  uncomp  comp  tape"
for d in /mn/stornext/d22/cmbco/comap/protodir/level1/20*/ ; do
    uncomp=$(ls $d\comap-*.hd5 2> /dev/null | wc -l)
    comp=$(ls $d\comp*.hd5 2> /dev/null | wc -l)
    tape=$(ls /mn/stornext/d16/cmbco/archive/comap/data/pathfinder/ovro/$d\comap-*.hd5 2> /dev/null | wc -l)
    month=${d:45:7}
    echo "$month $uncomp $comp $tape"
done
