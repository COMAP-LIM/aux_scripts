#!/bin/bash -l
echo "Running daily level1 rsync from OVRO at $(date)."
MONTH="$1"
DAY="$2"
rsync -Pavt --no-links comap_analysis@presto.caltech.edu:/comapdata*/pathfinder/level1/$(MONTH)*/comap-[0-9][0-9][0-9][0-9][0-9][0-9][0-9]-$(DAY)-[0-9][0-9][0-9][0-9][0-9][0-9].log /mn/stornext/d22/cmbco/comap/protodir/presto_level1_logs/$(MONTH)
rsync -Pavt --no-links comap_analysis@presto.caltech.edu:/comapdata*/pathfinder/level1/$(MONTH)*/comap-[0-9][0-9][0-9][0-9][0-9][0-9][0-9]-$(DAY)-[0-9][0-9][0-9][0-9][0-9][0-9]_s1.hd5 /mn/stornext/d22/cmbco/comap/protodir/tsys_presto/$(MONTH)
rsync --dry-run --info=name --archive --relative --no-implied-dirs --no-links comap_analysis@presto.caltech.edu:/comapdata*/pathfinder/level1/$(MONTH)*/comap-[0-9][0-9][0-9][0-9][0-9][0-9][0-9]-$(DAY)-[0-9][0-9][0-9][0-9][0-9][0-9].hd5 /mn/stornext/d22/cmbco/comap/protodir/level1/$(MONTH)/ > /mn/stornext/d22/cmbco/comap/protodir/level1/transfer_lists/l1_transfer_$(DAY).log
sleep 2
cat /mn/stornext/d22/cmbco/comap/protodir/level1/transfer_lists/l1_transfer_$(DAY).log | /astro/local/bin/parallel --will-cite -j 12 rsync -Pavt --no-links comap_analysis@presto.caltech.edu:/{} /mn/stornext/d22/cmbco/comap/protodir/level1/l1_temp/from_ovro/$(MONTH)/