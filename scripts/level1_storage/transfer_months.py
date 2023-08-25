import sys
import os

months = sys.argv[1:]

print(f"Months: {months}")

for month in months:
    print(f">>>>>>>>>> Transfering log files. <<<<<<<<<<")
    command = f"rsync -Pav --no-links comap_analysis@presto.caltech.edu:/comapdata*/pathfinder/level1/{month}*/comap-[0-9][0-9][0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9][0-9][0-9][0-9][0-9].log /mn/stornext/d22/cmbco/comap/protodir/presto_level1_logs/{month}/"
    try:
        os.system(command)
    except:
        print(f"Found no files.")

    print(f">>>>>>>>>> Transfering tys files. <<<<<<<<<<")
    command = f"rsync -Pav --no-links comap_analysis@presto.caltech.edu:/comapdata*/pathfinder/level1/{month}*/comap-[0-9][0-9][0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9][0-9][0-9][0-9][0-9]_s1.hd5 /mn/stornext/d22/cmbco/comap/protodir/tsys_presto/{month}/"
    try:
        os.system(command)
    except:
        print(f"Found no files.")

    # print(f">>>>>>>>>> Transfering level1 files. <<<<<<<<<<")
    # command = f"rsync -Pav --no-links comap_analysis@presto.caltech.edu:/comapdata*/pathfinder/level1/{month}*/comap-[0-9][0-9][0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]-[0-9][0-9][0-9][0-9][0-9][0-9].hd5 /mn/stornext/d22/cmbco/comap/protodir/level1/{month}/"
    # try:
    #     os.system(command)
    # except:
    #     print(f"Found no files.")
