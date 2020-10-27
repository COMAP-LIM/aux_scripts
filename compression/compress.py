"""
Program to compress a series of hdf5 files in a series of directories, using the h5repack gzip compression algorithm.
In a series of specified direcotories, it compresses all files following the format comap-*.hd5, using a set number of threads.

Takes two or more command-line arguments.
The first argument is the number of threads to use, as an integer.
An arbritrary number of arguments can follow, specifying the folders in which to look for hdf5 files.
Use relative paths from where the program is executed, and include a trailing /

Example compressing all files in the folders 2019-05 and 2019-06 using 48 threads:
python3 compress.py 48 2019-05/ 2019-06/

A single logfile is written, on the format comp-[datetime].log, to the folder where the program was executed.
"""


import os
from os import listdir
from os.path import isfile, join
import sys
from multiprocessing.pool import Pool
import logging
import time
import datetime


class Compress:
    def __init__(self, nr_threads, path):
        self.nr_threads = nr_threads
        self.path = path
        self.filenames = []
        self.nr_errors = 0
        self.raw_size_sum = 0
        self.comp_size_sum = 0

        logging.info("Initializing compression for directory %s", path)

        self.find_filenames()



    def find_filenames(self):
        #### Parse through directory to find filenames matching pattern. ####
        logging.info("Finding files matching pattern.")
        path = self.path
        nonzerofiles = 0
        zerofiles = 0
        for filename in listdir(path):
            if isfile(path + filename): # Check if file. Could be dir.
                if filename[:6] == "comap-" and filename[-4:] == ".hd5": # Do only files starting with comap- and ending with .hd5
                    self.filenames.append(filename)
                    if os.path.getsize(path + filename) > 0:
                        nonzerofiles += 1
                    else:
                        zerofiles += 1
        logging.info("Finished file searching directory %s. Found %d non-zero files, and %d 0-byte files." % (path, nonzerofiles, zerofiles))



    def compress(self):
        #### Main compression function. Start compression of all files over multiple threads using multiprocessing.Pool. ####
        filenames = self.filenames
        path = self.path
        logging.info("Starting compression.")
        logging.info("Compressing %d files with %d threads in dir %s." % (len(filenames), nr_threads, path))
        t0 = time.time()
        with Pool(nr_threads) as p:
            p.map(self.compress_file, range(len(filenames)))
        t1 = time.time()
        for filename in self.filenames:
            raw_filepath = path + filename
            comp_filepath = path + "comp_" + filename
            self.raw_size_sum += os.path.getsize(raw_filepath)
            self.comp_size_sum += os.path.getsize(comp_filepath)
        logging.info("Finished compression job. Old totsize=%.2fGB  new totsize=%.2fGB  ratio=%.2f" % (self.raw_size_sum*1e-9, self.comp_size_sum*1e-9, self.raw_size_sum/max(1, self.comp_size_sum)))
        logging.info("Total execution time %.2f minutes." % ((t1-t0)/60.0))


    def validate(self):
        #### Main validation function. Starts validation function for all files over multiple threads. ####

        logging.info("Starting validation.")
        # Starting file-counting validation.
        logging.info("Counting files...")
        for filename in self.filenames:
            raw_filepath = path + filename
            comp_filepath = path + "comp_" + filename
            if not os.path.isfile(raw_filepath):
                logging.error("THE ORIGINAL FILE %s SHOULD EXIST, BUT I CAN'T FIND IT." % raw_filepath)
            if not os.path.isfile(comp_filepath):
                logging.error("THE COMPRESSED FILE %s SHOULD EXIST, BUT I CAN'T FIND IT." % comp_filepath)
                
        # Starting content-comparison validation.
        logging.info("Comparing compressed to uncompressed version.")
        t2 = time.time()
        with Pool(nr_threads) as p:
            p.map(self.validate_file, range(len(self.filenames)))
        t3 = time.time()
        logging.info("Finished validation in %.2f seconds" % (t3-t2))
        if self.nr_errors > 0:
            logging.error("THERE WERE %d ERRORS DURING VALIDATION!" % self.nr_errors)
        else:
            logging.info("There were no errors.")



    def compress_file(self, file_index):
        #### Compression function for single file, which will be run by each thread. A unique file index is provided by the Pool.map func. ####
        t00 = time.time()
        filename = self.filenames[file_index]
        raw_filepath = path + filename  # Filepath of old uncompressed file.
        comp_filepath = path + "comp_" + filename  # Add "comp_" to compressed filename and write it to the same folder.
        if os.path.getsize(raw_filepath) > 0:
            command = "h5repack -f /spectrometer/tod:GZIP=1 %s %s" % (raw_filepath, comp_filepath)
        else:
            command = "cp %s %s" % (raw_filepath, comp_filepath)
        exitstatus = os.system(command)
        t01 = time.time()
        if exitstatus != 0:
            logging.error("h5repack ON FILE %s FINISHED WITH NON-ZERO EXIT STATUS %d." % (comp_filepath, exitstatus))
        
        minutes_spent = (t01-t00)/60.0
        raw_size = os.path.getsize(raw_filepath)
        comp_size = os.path.getsize(comp_filepath)
        logging.info("Finished file %s (%d/%d)  oldsize=%.2fGB  newsize=%.2fGB  ratio=%.2f  time spent=%.1fm" % (raw_filepath, file_index+1, len(self.filenames), raw_size*1e-9, comp_size*1e-9, raw_size/max(1, comp_size), minutes_spent))



    def validate_file(self, file_index):
        #### Validation function for a single file, to be run in parallel by Pool.map. ####
        filename = self.filenames[file_index]
        raw_filepath = path + filename
        comp_filepath = path + "comp_" + filename
        
        if os.path.getsize(raw_filepath) > 0 and os.path.getsize(comp_filepath) > 0:  # If both filesizes are nonzero.
            errorcode = os.system("h5diff %s %s" % (raw_filepath, comp_filepath))
        elif os.path.getsize(raw_filepath) == 0 and os.path.getsize(comp_filepath) == 0:  # If both filesizes are zero, everything is fine.
            errorcode = 0
        else: # If only one filesize is zero, something is wrong.
            errorcode = 1
        
        if errorcode != 0:
            self.nr_errors += 1
            logging.error("FILE %s DOES NOT MATCH COMPRESSED VERSION %s!" % (raw_filepath, comp_filepath))




if __name__ == "__main__":
    #### Read command-line arguments. ####
    if len(sys.argv) < 2:
        raise IndexError("Must provide at least two arguments: Number of threads, and at least one folder path for the uncompressed files.")
    else:
        nr_threads = int(sys.argv[1])
        paths = []
        for i in range(2, len(sys.argv)):
            paths.append(sys.argv[i])

    #### Set up logging ####
    timestring = str(datetime.datetime.now()).replace(" ", "-").replace(":", "-")
    logfilename = "comp-%s.log" % timestring
    logging.basicConfig(filename=logfilename, filemode="a", format="%(levelname)s %(asctime)s - %(message)s", level=logging.DEBUG)
    logging.info("Log started. Compressing files in directories %s", paths)
    print("Compression script started. Logging to file %s" % logfilename)

    #### Loop over paths and run the compression class on each directory. ####
    for path in paths:
        compress = Compress(nr_threads, path)
        compress.compress()
        compress.validate()