"""
COMAP HDF5 Batch Downsampler

This script manages the parallel execution of the COMAP HDF5 downsampling. It scans an input
directory for level-1 spectrometer files, spawns a multiprocessing pool, and processes multiple
files concurrently.

Usage Example:
--------------
Run the script from the command line, pointing it to your raw data directory and your intended
output directory. Specify the downsampling factors (X and Y) and the number of parallel worker
processes to spawn.

    python3 downsample_level1_directory.py \\
        -i /mn/stornext/d22/cmbco_nobackup/comap/data/level1/2025-07/ \\
        -o /path/to/your/archival/output/ \\
        -x 16 \\
        -y 10 \\
        -p 16

Arguments:
----------
    -i, --input-dir   : Directory containing the raw 'comap-*.hd5' files.
    -o, --output-dir  : Directory where the downsampled files will be saved.
    -x, --freq-factor : Frequency downsampling factor (must cleanly divide 1024).
    -y, --time-factor : Time downsampling factor.
    -p, --processes   : Number of concurrent parallel workers (default: 4).
                        WARNING: Do not exceed your node's allocated CPU cores.

* Fragmentation and disk usage: I have noticed that files tend to reduce in size if I perform a raw
  h5repack on the final output from this script. This is probably because it defragments the files.
  I would recommend doing so to reduce final disk usage by something like 5-10%.
  Doing so serially after all compression is finished is also probably optimal.
"""

import argparse
import multiprocessing
from pathlib import Path
import traceback
import time

from downsample_level1_file import downsample_comap_file

def worker_wrapper(args):
    """
    Unpacks arguments and safely executes the downsampling function.
    Acts as a protective barrier so one failed file doesn't crash the pool.
    """
    input_path, output_path, x, y = args
    
    try:
        print(f"[START] Processing: {input_path.name}")
        start_time = time.time()

        created_output = downsample_comap_file(str(input_path), str(output_path), x, y)
        
        elapsed = time.time() - start_time
        if created_output:
            print(f"[SUCCESS] Finished: {input_path.name} in {elapsed:.1f}s")
            return (input_path.name, "success", None)

        print(f"[SKIPPED] Skipped: {input_path.name} in {elapsed:.1f}s")
        return (input_path.name, "skipped", None)
        
    except Exception as e:
        print(f"[ERROR] Failed on {input_path.name}: {e}")
        # Print the traceback to the cluster log for debugging
        traceback.print_exc()
        return (input_path.name, "failed", str(e))

def main():
    parser = argparse.ArgumentParser(description="Batch process COMAP HDF5 files in parallel.")
    parser.add_argument("-i", "--input-dir", required=True, help="Directory containing raw HDF5 files")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory to save downsampled files")
    parser.add_argument("-x", "--freq-factor", type=int, required=True, help="Frequency downsampling factor")
    parser.add_argument("-y", "--time-factor", type=int, required=True, help="Time downsampling factor")
    parser.add_argument("-p", "--processes", type=int, default=4, help="Number of parallel processes to spawn")
    
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Validate directories
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all files matching 'comap-*.hd5' or 'comap-*.h5'
    # Using a list comprehension to filter the directory contents safely
    target_files = [
        p for p in input_dir.iterdir() 
        if p.is_file() and p.name.startswith("comap-") and p.suffix in [".hd5", ".h5"]
    ]

    if not target_files:
        print(f"No matching COMAP files found in {input_dir}.")
        return

    print(f"Found {len(target_files)} files. Starting pool with {args.processes} processes...")

    # Prepare the arguments for the multiprocessing pool
    # The output file will maintain the exact same name, just in the new directory
    task_args = [
        (file_path, output_dir / file_path.name, args.freq_factor, args.time_factor)
        for file_path in target_files
    ]

    # Initialize the multiprocessing Pool
    successful_count = 0
    skipped_files = []
    failed_files = []

    with multiprocessing.Pool(processes=args.processes) as pool:
        # imap_unordered is great for clusters as it yields results as soon as they finish,
        # rather than waiting for chunks to complete sequentially.
        for filename, status, error_msg in pool.imap_unordered(worker_wrapper, task_args):
            if status == "success":
                successful_count += 1
            elif status == "skipped":
                skipped_files.append(filename)
            else:
                failed_files.append(filename)

    # Final summary report
    print("\n" + "="*40)
    print("BATCH PROCESSING COMPLETE")
    print("="*40)
    print(f"Total files processed: {len(target_files)}")
    print(f"Successfully created:  {successful_count}")
    print(f"Skipped:               {len(skipped_files)}")
    print(f"Failed:                {len(failed_files)}")

    if skipped_files:
        print("\nSkipped Files:")
        for file_name in skipped_files:
            print(f"  - {file_name}")

    if failed_files:
        print("\nFailed Files:")
        for file_name in failed_files:
            print(f"  - {file_name}")

if __name__ == "__main__":
    main()