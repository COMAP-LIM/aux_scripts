"""
COMAP HDF5 Spectrometer Data Downsampler (written by Gemini and inspected/modified by Jonas)

This script performs downsampling of COMAP Level 1 spectrometer data along both time and frequency
axes. It traverses an input HDF5 file, applies downsampling to mapped datasets, and writes the
output to a new, cleanly constructed HDF5 file. The script reads the full datasets into system RAM.

Handling of Edge Cases & Data Artifacts:
--------------------------------------
* NaN/Inf Propagation: The script uses standard IEEE 754 arithmetic (`numpy.mean`). 
  If any component value within an N-length averaging block is `NaN` or `Inf`, 
  the entire resulting downsampled entry is set to `NaN` or `Inf`. No masking is 
  performed.
* Edge Cropping: The time axis length varies and is rarely perfectly divisible 
    by the downsampling factor (Y). The script calculates the remainder and crops 
    the excess samples from the end of the time axis before averaging.

HDF5 Feature Support & Limitations:
-----------------------------------
Because the script extracts raw payloads into NumPy arrays and rebuilds the HDF5 
structure from scratch, it inherently alters advanced internal HDF5 features:
* Supported (Preserved): Deep directory hierarchies, standard numerical arrays, 
  and explicit metadata (attributes/headers) on both groups and datasets.
* Not Supported (Severed/Broken): 
    - Soft/External Links: Automatically dereferenced and copied as hard, independent datasets.
    - Dimension Scales: Internal dataset linkages are severed.
    - Object References: Pointer memory addresses are invalidated.
    - Variable-Length (VLEN) Strings: Calling `obj[...]` on VLEN string arrays 
      will cause `h5py` to cast them as generic Python objects, crashing the 
      downstream `create_dataset` function with a `TypeError`.

System Considerations:
--------------------
* Fragmentation and disk usage: I have noticed that files tend to reduce in size if I perform a raw
  h5repack on the final output from this script. This is probably because it defragments the files.
  I would recommend doing so to reduce final disk usage by something like 5-10%.
  Doing so serially after all compression is finished is also probably optimal.
"""

import argparse
from pathlib import Path
import warnings

import h5py

# The cache size we want hdf5 to build up before triggering a write. Larger = less frequent I/O.
# 4 * 1024 * 1024 * 1024 = 4 GB cache per file
CACHE_SIZE = 4 * 1024**3

# This dictionary specifies which dimensions are the time- and frequency-dimensions in all hdf5
# datasets that are to be downsampled. A value of -1 means the dataset does not contain the
# respective dimension at all.
AXIS_MAPPING = {
    'spectrometer/MJD':                     {'time':  0, 'freq': -1},
    'spectrometer/band_average':            {'time':  2, 'freq': -1},
    'spectrometer/features':                {'time':  0, 'freq': -1},
    'spectrometer/frequency':               {'time': -1, 'freq':  1},
    'spectrometer/time_average':            {'time': -1, 'freq':  2},
    'spectrometer/tod':                     {'time':  3, 'freq':  2},
    'spectrometer/pixel_pointing/pixel_az': {'time':  1, 'freq': -1},
    'spectrometer/pixel_pointing/pixel_dec':{'time':  1, 'freq': -1},
    'spectrometer/pixel_pointing/pixel_el': {'time':  1, 'freq': -1},
    'spectrometer/pixel_pointing/pixel_ra': {'time':  1, 'freq': -1},
}


def validate_inputs(input_path, output_path, freq_factor, time_factor):
    """
    Performs lightweight input validation before opening files or processing data.
    """
    if freq_factor <= 0:
        raise ValueError("Frequency downsampling factor must be positive and non-zero.")

    if time_factor <= 0:
        raise ValueError("Time downsampling factor must be positive and non-zero.")

    if 1024 % freq_factor != 0:
        raise ValueError(f"Frequency downsample factor X ({freq_factor}) must cleanly divide 1024.")

    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist or is not a regular file: {input_path}")

    if not output_path.parent.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_path.parent}")

    if input_path.resolve() == output_path.resolve(strict=False):
        raise ValueError("Input and output files must be different.")

    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")

    return input_path, output_path


def get_missing_required_datasets(h5file):
    """
    Returns the required mapped datasets that are missing from the input file.
    """
    return [dataset_name for dataset_name in AXIS_MAPPING if dataset_name not in h5file]

def slice_and_downsample(data, time_axis, freq_axis, X, Y, crop_end):
    """
    Applies end-only time cropping and dimension restructuring to perform 
    vectorized averaging in-core.
    
    Parameters:
        data (np.ndarray): The raw n-dimensional data array.
        time_axis (int): The index of the time axis (-1 if absent).
        freq_axis (int): The index of the frequency axis (-1 if absent).
        X (int): Frequency downsampling factor.
        Y (int): Time downsampling factor.
        crop_end (int): Number of time samples to discard from the end.
        
    Returns:
        np.ndarray: The cropped and downsampled array.
    """
    # 1. End-only time axis cropping
    if time_axis != -1:
        slices = [slice(None)] * data.ndim
        slices[time_axis] = slice(None, data.shape[time_axis] - crop_end)
        data = data[tuple(slices)]

    # 2. Vectorized frequency downsampling
    # Introduces a new dimension of length X, then averages across it.
    if freq_axis != -1:
        shape = list(data.shape)
        shape.insert(freq_axis + 1, X)
        shape[freq_axis] //= X
        data = data.reshape(shape).mean(axis=freq_axis + 1)

    # 3. Vectorized time downsampling
    # Introduces a new dimension of length Y, then averages across it.
    if time_axis != -1:
        shape = list(data.shape)
        shape.insert(time_axis + 1, Y)
        shape[time_axis] //= Y
        data = data.reshape(shape).mean(axis=time_axis + 1)

    return data


def downsample_comap_file(input_path, output_path, freq_factor, time_factor):
    """
    Core function to process and downsample a single COMAP HDF5 file. 
    Can be imported and called programmatically by other Python scripts.

    Parameters:
        input_path (str): Path to the input HDF5 file.
        output_path (str): Path to write the new output HDF5 file.
        freq_factor (int): Frequency downsampling factor (must cleanly divide 1024).
        time_factor (int): Time downsampling factor.
    """
    input_path, output_path = validate_inputs(
        input_path, output_path, freq_factor, time_factor
    )

    X = freq_factor
    Y = time_factor

    with h5py.File(input_path, 'r') as fin:
        missing_datasets = get_missing_required_datasets(fin)
        if missing_datasets:
            warnings.warn(
                f"{input_path}: missing required datasets: {', '.join(missing_datasets)}"
            )
            return False

        # Determine trailing cropping required to make the time axis cleanly divisible by Y
        n_time_orig = fin['spectrometer/MJD'].shape[0]
        if Y > n_time_orig:
            raise ValueError(
                f"Time downsampling factor Y ({Y}) cannot exceed the number of time samples ({n_time_orig})."
            )

        remainder = n_time_orig % Y
        crop_end = remainder
        n_time_downsampled = (n_time_orig - crop_end) // Y

        # HDF5 forbids a chunk dimension larger than the dataset dimension, so short observations
        # (a handful exist, down to ~500 raw samples) get the TOD chunk clipped to the full axis.
        tod_time_chunk = min(4000 // Y, n_time_downsampled)
        if tod_time_chunk <= 0:
            raise ValueError(
                f"Time downsampling factor Y ({Y}) is too large for the fixed TOD chunking policy."
            )

        with h5py.File(output_path, 'w', rdcc_nbytes=CACHE_SIZE) as fout:
            # Propagate file-level root attributes (e.g., observation metadata)
            for k, v in fin.attrs.items():
                fout.attrs[k] = v

            def process_node(name, obj):
                """
                Callback function used by h5py.visititems(). 
                Reconstructs groups and processes datasets.
                """
                if isinstance(obj, h5py.Group):
                    # Ensure the directory structure exists in the output file
                    grp = fout.require_group(name)
                    for k, v in obj.attrs.items():
                        grp.attrs[k] = v

                elif isinstance(obj, h5py.Dataset):
                    info = AXIS_MAPPING.get(name, {'time': -1, 'freq': -1})
                    
                    # Extract all data to RAM. This strips the HDF5 object wrapper, 
                    # effectively discarding the input file's compression filters and 
                    # internal metadata linkages (like Dimension Scales).
                    data = obj[...]

                    # Apply vectorized downsampling if axes are mapped
                    if info['time'] != -1 or info['freq'] != -1:
                        data = slice_and_downsample(data, info['time'], info['freq'], X, Y, crop_end)

                    creation_kwargs = {}

                    if name == 'spectrometer/tod':
                        # Apply specific storage layout constraints to the heavy TOD array
                        creation_kwargs['compression'] = 'gzip'
                        creation_kwargs['compression_opts'] = 3
                        creation_kwargs['shuffle'] = True
                        creation_kwargs['chunks'] = (1, 4, 1024 // X, tod_time_chunk)
                    else:
                        # Enforce strict uncompressed state for all auxiliary and metadata entries
                        creation_kwargs['compression'] = None
                        creation_kwargs['shuffle'] = False
                        
                        # Enable auto-chunking for arrays larger than 1MB. HDF5 enforces a strict 4GB 
                        # limit on contiguous (unchunked) datasets. This avoids file structure crashes.
                        if data.nbytes > 1024 * 1024:
                            creation_kwargs['chunks'] = True

                    # Write the processed numpy array back out to the new file
                    out_ds = fout.create_dataset(name, data=data, **creation_kwargs)
                    
                    # Propagate specific dataset-level attributes
                    for k, v in obj.attrs.items():
                        out_ds.attrs[k] = v

            # Recursively visit all tree items starting from the HDF5 root directory
            fin.visititems(process_node)

    return True


def main():
    """
    CLI entry point. Parses arguments and calls the core downsampling function.
    """
    parser = argparse.ArgumentParser(description="Downsample COMAP HDF5 spectrometer data (In-Core, Custom Compression).")
    parser.add_argument("-i", "--input", required=True, help="Path to input HDF5 file")
    parser.add_argument("-o", "--output", required=True, help="Path to output HDF5 file")
    parser.add_argument("-x", "--freq-factor", type=int, required=True, help="Frequency downsampling factor")
    parser.add_argument("-y", "--time-factor", type=int, required=True, help="Time downsampling factor")
    args = parser.parse_args()

    # Call the standalone processing function
    downsample_comap_file(args.input, args.output, args.freq_factor, args.time_factor)

if __name__ == "__main__":
    main()