import sys
import os
from tqdm import tqdm

path = sys.argv[1]
all_files = os.listdir(path)
filenames = []
for filename in os.listdir(path):
    if os.path.isfile(os.path.join(path, filename)):
        if filename[:5] == "comp_" and filename[-4:] == ".hd5":
            filenames.append(filename)

print(f"Found {len(filenames)} compressed files in directory {path}.")
print(f"Renaming...")
for filename in tqdm(filenames):
    new_filename = filename[5:]
    new_filepath = os.path.join(path, new_filename)
    filepath = os.path.join(path, filename)
    # print(filepath, new_filepath)
    os.rename(filepath, new_filepath)