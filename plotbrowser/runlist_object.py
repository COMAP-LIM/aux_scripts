import re

class Runlist():
    def __init__(self, filename):
        self.filename = filename 

    def read_runlist(self):
        runlist_file = open(self.filename, "r")         # Opening 
        runlist = runlist_file.read()
        
        tod_in_list = re.findall(r"\/.*?\.\w+", runlist)
        self.tod_in_list = tod_in_list

        patch_name = re.findall(r"\s([a-zA-Z0-9]+)\s", runlist)
        self.patch_names = str(patch_name.group(1))

        obsIDs = re.findall(r"\s\d{6}\s", runlist)         # Regex pattern to find all obsIDs in runlist
        self.obsIDs = [num.strip() for num in obsIDs]
        self.nobsIDs = len(self.obsIDs)                    # Number of obsIDs in runlist
        
        scans_per_obsid = re.findall(r"\d\s+(\d+)\s+\/", runlist)
        self.scans_per_obsid = [int(num.strip()) - 2 for num in scans_per_obsid]

        runlist_file.close()