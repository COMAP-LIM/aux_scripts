import time
import sys
import getopt
import numpy as np
from scipy import signal
import h5py as h5
import textwrap
import ctypes

class Atlas:
    def __init__(self):
        """
        Initiating Atlas class and setting class attributes and default command line arguments
        """
        self.jk_choices   = ["odde", "dayn", "half", "sdlb", "sidr"]    # Possible choices of jackknife modes.
        self.jk           = None                                        # If no jackknife argument is given self.jk will be None.
        self.jack         = False                                       # If ture operations are performed on jackknives, changes to 
                                                                        #true if jackknife command line input is given.

        self.tool_choices   = ["coadd", "subtract", "add", "dgradeXY", "dgradeZ", "dgradeXYZ",
                                                           "ugradeXY", "ugradeZ", "ugradeXYZ",
                                                           "smoothXY", "smoothZ", "smoothXYZ"]     # Tool choices.
        self.tool         = None               # Default tool is coadd.
        self.det_list     = np.arange(1,20)         # List of detectors to use, default all.
        self.sb_list      = np.arange(1,5)          # List of sidebands to use, default all.
        self.freq_list    = np.arange(1,65)         # List of frequency channels per sideband, default all.
        self.outfile      = "outfile.h5"            # Output file name.
        self.n_sigma      = 5                       # Default n_sigma; used for defining the Gaussian smoothing kernel's grid.

        self.beam        = False    # If true subsequent operations are only performed on _beam dataset.
        self.full        = False    # If true subsequent operations are only performed on full dataset.
        self.everything  = False    # If true full, beam and jackknive datasets are all processed.
        self.patch1       = ''      # Patch name of first infile.
        self.patch2       = ''      # Patch name of second infile.
        self.infile1      = None    # Fist infile name.
        self.infile2      = None    # Second infile name.
        self.maputilslib = ctypes.cdll.LoadLibrary("/mn/stornext/d16/cmbco/comap/protodir/auxiliary/maputilslib.so.1")  # Load shared C utils library.
        #self.maputilslib = ctypes.cdll.LoadLibrary("maputilslib.so.1")  # Load shared C utils library.

        self.input()    # Calling the input function to set variables dependent on command line input.

        if self.infile1 != None and self.infile2 != None:
            """Checking whether both input files have jackknife datasets"""
            if self.jack and ("jackknives" not in self.dfile1 or "jackknives" not in self.dfile2):
                print("One or both of the input files does not contain any jackknife information!")
                sys.exit()

        if not self.full and not self.beam and not self.jack:
            """Checking whether to process all datasets"""
            self.everything = True
            if self.infile1 != None and self.infile2 != None:
                if "jackknives" in self.dfile1 and "jackknives" in self.dfile2:
                    nhit_lst = [i for i in self.dfile1["jackknives"].keys() if "nhit_" in i]
                    self.jk =  [i.split("_")[1] for i in nhit_lst]
            else: 
                if "jackknives" in self.dfile1:
                    nhit_lst = [i for i in self.dfile1["jackknives"].keys() if "nhit_" in i]
                    self.jk =  [i.split("_")[1] for i in nhit_lst]
        
        self.ofile = h5.File(self.outfile, "w")             # Opening outfile object with write access.   
        self.operation()                                    # Calling operations to perform operation with tool given in command line.
        self.dfile1.close()                                 # Closing first input file.
        if self.infile1 != None and self.infile2 != None:   
            self.dfile2.close()                             # Closing second input file if provided.
        self.ofile.close()                                  # Closing output file.

    def input(self):
        """
        Function processing the command line input using getopt.
        """

        if len(sys.argv) == 1:
            self.usage()

        try:
            opts, args = getopt.getopt(sys.argv[1:],"s:f:i:h:d:o:I:j:t:bF", ["sb=", "freq=", "infile1=", "help", "det=",
                                                                            "out=","infile2=", "jk=", "tool=", "beam", "full"])
        except getopt.GetoptError:
            self.usage()

        for opt, arg in opts:
            
            if opt in ("-j", "--jk"):
                self.jack = True
                self.jk = arg.split(",")
                self.jk = list(self.jk)
                for jk in self.jk:
                    if jk not in self.jk_choices:
                        print("Make sure you have chosen the correct jk choices")                                                                                                   
                        sys.exit() 
            
            elif opt in ("-t", "--tool"):
                """Processing input tool"""
                conditionXY   = "dgradeXY" in arg.split(",") or "ugradeXY" in arg.split(",")
                conditionZ    = "dgradeZ" in arg.split(",") or "ugradeZ" in arg.split(",")
                conditionXYZ  = "dgradeXYZ" in arg.split(",") or "ugradeXYZ" in arg.split(",")

                smoothXY    = "smoothXY" in arg.split(",")
                smoothZ    = "smoothZ" in arg.split(",")
                smoothXYZ    = "smoothXYZ" in arg.split(",")

                if "coadd" in arg or "subtract" in arg or "add" in arg:
                    self.tool = arg
                    if self.infile1 != None and self.infile2 == None:
                        print("To perform a coadd or subtraction two input files must be given!")
                        sys.exit()
                
                elif conditionXY and len(arg.split(",")) == 2:
                    if self.infile1 != None and self.infile2 != None:
                        print("Tool ugradeXY and dgradeXY are only supported for single input file!")
                        sys.exit()
                    self.tool, self.merge_numXY = arg.split(",")
                    self.merge_numXY = int(self.merge_numXY)
                    n_x = np.array(self.dfile1["n_x"])
                    n_y = np.array(self.dfile1["n_y"])
                    if n_x % self.merge_numXY != 0 or n_y % self.merge_numXY != 0: 
                        message = """\
                        Make sure that the pixel grid resolution of input map 
                        file is a multiple of the number of merging pixels!
                        """
                        print(textwrap.dedent(message))
                        sys.exit()
                
                elif conditionXY and len(arg.split(",")) != 2:
                    message = """\
                    To use ugradeXY or dgradeXY tool please provide a number of pixels to co-merge along each 
                    axis; e.g. -t dgrade,2 (don't forget the comma!!)!
                    """
                    print(textwrap.dedent(message))
                    sys.exit()
                
                elif conditionZ and len(arg.split(",")) == 2:
                    if self.infile1 != None and self.infile2 != None:
                        print("Tool ugradeZ and dgradeZ are only supported for single input file!")
                        sys.exit()
                    self.tool, self.merge_numZ = arg.split(",")
                    self.merge_numZ = int(self.merge_numZ)
                    n_z = self.dfile1["freq"].shape[1]
                    if n_z % self.merge_numZ != 0: 
                        message = """\
                        Make sure that the pixel grid resolution of input map file is a multiple
                        of the number of merging pixels!
                        """
                        print(textwrap.dedent(message))
                        sys.exit()
                
                elif conditionZ and len(arg.split(",")) != 2:
                    message = """\
                    To use ugradeZ or dgradeZ tool please provide a number of frequency channels to co-merge along 
                    each axis; e.g. -t dgrade,2 (don't forget the comma!!)!
                    """
                    print(textwrap.dedent(message))
                    sys.exit()
                
                elif conditionXYZ and len(arg.split(",")) == 3:
                    if self.infile1 != None and self.infile2 != None:
                        print("Tool ugradeXYZ and dgradeXYZ are only supported for single input file!")
                        sys.exit()
                    self.tool, self.merge_numXY, self.merge_numZ = arg.split(",")
                    self.merge_numXY, self.merge_numZ = int(self.merge_numXY), int(self.merge_numZ)
                    n_x = np.array(self.dfile1["n_x"])
                    n_y = np.array(self.dfile1["n_y"])
                    n_z = self.dfile1["freq"].shape[1]
                    conditionX = n_x % self.merge_numXY != 0
                    conditionY = n_y % self.merge_numXY != 0
                    conditionZ = n_z % self.merge_numZ  != 0
                    condition  = conditionX or conditionY or conditionZ
                    if condition: 
                        message = """\
                        Make sure that the pixel grid resolution and number of frequency channels of 
                        input map file is a multiple of the number of merging pixels!
                        """
                        print(textwrap.dedent(message))
                        sys.exit()
                
                elif conditionXYZ and len(arg.split(",")) != 3:
                    message = """
                    To use ugradeXYZ or dgradeXYZ tool please provide a number of pixels 
                    to co-merge along each axis as well as a number of frequency 
                    channels to co-merge; e.g. -t dgrade,2,2 (don't forget the commas!!)!
                    """
                    print(textwrap.dedent(message))
                    sys.exit()
                
                elif smoothXY and (len(arg.split(",")) == 3 or len(arg.split(",")) == 4):
                    if len(arg.split(",")) == 3:
                        self.tool, self.sigmaX, self.sigmaY = arg.split(",")
                        self.sigmaX, self.sigmaY = float(self.sigmaX), float(self.sigmaY)

                    elif len(arg.split(",")) == 4:
                        self.tool, self.sigmaX, self.sigmaY, self.n_sigma = arg.split(",")
                        self.sigmaX, self.sigmaY, self.n_sigma = float(self.sigmaX), float(self.sigmaY), int(self.n_sigma)
                    
                    if self.infile1 != None and self.infile2 != None:
                        print("Gaussian smoothing is only supported for single input file!")
                        sys.exit()
                    
                elif smoothXY and (len(arg.split(",")) != 3 or len(arg.split(",")) != 4):
                    message = """
                    To use the smooothing tool please provide a sigmaX, sigmaY and n_sigma; e.g. -t smoothXY,10,10,5  
                    or -t smoothXY,10,10 [default n_sigma = 5] (don't forget the commas!!)!
                    """
                    print(textwrap.dedent(message))
                    sys.exit()

                elif smoothZ and (len(arg.split(",")) == 2 or len(arg.split(",")) == 3):
                    if len(arg.split(",")) == 2:
                        self.tool, self.sigmaZ = arg.split(",")
                        self.sigmaZ = float(self.sigmaZ)
                    
                    elif len(arg.split(",")) == 3:
                        self.tool, self.sigmaZ, self.n_sigma = arg.split(",")
                        self.sigmaZ, self.n_sigma = float(self.sigmaZ), int(self.n_sigma)
                    
                    if self.infile1 != None and self.infile2 != None:
                        print("Gaussian smoothing is only supported for single input file!")
                        sys.exit()
                    
                elif smoothZ and (len(arg.split(",")) != 2 or len(arg.split(",")) != 3):
                    message = """
                    To use the smooothing tool please provide a sigmaZ and n_sigma; e.g. -t smoothZ,10,5 
                    or -t smoothXY,10 [default n_sigma = 5] (don't forget the commas!!)!
                    """
                    print(textwrap.dedent(message))
                    sys.exit()
                
                elif smoothXYZ and (len(arg.split(",")) == 4 or len(arg.split(",")) == 5):
                    if len(arg.split(",")) == 4:
                        self.tool, self.sigmaX, self.sigmaY, self.sigmaZ = arg.split(",")
                        self.sigmaX, self.sigmaY, self.sigmaZ = float(self.sigmaX), float(self.sigmaY), float(self.sigmaZ)

                    elif len(arg.split(",")) == 5:
                        self.tool, self.sigmaX, self.sigmaY, self.sigmaZ, self.n_sigma = arg.split(",")
                        self.sigmaX, self.sigmaY, self.sigmaZ, self.n_sigma = float(self.sigmaX), float(self.sigmaY), float(self.sigmaZ), int(self.n_sigma)
                
                    if self.infile1 != None and self.infile2 != None:
                        print("Gaussian smoothing is only supported for single input file!")
                        sys.exit()
                    
                elif smoothXYZ and (len(arg.split(",")) != 4 or len(arg.split(",")) != 5):
                    message = """
                    To use the smooothing tool please provide a sigmaX, sigmaY, sigmaZ and n_sigma; e.g. -t smoothXYZ,10,10,10,5 
                    or -t smoothXY,10,10,10 [default n_sigma = 5] (don't forget the commas!!)!
                    """
                    print(textwrap.dedent(message))
                    sys.exit()

                if self.tool not in self.tool_choices:
                    print("Make sure you have chosen the correct tool choices")                                                                                                   
                    sys.exit() 
        
            elif opt in ("-b", "--beam"):
                """If beam is chosen, subsequent operations are only performed on _beam datasets"""
                self.beam = True
            
            elif opt in ("-F", "--full"):
                """If full is chosen, subsequent operations are only done on the full datasets"""
                self.full = True
            
            elif opt in ("-o", "--out"):
                """Set custom output file name"""
                self.outfile = arg
            
            elif opt in ("-d", "--det"):
                """Set custom list of detector feeds to use"""
                self.det_list = eval(arg)
                if type(self.det_list) != list and type(self.det_list) != np.ndarray:
                    print("Detectors my be inserted as a list or array, ie. -d [1,2,5,7]")
                    sys.exit()
                else:
                    if 0 in self.det_list:
                        print("Use 1-base, not 0-base please")
                        sys.exit()
                self.det_list = np.array(self.det_list, dtype = int)
            
            elif opt in ("-s", "--sb"):
                """Set custom upper/lower bound of sidebands to use"""
                self.sb_list = eval(arg)
                if type(self.sb_list) != list and type(self.sb_list) != np.ndarray:
                    print("Side bands my be inserted as a list or array, ie. -d [1,2,4]")
                    sys.exit()
                else:
                    if 0 in self.sb_list:
                        print("Use 1-base, not 0-base please")
                        sys.exit()
                self.sb_list = np.array(self.sb_list, dtype = int)
            
            elif opt in ("-f", "--freq"):
                """Set custom upper/lower bound of frequency channels to use"""
                self.freq_list = eval(arg)
                if type(self.freq_list)!= list and type(self.freq_list) != np.ndarray:
                    print("Frequencies my be inserted as a list or array, ie. -n [1,34,50]")
                else:
                    if 0 in self.freq_list:
                        print("Use 1-base, not 0-base please")
                        sys.exit()
                    if np.any(np.array(self.freq_list) > 64):
                        print("There are only 64 frequencies pr. side band")
                        sys.exit()
                self.freq_list = np.array(self.freq_list, dtype = int)
            
            elif opt in ("-i", "--infile1"):
                """Set first input file name (must always be given)"""
                self.infile1 = arg
                self.dfile1  = h5.File(self.infile1,'r')
                temp = self.infile1.split('/')[-1]
                self.patch1 = temp.split('_')[0]
            
            elif opt in ("-I", "--infile2"):
                """Set second infile file name (must be given if to perform coadd, subtract or add)"""
                self.infile2 = arg
                self.dfile2        = h5.File(self.infile2,'r')
                temp = self.infile1.split('/')[-1]
                self.patch2 = temp.split('_')[0]
            
            elif ("-i" in opt or "--infile1" in opt) and ("-I" in opt or "--infile2" in opt):
                if self.patch1 != self.patch2:
                    print("Can only perform operations on two maps if they belong to the same sky-patch.") 
                    sys.exit()
            
            elif opt in ("-h", "--help"):
                self.usage()
            
            else:
                self.usage()

    def usage(self):
        prefix = ""
        preferredWidth = 150
        wrapper = textwrap.TextWrapper(initial_indent=prefix, width=preferredWidth,subsequent_indent=' ' * 30)
        m1 = "[(Side band as a list of upper and lower bound, ie. [1,4]. Default all sidebands]"
        m2 = "[(Frequency, given as list of upper and lower bounds, ie. [1,26]. Default all frequencies] "
        m3 = "[(Tool to use in operation performed on map file(s).]" 
        m3  += "Choices: For one input file coadd, subtract add;" 
        m3  += "for two input files dgradeXY,i, dgradeZ,j and dgradeXYZ,i,j" 
        m3  += "where i and j are the number of px and channels to merge"
        m3  += "ugrade currently not fully supported)]."
        m4 = "[First input file (must always be given)]"
        m5 = "[Second input file (must be given if coadding, subtracting or adding two maps)]"
        m6 = "[(Which detector as a list, ie. [4,11,18]. Default all]"
        m7 = "[Outfile name, default 'outfile.h5']"
        m8 = "[Beam argument, if given operation is only performed on '_beam' datasets.]"
        m8 += "[Default are beam, full and jackknives ]"
        m9 = "[If given performes operation only on full datasets (ie map, nhit and rms."
        m9 += "Default are beam, full and jackknives]"
        m10 = "[Jackknife mode to perform operation on. Choices are dayn, half, odde and sdlb." 
        m10 += "Default are beam, full and jackknives]"
        m11 = "[Help which diplays this text]"
        
        print("\nThis is the usage function\n")
        print("Before running the program one must comile the comanion C library:")
        print("$~ [compiler] -shared -O3 -fPIC maputilslib.c -o maputilslib.so.1\n")
        print("Run example:")
        print("$~ python mapeditor.py -i inmap1.h5 -I inmap2.h5 -t coadd\n")
        print("Flags:")
        print("-i ----> optional --infile1..." + wrapper.fill(m4))
        print("-I ----> optional --infile2..." + wrapper.fill(m5))
        print("-o ----> optional --out......." + wrapper.fill(m7))
        print("-d ----> optional --det......." + wrapper.fill(m6))
        print("-s ----> optional --sb........" + wrapper.fill(m1))
        print("-f ----> optional --freq......" + wrapper.fill(m2))
        print("-t ----> optional --tool......" + wrapper.fill(m3))
        print("-b ----> optional --beam......" + wrapper.fill(m8))
        print("-F ----> optional --full......" + wrapper.fill(m9))
        print("-j ----> optional --jk........" + wrapper.fill(m10))
        print("-h ----> optional --help......" + wrapper.fill(m11))
        sys.exit()

    def readMap(self, first_file = True, jackmode = None):
        """
        Function reading an input file and returns map, nhit and rms (for beam, full or jackknife) datasets.
        
        Parameters:
        ------------------------------
        first_infile: bool
            If true the data is read from the first input file, else the second input file is read.
        jackkmode: str
            A string containing the name of the chosen jackknife name; dayn, half, odde or sdlb.
        ------------------------------
        Returns:
            map: numpy.ndarray
                Array containing the map dataset from the given input file.    
            nhit: numpy.ndarray
                Array containing the nhit dataset from the given input file.
            rms: numpy.ndarray
                Array containing the rms dataset from the given input file.
        """

        if first_file:
            dfile = self.dfile1
        else:
            dfile = self.dfile2

        """Extracting frequency and sideband upper and lower bounds"""
        freq_start = self.freq_list[0] - 1
        freq_end   = self.freq_list[-1] - 1
        sb_start = self.sb_list[0] - 1
        sb_end   = self.sb_list[-1] - 1
        
        if jackmode != None:
            """Reading jackknife datasets"""
            if jackmode == "dayn" or jackmode == "half":
                map  =  dfile["jackknives/map_" + jackmode][:, self.det_list - 1, sb_start:sb_end + 1, freq_start:freq_end + 1, ...]
                nhit =  dfile["jackknives/nhit_" + jackmode][:, self.det_list - 1, sb_start:sb_end + 1, freq_start:freq_end + 1, ...]
                rms  =  dfile["jackknives/rms_" + jackmode][:, self.det_list - 1, sb_start:sb_end + 1, freq_start:freq_end + 1, ...]
            else: 
                map  =  dfile["jackknives/map_" + jackmode][:, sb_start:sb_end + 1, freq_start:freq_end + 1, ...]
                nhit =  dfile["jackknives/nhit_" + jackmode][:, sb_start:sb_end + 1, freq_start:freq_end + 1, ...]
                rms  =  dfile["jackknives/rms_" + jackmode][:, sb_start:sb_end + 1, freq_start:freq_end + 1, ...]
        else:
            if self.beam:
                """Reading beam dataset"""
                try:
                    map =  dfile["map_beam"][sb_start:sb_end + 1, freq_start:freq_end + 1, ...]
                    nhit =  dfile["nhit_beam"][sb_start:sb_end + 1, freq_start:freq_end + 1, ...]
                    rms =  dfile["rms_beam"][sb_start:sb_end + 1, freq_start:freq_end + 1, ...]
                except KeyError:
                    map =  dfile["map_coadd"][sb_start:sb_end + 1, freq_start:freq_end + 1, ...]
                    nhit =  dfile["nhit_coadd"][sb_start:sb_end + 1, freq_start:freq_end + 1, ...]
                    rms =  dfile["rms_coadd"][sb_start:sb_end + 1, freq_start:freq_end + 1, ...]
                    

            elif self.full:
                """Reading full dataset"""
                map =  dfile["map"][self.det_list - 1, sb_start:sb_end + 1, freq_start:freq_end + 1, ...]
                nhit =  dfile["nhit"][self.det_list - 1, sb_start:sb_end + 1, freq_start:freq_end + 1, ...]
                rms =  dfile["rms"][self.det_list - 1, sb_start:sb_end + 1, freq_start:freq_end + 1, ...]

        return map, nhit, rms
        
    def writeMap(self, jackmode = None, write_the_rest = False):
        """
        Function that saves self.map, self.nhit and self.rms class attributes are saved 
        to output file. 

        Parameters:
        ------------------------------
        jackkmode: str
            A string containing the name of the chosen jackknife name; dayn, half, odde or sdlb.
        write_the_rest: bool
            If true all the other datasets and attributes from the first input file are 
            (possibly modified accordingly and) copied to outfile.
        ------------------------------
        """

        if (jackmode != None )and not write_the_rest:
            """
            Generating dataset names and writing jackknives 
            datasets to outfile.
            """
            map_name    = "jackknives/map_" + jackmode
            nhit_name   = "jackknives/nhit_" + jackmode
            rms_name    = "jackknives/rms_" + jackmode
            
            self.ofile.create_dataset(map_name, data = self.map)
            self.ofile.create_dataset(nhit_name, data = self.nhit)
            self.ofile.create_dataset(rms_name, data = self.rms)

        elif (self.beam or self.full) and not write_the_rest:
            if self.beam:
                """Generating dataset names for beam datasets"""
                if "map_beam" in self.dfile1.keys():
                    map_name    = "map_beam"
                    nhit_name   = "nhit_beam"
                    rms_name    = "rms_beam"
                elif "map_coadd" in self.dfile1.keys():
                    map_name    = "map_coadd"
                    nhit_name   = "nhit_coadd"
                    rms_name    = "rms_coadd"

            elif self.full: 
                """Generating dataset names for full datasets"""
                map_name    = "map"
                nhit_name   = "nhit"
                rms_name    = "rms"
            
            """Writing beam or full dataset to outfile"""
            self.ofile.create_dataset(map_name, data = self.map)
            self.ofile.create_dataset(nhit_name, data = self.nhit)
            self.ofile.create_dataset(rms_name, data = self.rms)
        
        if write_the_rest:
            """Copieing and modifying other datasets to outfile"""
            if self.infile1 != None and self.infile2 != None:
                """Things not to copy because they are already copied or will be copied at different time""" 
                
                if "map_beam" in self.dfile1.keys():
                    data_not_to_copy = ["jackknives", "map", "map_beam", "nhit", 
                                        "nhit_beam", "rms", "rms_beam"]
                
                elif "map_coadd" in self.dfile1.keys():
                    data_not_to_copy = ["jackknives", "map", "map_coadd", "nhit", 
                                        "nhit_coadd", "rms", "rms_coadd"]
                
                
                jk_data_not_to_copy = ["map_dayn",  "map_half",  "map_odde",  "map_sdlb",
                                       "nhit_dayn", "nhit_half", "nhit_odde", "nhit_sdlb",
                                       "rms_dayn",  "rms_half",  "rms_odde",  "rms_sdlb"]
                """Looping through and copying over"""                      
                for name in self.dfile1.keys():
                    if name not in self.ofile.keys() and name not in data_not_to_copy:
                        self.ofile.create_dataset(name, data = self.dfile1[name])    
                
                    if "jackknives" in self.dfile1 and "jackknives" in self.dfile2 and "jackknives" in self.ofile:
                        for name in self.dfile1["jackknives"].keys():
                            if name not in self.ofile["jackknives"].keys() and name not in jk_data_not_to_copy:
                                self.ofile.create_dataset("jackknives/" + name, 
                                                            data = self.dfile1["jackknives/" + name])   
            else:   
                if self.tool == "dgradeXY" or self.tool == "ugradeXY":
                    self.merge_numZ = 1
                elif self.tool == "dgradeZ" or self.tool == "ugradeZ":
                    self.merge_numXY = 1

                condition1 = "x" in self.ofile and "y" in self.ofile 
                condition2 = "n_x" in self.ofile and "n_y" in self.ofile 
                condition3 = "nside" in self.ofile
                condition4 = "freq" in self.ofile
                condition  = condition1 and condition2 and condition3 and condition4

                if not condition and "dgrade" in self.tool:
                    """Reading and modify other datasets"""
                    x1, y1 = self.dfile1["x"][:], self.dfile1["y"][:]
                    x      = x1.reshape(int(len(x1) / self.merge_numXY), self.merge_numXY) 
                    y      = y1.reshape(int(len(y1) / self.merge_numXY), self.merge_numXY)
                    x      = np.mean(x, axis = 1)   # Finding new pixel center by averaging neighboring pixel x coordinates
                    y      = np.mean(y, axis = 1)   # Finding new pixel center by averaging neighboring pixel x coordinates
                    nside  = np.array(self.dfile1["nside"]) / self.merge_numXY                
                    freq   = self.dfile1["freq"][:]
                    freq   = freq.reshape(freq.shape[0], int(freq.shape[1] / self.merge_numZ), self.merge_numZ)
                    freq   = np.mean(freq, axis = 2)    # Finding new frequency channel center by averaging 
                                                        # neighbouring frequencies.

                    """Copying over data"""
                    self.ofile.create_dataset("x",      data = x)
                    self.ofile.create_dataset("y",      data = y)
                    self.ofile.create_dataset("n_x",    data = len(x))
                    self.ofile.create_dataset("n_y",    data = len(y))
                    self.ofile.create_dataset("nside",  data = nside)
                    self.ofile.create_dataset("freq",   data = freq)

                elif not condition and "ugrade" in self.tool:
                    """Reading and modifying other datasets"""
                    x1, y1 = self.dfile1["x"][:], self.dfile1["y"][:]
                    dx, dy = x1[1] - x1[0], y1[1] - y1[0]
                    first_center_x = x1[0] - dx / 4  # Finding center of new pixels x value
                    first_center_y = y1[0] - dy / 4  # Finding center of new pixel y value
                    x      = np.zeros(len(x1) * self.merge_numXY) 
                    y      = np.zeros(len(y1) * self.merge_numXY)

                    """Filling up x and y arrays with new pixel centers"""
                    for i in range(len(x1)):
                        x[i] = first_center_x + i * dx / 2
                        y[i] = first_center_y + i * dy / 2

                    nside  = np.array(self.dfile1["nside"]) / self.merge_numXY                
                    
                    freq1   = self.dfile1["freq"][:]
                    freq    = np.zeros((freq1.shape[0], freq1.shape[1] * self.merge_numZ))
                    """Filling up frequency channe array with new bin centers"""
                    df = freq1[:, 1] - freq1[:, 0]
                    first_center_freq = freq1[:, 0] - df / 4    # Finding center of new pixel y value
                    for i in range(freq.shape[1]):
                        freq[:, i] = first_center_freq + i * df / 2
                    
                    """Copying over data to outfile"""
                    self.ofile.create_dataset("x",      data = x)
                    self.ofile.create_dataset("y",      data = y)
                    self.ofile.create_dataset("n_x",    data = len(x))
                    self.ofile.create_dataset("n_y",    data = len(y))
                    self.ofile.create_dataset("nside",  data = nside)
                    self.ofile.create_dataset("freq",   data = freq)

                """Copying over the remainder of the other datasets not yet copied"""
                if "map_beam" in self.dfile1.keys():
                    data_not_to_copy = ["jackknives", "map", "map_beam", "nhit", 
                                        "nhit_beam", "rms", "rms_beam",
                                        "x", "y", "n_x", "n_y", "nside", "freq"]

                elif "map_coadd" in self.dfile1.keys():
                    data_not_to_copy = ["jackknives", "map", "map_coadd", "nhit", 
                                        "nhit_coadd", "rms", "rms_coadd",
                                        "x", "y", "n_x", "n_y", "nside", "freq"]

                jk_data_not_to_copy = ["map_dayn",  "map_half",  "map_odde",  "map_sdlb",
                                       "nhit_dayn", "nhit_half", "nhit_odde", "nhit_sdlb",
                                       "rms_dayn",  "rms_half",  "rms_odde",  "rms_sdlb"]
                for name in self.dfile1.keys():
                    if name not in self.ofile.keys() and name not in data_not_to_copy:
                        self.ofile.create_dataset(name, data = self.dfile1[name])    
                
                    if "jackknives" in self.dfile1 and "jackknives" in self.ofile:
                        for name in self.dfile1["jackknives"].keys():
                            if name not in self.ofile["jackknives"].keys() and name not in jk_data_not_to_copy:
                                self.ofile.create_dataset("jackknives/" + name, 
                                                            data = self.dfile1["jackknives/" + name])   
            
    def operation(self):  
        """
        Function performes the operation with the tool and input file(s) given 
        by the command line input arguments.
        """           
        if self.infile1 != None and self.infile2 != None:
            """Operations of two infiles"""
            if self.everything:
                if "jackknives" in self.dfile1 and "jackknives" in self.dfile2:
                    for jack in self.jk:
                        self.map1, self.nhit1, self.rms1 = self.readMap(True, jack)
                        self.map2, self.nhit2, self.rms2 = self.readMap(False, jack)
                        
                        if self.tool == "coadd":
                            if len(self.map1.shape) == 6:
                                self.C_coadd6D(self.map1, self.nhit1, self.rms1,
                                               self.map2, self.nhit2, self.rms2)  
                    
                            elif len(self.map1.shape) == 5: 
                                self.C_coadd5D(self.map1, self.nhit1, self.rms1,
                                               self.map2, self.nhit2, self.rms2)  

                        elif self.tool == "subtract": 
                            if len(self.map1.shape) == 6:
                                self.C_subtract6D(self.map1, self.nhit1, self.rms1,
                                                  self.map2, self.nhit2, self.rms2)  
                            
                            if len(self.map1.shape) == 5:
                                self.C_subtract5D(self.map1, self.nhit1, self.rms1,
                                                  self.map2, self.nhit2, self.rms2) 
                        elif self.tool == "add": 
                            if len(self.map1.shape) == 6:
                                self.C_add6D(self.map1, self.nhit1, self.rms1,
                                             self.map2, self.nhit2, self.rms2)  
                            
                            if len(self.map1.shape) == 5:
                                self.C_add5D(self.map1, self.nhit1, self.rms1,
                                             self.map2, self.nhit2, self.rms2) 
                        self.writeMap(jack)
                
                self.full = True
                self.map1, self.nhit1, self.rms1 = self.readMap(True)
                self.map2, self.nhit2, self.rms2 = self.readMap(False)
                
                if self.tool == "coadd":
                    self.C_coadd5D(self.map1, self.nhit1, self.rms1,
                                   self.map2, self.nhit2, self.rms2)

                elif self.tool == "subtract": 
                    self.C_subtract5D(self.map1, self.nhit1, self.rms1,
                                      self.map2, self.nhit2, self.rms2)
                elif self.tool == "add": 
                    self.C_add5D(self.map1, self.nhit1, self.rms1,
                                 self.map2, self.nhit2, self.rms2)
                self.writeMap()
                
                self.full = False
                self.beam = True
                self.map1, self.nhit1, self.rms1 = self.readMap(True)
                self.map2, self.nhit2, self.rms2 = self.readMap(False)
                
                if self.tool == "coadd":
                    self.C_coadd4D(self.map1, self.nhit1, self.rms1,
                                   self.map2, self.nhit2, self.rms2)

                elif self.tool == "subtract": 
                    self.C_subtract4D(self.map1, self.nhit1, self.rms1,
                                      self.map2, self.nhit2, self.rms2)  

                elif self.tool == "add": 
                    self.C_add4D(self.map1, self.nhit1, self.rms1,
                                self.map2, self.nhit2, self.rms2)  

                self.writeMap()
                self.beam = False
            
            if self.jack:
                for jack in self.jk:
                    self.map1, self.nhit1, self.rms1 = self.readMap(True, jack)
                    self.map2, self.nhit2, self.rms2 = self.readMap(False, jack)
                    
                    if self.tool == "coadd":
                        if len(self.map1.shape) == 6:
                            self.C_coadd6D(self.map1, self.nhit1, self.rms1,
                                           self.map2, self.nhit2, self.rms2)  
                
                        elif len(self.map1.shape) == 5: 
                            self.C_coadd5D(self.map1, self.nhit1, self.rms1,
                                           self.map2, self.nhit2, self.rms2)

                    elif self.tool == "subtract": 
                        if len(self.map1.shape) == 6:
                            self.C_subtract6D(self.map1, self.nhit1, self.rms1,
                                        self.map2, self.nhit2, self.rms2)  
                        
                        if len(self.map1.shape) == 5:
                            self.C_subtract5D(self.map1, self.nhit1, self.rms1,
                                        self.map2, self.nhit2, self.rms2) 
                        
                        elif self.tool == "add": 
                            if len(self.map1.shape) == 6:
                                self.C_add6D(self.map1, self.nhit1, self.rms1,
                                            self.map2, self.nhit2, self.rms2)  
                            
                            if len(self.map1.shape) == 5:
                                self.C_add5D(self.map1, self.nhit1, self.rms1,
                                            self.map2, self.nhit2, self.rms2) 
                    self.writeMap(jack)

            if self.full:
                _beam = self.beam
                self.beam = False
                self.map1, self.nhit1, self.rms1 = self.readMap(True)
                self.map2, self.nhit2, self.rms2 = self.readMap(False)
                
                if self.tool == "coadd":
                    self.C_coadd5D(self.map1, self.nhit1, self.rms1,
                                    self.map2, self.nhit2, self.rms2)
    
                elif self.tool == "subtract": 
                    self.C_subtract5D(self.map1, self.nhit1, self.rms1,
                                      self.map2, self.nhit2, self.rms2)  
                elif self.tool == "add": 
                    self.C_add5D(self.map1, self.nhit1, self.rms1,
                                self.map2, self.nhit2, self.rms2)  

                self.writeMap()
                self.beam = _beam
        
            if self.beam:
                _full = self.full
                self.full = False
                self.map1, self.nhit1, self.rms1 = self.readMap(True)
                self.map2, self.nhit2, self.rms2 = self.readMap(False)
                
                if self.tool == "coadd":
                    self.C_coadd4D(self.map1, self.nhit1, self.rms1,
                                   self.map2, self.nhit2, self.rms2)

                elif self.tool == "subtract": 
                    self.C_subtract4D(self.map1, self.nhit1, self.rms1,
                                      self.map2, self.nhit2, self.rms2)  
                elif self.tool == "add": 
                    self.C_add4D(self.map1, self.nhit1, self.rms1,
                                self.map2, self.nhit2, self.rms2)

                self.writeMap()
                self.full = _full
            self.writeMap(write_the_rest = True)

        if self.infile1 != None and self.infile2 == None:
            """Operations to perform on single input file"""
            if self.everything:
                if "jackknives" in self.dfile1:
                    for jack in self.jk:
                        self.map1, self.nhit1, self.rms1 = self.readMap(True, jack)
                        if self.tool == "dgradeXY":
                            if len(self.map1.shape) == 6:
                                self.C_dgradeXY6D(self.map1, self.nhit1, self.rms1)

                            elif len(self.map1.shape) == 5:
                                self.C_dgradeXY5D(self.map1, self.nhit1, self.rms1)

                        elif self.tool == "dgradeZ":
                            if len(self.map1.shape) == 6:
                                self.C_dgradeZ6D(self.map1, self.nhit1, self.rms1)
                            
                            elif len(self.map1.shape) == 5:
                                self.C_dgradeZ5D(self.map1, self.nhit1, self.rms1)

                        elif self.tool == "dgradeXYZ":
                            if len(self.map1.shape) == 6:
                                self.C_dgradeXYZ6D(self.map1, self.nhit1, self.rms1)
                            
                            elif len(self.map1.shape) == 5:
                                self.C_dgradeXYZ5D(self.map1, self.nhit1, self.rms1)
                        
                        elif self.tool == "ugradeXY":
                            if len(self.map1.shape) == 6:
                                self.C_ugradeXY6D(self.map1, self.nhit1, self.rms1)

                            elif len(self.map1.shape) == 5:
                                self.C_ugradeXY5D(self.map1, self.nhit1, self.rms1)

                        elif self.tool == "ugradeZ":
                            if len(self.map1.shape) == 6:
                                self.C_ugradeZ6D(self.map1, self.nhit1, self.rms1)

                            elif len(self.map1.shape) == 5:
                                self.C_ugradeZ5D(self.map1, self.nhit1, self.rms1)
                                
                        elif self.tool == "ugradeXYZ":
                            if len(self.map1.shape) == 6:
                                self.C_ugradeXYZ6D(self.map1, self.nhit1, self.rms1)
                            
                            elif len(self.map1.shape) == 5:
                                self.C_ugradeXYZ5D(self.map1, self.nhit1, self.rms1)
                        
                        elif self.tool == "smoothXY":
                            self.gaussian_smoothXY(self.map1, self.nhit1, self.rms1)
            
                        elif self.tool == "smoothZ":
                            self.gaussian_smoothZ(self.map1, self.nhit1, self.rms1)
                        
                        elif self.tool == "smoothXYZ":
                            self.gaussian_smoothXYZ(self.map1, self.nhit1, self.rms1)

                        self.writeMap(jack)

                self.full = True
                self.map1, self.nhit1, self.rms1 = self.readMap(True)
                
                if self.tool == "dgradeXY":
                    self.C_dgradeXY5D(self.map1, self.nhit1, self.rms1)
                
                elif self.tool == "dgradeZ":
                    self.C_dgradeZ5D(self.map1, self.nhit1, self.rms1)
                
                elif self.tool == "dgradeXYZ":
                    self.C_dgradeXYZ5D(self.map1, self.nhit1, self.rms1)
                
                elif self.tool == "ugradeXY":
                    self.C_ugradeXY5D(self.map1, self.nhit1, self.rms1)
                
                elif self.tool == "ugradeZ":
                    self.C_ugradeZ5D(self.map1, self.nhit1, self.rms1)
                
                elif self.tool == "ugradeXYZ":
                    self.C_ugradeXYZ5D(self.map1, self.nhit1, self.rms1)
                
                elif self.tool == "smoothXY":
                    self.gaussian_smoothXY(self.map1, self.nhit1, self.rms1)

                elif self.tool == "smoothZ":
                    self.gaussian_smoothZ(self.map1, self.nhit1, self.rms1)

                elif self.tool == "smoothXYZ":
                    self.gaussian_smoothXYZ(self.map1, self.nhit1, self.rms1)

                self.writeMap()
                
                self.full = False
                self.beam = True
                self.map1, self.nhit1, self.rms1 = self.readMap(True)
                
                if self.tool == "dgradeXY":
                    self.C_dgradeXY4D(self.map1, self.nhit1, self.rms1)

                elif self.tool == "dgradeZ":
                    self.C_dgradeZ4D(self.map1, self.nhit1, self.rms1)

                elif self.tool == "dgradeXYZ":
                    self.C_dgradeXYZ4D(self.map1, self.nhit1, self.rms1)

                elif self.tool == "ugradeXY":
                    self.C_ugradeXY4D(self.map1, self.nhit1, self.rms1)
                
                elif self.tool == "ugradeZ":
                    self.C_ugradeZ4D(self.map1, self.nhit1, self.rms1)

                elif self.tool == "ugradeXYZ":
                    self.C_ugradeXYZ4D(self.map1, self.nhit1, self.rms1)

                elif self.tool == "smoothXY":
                    self.gaussian_smoothXY(self.map1, self.nhit1, self.rms1)

                elif self.tool == "smoothZ":
                    self.gaussian_smoothZ(self.map1, self.nhit1, self.rms1)

                elif self.tool == "smoothXYZ":
                    self.gaussian_smoothXYZ(self.map1, self.nhit1, self.rms1)

                self.writeMap()
                self.beam = False

            if self.jack:
                for jack in self.jk:
                    self.map1, self.nhit1, self.rms1 = self.readMap(True, jack)
                    if self.tool == "dgradeXY":
                        if len(self.map1.shape) == 6:
                            self.C_dgradeXY6D(self.map1, self.nhit1, self.rms1)
                        
                        elif len(self.map1.shape) == 5:
                            self.C_dgradeXY5D(self.map1, self.nhit1, self.rms1)
                        
                    elif self.tool == "dgradeZ":
                        if len(self.map1.shape) == 6:
                            self.C_dgradeZ6D(self.map1, self.nhit1, self.rms1)
                        
                        elif len(self.map1.shape) == 5:
                            self.C_dgradeZ5D(self.map1, self.nhit1, self.rms1)

                    elif self.tool == "dgradeXYZ":
                        if len(self.map1.shape) == 6:
                            self.C_dgradeXYZ6D(self.map1, self.nhit1, self.rms1)
                        
                        elif len(self.map1.shape) == 5:
                            self.C_dgradeXYZ5D(self.map1, self.nhit1, self.rms1)

                    elif self.tool == "ugradeXY":
                        if len(self.map1.shape) == 6:
                            self.C_ugradeXY6D(self.map1, self.nhit1, self.rms1)

                        elif len(self.map1.shape) == 5:
                            self.C_ugradeXY5D(self.map1, self.nhit1, self.rms1)

                    elif self.tool == "ugradeZ":
                        if len(self.map1.shape) == 6:
                            self.C_ugradeZ6D(self.map1, self.nhit1, self.rms1)

                        elif len(self.map1.shape) == 5:
                            self.C_ugradeZ5D(self.map1, self.nhit1, self.rms1)
                            
                    elif self.tool == "ugradeXYZ":
                        if len(self.map1.shape) == 6:
                            self.C_ugradeXYZ6D(self.map1, self.nhit1, self.rms1)
                        
                        elif len(self.map1.shape) == 5:
                            self.C_ugradeXYZ5D(self.map1, self.nhit1, self.rms1)
                    
                    elif self.tool == "smoothXY":
                        self.gaussian_smoothXY(self.map1, self.nhit1, self.rms1)
                    
                    elif self.tool == "smoothZ":
                        self.gaussian_smoothZ(self.map1, self.nhit1, self.rms1)

                    elif self.tool == "smoothXYZ":
                        self.gaussian_smoothXYZ(self.map1, self.nhit1, self.rms1)

                    self.writeMap(jack)

            if self.full:
                _beam = self.beam
                self.beam = False
                self.map1, self.nhit1, self.rms1 = self.readMap(True)
                
                if self.tool == "dgradeXY":
                    self.C_dgradeXY5D(self.map1, self.nhit1, self.rms1)
                
                elif self.tool == "dgradeZ":
                    self.C_dgradeZ5D(self.map1, self.nhit1, self.rms1)
                
                elif self.tool == "dgradeXYZ":
                    self.C_dgradeXYZ5D(self.map1, self.nhit1, self.rms1)

                elif self.tool == "ugradeXY":
                    self.C_ugradeXY5D(self.map1, self.nhit1, self.rms1)
                
                elif self.tool == "ugradeZ":
                    self.C_ugradeZ5D(self.map1, self.nhit1, self.rms1)
                
                elif self.tool == "ugradeXYZ":
                    self.C_ugradeXYZ5D(self.map1, self.nhit1, self.rms1)
                
                elif self.tool == "smoothXY":
                    self.gaussian_smoothXY(self.map1, self.nhit1, self.rms1)

                elif self.tool == "smoothZ":
                    self.gaussian_smoothZ(self.map1, self.nhit1, self.rms1)

                elif self.tool == "smoothXYZ":
                    self.gaussian_smoothXYZ(self.map1, self.nhit1, self.rms1)

                self.writeMap()
                self.beam = _beam
        
            if self.beam:
                _full = self.full
                self.full = False
                self.map1, self.nhit1, self.rms1 = self.readMap(True)
                
                if self.tool == "dgradeXY":
                    self.C_dgradeXY4D(self.map1, self.nhit1, self.rms1)
                
                elif self.tool == "dgradeZ":
                    self.C_dgradeZ4D(self.map1, self.nhit1, self.rms1)

                elif self.tool == "dgradeXYZ":
                    self.C_dgradeXYZ4D(self.map1, self.nhit1, self.rms1)

                elif self.tool == "ugradeXY":
                    self.C_ugradeXY4D(self.map1, self.nhit1, self.rms1)
                
                elif self.tool == "ugradeZ":
                    self.C_ugradeZ4D(self.map1, self.nhit1, self.rms1)

                elif self.tool == "ugradeXYZ":
                    self.C_ugradeXYZ4D(self.map1, self.nhit1, self.rms1)

                elif self.tool == "smoothXY":
                    self.gaussian_smoothXY(self.map1, self.nhit1, self.rms1)

                elif self.tool == "smoothZ":
                    self.gaussian_smoothZ(self.map1, self.nhit1, self.rms1)

                elif self.tool == "smoothXYZ":
                    self.gaussian_smoothXYZ(self.map1, self.nhit1, self.rms1)
                
                self.writeMap()
                self.full = _full
            self.writeMap(write_the_rest = True)            



    def C_coadd4D(self, map1, nhit1, rms1,
                        map2, nhit2, rms2):
        """
        Function taking inn 4D datasets from two infiles, and coadds them.
        Result is saved to class attributes self.map, self.nhit and self.rms.

        Parameters:
        ------------------------------
        map1: numpy.ndarray
            Map dataset of first input file. Must be 4D array.
        nhit1: numpy.ndarray
            Number of hits dataset of first input file. Must be 4D array
        rms1: numpy.ndarray
            Rms dataset of first input file. Must be 4D array
        map2: numpy.ndarray
            Map dataset of second input file. Must be 4D array.
        nhit2: numpy.ndarray
            Number of hits dataset of second input file. Must be 4D array
        rms2: numpy.ndarray
            Rms dataset of second input file. Must be 4D array

        ------------------------------
        """
        float32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=4, flags="contiguous")   # 4D array 32-bit float pointer object.
        int32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=4, flags="contiguous")       # 4D array 32-bit integer pointer object.
        
        self.maputilslib.coadd4D.argtypes = [float32_array4, int32_array4, float32_array4,          # Specifying input types for C library function.
                                             float32_array4, int32_array4, float32_array4,
                                             float32_array4, int32_array4, float32_array4,
                                             ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                             ctypes.c_int]
        n0, n1, n2, n3  = self.map1.shape                               # Extracting axis lengths of 4D array to loop over in C.
        self.map        = np.zeros_like(map1,   dtype = ctypes.c_float) # Generating arrays to fill up with coadded data.
        self.nhit       = np.zeros_like(nhit1,  dtype = ctypes.c_int)
        self.rms        = np.zeros_like(rms1,   dtype = ctypes.c_float)

        self.maputilslib.coadd4D(map1, nhit1, rms1,                     # Filling self.map, self.nhit and self.rms by 
                                 map2, nhit2, rms2,                     # call-by-pointer to C library
                                 self.map, self.nhit, self.rms,
                                 n0, n1, n2, n3)
    
    def C_coadd5D(self, map1, nhit1, rms1,
                        map2, nhit2, rms2):
        """
        Function taking inn 5D datasets from two infiles, and coadds them.
        Result is saved to class attributes self.map, self.nhit and self.rms.

        Parameters:
        ------------------------------
        map1: numpy.ndarray
            Map dataset of first input file. Must be 5D array.
        nhit1: numpy.ndarray
            Number of hits dataset of first input file. Must be 5D array
        rms1: numpy.ndarray
            Rms dataset of first input file. Must be 5D array
        map2: numpy.ndarray
            Map dataset of second input file. Must be 5D array.
        nhit2: numpy.ndarray
            Number of hits dataset of second input file. Must be 5D array
        rms2: numpy.ndarray
            Rms dataset of second input file. Must be 5D array

        ------------------------------
        """
        float32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=5, flags="contiguous")   # 5D array 32-bit float pointer object.
        int32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=5, flags="contiguous")       # 5D array 32-bit integer pointer object.

        self.maputilslib.coadd5D.argtypes = [float32_array5, int32_array5, float32_array5,          # Specifying input types for C library function.
                                            float32_array5, int32_array5, float32_array5,
                                            float32_array5, int32_array5, float32_array5,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int]
        n0, n1, n2, n3, n4 = self.map1.shape                                # Extracting axis lengths of 5D array to loop over in C.
        self.map = np.zeros_like(map1, dtype = ctypes.c_float)              # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros_like(nhit1, dtype = ctypes.c_int)
        self.rms = np.zeros_like(rms1, dtype = ctypes.c_float)

        self.maputilslib.coadd5D(map1, nhit1, rms1,                 # Filling self.map, self.nhit and self.rms by 
                                 map2, nhit2, rms2,                 # call-by-pointer to C library
                                 self.map, self.nhit, self.rms,
                                 n0, n1, n2, n3, n4)
                            
    def C_coadd6D(self, map1, nhit1, rms1,
                        map2, nhit2, rms2):
        """
        Function taking inn 6D datasets from two infiles, and coadds them.
        Result is saved to class attributes self.map, self.nhit and self.rms.

        Parameters:
        ------------------------------
        map1: numpy.ndarray
            Map dataset of first input file. Must be 6D array.
        nhit1: numpy.ndarray
            Number of hits dataset of first input file. Must be 6D array
        rms1: numpy.ndarray
            Rms dataset of first input file. Must be 6D array
        map2: numpy.ndarray
            Map dataset of second input file. Must be 6D array.
        nhit2: numpy.ndarray
            Number of hits dataset of second input file. Must be 6D array
        rms2: numpy.ndarray
            Rms dataset of second input file. Must be 6D array

        ------------------------------
        """
        float32_array6 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=6, flags="contiguous")   # 6D array 32-bit float pointer object.
        int32_array6 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=6, flags="contiguous")       # 6D array 32-bit integer pointer object.

        self.maputilslib.coadd6D.argtypes = [float32_array6, int32_array6, float32_array6,          # Specifying input types for C library function.
                                            float32_array6, int32_array6, float32_array6,
                                            float32_array6, int32_array6, float32_array6,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_int]
        n0, n1, n2, n3, n4, n5 = self.map1.shape                                        # Extracting axis lengths of 6D array to loop over in C.
        self.map = np.zeros_like(map1, dtype = ctypes.c_float)                          # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros_like(nhit1, dtype = ctypes.c_int)
        self.rms = np.zeros_like(rms1, dtype = ctypes.c_float)

        self.maputilslib.coadd6D(map1,     nhit1,     rms1,             # Filling self.map, self.nhit and self.rms by
                                 map2,     nhit2,     rms2,             # call-by-pointer to C library.
                                 self.map, self.nhit, self.rms,
                                 n0,       n1,        n2, 
                                 n3,       n4,        n5)



    def C_subtract4D(self, map1, nhit1, rms1,
                           map2, nhit2, rms2):
        """
        Function taking inn 4D datasets from two infiles, and subracts the second form the first.
        Result is saved to class attributes self.map, self.nhit and self.rms.

        Parameters:
        ------------------------------
        map1: numpy.ndarray
            Map dataset of first input file. Must be 4D array.
        nhit1: numpy.ndarray
            Number of hits dataset of first input file. Must be 4D array
        rms1: numpy.ndarray
            Rms dataset of first input file. Must be 4D array
        map2: numpy.ndarray
            Map dataset of second input file. Must be 4D array.
        nhit2: numpy.ndarray
            Number of hits dataset of second input file. Must be 4D array
        rms2: numpy.ndarray
            Rms dataset of second input file. Must be 4D array

        ------------------------------
        """
        float32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=4, flags="contiguous")   # 4D array 32-bit float pointer object.
        int32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=4, flags="contiguous")       # 4D array 32-bit integer pointer object.

        self.maputilslib.subtract4D.argtypes = [float32_array4, int32_array4, float32_array4,       # Specifying input types for C library function.
                                                float32_array4, int32_array4, float32_array4,
                                                float32_array4, int32_array4, float32_array4,
                                                ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int]
        n0, n1, n2, n3  = self.map1.shape                                   # Extracting axis lengths of 4D array to loop over in C.
        self.map        = np.zeros_like(map1,   dtype = ctypes.c_float)     # Generating arrays to fill up with coadded data.
        self.nhit       = np.zeros_like(nhit1,  dtype = ctypes.c_int)
        self.rms        = np.zeros_like(rms1,   dtype = ctypes.c_float)

        self.maputilslib.subtract4D(map1, nhit1, rms1,              # Filling self.map, self.nhit and self.rms by
                                 map2, nhit2, rms2,                 # call-by-pointer to C library.
                                 self.map, self.nhit, self.rms,
                                 n0, n1, n2, n3)
    
    def C_subtract5D(self, map1, nhit1, rms1,
                           map2, nhit2, rms2):
        """
        Function taking inn 5D datasets from two infiles, and subracts the second form the first.
        Result is saved to class attributes self.map, self.nhit and self.rms.

        Parameters:
        ------------------------------
        map1: numpy.ndarray
            Map dataset of first input file. Must be 5D array.
        nhit1: numpy.ndarray
            Number of hits dataset of first input file. Must be 5D array
        rms1: numpy.ndarray
            Rms dataset of first input file. Must be 5D array
        map2: numpy.ndarray
            Map dataset of second input file. Must be 5D array.
        nhit2: numpy.ndarray
            Number of hits dataset of second input file. Must be 5D array
        rms2: numpy.ndarray
            Rms dataset of second input file. Must be 5D array

        ------------------------------
        """
        float32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=5, flags="contiguous")   # 5D array 32-bit float pointer object.
        int32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=5, flags="contiguous")       # 5D array 32-bit integer pointer object.

        self.maputilslib.subtract5D.argtypes = [float32_array5, int32_array5, float32_array5,       # Specifying input types for C library function.
                                                float32_array5, int32_array5, float32_array5,
                                                float32_array5, int32_array5, float32_array5,
                                                ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int,   ctypes.c_int]
        n0, n1, n2, n3, n4 = self.map1.shape                                    # Extracting axis lengths of 5D array to loop over in C.
        self.map = np.zeros_like(map1, dtype = ctypes.c_float)                  # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros_like(nhit1, dtype = ctypes.c_int)  
        self.rms = np.zeros_like(rms1, dtype = ctypes.c_float)
        
        self.maputilslib.subtract5D(map1, nhit1, rms1,              # Filling self.map, self.nhit and self.rms by
                                 map2, nhit2, rms2,                 # call-by-pointer to C library.
                                 self.map, self.nhit, self.rms,
                                 n0, n1, n2, n3, n4)
                            
    def C_subtract6D(self, map1, nhit1, rms1,
                           map2, nhit2, rms2):
        """
        Function taking inn 6D datasets from two infiles, and subracts the second form the first.
        Result is saved to class attributes self.map, self.nhit and self.rms.

        Parameters:
        ------------------------------
        map1: numpy.ndarray
            Map dataset of first input file. Must be 6D array.
        nhit1: numpy.ndarray
            Number of hits dataset of first input file. Must be 6D array
        rms1: numpy.ndarray
            Rms dataset of first input file. Must be 6D array
        map2: numpy.ndarray
            Map dataset of second input file. Must be 6D array.
        nhit2: numpy.ndarray
            Number of hits dataset of second input file. Must be 6D array
        rms2: numpy.ndarray
            Rms dataset of second input file. Must be 6D array

        ------------------------------
        """
        float32_array6 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=6, flags="contiguous")   # 6D array 32-bit float pointer object.
        int32_array6 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=6, flags="contiguous")       # 6D array 32-bit integer pointer object.

        self.maputilslib.subtract6D.argtypes = [float32_array6, int32_array6, float32_array6,       # Specifying input types for C library function.
                                                float32_array6, int32_array6, float32_array6,
                                                float32_array6, int32_array6, float32_array6,
                                                ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int,   ctypes.c_int, ctypes.c_int]
        n0, n1, n2, n3, n4, n5 = self.map1.shape                    # Extracting axis lengths of 6D array to loop over in C.
        self.map = np.zeros_like(map1, dtype = ctypes.c_float)      # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros_like(nhit1, dtype = ctypes.c_int)
        self.rms = np.zeros_like(rms1, dtype = ctypes.c_float)

        self.maputilslib.subtract6D(map1,     nhit1,     rms1,      # Filling self.map, self.nhit and self.rms by
                                 map2,     nhit2,     rms2,         # call-by-pointer to C library.
                                 self.map, self.nhit, self.rms,
                                 n0,       n1,        n2, 
                                 n3,       n4,        n5)

    def C_add4D(self, map1, nhit1, rms1,
                      map2, nhit2, rms2):
        """
        Function taking inn 4D datasets from two infiles, and directely add them.
        Result is saved to class attributes self.map, self.nhit and self.rms.

        Parameters:
        ------------------------------
        map1: numpy.ndarray
            Map dataset of first input file. Must be 4D array.
        nhit1: numpy.ndarray
            Number of hits dataset of first input file. Must be 4D array
        rms1: numpy.ndarray
            Rms dataset of first input file. Must be 4D array
        map2: numpy.ndarray
            Map dataset of second input file. Must be 4D array.
        nhit2: numpy.ndarray
            Number of hits dataset of second input file. Must be 4D array
        rms2: numpy.ndarray
            Rms dataset of second input file. Must be 4D array

        ------------------------------
        """
        float32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=4, flags="contiguous")   # 4D array 32-bit float pointer object.
        int32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=4, flags="contiguous")       # 4D array 32-bit integer pointer object.

        self.maputilslib.add4D.argtypes = [float32_array4, int32_array4, float32_array4,       # Specifying input types for C library function.
                                           float32_array4, int32_array4, float32_array4,
                                           float32_array4, int32_array4, float32_array4,
                                           ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int]
        n0, n1, n2, n3  = self.map1.shape                                   # Extracting axis lengths of 4D array to loop over in C.
        self.map        = np.zeros_like(map1,   dtype = ctypes.c_float)     # Generating arrays to fill up with coadded data.
        self.nhit       = np.zeros_like(nhit1,  dtype = ctypes.c_int)
        self.rms        = np.zeros_like(rms1,   dtype = ctypes.c_float)

        self.maputilslib.add4D(map1, nhit1, rms1,              # Filling self.map, self.nhit and self.rms by
                               map2, nhit2, rms2,                 # call-by-pointer to C library.
                               self.map, self.nhit, self.rms,
                               n0, n1, n2, n3)
    
    def C_add5D(self, map1, nhit1, rms1,
                      map2, nhit2, rms2):
        """
        Function taking inn 5D datasets from two infiles, and directely add them.
        Result is saved to class attributes self.map, self.nhit and self.rms.

        Parameters:
        ------------------------------
        map1: numpy.ndarray
            Map dataset of first input file. Must be 5D array.
        nhit1: numpy.ndarray
            Number of hits dataset of first input file. Must be 5D array
        rms1: numpy.ndarray
            Rms dataset of first input file. Must be 5D array
        map2: numpy.ndarray
            Map dataset of second input file. Must be 5D array.
        nhit2: numpy.ndarray
            Number of hits dataset of second input file. Must be 5D array
        rms2: numpy.ndarray
            Rms dataset of second input file. Must be 5D array

        ------------------------------
        """
        float32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=5, flags="contiguous")   # 5D array 32-bit float pointer object.
        int32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=5, flags="contiguous")       # 5D array 32-bit integer pointer object.

        self.maputilslib.add5D.argtypes = [float32_array5, int32_array5, float32_array5,       # Specifying input types for C library function.
                                           float32_array5, int32_array5, float32_array5,
                                           float32_array5, int32_array5, float32_array5,
                                           ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int,   ctypes.c_int]
        n0, n1, n2, n3, n4 = self.map1.shape                                    # Extracting axis lengths of 5D array to loop over in C.
        self.map = np.zeros_like(map1, dtype = ctypes.c_float)                  # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros_like(nhit1, dtype = ctypes.c_int)  
        self.rms = np.zeros_like(rms1, dtype = ctypes.c_float)
        
        self.maputilslib.add5D(map1, nhit1, rms1,              # Filling self.map, self.nhit and self.rms by
                               map2, nhit2, rms2,                 # call-by-pointer to C library.
                               self.map, self.nhit, self.rms,
                               n0, n1, n2, n3, n4)
                            
    def C_add6D(self, map1, nhit1, rms1,
                      map2, nhit2, rms2):
        """
        Function taking inn 6D datasets from two infiles, and directely add them.
        Result is saved to class attributes self.map, self.nhit and self.rms.

        Parameters:
        ------------------------------
        map1: numpy.ndarray
            Map dataset of first input file. Must be 6D array.
        nhit1: numpy.ndarray
            Number of hits dataset of first input file. Must be 6D array
        rms1: numpy.ndarray
            Rms dataset of first input file. Must be 6D array
        map2: numpy.ndarray
            Map dataset of second input file. Must be 6D array.
        nhit2: numpy.ndarray
            Number of hits dataset of second input file. Must be 6D array
        rms2: numpy.ndarray
            Rms dataset of second input file. Must be 6D array

        ------------------------------
        """
        float32_array6 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=6, flags="contiguous")   # 6D array 32-bit float pointer object.
        int32_array6 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=6, flags="contiguous")       # 6D array 32-bit integer pointer object.

        self.maputilslib.add6D.argtypes = [float32_array6, int32_array6, float32_array6,       # Specifying input types for C library function.
                                           float32_array6, int32_array6, float32_array6,
                                           float32_array6, int32_array6, float32_array6,
                                           ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int,   ctypes.c_int, ctypes.c_int]
        n0, n1, n2, n3, n4, n5 = self.map1.shape                    # Extracting axis lengths of 6D array to loop over in C.
        self.map = np.zeros_like(map1, dtype = ctypes.c_float)      # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros_like(nhit1, dtype = ctypes.c_int)
        self.rms = np.zeros_like(rms1, dtype = ctypes.c_float)

        self.maputilslib.add6D(map1,     nhit1,     rms1,      # Filling self.map, self.nhit and self.rms by
                               map2,     nhit2,     rms2,         # call-by-pointer to C library.
                               self.map, self.nhit, self.rms,
                               n0,       n1,        n2, 
                               n3,       n4,        n5)

    def C_dgradeXY4D(self, map_h, nhit_h, rms_h):
        """
        Function taking inn 4D datasets from one infile, and performes a co-merging of a given
        number of neighboring pixels. Result is saved to class attributes self.map, self.nhit and self.rms.

        Parameters:
        ------------------------------
        map_h: numpy.ndarray
            Map dataset of first input file. Must be 4D array.
        nhit_h: numpy.ndarray
            Number of hits dataset of first input file. Must be 4D array
        rms_h: numpy.ndarray
            Rms dataset of first input file. Must be 4D array
        ------------------------------
        """
        float32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=4, flags="contiguous")   # 4D array 32-bit float pointer object.
        int32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=4, flags="contiguous")       # 4D array 32-bit integer pointer object.

        self.maputilslib.dgradeXY4D.argtypes = [float32_array4, int32_array4, float32_array4,       # Specifying input types for C library function.
                                                float32_array4, int32_array4, float32_array4,
                                                ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int]
        n0, n1, n2, n3 = map_h.shape                                        # Extracting axis lengths of 4D array to loop over in C.
        N2, N3 = int(n2 / self.merge_numXY), int(n3 / self.merge_numXY)     # Size of the new pixel images
        
        self.map = np.zeros( (n0, n1, N2, N3), dtype = ctypes.c_float)      # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros((n0, n1, N2, N3), dtype = ctypes.c_int)
        self.rms = np.zeros( (n0, n1, N2, N3), dtype = ctypes.c_float)

        self.maputilslib.dgradeXY4D(map_h,    nhit_h,     rms_h,        # Filling self.map, self.nhit and self.rms by
                                self.map,   self.nhit,  self.rms,       # call-by-pointer to C library.
                                n0,         n1,         n2,
                                n3,         N2,         N3,
                                self.merge_numXY)
        
    def C_dgradeXY5D(self, map_h, nhit_h, rms_h):
        """
        Function taking inn 5D datasets from one infile, and performes a co-merging of a given
        number of neighboring pixels. Result is saved to class attributes self.map, self.nhit and self.rms.

        Parameters:
        ------------------------------
        map_h: numpy.ndarray
            Map dataset of first input file. Must be 5D array.
        nhit_h: numpy.ndarray
            Number of hits dataset of first input file. Must be 5D array
        rms_h: numpy.ndarray
            Rms dataset of first input file. Must be 5D array
        ------------------------------
        """
        float32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=5, flags="contiguous")   # 5D array 32-bit float pointer object
        int32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=5, flags="contiguous")       # 5D array 32-bit integer pointer object.

        self.maputilslib.dgradeXY5D.argtypes = [float32_array5, int32_array5, float32_array5,       # Specifying input types for C library function.
                                              float32_array5, int32_array5, float32_array5,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int]
        n0, n1, n2, n3, n4 = map_h.shape                                    # Extracting axis lengths of 5D array to loop over in C.
        N3, N4 = int(n3 / self.merge_numXY), int(n4 / self.merge_numXY)     # Size of the new pixel images
        
        self.map = np.zeros( (n0, n1, n2, N3, N4), dtype = ctypes.c_float)  # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros((n0, n1, n2, N3, N4), dtype = ctypes.c_int)
        self.rms = np.zeros( (n0, n1, n2, N3, N4), dtype = ctypes.c_float)

        self.maputilslib.dgradeXY5D(map_h,    nhit_h,     rms_h,            # Filling self.map, self.nhit and self.rms by
                                  self.map, self.nhit,  self.rms,           # call-by-pointer to C library.
                                  n0,       n1,         n2,
                                  n3,       n4,         N3,
                                  N4,       self.merge_numXY)

    def C_dgradeXY6D(self, map_h, nhit_h, rms_h):
        """
        Function taking inn 6D datasets from one infile, and performes a co-merging of a given
        number of neighboring pixels. Result is saved to class attributes self.map, self.nhit and self.rms.

        Parameters:
        ------------------------------
        map_h: numpy.ndarray
            Map dataset of first input file. Must be 6D array.
        nhit_h: numpy.ndarray
            Number of hits dataset of first input file. Must be 6D array
        rms_h: numpy.ndarray
            Rms dataset of first input file. Must be 6D array
        ------------------------------
        """
        float32_array6 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=6, flags="contiguous")   # 6D array 32-bit float pointer object
        int32_array6 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=6, flags="contiguous")       # 6D array 32-bit integer pointer object.

        self.maputilslib.dgradeXY6D.argtypes = [float32_array6, int32_array6, float32_array6,       # Specifying input types for C library function.
                                              float32_array6, int32_array6, float32_array6,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int]
        n0, n1, n2, n3, n4, n5 = map_h.shape                                                # Extracting axis lengths of 6D array to loop over in C.
        N4, N5 = int(n4 / self.merge_numXY), int(n5 / self.merge_numXY)                     # Size of the new pixel images
        
        self.map = np.zeros( (n0, n1, n2, n3, N4, N5), dtype = ctypes.c_float)      # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros((n0, n1, n2, n3, N4, N5), dtype = ctypes.c_int)
        self.rms = np.zeros( (n0, n1, n2, n3, N4, N5), dtype = ctypes.c_float)

        self.maputilslib.dgradeXY6D(map_h,        nhit_h,         rms_h,            # Filling self.map, self.nhit and self.rms by
                                  self.map,     self.nhit,      self.rms,           # call-by-pointer to C library.
                                  n0,           n1,             n2,
                                  n3,           n4,             n5, 
                                  N4,           N5,             self.merge_numXY)

    def C_dgradeZ4D(self, map_h, nhit_h, rms_h):
        """
        Function taking inn 4D datasets from one infile, and performes a co-merging of a given
        number of neighboring frequency channels. Result is saved to class attributes self.map, 
        self.nhit and self.rms.

        Parameters:
        ------------------------------
        map_h: numpy.ndarray
            Map dataset of first input file. Must be 4D array.
        nhit_h: numpy.ndarray
            Number of hits dataset of first input file. Must be 4D array
        rms_h: numpy.ndarray
            Rms dataset of first input file. Must be 4D array
        ------------------------------
        """
        float32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=4, flags="contiguous")   # 4D array 32-bit float pointer object
        int32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=4, flags="contiguous")       # 4D array 32-bit integer pointer object.

        self.maputilslib.dgradeZ4D.argtypes = [float32_array4, int32_array4, float32_array4,        # Specifying input types for C library function.
                                                float32_array4, int32_array4, float32_array4,
                                                ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int,   ctypes.c_int, ctypes.c_int]
        n0, n1, n2, n3  = map_h.shape                   # Extracting axis lengths of 4D array to loop over in C.
        N1              = int(n1 / self.merge_numZ)     # Size of the new pixel channel axis
        
        self.map = np.zeros( (n0, N1, n2, n3), dtype = ctypes.c_float)  # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros((n0, N1, n2, n3), dtype = ctypes.c_int)
        self.rms = np.zeros( (n0, N1, n2, n3), dtype = ctypes.c_float)

        self.maputilslib.dgradeZ4D(map_h,    nhit_h,     rms_h,                 # Filling self.map, self.nhit and self.rms by
                                    self.map,   self.nhit,  self.rms,           # call-by-pointer to C library.
                                    n0,         n1,         n2,
                                    n3,         N1,         self.merge_numZ)

    def C_dgradeZ5D(self, map_h, nhit_h, rms_h):
        """
        Function taking inn 5D datasets from one infile, and performes a co-merging of a given
        number of neighboring frequency channels. Result is saved to class attributes self.map, 
        self.nhit and self.rms.

        Parameters:
        ------------------------------
        map_h: numpy.ndarray
            Map dataset of first input file. Must be 5D array.
        nhit_h: numpy.ndarray
            Number of hits dataset of first input file. Must be 5D array
        rms_h: numpy.ndarray
            Rms dataset of first input file. Must be 5D array
        ------------------------------
        """
        float32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=5, flags="contiguous")   # 5D array 32-bit float pointer object
        int32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=5, flags="contiguous")       # 5D array 32-bit integer pointer object.

        self.maputilslib.dgradeZ5D.argtypes = [float32_array5, int32_array5, float32_array5,        # Specifying input types for C library function.
                                              float32_array5, int32_array5, float32_array5,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int]
        n0, n1, n2, n3, n4 = map_h.shape        # Extracting axis lengths of 5D array to loop over in C.
        N2 = int(n2 / self.merge_numZ)          # Size of the new pixel channel axis
        
        self.map = np.zeros( (n0, n1, N2, n3, n4), dtype = ctypes.c_float)  # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros((n0, n1, N2, n3, n4), dtype = ctypes.c_int)
        self.rms = np.zeros( (n0, n1, N2, n3, n4), dtype = ctypes.c_float)

        self.maputilslib.dgradeZ5D(map_h,    nhit_h,     rms_h,     # Filling self.map, self.nhit and self.rms by
                                  self.map, self.nhit,  self.rms,   # call-by-pointer to C library.
                                  n0,       n1,         n2,
                                  n3,       n4,         N2,
                                  self.merge_numZ)

    def C_dgradeZ6D(self, map_h, nhit_h, rms_h):
        """
        Function taking inn 6D datasets from one infile, and performes a co-merging of a given
        number of neighboring frequency channels. Result is saved to class attributes self.map, 
        self.nhit and self.rms.

        Parameters:
        ------------------------------
        map_h: numpy.ndarray
            Map dataset of first input file. Must be 6D array.
        nhit_h: numpy.ndarray
            Number of hits dataset of first input file. Must be 6D array
        rms_h: numpy.ndarray
            Rms dataset of first input file. Must be 6D array
        ------------------------------
        """
        float32_array6 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=6, flags="contiguous")   # 6D array 32-bit float pointer object
        int32_array6 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=6, flags="contiguous")       # 6D array 32-bit integer pointer object.

        self.maputilslib.dgradeZ6D.argtypes = [float32_array6, int32_array6, float32_array6,        # Specifying input types for C library function.
                                              float32_array6, int32_array6, float32_array6,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int]
        n0, n1, n2, n3, n4, n5 = map_h.shape        # Extracting axis lengths of 6D array to loop over in C.
        N3 = int(n3 / self.merge_numZ)              # Size of the new pixel channel axis
        
        self.map = np.zeros( (n0, n1, n2, N3, n4, n5), dtype = ctypes.c_float)  # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros((n0, n1, n2, N3, n4, n5), dtype = ctypes.c_int)
        self.rms = np.zeros( (n0, n1, n2, N3, n4, n5), dtype = ctypes.c_float)

        self.maputilslib.dgradeZ6D(map_h,        nhit_h,         rms_h,     # Filling self.map, self.nhit and self.rms by
                                  self.map,     self.nhit,      self.rms,   # call-by-pointer to C library.
                                  n0,           n1,             n2,
                                  n3,           n4,             n5, 
                                  N3,           self.merge_numZ)
    
    def C_dgradeXYZ4D(self, map_h, nhit_h, rms_h):
        """
        Function taking inn 4D datasets from one infile, and performes a co-merging of a given
        number of neighboring pixels and frequency channels. Result is saved to class attributes self.map, 
        self.nhit and self.rms.

        Parameters:
        ------------------------------
        map_h: numpy.ndarray
            Map dataset of first input file. Must be 4D array.
        nhit_h: numpy.ndarray
            Number of hits dataset of first input file. Must be 4D array
        rms_h: numpy.ndarray
            Rms dataset of first input file. Must be 4D array
        ------------------------------
        """
        float32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=4, flags="contiguous")   # 4D array 32-bit float pointer object.
        int32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=4, flags="contiguous")       # 4D array 32-bit integer pointer object.

        self.maputilslib.dgradeXYZ4D.argtypes = [float32_array4, int32_array4, float32_array4,  # Specifying input types for C library function.
                                                float32_array4, int32_array4, float32_array4,
                                                ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int,   ctypes.c_int, ctypes.c_int]
        n0, n1, n2, n3 = map_h.shape        # Extracting axis lengths of 4D array to loop over in C.

        N1, N2, N3 = int(n1 / self.merge_numZ), int(n2 / self.merge_numXY), int(n3 / self.merge_numXY)  # Size of the new pixel-freq cube
        
        self.map = np.zeros( (n0, N1, N2, N3), dtype = ctypes.c_float)  # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros((n0, N1, N2, N3), dtype = ctypes.c_int)
        self.rms = np.zeros( (n0, N1, N2, N3), dtype = ctypes.c_float)

        self.maputilslib.dgradeXYZ4D(map_h,    nhit_h,          rms_h,          # Filling self.map, self.nhit and self.rms by
                                    self.map, self.nhit,       self.rms,        # call-by-pointer to C library.
                                    n0,       n1,              n2,
                                    n3,       N1,              N2,
                                    N3,       self.merge_numZ,  self.merge_numXY)

    def C_dgradeXYZ5D(self, map_h, nhit_h, rms_h):
        """
        Function taking inn 5D datasets from one infile, and performes a co-merging of a given
        number of neighboring pixels and frequency channels. Result is saved to class attributes self.map, 
        self.nhit and self.rms.

        Parameters:
        ------------------------------
        map_h: numpy.ndarray
            Map dataset of first input file. Must be 5D array.
        nhit_h: numpy.ndarray
            Number of hits dataset of first input file. Must be 5D array
        rms_h: numpy.ndarray
            Rms dataset of first input file. Must be 5D array
        ------------------------------
        """
        float32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=5, flags="contiguous")   # 5D array 32-bit float pointer object.
        int32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=5, flags="contiguous")       # 5D array 32-bit integer pointer object.

        self.maputilslib.dgradeXYZ5D.argtypes = [float32_array5, int32_array5, float32_array5,      # Specifying input types for C library function.
                                              float32_array5, int32_array5, float32_array5,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int]
        n0, n1, n2, n3, n4 = map_h.shape    # Extracting axis lengths of 5D array to loop over in C.

        N2, N3, N4 = int(n2 / self.merge_numZ), int(n3 / self.merge_numXY), int(n4 / self.merge_numXY)  # Size of the new pixel-freq cube
        
        self.map = np.zeros( (n0, n1, N2, N3, N4), dtype = ctypes.c_float)  # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros((n0, n1, N2, N3, N4), dtype = ctypes.c_int)
        self.rms = np.zeros( (n0, n1, N2, N3, N4), dtype = ctypes.c_float)

        self.maputilslib.dgradeXYZ5D(map_h,    nhit_h,     rms_h,       # Filling self.map, self.nhit and self.rms by
                                    self.map, self.nhit,  self.rms,     # call-by-pointer to C library.
                                    n0,       n1,         n2,
                                    n3,       n4,         N2,
                                    N3,       N4,         self.merge_numZ,
                                    self.merge_numXY)

    def C_dgradeXYZ6D(self, map_h, nhit_h, rms_h):
        """
        Function taking inn 6D datasets from one infile, and performes a co-merging of a given
        number of neighboring pixels and frequency channels. Result is saved to class attributes self.map, 
        self.nhit and self.rms.

        Parameters:
        ------------------------------
        map_h: numpy.ndarray
            Map dataset of first input file. Must be 6D array.
        nhit_h: numpy.ndarray
            Number of hits dataset of first input file. Must be 6D array
        rms_h: numpy.ndarray
            Rms dataset of first input file. Must be 6D array
        ------------------------------
        """
        float32_array6 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=6, flags="contiguous")   # 6D array 32-bit float pointer object.
        int32_array6 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=6, flags="contiguous")       # 6D array 32-bit integer pointer object.

        self.maputilslib.dgradeXYZ6D.argtypes = [float32_array6, int32_array6, float32_array6,      # Specifying input types for C library function.
                                              float32_array6, int32_array6, float32_array6,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int]
        n0, n1, n2, n3, n4, n5 = map_h.shape        # Extracting axis lengths of 6D array to loop over in C.

        N3, N4, N5 = int(n3 / self.merge_numZ), int(n4 / self.merge_numXY), int(n5 / self.merge_numXY)  # Size of the new pixel-freq cube
        
        self.map = np.zeros( (n0, n1, n2, N3, N4, N5), dtype = ctypes.c_float)  # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros((n0, n1, n2, N3, N4, N5), dtype = ctypes.c_int)
        self.rms = np.zeros( (n0, n1, n2, N3, N4, N5), dtype = ctypes.c_float)

        self.maputilslib.dgradeXYZ6D(map_h,         nhit_h,         rms_h,      # Filling self.map, self.nhit and self.rms by
                                  self.map,         self.nhit,      self.rms,   # call-by-pointer to C library.
                                  n0,               n1,             n2,
                                  n3,               n4,             n5, 
                                  N3,               N4,             N5,
                                  self.merge_numZ,  self.merge_numXY)



    def C_ugradeXY4D(self, map_l, nhit_l, rms_l):
        """
        Function taking inn 4D datasets from one infile, and performes a transformation of a low-resolution
        pixel grid to a high-resolution one. Result is saved to class attributes self.map, 
        self.nhit and self.rms.

        Parameters:
        ------------------------------
        map_l: numpy.ndarray
            Map dataset of first input file. Must be 4D array.
        nhit_l: numpy.ndarray
            Number of hits dataset of first input file. Must be 4D array
        rms_l: numpy.ndarray
            Rms dataset of first input file. Must be 4D array
        ------------------------------
        """
        float32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=4, flags="contiguous")   # 4D array 32-bit float pointer object.
        int32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=4, flags="contiguous")       # 4D array 32-bit integer pointer object.

        self.maputilslib.ugradeXY4D.argtypes = [float32_array4, int32_array4, float32_array4,       # Specifying input types for C library function.
                                                float32_array4, int32_array4, float32_array4,
                                                ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int]
        n0, n1, n2, n3 = map_l.shape                            # Extracting axis lengths of 4D array to loop over in C.
        N2, N3 = n2 * self.merge_numXY, n3 * self.merge_numXY   # Size of the new pixel image
        
        self.map = np.zeros( (n0, n1, N2, N3), dtype = ctypes.c_float)  # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros((n0, n1, N2, N3), dtype = ctypes.c_int)
        self.rms = np.zeros( (n0, n1, N2, N3), dtype = ctypes.c_float)

        self.maputilslib.ugradeXY4D(self.map,   self.nhit,  self.rms,   # Filling self.map, self.nhit and self.rms by
                                    map_l,    nhit_l,     rms_l,        # call-by-pointer to C library.
                                    n0,         n1,         n2,
                                    n3,         N2,         N3,
                                    self.merge_numXY)

    def C_ugradeXY5D(self, map_l, nhit_l, rms_l):
        """
        Function taking inn 5D datasets from one infile, and performes a transformation of a low-resolution
        pixel grid to a high-resolution one. Result is saved to class attributes self.map, 
        self.nhit and self.rms.

        Parameters:
        ------------------------------
        map_l: numpy.ndarray
            Map dataset of first input file. Must be 5D array.
        nhit_l: numpy.ndarray
            Number of hits dataset of first input file. Must be 5D array
        rms_l: numpy.ndarray
            Rms dataset of first input file. Must be 5D array
        ------------------------------
        """
        float32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=5, flags="contiguous")   # 5D array 32-bit float pointer object.
        int32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=5, flags="contiguous")       # 5D array 32-bit integer pointer object.

        self.maputilslib.ugradeXY5D.argtypes = [float32_array5, int32_array5, float32_array5,       # Specifying input types for C library function.
                                              float32_array5, int32_array5, float32_array5,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int]
        n0, n1, n2, n3, n4 = map_l.shape                        # Extracting axis lengths of 5D array to loop over in C.
        N3, N4 = n3 * self.merge_numXY, n4 * self.merge_numXY   # Size of the new pixel image
        
        self.map = np.zeros( (n0, n1, n2, N3, N4), dtype = ctypes.c_float)  # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros((n0, n1, n2, N3, N4), dtype = ctypes.c_int)
        self.rms = np.zeros( (n0, n1, n2, N3, N4), dtype = ctypes.c_float)

        self.maputilslib.ugradeXY5D(self.map, self.nhit,  self.rms,         # Filling self.map, self.nhit and self.rms by
                                    map_l,    nhit_l,     rms_l,            # call-by-pointer to C library.
                                    n0,       n1,         n2,
                                    n3,       n4,         N3,
                                    N4,       self.merge_numXY)

    def C_ugradeXY6D(self, map_l, nhit_l, rms_l):
        """
        Function taking inn 6D datasets from one infile, and performes a transformation of a low-resolution
        pixel grid to a high-resolution one. Result is saved to class attributes self.map, 
        self.nhit and self.rms.

        Parameters:
        ------------------------------
        map_l: numpy.ndarray
            Map dataset of first input file. Must be 6D array.
        nhit_l: numpy.ndarray
            Number of hits dataset of first input file. Must be 6D array
        rms_l: numpy.ndarray
            Rms dataset of first input file. Must be 6D array
        ------------------------------
        """
        float32_array6 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=6, flags="contiguous")   # 6D array 32-bit float pointer object.
        int32_array6 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=6, flags="contiguous")       # 6D array 32-bit integer pointer object.

        self.maputilslib.ugradeXY6D.argtypes = [float32_array6, int32_array6, float32_array6,       # Specifying input types for C library function.
                                              float32_array6, int32_array6, float32_array6,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int]
        n0, n1, n2, n3, n4, n5 = map_l.shape                    # Extracting axis lengths of 6D array to loop over in C.
        N4, N5 = n4 * self.merge_numXY, n5 * self.merge_numXY   # Size of the new pixel image
        
        self.map = np.zeros( (n0, n1, n2, n3, N4, N5), dtype = ctypes.c_float)  # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros((n0, n1, n2, n3, N4, N5), dtype = ctypes.c_int)
        self.rms = np.zeros( (n0, n1, n2, n3, N4, N5), dtype = ctypes.c_float)

        self.maputilslib.ugradeXY6D(self.map,     self.nhit,      self.rms,         # Filling self.map, self.nhit and self.rms by
                                    map_l,        nhit_l,         rms_l,            # call-by-pointer to C library.
                                    n0,           n1,             n2,
                                    n3,           n4,             n5, 
                                    N4,           N5,             self.merge_numXY)

    def C_ugradeZ4D(self, map_l, nhit_l, rms_l):
        """
        Function taking inn 4D datasets from one infile, and performes a transformation of a low-resolution
        frequency grid to a high-resolution one. Result is saved to class attributes self.map, 
        self.nhit and self.rms.

        Parameters:
        ------------------------------
        map_l: numpy.ndarray
            Map dataset of first input file. Must be 4D array.
        nhit_l: numpy.ndarray
            Number of hits dataset of first input file. Must be 4D array
        rms_l: numpy.ndarray
            Rms dataset of first input file. Must be 4D array
        ------------------------------
        """
        float32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=4, flags="contiguous")   # 4D array 32-bit float pointer object.
        int32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=4, flags="contiguous")       # 4D array 32-bit integer pointer object.

        self.maputilslib.ugradeZ4D.argtypes = [float32_array4, int32_array4, float32_array4,        # Specifying input types for C library function.
                                                float32_array4, int32_array4, float32_array4,
                                                ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int,   ctypes.c_int, ctypes.c_int]
        n0, n1, n2, n3  = map_l.shape               # Extracting axis lengths of 4D array to loop over in C.
        N1              = n1 * self.merge_numZ      # Size of the new frequency axis
        
        self.map = np.zeros( (n0, N1, n2, n3), dtype = ctypes.c_float)      # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros((n0, N1, n2, n3), dtype = ctypes.c_int)
        self.rms = np.zeros( (n0, N1, n2, n3), dtype = ctypes.c_float)

        self.maputilslib.ugradeZ4D(self.map,   self.nhit,  self.rms,        # Filling self.map, self.nhit and self.rms by
                                   map_l,      nhit_l,     rms_l,           # call-by-pointer to C library.
                                   n0,         n1,         n2,
                                   n3,         N1,         self.merge_numZ)

    def C_ugradeZ5D(self, map_l, nhit_l, rms_l):
        """
        Function taking inn 5D datasets from one infile, and performes a transformation of a low-resolution
        frequency grid to a high-resolution one. Result is saved to class attributes self.map, 
        self.nhit and self.rms.

        Parameters:
        ------------------------------
        map_l: numpy.ndarray
            Map dataset of first input file. Must be 5D array.
        nhit_l: numpy.ndarray
            Number of hits dataset of first input file. Must be 5D array
        rms_l: numpy.ndarray
            Rms dataset of first input file. Must be 5D array
        ------------------------------
        """
        float32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=5, flags="contiguous")   # 5D array 32-bit float pointer object.
        int32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=5, flags="contiguous")       # 5D array 32-bit integer pointer object.

        self.maputilslib.ugradeZ5D.argtypes = [float32_array5, int32_array5, float32_array5,        # Specifying input types for C library function.
                                              float32_array5, int32_array5, float32_array5,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int]
        n0, n1, n2, n3, n4 = map_l.shape        # Extracting axis lengths of 5D array to loop over in C.
        N2 = n2 * self.merge_numZ               # Size of the new frequency axis
        
        self.map = np.zeros( (n0, n1, N2, n3, n4), dtype = ctypes.c_float)      # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros((n0, n1, N2, n3, n4), dtype = ctypes.c_int)
        self.rms = np.zeros( (n0, n1, N2, n3, n4), dtype = ctypes.c_float)

        self.maputilslib.ugradeZ5D(self.map, self.nhit,  self.rms,          # Filling self.map, self.nhit and self.rms by
                                   map_l,    nhit_l,     rms_l,             # call-by-pointer to C library.
                                   n0,       n1,         n2,
                                   n3,       n4,         N2,
                                   self.merge_numZ)

    def C_ugradeZ6D(self, map_l, nhit_l, rms_l):
        """
        Function taking inn 6D datasets from one infile, and performes a transformation of a low-resolution
        frequency grid to a high-resolution one. Result is saved to class attributes self.map, 
        self.nhit and self.rms.

        Parameters:
        ------------------------------
        map_l: numpy.ndarray
            Map dataset of first input file. Must be 6D array.
        nhit_l: numpy.ndarray
            Number of hits dataset of first input file. Must be 6D array
        rms_l: numpy.ndarray
            Rms dataset of first input file. Must be 6D array
        ------------------------------
        """
        float32_array6 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=6, flags="contiguous")   # 6D array 32-bit float pointer object.
        int32_array6 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=6, flags="contiguous")       # 6D array 32-bit integer pointer object.

        self.maputilslib.ugradeZ6D.argtypes = [float32_array6, int32_array6, float32_array6,        # Specifying input types for C library function.
                                              float32_array6, int32_array6, float32_array6,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int]
        n0, n1, n2, n3, n4, n5 = map_l.shape        # Extracting axis lengths of 6D array to loop over in C.
        N3 = n3 * self.merge_numZ                   # Size of the new frequency axis
        
        self.map = np.zeros( (n0, n1, n2, N3, n4, n5), dtype = ctypes.c_float)  # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros((n0, n1, n2, N3, n4, n5), dtype = ctypes.c_int)
        self.rms = np.zeros( (n0, n1, n2, N3, n4, n5), dtype = ctypes.c_float)

        self.maputilslib.ugradeZ6D(self.map,     self.nhit,      self.rms,      # Filling self.map, self.nhit and self.rms by
                                   map_l,        nhit_l,         rms_l,         # call-by-pointer to C library.
                                   n0,           n1,             n2,
                                   n3,           n4,             n5, 
                                   N3,           self.merge_numZ)
    
    def C_ugradeXYZ4D(self, map_l, nhit_l, rms_l):
        """
        Function taking inn 4D datasets from one infile, and performes a transformation of a low-resolution
        pixel-frequency grid to a high-resolution one. Result is saved to class attributes self.map, 
        self.nhit and self.rms.

        Parameters:
        ------------------------------
        map_l: numpy.ndarray
            Map dataset of first input file. Must be 4D array.
        nhit_l: numpy.ndarray
            Number of hits dataset of first input file. Must be 4D array
        rms_l: numpy.ndarray
            Rms dataset of first input file. Must be 4D array
        ------------------------------
        """
        float32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=4, flags="contiguous")   # 4D array 32-bit float pointer object.
        int32_array4 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=4, flags="contiguous")       # 4D array 32-bit integer pointer object.

        self.maputilslib.ugradeXYZ4D.argtypes = [float32_array4, int32_array4, float32_array4,      # Specifying input types for C library function.
                                                float32_array4, int32_array4, float32_array4,
                                                ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                                ctypes.c_int,   ctypes.c_int, ctypes.c_int]
        n0, n1, n2, n3 = map_l.shape            # Extracting axis lengths of 4D array to loop over in C.

        N1, N2, N3 = n1 * self.merge_numZ, n2 * self.merge_numXY, n3 * self.merge_numXY     # Size of the new pixel-freq cube
        
        self.map = np.zeros( (n0, N1, N2, N3), dtype = ctypes.c_float)      # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros((n0, N1, N2, N3), dtype = ctypes.c_int)
        self.rms = np.zeros( (n0, N1, N2, N3), dtype = ctypes.c_float)

        self.maputilslib.ugradeXYZ4D(self.map, self.nhit,       self.rms,           # Filling self.map, self.nhit and self.rms by
                                     map_l,    nhit_l,          rms_l,              # call-by-pointer to C library.
                                     n0,       n1,              n2,
                                     n3,       N1,              N2,
                                     N3,       self.merge_numZ,  self.merge_numXY)

    def C_ugradeXYZ5D(self, map_l, nhit_l, rms_l):
        """
        Function taking inn 5D datasets from one infile, and performes a transformation of a low-resolution
        pixel-frequency grid to a high-resolution one. Result is saved to class attributes self.map, 
        self.nhit and self.rms.

        Parameters:
        ------------------------------
        map_l: numpy.ndarray
            Map dataset of first input file. Must be 5D array.
        nhit_l: numpy.ndarray
            Number of hits dataset of first input file. Must be 5D array
        rms_l: numpy.ndarray
            Rms dataset of first input file. Must be 5D array
        ------------------------------
        """
        float32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=5, flags="contiguous")   # 5D array 32-bit float pointer object.
        int32_array5 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=5, flags="contiguous")       # 5D array 32-bit integer pointer object.

        self.maputilslib.ugradeXYZ5D.argtypes = [float32_array5, int32_array5, float32_array5,  # Specifying input types for C library function.
                                              float32_array5, int32_array5, float32_array5,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int]
        n0, n1, n2, n3, n4 = map_l.shape    # Extracting axis lengths of 5D array to loop over in C.

        N2, N3, N4 = n2 * self.merge_numZ, n3 * self.merge_numXY, n4 * self.merge_numXY # Size of the new pixel-freq cube
        
        self.map = np.zeros( (n0, n1, N2, N3, N4), dtype = ctypes.c_float)  # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros((n0, n1, N2, N3, N4), dtype = ctypes.c_int)
        self.rms = np.zeros( (n0, n1, N2, N3, N4), dtype = ctypes.c_float)

        self.maputilslib.ugradeXYZ5D(self.map, self.nhit,  self.rms,            # Filling self.map, self.nhit and self.rms by
                                     map_l,    nhit_l,     rms_l,               # call-by-pointer to C library.
                                     n0,       n1,         n2,
                                     n3,       n4,         N2,
                                     N3,       N4,         self.merge_numZ,
                                     self.merge_numXY)

    def C_ugradeXYZ6D(self, map_l, nhit_l, rms_l):
        """
        Function taking inn 6D datasets from one infile, and performes a transformation of a low-resolution
        pixel-frequency grid to a high-resolution one. Result is saved to class attributes self.map, 
        self.nhit and self.rms.

        Parameters:
        ------------------------------
        map_l: numpy.ndarray
            Map dataset of first input file. Must be 6D array.
        nhit_l: numpy.ndarray
            Number of hits dataset of first input file. Must be 6D array
        rms_l: numpy.ndarray
            Rms dataset of first input file. Must be 6D array
        ------------------------------
        """
        float32_array6 = np.ctypeslib.ndpointer(dtype=ctypes.c_float, ndim=6, flags="contiguous")   # 6D array 32-bit float pointer object.
        int32_array6 = np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=6, flags="contiguous")       # 6D array 32-bit integer pointer object.

        self.maputilslib.ugradeXYZ6D.argtypes = [float32_array6, int32_array6, float32_array6,      # Specifying input types for C library function.
                                              float32_array6, int32_array6, float32_array6,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int, ctypes.c_int,
                                              ctypes.c_int,   ctypes.c_int]
        n0, n1, n2, n3, n4, n5 = map_l.shape        # Extracting axis lengths of 6D array to loop over in C.

        N3, N4, N5 = n3 * self.merge_numZ, n4 * self.merge_numXY, n5 * self.merge_numXY # Size of the new pixel-freq cube
        
        self.map = np.zeros( (n0, n1, n2, N3, N4, N5), dtype = ctypes.c_float)  # Generating arrays to fill up with coadded data.
        self.nhit = np.zeros((n0, n1, n2, N3, N4, N5), dtype = ctypes.c_int)
        self.rms = np.zeros( (n0, n1, n2, N3, N4, N5), dtype = ctypes.c_float)

        self.maputilslib.ugradeXYZ6D(self.map,         self.nhit,      self.rms,    # Filling self.map, self.nhit and self.rms by
                                     map_l,            nhit_l,         rms_l,       # call-by-pointer to C library.
                                     n0,               n1,             n2,
                                     n3,               n4,             n5, 
                                     N3,               N4,             N5,
                                     self.merge_numZ,  self.merge_numXY)



    def gaussian_kernelXY(self):
        """
        Function returning a 2D Gaussian kernal for pixel smoothing.
        """
        size_x = int(self.n_sigma * self.sigmaX)                    # Grid boundaries for kernal
        size_y = int(self.n_sigma * self.sigmaY)
        x, y = np.mgrid[-size_x:size_x + 1, -size_y:size_y + 1]                     # Seting up the kernal's grid
        g = np.exp(-(x**2 / (2. * self.sigmaX**2) + y**2 / (2. * self.sigmaY**2)))  # Computing the Gaussian Kernal
        return g / g.sum()                                          
    
    def gaussian_kernelZ(self):
        """
        Function returning a 1D Gaussian kernal for frequency smoothing.
        """
        size_z = int(self.n_sigma * self.sigmaZ)        # Grid bounday for kernal
        z = np.arange(-size_z, size_z + 1)              # Seting up the kernal's grid
        g = np.exp(-(z**2 / (2. * self.sigmaZ ** 2)))   # Computing the Gaussian Kernal
        return g / g.sum()

    def gaussian_kernelXYZ(self):
        """
        Function returning a 3D Gaussian kernal for voxel smoothing.
        """
        size_x = int(self.n_sigma * self.sigmaX)        # Grid bounday for kernal
        size_y = int(self.n_sigma * self.sigmaY)
        size_z = int(self.n_sigma * self.sigmaZ)
        z, x, y = np.mgrid[ -size_z:size_z + 1, 
                            -size_x:size_x + 1, 
                            -size_y:size_y + 1]         # Seting up the kernal's grid
        g = np.exp(-(x**2 / (2. * self.sigmaX**2) 
                   + y**2 / (2. * self.sigmaY**2) 
                   + z**2 / (2. * self.sigmaZ**2)))     # Computing the Gaussian Kernal
        return g / g.sum()


    def gaussian_smoothXY(self, map, nhit, rms):
        """
        Function computing a map of smoothed pixels by convolving 
        the map with a 2D Gaussian Kernal.

        Parameters:
        ------------------------------
        map_l: numpy.ndarray
            Map dataset of first input file. Must be 4D, 5D or 6D array.
        nhit_l: numpy.ndarray
            Number of hits dataset of first input file. Must be 4D, 5D or 6D array.
        rms_l: numpy.ndarray
            Rms dataset of first input file. Must be 4D, 5D or 6D array.
        ------------------------------
        """

        kernel = self.gaussian_kernelXY()                   # Computing 2D Gaussian Kernal
        axes   = [len(map.shape)-2,  len(map.shape) - 1]    # Pixel axes of map matrix over which to compute the convolution
        
        """Computing convolution"""
        if len(map.shape) == 4:
            self.map    = signal.fftconvolve(map,   kernel[np.newaxis, np.newaxis, :, :], 
                                            mode='same', axes = axes)
            self.nhit   = signal.fftconvolve(nhit,  kernel[np.newaxis, np.newaxis, :, :], 
                                            mode='same', axes = axes)
            self.rms    = signal.fftconvolve(rms,   kernel[np.newaxis, np.newaxis, :, :], 
                                            mode='same', axes = axes)

        elif len(map.shape) == 5:
            self.map    = signal.fftconvolve(map,  kernel[np.newaxis, np.newaxis, np.newaxis, :, :], 
                                            mode='same', axes = axes)
            self.nhit   = signal.fftconvolve(nhit, kernel[np.newaxis, np.newaxis, np.newaxis, :, :], 
                                            mode='same', axes = axes)
            self.rms    = signal.fftconvolve(rms,  kernel[np.newaxis, np.newaxis, np.newaxis, :, :], 
                                            mode='same', axes = axes)

        elif len(map.shape) == 6:
            self.map    = signal.fftconvolve(map, kernel[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :], 
                                            mode='same', axes = axes)
            self.nhit   = signal.fftconvolve(nhit, kernel[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :], 
                                            mode='same', axes = axes)
            self.rms    = signal.fftconvolve(rms, kernel[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :], 
                                            mode='same', axes = axes)

    def gaussian_smoothZ(self, map, nhit, rms):
        """
        Function computing a map of smoothed frequencies by convolving 
        the map with a 1D Gaussian Kernal.

        Parameters:
        ------------------------------
        map_l: numpy.ndarray
            Map dataset of first input file. Must be 4D, 5D or 6D array.
        nhit_l: numpy.ndarray
            Number of hits dataset of first input file. Must be 4D, 5D or 6D array.
        rms_l: numpy.ndarray
            Rms dataset of first input file. Must be 4D, 5D or 6D array.
        ------------------------------
        """

        kernel = self.gaussian_kernelZ()            # Computing 2D Gaussian Kernal
        axes   = len(map.shape) - 4                 # Frequency axes of map matrix over which to compute the convolution

        """Computing convolution"""
        if len(map.shape) == 4:
            n0, n1, n2, n3 = map.shape
            
            map = map.reshape(n0 * n1, n2, n3)      # Reshaping [..., sb, channel, ...] into continous [..., frequency, ...] = [..., sb * channel, ...] 
            nhit = nhit.reshape(n0 * n1, n2, n3)    # for convolution in frequency direction.
            rms = rms.reshape(n0 * n1, n2, n3)
            
            self.map    = signal.fftconvolve(map,   kernel[:, np.newaxis, np.newaxis], 
                                            mode='same', axes = axes)
            self.nhit   = signal.fftconvolve(nhit,  kernel[:, np.newaxis, np.newaxis], 
                                            mode='same', axes = axes)
            self.rms    = signal.fftconvolve(rms,   kernel[:, np.newaxis, np.newaxis], 
                                            mode='same', axes = axes)

            self.map = self.map.reshape(n0, n1, n2, n3)         # Reshaping back to original [..., sb, channel, ...] format
            self.nhit = self.nhit.reshape(n0, n1, n2, n3)
            self.rms = self.rms.reshape(n0, n1, n2, n3)
            
        elif len(map.shape) == 5:
            n0, n1, n2, n3, n4 = map.shape
            map = map.reshape(n0, n1 * n2, n3, n4)      # Reshaping [..., sb, channel, ...] into continous [..., frequency, ...] = [..., sb * channel, ...]
            nhit = nhit.reshape(n0, n1 * n2, n3, n4)    # for convolution in frequency direction.
            rms = rms.reshape(n0, n1 * n2, n3, n4)
            
            self.map    = signal.fftconvolve(map,  kernel[np.newaxis, :, np.newaxis,  np.newaxis], 
                                            mode='same', axes = axes)
            self.nhit   = signal.fftconvolve(nhit, kernel[np.newaxis, :, np.newaxis, np.newaxis], 
                                            mode='same', axes = axes)
            self.rms    = signal.fftconvolve(rms,  kernel[np.newaxis, :, np.newaxis, np.newaxis], 
                                            mode='same', axes = axes)

            self.map = self.map.reshape(n0, n1, n2, n3, n4)     # Reshaping back to original [..., sb, channel, ...] format
            self.nhit = self.nhit.reshape(n0, n1, n2, n3, n4)
            self.rms = self.rms.reshape(n0, n1, n2, n3, n4)
            
        elif len(map.shape) == 6:
            n0, n1, n2, n3, n4, n5 = map.shape
            map = map.reshape(n0, n1, n2 * n3, n4, n5)      # Reshaping [..., sb, channel, ...] into continous [..., frequency, ...] = [..., sb * channel, ...]
            nhit = nhit.reshape(n0, n1, n2 * n3, n4, n5)    # for convolution in frequency direction.
            rms = rms.reshape(n0, n1, n2 * n3, n4, n5)

            self.map    = signal.fftconvolve(map, kernel[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis], 
                                            mode='same', axes = axes)
            self.nhit   = signal.fftconvolve(nhit, kernel[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis], 
                                            mode='same', axes = axes)
            self.rms    = signal.fftconvolve(rms, kernel[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis], 
                                            mode='same', axes = axes)

            self.map = self.map.reshape(n0, n1, n2, n3, n4, n5)     # Reshaping back to original [..., sb, channel, ...] format
            self.nhit = self.nhit.reshape(n0, n1, n2, n3, n4, n5)
            self.rms = self.rms.reshape(n0, n1, n2, n3, n4, n5)

    def gaussian_smoothXYZ(self, map, nhit, rms):
        """
        Function computing a map of smoothed voxels by convolving 
        the map with a 3D Gaussian Kernal.

        Parameters:
        ------------------------------
        map_l: numpy.ndarray
            Map dataset of first input file. Must be 4D, 5D or 6D array.
        nhit_l: numpy.ndarray
            Number of hits dataset of first input file. Must be 4D, 5D or 6D array.
        rms_l: numpy.ndarray
            Rms dataset of first input file. Must be 4D, 5D or 6D array.
        ------------------------------
        """
        kernel = self.gaussian_kernelXYZ()      # Computing 2D Gaussian Kernal
        axes = [len(map.shape) - 4, 
                len(map.shape) - 3, 
                len(map.shape) - 2]             # Voxel axes of map matrix over which to compute the convolution
        
        """Computing convolution"""
        if len(map.shape) == 4:
            n0, n1, n2, n3 = map.shape
            
            map = map.reshape(n0 * n1, n2, n3)      # Reshaping [..., sb, channel, ...] into continous [..., frequency, ...] = [..., sb * channel, ...]
            nhit = nhit.reshape(n0 * n1, n2, n3)    # for convolution in frequency direction.
            rms = rms.reshape(n0 * n1, n2, n3)

            self.map    = signal.fftconvolve(map,   kernel, mode='same', 
                                            axes = axes)
            self.nhit   = signal.fftconvolve(nhit,  kernel, mode='same', 
                                            axes = axes)
            self.rms    = signal.fftconvolve(rms,   kernel, mode='same', 
                                            axes = axes)

            self.map = self.map.reshape(n0, n1, n2, n3)     # Reshaping back to original [..., sb, channel, ...] format
            self.nhit = self.nhit.reshape(n0, n1, n2, n3)
            self.rms = self.rms.reshape(n0, n1, n2, n3)
            
        elif len(map.shape) == 5:
            n0, n1, n2, n3, n4 = map.shape
            map = map.reshape(n0, n1 * n2, n3, n4)      # Reshaping [..., sb, channel, ...] into continous [..., frequency, ...] = [..., sb * channel, ...]
            nhit = nhit.reshape(n0, n1 * n2, n3, n4)    # for convolution in frequency direction.
            rms = rms.reshape(n0, n1 * n2, n3, n4)
            
            self.map    = signal.fftconvolve(map,  kernel[np.newaxis, :, :, :], 
                                            mode='same', axes = axes) 
                                                                
            self.nhit   = signal.fftconvolve(nhit, kernel[np.newaxis, :, :, :], 
                                            mode='same', axes = axes)

            self.rms    = signal.fftconvolve(rms,  kernel[np.newaxis, :, :, :], 
                                            mode='same', axes = axes)

            self.map = self.map.reshape(n0, n1, n2, n3, n4)     # Reshaping back to original [..., sb, channel, ...] format
            self.nhit = self.nhit.reshape(n0, n1, n2, n3, n4)
            self.rms = self.rms.reshape(n0, n1, n2, n3, n4)
            
        elif len(map.shape) == 6:
            n0, n1, n2, n3, n4, n5 = map.shape
            map = map.reshape(n0, n1, n2 * n3, n4, n5)      # Reshaping [..., sb, channel, ...] into continous [..., frequency, ...] = [..., sb * channel, ...]
            nhit = nhit.reshape(n0, n1, n2 * n3, n4, n5)    # for convolution in frequency direction.
            rms = rms.reshape(n0, n1, n2 * n3, n4, n5)
            
            self.map    = signal.fftconvolve(map, kernel[np.newaxis, np.newaxis, :, :, :], 
                                            mode='same', axes = axes)

            self.nhit   = signal.fftconvolve(nhit, kernel[np.newaxis, np.newaxis, :, :, :], 
                                            mode='same', axes = axes)

            self.rms    = signal.fftconvolve(rms, kernel[np.newaxis, np.newaxis, :, :, :], 
                                            mode='same', axes = axes)

            self.map = self.map.reshape(n0, n1, n2, n3, n4, n5)     # Reshaping back to original [..., sb, channel, ...] format
            self.nhit = self.nhit.reshape(n0, n1, n2, n3, n4, n5)
            self.rms = self.rms.reshape(n0, n1, n2, n3, n4, n5)

if __name__ == "__main__":
    t = time.time()
    map = Atlas()
    print("Run time: ", time.time() - t, " sec")
    """
    dummy_map = np.zeros((1,2,2,2), dtype = np.float32)
    dummy_nhit = np.zeros((1,2,2,2), dtype = np.int32)
    dummy_rms = np.zeros((1,2,2,2), dtype  = np.float32)

    dummy_map[0, 0, :, :] = 0
    dummy_map[0, 1, 1, :] = 0
    dummy_map[0, 1, 0, 0] = 1
    dummy_map[0, 1, 0, 1] = 1

    dummy_nhit[0, 0, :, :] = 0
    dummy_nhit[0, 1, 1, :] = 0
    dummy_nhit[0, 1, 0, 0] = 2
    dummy_nhit[0, 1, 0, 1] = 1

    dummy_rms[0, 0, :, :] = 0
    dummy_rms[0, 1, 1, :] = 0
    dummy_rms[0, 1, 0, 0] = 1 / np.sqrt(2)
    dummy_rms[0, 1, 0, 1] = 1   
    
    map.C_dgradeXY4D(dummy_map, dummy_nhit, dummy_rms)
    print("map")
    print(dummy_map[0, 0, :, :])
    print(dummy_map[0, 1, :, :])
    print("nhit")

    print(dummy_nhit[0, 0, :, :])
    print(dummy_nhit[0, 1, :, :])
    print("rms")

    print(dummy_rms[0, 0, :, :])
    print(dummy_rms[0, 1, :, :])

    print("map")
    print(map.map[0, 0, :, :])
    print(map.map[0, 1, :, :])

    print("nhit")
    print(map.nhit[0, 0, :, :])
    print(map.nhit[0, 1, :, :])

    print("rms")
    print(map.rms[0, 0, :, :])
    print(map.rms[0, 1, :, :])
    """






    
