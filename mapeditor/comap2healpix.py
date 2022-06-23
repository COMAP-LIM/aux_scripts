import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import healpy as hp
import h5py as h5
import sys
import getopt
import textwrap
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy_healpix import HEALPix
from astropy.io import fits


def usage():
    """
    Function printing out usage of the program.
    """
    prefix = ""
    preferredWidth = 150
    wrapper = textwrap.TextWrapper(
        initial_indent=prefix, width=preferredWidth, subsequent_indent=" " * len(prefix)
    )
    m1 = "(Side band as a single number, ie. '-s 2'. Default is side band 2)"
    m2 = "(Frequency, given as single number, ie. '-f 30'. Default is frequency nr. 30)"
    m3 = "(Filename of first input map)"
    m4 = "(Filename of second input map)"
    m5 = "(Filename of third input map)"
    m6 = "(color_limits of colorbar) as nested list. First plot --> first list in main list. Default none)"
    m7 = "(Outfile name, default 'outfile')"
    m8 = "(Save HEALPix map as .png or .fits format. Default FITS)"

    print("\nThis is the usage function\n")
    print("Flags:")
    print("-i ----> optional --in1 " + wrapper.fill(m3))
    print("-I ----> optional --in2 " + wrapper.fill(m4))
    print("-J ----> optional --in3 " + wrapper.fill(m5))
    print("-o ----> optional --out " + wrapper.fill(m7))
    print("-s ----> optional --sb " + wrapper.fill(m1))
    print("-f ----> optional --freq " + wrapper.fill(m2))
    print("-c ----> optional --colorlim " + wrapper.fill(m5))
    print("-m ----> optional --mapfile " + wrapper.fill(m5))
    sys.exit()


sb = 2
freq = 30
color_lim = [None, None]
mapfiletype = "fits"

if len(sys.argv) == 1:
    usage()

try:
    opts, args = getopt.getopt(
        sys.argv[1:],
        "s:f:i:I:J:h:c:o:m:",
        [
            "sb=",
            "freq=",
            "in1=",
            "in2=",
            "in3=",
            "help=",
            "colorlim=",
            "out=",
            "mapfile=",
        ],
    )
except getopt.GetoptError:
    usage()

for opt, arg in opts:
    if opt in ("-o", "--out"):
        outfile = arg
    elif opt in ("-s", "--sb"):
        sb = int(eval(arg))
        if sb == 0:
            print("Use 1-base, not 0-base please")
            sys.exit()
        if type(sb) != int:
            print("Please provide only one side band at a time!")
    elif opt in ("-f", "--freq"):
        freq = int(eval(arg))
        if freq == 0:
            print("Use 1-base, not 0-base please")
            sys.exit()
        if type(sb) != int:
            print("Please provide only one frequency at a time!")
            sys.exit()
    elif opt in ("-i", "--in1"):
        infile1 = arg
        temp = infile1.split("/")[-1]
        patch1 = temp.split("_")[0]
    elif opt in ("-I", "--in2"):
        infile2 = arg
        temp = infile2.split("/")[-1]
        patch2 = temp.split("_")[0]
    elif opt in ("-J", "--in3"):
        infile3 = arg
        temp = infile3.split("/")[-1]
        patch3 = temp.split("_")[0]
    elif opt in ("-h", "--help"):
        usage()
    elif opt in ("-c", "--colorlim"):
        color_lim = eval(arg)
    elif opt in ("-m", "--mapfile"):
        mapfiletype = arg.strip()
    else:
        usage()


def readMap(filename):
    """
    Function reading in map file to return angular pixel coordinates 
    and number of hits.
    --------------------
    filename: str
        Filename of HDF-file (.h5 format) containing a map to read in.
    """

    dfile = h5.File(filename, "r")  # Reading in file
    x = dfile["x"]
    x = np.array(x[:]).astype(float)  # Right ascension of pixels
    y = dfile["y"]
    y = np.array(y[:]).astype(float)  # Declination of pixels
    hitname = "nhit"
    hits = dfile[hitname]
    hits = np.array(hits[...]).astype(float)  # Number of hits per pixel

    return x, y, hits


def map2healpix(mapname1, mapname2, mapname3, sb, freq):
    """
    Function loading three maps at a given sideband and frequency to project 
    their pixels into a healpix format to obtain a full-sky map with
    all three maps. Assuming that all three input maps have the same resolution,
    to compute the Nside parameter.
    --------------------
    mapname1: str
        Filename of first map file to read (.h5 format HDF-file)
    mapname2: str
        Filename of second map file to read (.h5 format HDF-file)
    mapname3: str
        Filename of third map file to read (.h5 format HDF-file)
    sb: int
        Number of sideband to use (base 1 not 0, i.e first and second sideband
        are sb = 1 and sb = 2 respectively)
    freq: int
        Number of the frequency channel to use (base 1 not 0, i.e. first and second 
        are freq = 1 and freq = 2 respectively)
    """

    x1, y1, hits1 = readMap(mapname1)  # Angular coord and # of hits per px of 1st map
    x2, y2, hits2 = readMap(mapname2)  # Angular coord and # of hits per px of 2nd map
    x3, y3, hits3 = readMap(mapname3)  # Angular coord and # of hits per px of 3rd map
    dx = np.array([x1[1] - x1[0], x2[1] - x2[0], x3[1] - x3[0]])
    dy = np.array([y1[1] - y1[0], y2[1] - y2[0], y3[1] - y3[0]])
    dx = np.max(
        dx
    )  # Right ascension resolution (the largest to ensure a nice HEALPix for all hitmaps)
    dy = np.max(
        dy
    )  # Declination resolution (the largest to ensure a nice HEALPix for all hitmaps)
    nx = len(x1)  # Number of pixels in horizontal direction
    ny = len(y1)  # Number of pixels in vertical direction
    Nside = int(
        round(np.sqrt(360 * 360 / (12 * np.pi * dx * dy)))
    )  # Nside parameter for healpix map (under the
    # assumption that each pixel in the input maps
    # is equal in area to the output healpix pixels).
    log2_Nside = np.log2(Nside)
    Nside = 2 ** (round(log2_Nside))  # Finding closest power of two for nside

    hits = np.zeros((3, nx, ny))  # Array of hits per pixel of all three maps

    lon1, lat1 = np.meshgrid(
        x1, y1, indexing="ij"
    )  # Making meshgrids om longitudes and latitudes
    lon2, lat2 = np.meshgrid(x2, y2, indexing="ij")
    lon3, lat3 = np.meshgrid(x3, y3, indexing="ij")

    lon1, lat1 = lon1.flatten(), lat1.flatten()  # Making 1D arrays of coordinate pairs
    lon2, lat2 = lon2.flatten(), lat2.flatten()
    lon3, lat3 = lon3.flatten(), lat3.flatten()

    c1_icrs = SkyCoord(
        ra=lon1, dec=lat1, frame="icrs", unit="deg"
    )  # Generate sky-coordinates in ICRS coordinates
    c2_icrs = SkyCoord(ra=lon2, dec=lat2, frame="icrs", unit="deg")
    c3_icrs = SkyCoord(ra=lon3, dec=lat3, frame="icrs", unit="deg")

    c1_gal = c1_icrs.transform_to(
        "galactic"
    )  # Transforming hitmap coordinates from ICRS to Galactic coordinates
    c2_gal = c2_icrs.transform_to("galactic")
    c3_gal = c3_icrs.transform_to("galactic")

    heal = HEALPix(
        Nside, order="ring", frame="galactic"
    )  # Generating HEALPix object with ring-numbering

    px_indices1 = heal.skycoord_to_healpix(
        c1_gal
    )  # Generating HEALPix indices for each hitmap
    px_indices2 = heal.skycoord_to_healpix(c2_gal)
    px_indices3 = heal.skycoord_to_healpix(c3_gal)

    hits[0, ...] = np.sum(
        hits1[:, sb - 1, freq - 1, :, :], axis=0
    )  # Co-adding hits of all detectors
    hits[1, ...] = np.sum(hits2[:, sb - 1, freq - 1, :, :], axis=0)
    hits[2, ...] = np.sum(hits3[:, sb - 1, freq - 1, :, :], axis=0)

    hits_list = np.zeros(
        (3, nx * ny)
    )  # List to contain hits of each pixel to be mapped to HEALPix format
    m = np.zeros(heal.npix)  # Array to contain the projected HEALPix pixels

    m[px_indices1] = hits[0, ...].flatten()  # Filling up HEALPix array with hitmaps
    m[px_indices2] = hits[1, ...].flatten()
    m[px_indices3] = hits[2, ...].flatten()
    """
    lons = np.zeros_like(m)
    lats = np.zeros_like(m)
    lons[px_indices1], lons[px_indices2], lons[px_indices3] = lon1, lon2, lon3
    lats[px_indices1], lats[px_indices2], lats[px_indices3] = lat1, lat2, lat3
    temp = np.zeros((len(m[np.where(m!=0)]), 3))
    temp[:, 0] = np.arange(0, len(m))[np.where(m!=0)]
    temp[:, 1], temp[:, 2] = lons[np.where(m!=0)], lats[np.where(m!=0)] 
    np.savetxt("comap2healpix.txt", temp, header = "HEALPix number, lon, lat (for some of the hits, where lon/lat are non-zero)")
    """
    return m


def savehealpix(infile1, infile2, infile3, outfile, sb, freq, mapfiletype):
    """
    Function saving the HEALPix map projection of three input maps at a given
    sideband and frequency.
    --------------------
    infile1: str
        Filename of first input file. HDF-file, .h5 format
    infile2: str
        Filename of second input file. HDF-file, .h5 format
    infile3: str 
        Filename of third input file. HDF-file, .h5 format  
    outfile: str
        Filename of file to save final HEALPix map to. 
    sb: int
        Number of sideband to use (base 1 not 0, i.e first and second sideband
        are sb = 1 and sb = 2 respectively)
    freq: int
        Number of the frequency channel to use (base 1 not 0, i.e. first and second 
        are freq = 1 and freq = 2 respectively)
    mapfiletype: str
        String input that desides whether to save HEALPix map as FITS of png file. 
        Must be either "FITS" of "png".
    """

    m = map2healpix(infile1, infile2, infile3, sb, freq)
    if mapfiletype == "fits":
        outfile += ".fits"
        hp.fitsfunc.write_map(
            outfile, m, fits_IDL=False, overwrite=True, coord="G", dtype=int
        )

    elif mapfiletype == "png":
        outfile += ".png"
        hp.mollview(
            m,
            cmap=cm.Oranges,
            coord="G",
            min=color_lim[0],
            max=color_lim[1],
            nest=False,
            title=None,
            unit="Hits",
        )
        plt.savefig(outfile)
    else:
        print(
            "Please provide either FITS or png file format for final HEALPix map file to be saved!"
        )
        exit()


def open_and_show_FITS_map(infile, infiletype):
    """
    Function opening and showing the HEALPix hit full-sky map made by the 
    function savehealpix. Also the function transforms the opened FITS image 
    from one coordinate system to another (default from celestial to galactic
    coordinated).
    --------------------
    infile: str
        Filename of FITS file to open and show.
    coord: list of str
        List of two strings. First string determines input coordinate
        system and second gives output coordinate system. May use "C" for celestial,
        "G" for galactic and "E" for ecliptic system.
    show: bool
        If true the loaded FITS image will be displayed.
    """
    if mapfiletype == "fits":
        infile += ".fits"
        healmap = hp.fitsfunc.read_map(infile, dtype=int)
        hp.mollview(
            healmap,
            cmap=cm.Oranges,
            coord="G",
            min=color_lim[0],
            max=color_lim[1],
            nest=False,
            title=None,
            unit="Hits",
        )
        plt.show()
    else:
        print("Please provide FITS file format to open HEALPix map file!")
        exit()


if __name__ == "__main__":
    savehealpix(infile1, infile2, infile3, outfile, sb, freq, mapfiletype)
    # open_and_show_FITS_map(outfile, mapfiletype)
