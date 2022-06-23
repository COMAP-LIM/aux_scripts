from operator import inv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA

from cmcrameri import cm
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy

import sys
import numpy.fft as fft
import os
import errno
import h5py
from scipy import stats
from scipy.stats import norm, skew
from scipy import signal
import copy

import argparse
import re 
from tqdm import tqdm
import warnings
import time as tm

from mapeditor.mapeditor import Atlas

import accept_mod.stats_list as stats_list
stats_list = stats_list.stats_list

from spikes import spike_data, spike_list, get_spike_list
from mapdata import Map

from experimental_map4browser import plot_pca_maps

"""
Function parsing the command line input.
"""
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inpath", type = str,
                    help = """Path to input map.""")

parser.add_argument("-n", "--inname", type = str,
                    help = """Name of input map.""")


parser.add_argument("-o", "--outname", type = str,
                    help = """Name of output plot.""")


parser.add_argument("-p", "--outpath", type = str,
                    help = """Path to output direcotry.""")

args = parser.parse_args()

if args.param == None:
    message = """No input parameterfile given, please provide an input parameterfile"""
    raise NameError(message)
else:
    mappath     = args.inpath
    outpath     = args.outpath
    inname     = args.inname
    outname     = args.outname


mapeditor = Atlas(no_init = True)

obsname = mappath + inname

obsmap  = Map(obsname)
scans   = range(2, 16)

mapobj = Map(obsname)
mapobj.read_map()

mapobj.outname = outname

mapobj.scanmap = False
mapobj.obsmap  = True
mapobj.obsid   = None

plot_pca_maps(mapobj, outpath, feeds = list(range(19)))
