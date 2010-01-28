#!/usr/bin/env python
# -*- coding: utf-8 -*-

# TODO: clean + pylint

import optparse
from pprint import pprint
import csv
import sys
import os.path as path

from v1like_extract import v1like_extract

import scipy as sp

from npprogressbar import *

DEFAULT_INPUT_PATH = "./"
DEFAULT_NPROCESSORS = 1
DEFAULT_OVERWRITE = False
DEFAULT_VERBOSE = False

verbose = DEFAULT_VERBOSE

# ------------------------------------------------------------------------------
def pv1like_extract(params):
    config_fname, input_fname, output_fname, overwrite = params
    return v1like_extract(config_fname, 
                          input_fname, 
                          output_fname,
                          overwrite = overwrite)

# ------------------------------------------------------------------------------
def v1like_extract_fromcsv(config_fname,
                           input_csv_fname,
                           output_suffix,
                           input_path = DEFAULT_INPUT_PATH,
                           nprocessors = DEFAULT_NPROCESSORS,
                           overwrite = DEFAULT_OVERWRITE,
                           ):
    
    assert(nprocessors >= 1)

    csvr = csv.reader(open(input_csv_fname))
    rows = [ row for row in csvr ]
    fnames = sp.array([ row[:-2] for row in rows ]).ravel()
    #fnames.sort()
    sp.random.shuffle(fnames) # shuffle to enable multiple instances to run in //
    nfnames = len(fnames)

    # -- set up progress bar
    widgets = [RotatingMarker(), " Progress: ", Percentage(), " ",
               Bar(left='[',right=']'), ' ', 
               #" (", FilenameUpdate(fnames), ") ", ETA()]
               ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=nfnames)
    
    # - should we use multiprocessing ?
    if nprocessors > 1:
        # -- prepare all parameters
        import multiprocessing as mp
        pool = mp.Pool(nprocessors)

        print "Creating list of parameters for multiprocessing..."
        params = []
        for fname in fnames:

            input_fname = path.join(input_path, fname)
            output_fname = input_fname + output_suffix

            params += [(config_fname, 
                        input_fname, 
                        output_fname,
                        overwrite)]

        # -- async iterator map
        done = pool.imap_unordered(pv1like_extract, params)

        # -- update progress bar
        print "Processing %d images..." % (nfnames)
        i = 1
        pbar.start()
        for _ in done:        
            pbar.update(i)
            i += 1
    else:
        # -- process images
        pbar.start()
        for i, fname in enumerate(fnames):

            input_fname = path.join(input_path, fname)
            output_fname = input_fname + output_suffix

            v1like_extract(config_fname, 
                           input_fname, 
                           output_fname,
                           overwrite = overwrite)
            pbar.update(i+1)        

    pbar.finish()
    print

# ------------------------------------------------------------------------------
def main():

    usage = "usage: %prog [options] <config_filename> <input_csv_filename> <output_suffix>"
    
    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--input_path", "-i",
                      default=DEFAULT_INPUT_PATH,
                      type="str",
                      metavar="STR",
                      help="[default=%default]")
    
    parser.add_option("--nprocessors",
                      default=DEFAULT_NPROCESSORS,
                      type="int",
                      metavar="INT",
                      help="number of processors to use (with multiprocessing if > 1) [default=%default]")
    
    parser.add_option("--overwrite", 
                      default=DEFAULT_OVERWRITE,
                      action="store_true",
                      help="overwrite existing file [default=%default]")

    parser.add_option("--verbose", "-v" ,
                      default=DEFAULT_VERBOSE,
                      action="store_true",
                      help="[default=%default]")

    opts, args = parser.parse_args()

    if len(args) != 3:
        parser.print_help()
    else:
        config_fname = args[0]
        input_csv_fname = args[1]
        output_suffix = args[2]
        
        global verbose
        if opts.verbose:
            verbose = True
        
        v1like_extract_fromcsv(config_fname,
                               input_csv_fname,
                               output_suffix,
                               input_path = opts.input_path,
                               nprocessors = opts.nprocessors,
                               overwrite = opts.overwrite,
                               )


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
    
