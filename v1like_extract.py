#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os
from os import path
import warnings
import numpy as np
import scipy as sp
from scipy import io
import time
import pprint
import hashlib
import cPickle

from npclockit import clockit_onprofile

from v1like_funcs import get_image, get_image2, conv
from v1like_funcs import v1like_norm, v1like_filter, v1like_pool
from v1like_funcs import rephists
from v1like_math import gabor2d

# TODO: clean + pylint

DEFAULT_OVERWRITE = False
DEFAULT_VERBOSE = False
WRITE_RETRY = 10

#from OptParserExtended import OptionExtended

filt_l = None

verbose = DEFAULT_VERBOSE

import warnings
warnings.simplefilter('ignore', UserWarning)

import colorconv

class MinMaxError(Exception): pass

# ------------------------------------------------------------------------------
@clockit_onprofile(verbose)
def v1like_fromarray(arr, params, featsel):
    """ Applies a simple V1-like model and generates a feature vector from
    its outputs. 

    Inputs:
      arr -- image's array 
      params -- representation parameters (dict)
      featsel -- features to include to the vector (dict)

    Outputs:
      fvector -- corresponding feature vector                  

    """

    if 'conv_mode' not in params:
        params['conv_mode'] = 'same'
    if 'color_space' not in params:
        params['color_space'] = 'gray'

    arr = sp.atleast_3d(arr)

    smallest_edge = min(arr.shape[:2])

    rep = params
    
    preproc_lsum = rep['preproc']['lsum_ksize']
    if preproc_lsum is None:
        preproc_lsum = 1
    smallest_edge -= (preproc_lsum-1)
            
    normin_kshape = rep['normin']['kshape']
    smallest_edge -= (normin_kshape[0]-1)

    filter_kshape = rep['filter']['kshape']
    smallest_edge -= (filter_kshape[0]-1)
        
    normout_kshape = rep['normout']['kshape']
    smallest_edge -= (normout_kshape[0]-1)
        
    pool_lsum = rep['pool']['lsum_ksize']
    smallest_edge -= (pool_lsum-1)

    arrh, arrw, _ = arr.shape

    if smallest_edge <= 0 and rep['conv_mode'] == 'valid':
        if arrh > arrw:
            new_w = arrw - smallest_edge + 1
            new_h =  int(np.round(1.*new_w  * arrh/arrw))
            print new_w, new_h
            raise
        elif arrh < arrw:
            new_h = arrh - smallest_edge + 1
            new_w =  int(np.round(1.*new_h  * arrw/arrh))
            print new_w, new_h
            raise
        else:
            pass
    
    # TODO: finish image size adjustment
    assert min(arr.shape[:2]) > 0

    # use the first 3 channels only
    orig_imga = arr.astype("float32")[:,:,:3]

    # make sure that we don't have a 3-channel (pseudo) gray image
    if orig_imga.shape[2] == 3 \
            and (orig_imga[:,:,0]-orig_imga.mean(2) < 0.1*orig_imga.max()).all() \
            and (orig_imga[:,:,1]-orig_imga.mean(2) < 0.1*orig_imga.max()).all() \
            and (orig_imga[:,:,2]-orig_imga.mean(2) < 0.1*orig_imga.max()).all():
        orig_imga = sp.atleast_3d(orig_imga[:,:,0])

    # rescale to [0,1]
    #print orig_imga.min(), orig_imga.max()
    if orig_imga.min() == orig_imga.max():
        raise MinMaxError("[ERROR] orig_imga.min() == orig_imga.max() "
                          "orig_imga.min() = %f, orig_imga.max() = %f"
                          % (orig_imga.min(), orig_imga.max())
                          )
    
    orig_imga -= orig_imga.min()
    orig_imga /= orig_imga.max()

    # -- color conversion
    # insure 3 dims
    #print orig_imga.shape
    if orig_imga.ndim == 2 or orig_imga.shape[2] == 1:
        orig_imga_new = sp.empty(orig_imga.shape[:2] + (3,), dtype="float32")
        orig_imga.shape = orig_imga_new[:,:,0].shape
        orig_imga_new[:,:,0] = 0.2989*orig_imga
        orig_imga_new[:,:,1] = 0.5870*orig_imga
        orig_imga_new[:,:,2] = 0.1141*orig_imga
        orig_imga = orig_imga_new    

    # -
    if params['color_space'] == 'rgb':
        orig_imga_conv = orig_imga
#     elif params['color_space'] == 'rg':
#         orig_imga_conv = colorconv.rg_convert(orig_imga)
    elif params['color_space'] == 'rg2':
        orig_imga_conv = colorconv.rg2_convert(orig_imga)
    elif params['color_space'] == 'gray':
        orig_imga_conv = colorconv.gray_convert(orig_imga)
        orig_imga_conv.shape = orig_imga_conv.shape + (1,)
    elif params['color_space'] == 'opp':
        orig_imga_conv = colorconv.opp_convert(orig_imga)
    elif params['color_space'] == 'oppnorm':
        orig_imga_conv = colorconv.oppnorm_convert(orig_imga)
    elif params['color_space'] == 'chrom':
        orig_imga_conv = colorconv.chrom_convert(orig_imga)
#     elif params['color_space'] == 'opponent':
#         orig_imga_conv = colorconv.opponent_convert(orig_imga)
#     elif params['color_space'] == 'W':
#         orig_imga_conv = colorconv.W_convert(orig_imga)
    elif params['color_space'] == 'hsv':
        orig_imga_conv = colorconv.hsv_convert(orig_imga)
    else:
        raise ValueError, "params['color_space'] not understood"
    
    # -- process each map
    fvector_l = []

    for cidx in xrange(orig_imga_conv.shape[2]):
        imga0 = orig_imga_conv[:,:,cidx]

        assert(imga0.min() != imga0.max())

        # -- 0. preprocessing
        #imga0 = imga0 / 255.0

        # flip image ?
        if 'flip_lr' in params['preproc'] and params['preproc']['flip_lr']:
            imga0 = imga0[:,::-1]
            
        if 'flip_ud' in params['preproc'] and params['preproc']['flip_ud']:
            imga0 = imga0[::-1,:]            

        # smoothing
        lsum_ksize = params['preproc']['lsum_ksize']
        conv_mode = params['conv_mode']
        if lsum_ksize is not None:
             k = sp.ones((lsum_ksize), 'f') / lsum_ksize             
             imga0 = conv(conv(imga0, k[sp.newaxis,:], conv_mode), 
                          k[:,sp.newaxis], conv_mode)
             
        # whiten full image (assume True)
        if 'whiten' not in params['preproc'] or params['preproc']['whiten']:
            imga0 -= imga0.mean()
            if imga0.std() != 0:
                imga0 /= imga0.std()

        # -- 1. input normalization
        imga1 = v1like_norm(imga0[:,:,sp.newaxis], conv_mode, **params['normin'])
        #print imga1.shape

        # -- 2. linear filtering
        filt_l = get_gabor_filters(params['filter'])
        imga2 = v1like_filter(imga1[:,:,0], conv_mode, filt_l)
        #print imga2.shape

        #raise

        # -- 3. simple non-linear activation (clamping)
        minout = params['activ']['minout'] # sustain activity
        maxout = params['activ']['maxout'] # saturation
        imga3 = imga2.clip(minout, maxout)
        #print imga3.shape

        # -- 4. output normalization
        imga4 = v1like_norm(imga3, conv_mode, **params['normout'])
        #print imga4.shape

        # -- 5. sparsify ?
        if "sparsify" in params and params["sparsify"]:
            imga4 = (imga4.max(2)[:,:,None] == imga4)
            #print imga4.shape
            #raise

        # -- 6. volume dimension reduction
        imga5 = v1like_pool(imga4, conv_mode, **params['pool'])
        output = imga5
        #print imga5.shape

        # -- 7. handle features to include
        feat_l = []

        # include input norm histograms ? 
        f_normin_hists = featsel['normin_hists']
        if f_normin_hists is not None:
            division, nfeatures = f_norminhists
            feat_l += [rephists(imga1, division, nfeatures)]

        # include filter output histograms ? 
        f_filter_hists = featsel['filter_hists']
        if f_filter_hists is not None:
            division, nfeatures = f_filter_hists
            feat_l += [rephists(imga2, division, nfeatures)]

        # include activation output histograms ?     
        f_activ_hists = featsel['activ_hists']
        if f_activ_hists is not None:
            division, nfeatures = f_activ_hists
            feat_l += [rephists(imga3, division, nfeatures)]

        # include output norm histograms ?     
        f_normout_hists = featsel['normout_hists']
        if f_normout_hists is not None:
            division, nfeatures = f_normout_hists
            feat_l += [rephists(imga4, division, nfeatures)]

        # include representation output histograms ? 
        f_pool_hists = featsel['pool_hists']
        if f_pool_hists is not None:
            division, nfeatures = f_pool_hists
            feat_l += [rephists(imga5, division, nfeatures)]

        # include representation output ?
        f_output = featsel['output']
        if f_output and len(feat_l) != 0:
            fvector = sp.concatenate([output.ravel()]+feat_l)
        else:
            fvector = output

        fvector_l += [fvector]

    # -- 

    # include grayscale values ?
    f_input_gray = featsel['input_gray']
    if f_input_gray is not None:
        shape = f_input_gray
        #print orig_imga.shape
        fvector_l += [sp.misc.imresize(colorconv.gray_convert(orig_imga), shape).ravel()]

    # include color histograms ?
    f_input_colorhists = featsel['input_colorhists']
    if f_input_colorhists is not None:
        nbins = f_input_colorhists
        colorhists = sp.empty((3,nbins), 'f')
        if orig_imga.ndim == 3:
            for d in xrange(3):
                h = sp.histogram(orig_imga[:,:,d].ravel(),
                                 bins=nbins,
                                 range=[0,255])
                binvals = h[0].astype('f')
                colorhists[d] = binvals
        else:
            raise ValueError, "orig_imga.ndim == 3"
            #h = sp.histogram(orig_imga[:,:].ravel(),
            #                 bins=nbins,
            #                 range=[0,255])
            #binvals = h[0].astype('f')
            #colorhists[:] = binvals

        #feat_l += [colorhists.ravel()]
        fvector_l += [colorhists.ravel()]

    # -- done !    
    fvector_l = [fvector.ravel() for fvector in fvector_l]
    out = sp.concatenate(fvector_l).ravel()
    return out
   
# -------------------------------------------------------------------------
def get_gabor_filters(params):
    """ Return a Gabor filterbank (generate it if needed)
    
    Inputs:
    params -- filters parameters (dict)

    Outputs:
    filt_l -- filterbank (list)

    """
        
    global filt_l

    if filt_l is not None:
        return filt_l

    # -- get parameters
    fh, fw = params['kshape']
    orients = params['orients']
    freqs = params['freqs']
    phases = params['phases']
    nf =  len(orients) * len(freqs) * len(phases)
    fbshape = nf, fh, fw
    xc = fw/2
    yc = fh/2
    filt_l = []
    
    # -- build the filterbank
    for freq in freqs:
        for orient in orients:
            for phase in phases:
                # create 2d gabor
                filt = gabor2d(xc,yc,xc,yc,
                               freq,orient,phase,
                               (fw,fh))
                filt_l += [filt]
                
    return filt_l
       
# -------------------------------------------------------------------------
def v1like_fromfilename(config_fname,
                        input_fname,
                        ):
    
    """ TODO """

    # -- get parameters
    config_path = path.abspath(config_fname)
    if verbose: print "Config file:", config_path
    v1like_config = {}
    execfile(config_path, {}, v1like_config)

    model = v1like_config['model']

    if len(model) != 1:
        raise NotImplementedError

    rep, featsel = model[0]
    if verbose: 
        print '*'*80
        pprint.pprint(rep)

    resize_type = rep['preproc'].get('resize_type', 'input')
    if resize_type == 'output':
        if 'max_edge' not in rep['preproc']:
            raise NotImplementedError
        # add whatever is needed to get output = max_edge
        new_max_edge = rep['preproc']['max_edge']

        preproc_lsum = rep['preproc']['lsum_ksize']
        new_max_edge += preproc_lsum-1
            
        normin_kshape = rep['normin']['kshape']
        assert normin_kshape[0] == normin_kshape[1]
        new_max_edge += normin_kshape[0]-1

        filter_kshape = rep['filter']['kshape']
        assert filter_kshape[0] == filter_kshape[1]
        new_max_edge += filter_kshape[0]-1
        
        normout_kshape = rep['normout']['kshape']
        assert normout_kshape[0] == normout_kshape[1]
        new_max_edge += normout_kshape[0]-1
        
        pool_lsum = rep['pool']['lsum_ksize']
        new_max_edge += pool_lsum-1

        rep['preproc']['max_edge'] = new_max_edge
    

    if 'max_edge' in rep['preproc']:
        max_edge = rep['preproc']['max_edge']
        resize_method = rep['preproc']['resize_method']
        imgarr = get_image(input_fname, max_edge=max_edge,
                           resize_method=resize_method)
    else:
        resize = rep['preproc']['resize']
        resize_method = rep['preproc']['resize_method']        
        imgarr = get_image2(input_fname, resize=resize,
                            resize_method=resize_method)

    try:
        fvector = v1like_fromarray(imgarr, rep, featsel)
    except MinMaxError, err:
        raise err, "with %s" % input_fname
    except AssertionError, err:
        raise err, "with %s" % input_fname

    if verbose: print '*'*80

    return fvector
    

# -------------------------------------------------------------------------
def v1like_extract(config_fname,
                   input_fname,
                   output_fname,
                   overwrite = DEFAULT_OVERWRITE,
                   ):
    
    """ Extract v1-like features from an image """

    # add matlab's extension to the output filename if needed
    if path.splitext(output_fname)[-1] != ".mat":
        output_fname += ".mat"        

#     lock_fname = output_fname + ".lock"

#     # can we overwrite ?
#     if (path.exists(lock_fname) or path.exists(output_fname)) and not overwrite:
#         warnings.warn("not allowed to overwrite %s"  % output_fname)
#         return
        
#     # lock
#     open(lock_fname, "w+")

    # can we overwrite ?        
    if path.exists(output_fname) and not overwrite:
        warnings.warn("not allowed to overwrite %s"  % output_fname)
        return

    fvector = v1like_fromfilename(config_fname, input_fname)

    if verbose: print "saving data (shape=%s) in %s" % (fvector.shape, output_fname)

    # XXX: supporting mat files is a pain in the ass...

    out_dict = {
        "data": fvector.ravel().reshape(1,-1),
        "shape": sp.array(fvector.shape, dtype='float32').reshape(1,-1)
        }

    sha1_gt = hashlib.sha1(cPickle.dumps(out_dict, 2)).hexdigest()
    out_dict['sha1'] = sha1_gt

    ok = False
    for i in xrange(WRITE_RETRY):
        if i > 0:
            print "Writing %s (retry %d)" % (output_fname, i)
            
        io.savemat(output_fname,
                   out_dict,
                   format='4',
                   )
        try:
            in_dict = io.loadmat(output_fname)
            del in_dict['sha1']            
            in_dict.pop('__globals__', None)
            sha1 = hashlib.sha1(cPickle.dumps(in_dict, 2)).hexdigest()
            if sha1 == sha1_gt:
                ok = True
                break
        except TypeError, err:
            if err.message != "buffer is too small for requested array":
                raise err
        except KeyError, err:
            #if err.message != "'sha1'":
            #    raise err
            pass
            
        os.unlink(output_fname)
        import time
        time.sleep(.5)

    if not ok:
        raise IOError("Error while saving '%s' (WRITE_RETRY=%d)"
                      % (output_fname, WRITE_RETRY))

# -------------------------------------------------------------------------
def main():

    import optparse

    usage = "usage: %prog [options] <config_filename> <input_filename> <output_filename>"
    
    parser = optparse.OptionParser(usage=usage)
    
    parser.add_option("--overwrite", 
                      default=DEFAULT_OVERWRITE,
                      action="store_true",
                      help="overwrite existing file [default=%default]")

    parser.add_option("--verbose", "-v",
                      default=DEFAULT_VERBOSE,
                      action="store_true",
                      help="[default=%default]")

    opts, args = parser.parse_args()

    if len(args) != 3:
        parser.print_help()
    else:
        config_fname = args[0]
        input_fname = args[1]
        output_fname = args[2]
        
        global verbose
        if opts.verbose:
            verbose = True

        v1like_extract(config_fname,
                       input_fname,
                       output_fname,
                       overwrite = opts.overwrite,
                       )        


# --------------------------------
if __name__ == "__main__":
    main()
    
