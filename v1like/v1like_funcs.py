#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" v1like_funcs module

Key sub-operations performed in a simple V1-like model
(normalization, linear filtering, downsampling, etc.)

"""

import Image
import scipy as N
import scipy.signal

fftconv = scipy.signal.fftconvolve
conv = scipy.signal.convolve

from npclockit import clockit_onprofile
import time

PROFILE = False

# -------------------------------------------------------------------------
@clockit_onprofile(PROFILE)
def v1like_norm(hin, conv_mode, kshape, threshold):
    """ V1S local normalization

    Each pixel in the input image is divisively normalized by the L2 norm
    of the pixels in a local neighborhood around it, and the result of this
    division is placed in the output image.

    Inputs:
      hin -- a 3-dimensional array (width X height X rgb)
      kshape -- kernel shape (tuple) ex: (3,3) for a 3x3 normalization
                neighborhood
      threshold -- magnitude threshold, if the vector's length is below
                   it doesn't get resized ex: 1.

    Outputs:
      hout -- a normalized 3-dimensional array (width X height X rgb)

    """

    eps = 1e-5
    kh, kw = kshape
    dtype = hin.dtype
    hsrc = hin[:].copy()

    # -- prepare hout
    hin_h, hin_w, hin_d = hin.shape
    hout_h = hin_h - kh + 1
    hout_w = hin_w - kw + 1
    hout_d = hin_d
    hout = N.empty((hout_h, hout_w, hout_d), 'f')

    # -- compute numerator (hnum) and divisor (hdiv)
    # sum kernel
    hin_d = hin.shape[-1]
    kshape3d = list(kshape) + [hin_d]
    ker = N.ones(kshape3d, dtype=dtype)
    size = ker.size

    # compute sum-of-square
    hsq = hsrc ** 2.
    hssq = conv(hsq, ker, conv_mode).astype(dtype)

    # compute hnum and hdiv
    ys = kh / 2
    xs = kw / 2
    hout_h, hout_w, hout_d = hout.shape[-3:]
    hs = hout_h
    ws = hout_w
    hsum = conv(hsrc, ker, conv_mode).astype(dtype)
    hnum = hsrc[ys:ys+hs, xs:xs+ws] - (hsum/size)
    val = (hssq - (hsum**2.)/size)
    N.putmask(val, val<0, 0) # to avoid negative sqrt
    hdiv = val ** (1./2) + eps

    # -- apply normalization
    # 'volume' threshold
    N.putmask(hdiv, hdiv < (threshold+eps), 1.)
    result = (hnum / hdiv)

    hout[:] = result
    return hout

@clockit_onprofile(PROFILE)
def v1like_norm2(hin, conv_mode, kshape, threshold):
    """ V1LIKE local normalization

    Each pixel in the input image is divisively normalized by the L2 norm
    of the pixels in a local neighborhood around it, and the result of this
    division is placed in the output image.

    Inputs:
      hin -- a 3-dimensional array (width X height X rgb)
      kshape -- kernel shape (tuple) ex: (3,3) for a 3x3 normalization
                neighborhood
      threshold -- magnitude threshold, if the vector's length is below
                   it doesn't get resized ex: 1.

    Outputs:
      hout -- a normalized 3-dimensional array (width X height X rgb)

    """

    eps = 1e-5
    kh, kw = kshape
    dtype = hin.dtype
    hsrc = hin[:].copy()

    # -- prepare hout
    hin_h, hin_w, hin_d = hin.shape
    hout_h = hin_h# - kh + 1
    hout_w = hin_w# - kw + 1

    if conv_mode != "same":
        hout_h = hout_h - kh + 1
        hout_w = hout_w - kw + 1

    hout_d = hin_d
    hout = N.empty((hout_h, hout_w, hout_d), 'float32')

    # -- compute numerator (hnum) and divisor (hdiv)
    # sum kernel
    hin_d = hin.shape[-1]
    kshape3d = list(kshape) + [hin_d]
    ker = N.ones(kshape3d, dtype=dtype)
    size = ker.size

    # compute sum-of-square
    hsq = hsrc ** 2.
    #hssq = conv(hsq, ker, conv_mode).astype(dtype)
    kerH = ker[:,0,0][:, None]#, None]
    kerW = ker[0,:,0][None, :]#, None]
    kerD = ker[0,0,:][None, None, :]

    #s = time.time()
    #r = conv(hsq, kerD, 'valid')[:,:,0]
    #print time.time()-s

    #s = time.time()
    hssq = conv(
                conv(
                     conv(hsq, kerD, 'valid')[:,:,0].astype(dtype),
                     kerW,
                     conv_mode),
                kerH,
                conv_mode).astype(dtype)
    #hssq = conv(kerH,
                #conv(kerW,
                     #conv(hsq, kerD, 'valid')[:,:,0].astype(dtype),
                     #conv_mode),
                #conv_mode).astype(dtype)
    hssq = hssq[:,:,None]
    #print time.time()-s

    # compute hnum and hdiv
    ys = kh / 2
    xs = kw / 2
    hout_h, hout_w, hout_d = hout.shape[-3:]
    hs = hout_h
    ws = hout_w
    #hsum = conv(hsrc, ker, conv_mode).astype(dtype)
    hsum = conv(
                conv(
                     conv(hsrc,
                          kerD, 'valid')[:,:,0].astype(dtype),
                     kerW,
                     conv_mode),
                kerH,
                conv_mode).astype(dtype)
    #hsum = conv(kerH,
                #conv(kerW,
                     #conv(hsrc,
                          #kerD, 'valid')[:,:,0].astype(dtype),
                     #conv_mode),
                #conv_mode).astype(dtype)
    hsum = hsum[:,:,None]
    if conv_mode == 'same':
        hnum = hsrc - (hsum/size)
    else:
        hnum = hsrc[ys:ys+hs, xs:xs+ws] - (hsum/size)
    val = (hssq - (hsum**2.)/size)
    val[val<0] = 0
    hdiv = val ** (1./2) + eps

    # -- apply normalization
    # 'volume' threshold
    N.putmask(hdiv, hdiv < (threshold+eps), 1.)
    result = (hnum / hdiv)

    #print result.shape
    hout[:] = result
    #print hout.shape, hout.dtype
    return hout

v1like_norm = v1like_norm2

# -------------------------------------------------------------------------
fft_cache = {}

@clockit_onprofile(PROFILE)
def v1like_filter(hin, conv_mode, filterbank, use_fft_cache=False):
    """ V1LIKE linear filtering
    Perform separable convolutions on an image with a set of filters

    Inputs:
      hin -- input image (a 2-dimensional array)
      filterbank -- FIXME list of tuples with 1d filters (row, col)
                    used to perform separable convolution
      use_fft_cache -- Boolean, use internal fft_cache (works _well_ if the
      input shapes don't vary much, otherwise you'll blow away the memory)

    Outputs:
      hout -- a 3-dimensional array with outputs of the filters
              (width X height X n_filters)

    """

    nfilters = len(filterbank)

    filt0 = filterbank[0]
    fft_shape = N.array(hin.shape) + N.array(filt0.shape) - 1
    hin_fft = scipy.signal.fftn(hin, fft_shape)

    if conv_mode == "valid":
        hout_shape = list( N.array(hin.shape[:2]) - N.array(filt0.shape[:2]) + 1 ) + [nfilters]
        hout_new = N.empty(hout_shape, 'f')
        begy = filt0.shape[0]
        endy = begy + hout_shape[0]
        begx = filt0.shape[1]
        endx = begx + hout_shape[1]
    elif conv_mode == "same":
        hout_shape = hin.shape[:2] + (nfilters,)
        hout_new = N.empty(hout_shape, 'f')
        begy = filt0.shape[0] / 2
        endy = begy + hout_shape[0]
        begx = filt0.shape[1] / 2
        endx = begx + hout_shape[1]
    else:
        raise NotImplementedError

    for i in xrange(nfilters):
        filt = filterbank[i]

        if use_fft_cache:
            key = (filt.tostring(), tuple(fft_shape))
            if key in fft_cache:
                filt_fft = fft_cache[key]
            else:
                filt_fft = scipy.signal.fftn(filt, fft_shape)
                fft_cache[key] = filt_fft
        else:
            filt_fft = scipy.signal.fftn(filt, fft_shape)

        res_fft = scipy.signal.ifftn(hin_fft*filt_fft)
        res_fft = res_fft[begy:endy, begx:endx]
        hout_new[:,:,i] = N.real(res_fft)

    hout = hout_new

    return hout

# -------------------------------------------------------------------------
@clockit_onprofile(PROFILE)
#@profile
def v1like_pool(hin, conv_mode, lsum_ksize=None, outshape=None, order=1):
    """ V1LIKE Pooling
    Boxcar Low-pass filter featuremap-wise

    Inputs:
      hin -- a 3-dimensional array (width X height X n_channels)
      lsum_ksize -- kernel size of the local sum ex: 17
      outshape -- fixed output shape (2d slices)
      order -- XXX

    Outputs:
       hout -- resulting 3-dimensional array

    """

    order = float(order)
    assert(order >= 1)

    # -- local sum
    if lsum_ksize is not None:
        hin_h, hin_w, hin_d = hin.shape
        dtype = hin.dtype
        if conv_mode == "valid":
            aux_shape = auxh, auxw, auxd = hin_h-lsum_ksize+1, hin_w-lsum_ksize+1, hin_d
            aux = N.empty(aux_shape, dtype)
        else:
            aux = N.empty(hin.shape, dtype)
        k1d = N.ones((lsum_ksize), 'f')
        k2d = N.ones((lsum_ksize, lsum_ksize), 'f')
        krow = k1d[None,:]
        kcol = k1d[:,None]
        for d in xrange(aux.shape[2]):
            if order == 1:
                aux[:,:,d] = conv(conv(hin[:,:,d], krow, conv_mode), kcol, conv_mode)
            else:
                aux[:,:,d] = conv(conv(hin[:,:,d]**order, krow, conv_mode), kcol, conv_mode)**(1./order)

    else:
        aux = hin

    # -- resample output
    if outshape is None or outshape == aux.shape:
        hout = aux
    else:
        hout = sresample(aux, outshape)

    return hout

# -------------------------------------------------------------------------
@clockit_onprofile(PROFILE)
def sresample(src, outshape):
    """ Simple 3d array resampling

    Inputs:
      src -- a ndimensional array (dim>2)
      outshape -- fixed output shape for the first 2 dimensions

    Outputs:
       hout -- resulting n-dimensional array

    """

    inh, inw = inshape = src.shape[:2]
    outh, outw = outshape
    hslice = (N.arange(outh) * (inh-1.)/(outh-1.)).round().astype(int)
    wslice = (N.arange(outw) * (inw-1.)/(outw-1.)).round().astype(int)
    hout = src[hslice, :][:, wslice]
    return hout.copy()



# -------------------------------------------------------------------------
@clockit_onprofile(PROFILE)
def get_image(img_fname, max_edge=None, min_edge=None,
              resize_method='bicubic'):
    """ Return a resized image as a numpy array

    Inputs:
      img_fname -- image filename
      max_edge -- maximum edge length (None = no resize)
      min_edge -- minimum edge length (None = no resize)
      resize_method -- 'antialias' or 'bicubic'

    Outputs:
      imga -- result

    """

    # -- open image
    try:
        img = Image.open(img_fname)
    except IOError, err:
        print "ERROR with '%s':" % img_fname, err
        raise err

    if max_edge is not None:
        # -- resize so that the biggest edge is max_edge (keep aspect ratio)
        iw, ih = img.size
        if iw > ih:
            new_iw = max_edge
            new_ih = int(round(1.* max_edge * ih/iw))
        else:
            new_iw = int(round(1.* max_edge * iw/ih))
            new_ih = max_edge
        if resize_method.lower() == 'bicubic':
            img = img.resize((new_iw, new_ih), Image.BICUBIC)
        elif resize_method.lower() == 'antialias':
            img = img.resize((new_iw, new_ih), Image.ANTIALIAS)
        else:
            raise ValueError("resize_method '%s' not understood", resize_method)

    # -- convert to a numpy array
    imga = N.misc.fromimage(img)#/255.
    return imga

# -------------------------------------------------------------------------
@clockit_onprofile(PROFILE)
def get_image2(img_fname, resize=None):
    """ Return a resized image as a numpy array

    Inputs:
      img_fname -- image filename
      resize -- tuple of (type, size) where type='min_edge' or 'max_edge'
                if None = no resize

    Outputs:
      imga -- result

    """

    # -- open image
    img = Image.open(img_fname)

    # -- resize image if needed
    if resize is not None:
        rtype, rsize = resize

        if rtype == 'min_edge':
            # -- resize so that the smallest edge is rsize (keep aspect ratio)
            iw, ih = img.size
            if iw < ih:
                new_iw = rsize
                new_ih = int(round(1.* rsize * ih/iw))
            else:
                new_iw = int(round(1.* rsize * iw/ih))
                new_ih = rsize

        elif rtype == 'max_edge':
            # -- resize so that the biggest edge is rszie (keep aspect ratio)
            iw, ih = img.size
            if iw > ih:
                new_iw = rsize
                new_ih = int(round(1.* rsize * ih/iw))
            else:
                new_iw = int(round(1.* rsize * iw/ih))
                new_ih = rsize

        else:
            raise ValueError, "resize parameter not understood"

        if resize_method.lower() == 'bicubic':
            img = img.resize((new_iw, new_ih), Image.BICUBIC)
        elif resize_method.lower() == 'antialias':
            img = img.resize((new_iw, new_ih), Image.ANTIALIAS)
        else:
            raise ValueError("resize_method '%s' not understood", resize_method)

    # -- convert to a numpy array
    imga = N.misc.fromimage(img)#/255.
    return imga


# -------------------------------------------------------------------------
@clockit_onprofile(PROFILE)
def rephists(hin, division, nfeatures):
    """ Compute local feature histograms from a given 3d (width X height X
    n_channels) image.

    These histograms are intended to serve as easy-to-compute additional
    features that can be concatenated onto the V1-like output vector to
    increase performance with little additional complexity. These additional
    features are only used in the V1LIKE+ (i.e. + 'easy tricks') version of
    the model.

    Inputs:
      hin -- 3d image (width X height X n_channels)
      division -- granularity of the local histograms (e.g. 2 corresponds
                  to computing feature histograms in each quadrant)
      nfeatures -- desired number of resulting features

    Outputs:
      fvector -- feature vector

    """

    hin_h, hin_w, hin_d = hin.shape
    nzones = hin_d * division**2
    nbins = nfeatures / nzones
    sx = (hin_w-1.)/division
    sy = (hin_h-1.)/division
    fvector = N.zeros((nfeatures), 'f')
    hists = []
    for d in xrange(hin_d):
        h = [N.histogram(hin[j*sy:(j+1)*sy,i*sx:(i+1)*sx,d], bins=nbins)[0].ravel()
             for i in xrange(division)
             for j in xrange(division)
             ]
        hists += [h]

    hists = N.array(hists, 'f').ravel()
    fvector[:hists.size] = hists
    return fvector
