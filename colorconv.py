#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy as sp

# ------------------------------------------------------------------------------
def gray_convert(arr):
    #assert(arr.min()>=0 and arr.max()<=1)
    # grayscale conversion
    out = 0.2989*arr[:,:,0] + \
        0.5870*arr[:,:,1] + \
        0.1141*arr[:,:,2]
    #out.shape = out.shape + (1,)    
    return out


# ------------------------------------------------------------------------------
def opp_convert(arr):
    #assert(arr.min()>=0 and arr.max()<=1)
    out = sp.empty_like(arr)

    # red-green
    out[:,:,0] = arr[:,:,0] - arr[:,:,1]
    # blue-yellow
    out[:,:,1] = arr[:,:,2] - arr[:,:,[0,1]].min(2)
    # intensity
    out[:,:,2] = arr.max(2)

    return out

# ------------------------------------------------------------------------------
def oppnorm_convert(arr, threshold=0.1):
    #assert(arr.min()>=0 and arr.max()<=1)
    #out = sp.empty_like(arr)
    arr = arr.astype('float32')
    out = sp.empty(arr.shape[:2]+(2,), dtype='float32')

    print out.shape

    # red-green
    out[:,:,0] = arr[:,:,0] - arr[:,:,1]
    # blue-yellow
    out[:,:,1] = arr[:,:,2] - arr[:,:,[0,1]].min(2)
    # intensity
    denom = arr.max(2)

    mask = denom < threshold#*denom[:,:,2].mean()
    
    out[:,:,0] /= denom    
    out[:,:,1] /= denom

    sp.putmask(out[:,:,0], mask, 0)
    sp.putmask(out[:,:,1], mask, 0)

    return out

# ------------------------------------------------------------------------------
def chrom_convert(arr):
    #assert(arr.min()>=0 and arr.max()<=1)

    opp = opp_convert(arr)
    out = sp.empty_like(opp[:,:,[0,1]])

    rg = opp[:,:,0]
    by = opp[:,:,1]
    intensity = opp[:,:,2]

    lowi = intensity < 0.1*intensity.max()
    rg[lowi] = 0
    by[lowi] = 0

    denom = intensity
    denom[denom==0] = 1
    out[:,:,0] = rg / denom
    out[:,:,1] = by / denom

    return out

# ------------------------------------------------------------------------------
def rg2_convert(arr):
    #assert(arr.min()>=0 and arr.max()<=1)

    out = sp.empty_like(arr[:,:,[0,1]])

    red = arr[:,:,0]
    green = arr[:,:,1]
    blue = arr[:,:,2]
    intensity = arr.mean(2)

    lowi = intensity < 0.1*intensity.max()
    arr[lowi] = 0

    denom = arr.sum(2)
    denom[denom==0] = 1
    out[:,:,0] = red / denom
    out[:,:,1] = green / denom
    
    return out

# ------------------------------------------------------------------------------
def hsv_convert(arr):
    """ fast rgb_to_hsv using numpy array """
 
    # adapted from Arnar Flatberg
    # http://www.mail-archive.com/numpy-discussion@scipy.org/msg06147.html
    # it now handles NaN properly and mimics colorsys.rgb_to_hsv output

    import numpy as np

    #assert(arr.min()>=0 and arr.max()<=1)

    #arr = arr/255.
    arr = arr.astype("float32")
    out = np.empty_like(arr)

    arr_max = arr.max(-1)
    delta = arr.ptp(-1)
    s = delta / arr_max
    
    s[delta==0] = 0

    # red is max
    idx = (arr[:,:,0] == arr_max) 
    out[idx, 0] = (arr[idx, 1] - arr[idx, 2]) / delta[idx]

    # green is max
    idx = (arr[:,:,1] == arr_max) 
    out[idx, 0] = 2. + (arr[idx, 2] - arr[idx, 0] ) / delta[idx]

    # blue is max
    idx = (arr[:,:,2] == arr_max) 
    out[idx, 0] = 4. + (arr[idx, 0] - arr[idx, 1] ) / delta[idx]

    out[:,:,0] = (out[:,:,0]/6.0) % 1.0
    out[:,:,1] = s
    out[:,:,2] = arr_max

    # rescale back to [0, 255]
    #out *= 255.

    # remove NaN
    out[np.isnan(out)] = 0

    return out

# ------------------------------------------------------------------------------
def rgb_convert(arr):
    #assert(arr.min()>=0 and arr.max()<=1)

    # force 3 dims
    if arr.ndim == 2 or arr.shape[2] == 1:
        arr_new = sp.empty(arr.shape[:2] + (3,), dtype="float32")
        arr_new[:,:,0] = arr.copy()
        arr_new[:,:,1] = arr.copy()
        arr_new[:,:,2] = arr.copy()
        arr = arr_new    
    
    return arr

# ------------------------------------------------------------------------------
def oppsande_convert(arr):
    #assert(arr.min()>=0 and arr.max()<=1)

    r = arr[:,:,0]
    g = arr[:,:,1]
    b = arr[:,:,2]
    
    out = sp.empty_like(arr)
    out[:,:,0] = (r-g) / sp.sqrt(2.)
    out[:,:,1] = (r+g-2.*b) / sp.sqrt(6.)
    out[:,:,2] = (r+g+b) / sp.sqrt(3.)

    return out

# ------------------------------------------------------------------------------
def rgbwhiten_convert(arr):
    #assert(arr.min()>=0 and arr.max()<=1)

    r = arr[:,:,0]
    rmean = r.mean()
    rstd = r.std()
    if rstd == 0: rstd = 1

    g = arr[:,:,1]
    gmean = g.mean()
    gstd = g.std()
    if gstd == 0: gstd = 1

    b = arr[:,:,2]
    bmean = b.mean()
    bstd = b.std()
    if bstd == 0: bstd = 1

    out = sp.empty_like(arr)
    out[:,:,0] = (r - rmean) / rstd
    out[:,:,1] = (g - gmean) / gstd
    out[:,:,2] = (b - bmean) / bstd

    return out

# ------------------------------------------------------------------------------
def irg_convert(arr):
    #assert(arr.min()>=0 and arr.max()<=1)

    r = arr[:,:,0]
    g = arr[:,:,1]
    b = arr[:,:,2]

    intensity = arr.mean(2)
    lowi = intensity < 0.1*intensity.max()
    
    r[lowi] = 0
    g[lowi] = 0

    denom = intensity.copy()
    denom[denom==0] = 1

    out = sp.empty_like(arr)

    out[:,:,0] = intensity
    out[:,:,1] = r / denom
    out[:,:,2] = g / denom
    
    return out

# ------------------------------------------------------------------------------
def oppwalker_convert(arr):
    #assert(arr.min()>=0 and arr.max()<=1)
    out = sp.empty_like(arr)

    # red-green
    out[:,:,0] = arr[:,:,0] - arr[:,:,1]
    # blue-yellow
    out[:,:,1] = arr[:,:,2] - arr[:,:,[0,1]].min(2)
    # intensity
    out[:,:,2] = arr.max(2)

    return out

# # ------------------------------------------------------------------------------
# def chromi_convert(arr):
#     #assert(arr.min()>=0 and arr.max()<=1)

#     opp = oppwalker_convert(arr)
#     out = sp.empty_like(opp[:,:,[0,1]])

#     rg = opp[:,:,0]
#     by lfini= opp[:,:,1]
#     intensity = opp[:,:,2]

#     lowi = intensity < 0.1*intensity.max()
#     rg[lowi] = 0
#     by[lowi] = 0

#     denom = intensity
#     denom[denom==0] = 1
#     out[:,:,0] = rg / denom
#     out[:,:,1] = by / denom

#     return out

# ------------------------------------------------------------------------------
def invE_convert(arr):
    #assert(arr.min()>=0 and arr.max()<=1)
    
    red = arr[:,:,0]
    green = arr[:,:,1]
    blue = arr[:,:,2]

    out = sp.empty_like(arr)

    out[:,:,0] = (red + green + blue) / 3.
    out[:,:,1] = (red + green - 2.*blue) / 4.
    out[:,:,2] = (red - 2.*green + blue) / 4.

    return out
    
# ------------------------------------------------------------------------------
def invW_convert(arr):
    
    #assert(arr.min()>=0 and arr.max()<=1)
    
    invE = invE_convert(arr)

    out = sp.empty_like(arr)
    
    intensity = invE[:,:,0]
    rg = invE[:,:,1]
    yb = invE[:,:,2]

    lowi = intensity < 0.1*intensity.max()
    rg[lowi] = 0
    yb[lowi] = 0

    denom = intensity.copy()
    denom[denom==0] = 1

    out[:,:,0] = intensity
    out[:,:,1] = rg / denom
    out[:,:,2] = yb / denom

    return out

# ------------------------------------------------------------------------------
def color_convert(arr, color_space):
    #assert(arr.min()>=0 and arr.max()<=1)

    # -- insure rgb 
    arr = rgb_convert(arr)
    
    if color_space == 'gray':
        out = gray_convert(arr)
        out.shape = out.shape + (1,)
#     elif color_space == 'rgb':
#         pass
#     elif color_space == 'hsv':
#         pass
    elif color_space == 'oppsande':
        out = oppsande_convert(arr)

    elif color_space == "invE":
        out = invE_convert(arr)
        
    elif color_space == "invW":
        out = invW_convert(arr)


#     elif color_space == 'oppwalker':
#         pass
    elif color_space == 'rgbwhiten':
        out = rgbwhiten_convert(arr)
#     elif color_space == 'rg':
#         pass
    elif color_space == 'irg':
        out = irg_convert(arr)
#     elif color_space == 'chrom':
#         pass
    elif color_space == 'chromi':
        out = chromi_convert(arr)
    else:
        raise ValueError, "'color_space' not understood"

    return out
    

#     # -
#     if color_space == 'rgb':
#         arr_conv = arr
# #     elif color_space == 'rg':
# #         arr_conv = colorconv.rg_convert(arr)
#     elif color_space == 'rg2':
#         arr_conv = colorconv.rg2_convert(arr)
#     elif color_space == 'gray':
#         arr_conv = colorconv.gray_convert(arr)
#         arr_conv.shape = arr_conv.shape + (1,)
#     elif color_space == 'opp':
#         arr_conv = colorconv.opp_convert(arr)
#     elif color_space == 'chrom':
#         arr_conv = colorconv.chrom_convert(arr)
# #     elif color_space == 'opponent':
# #         arr_conv = colorconv.opponent_convert(arr)
# #     elif color_space == 'W':
# #         arr_conv = colorconv.W_convert(arr)
#     elif color_space == 'hsv':
#         arr_conv = colorconv.hsv_convert(arr)
#     else:
#         raise ValueError, "'color_space' not understood"



# def rg_convert(arr):
#     denom = arr.sum(2)
#     denom[denom==0] = 1.
#     out = arr / denom[:,:,None]
#     out = out[:,:,[0,1]]
#     return out    

# def opponent_convert(arr):
#     out = sp.empty_like(arr)
#     r = arr[:,:,0]
#     g = arr[:,:,1]
#     b = arr[:,:,2]
    
#     out[:,:,0] = (r-g) / sp.sqrt(2.)
#     out[:,:,1] = (r+g-2.*b) / sp.sqrt(6.)
#     out[:,:,2] = (r+g+b) / sp.sqrt(3.)

#     return out

# def W_convert(arr):
#     opp = opponent_convert(arr)
#     out = sp.empty_like(opp[:,:,[0,1]])
#     denom = opp[:,:,2]
#     denom[denom==0] = 1.
#     out[:,:,0] = opp[:,:,0] / denom
#     out[:,:,1] = opp[:,:,1] / denom

#     return out

# def W_convert2(arr):
#     opp = opponent_convert(arr)
#     out = sp.empty_like(opp[:,:,[0,1]])
#     intensity = opp[:,:,2]

#     low_intensity = intensity < intensity.max() / 10.

#     intensity[intensity==0] = 1.
#     out[:,:,0] = (opp[:,:,0] / intensity).clip(0, sp.inf)
#     out[:,:,1] = (opp[:,:,1] / intensity).clip(0, sp.inf)

#     out[:,:,:][low_intensity] = 0

#     return out


