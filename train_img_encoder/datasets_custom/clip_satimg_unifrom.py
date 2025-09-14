from numba import njit, prange
import numpy as np

@njit(parallel=True)
def clip_satimg(sat_img_np, rcs_girdcoords, size2clip):
    shape = (rcs_girdcoords.shape[0],rcs_girdcoords.shape[1],size2clip,size2clip,3)
    patches = np.empty(shape,dtype=sat_img_np.dtype)
    for i in prange(rcs_girdcoords.shape[0]):
        for j in prange(rcs_girdcoords.shape[1]):
            rb, re, cb, ce = rcs_girdcoords[i, j]
            patches[i, j] = sat_img_np[rb:re, cb:ce,:]
    return patches


