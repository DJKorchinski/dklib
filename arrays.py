import numpy as np 
from numba import jit 

#accepts any scalar, or any list that will 
def listoflists(dimensions):
    lengths = np.array(dimensions,copy=False) if not np.isscalar(dimensions) else np.array([dimensions])
    if(np.size(lengths) > 1):
        return [ listoflists(lengths[1:]) for i in range(0,lengths[0])]
    else:
        return [ [] for i in range(0,lengths[0])]

def flatten_list_of_lists(lst):
    flattened = []
    for sublist in lst:
        flattened.extend(sublist)
    return flattened


#A should be a list of one-d lists (or np arrays)
def flattenragged(A,arrdtype=np.int32):
    lens = np.array([np.shape(a)[0] for a in A],dtype=np.int32)
    starts = np.cumsum(lens) - lens
    # flat  = np.zeros(starts[-1]+lens[-1],arrdtype)
    return arrdtype(np.concatenate(A)),lens,starts

class ragged:
    def __init__(self,A,arrdtype=np.int32):
        self.dat,self.lens,self.inds = flattenragged(A,arrdtype)
    
    #probably want to write some nice indexing routines.
    def __getitem__(self,ind):
        #this might return a copy? 
        return self.dat[self.inds[ind]:self.inds[ind]+self.lens[ind]]
    def __iter__(self):
        yield from [ self[i] for i,l in enumerate(self.lens) ]
    


@jit(nopython=True,cache=True)
def sumreduce(L1,L2,L3):
    for i in range(L1.shape[0]):
        L3[L2[i]]+=L1[i]
    return L3

@jit(nopython=True,cache=True)
def sumreduce_strided(L1,L2,L3,offset,stride):
    for j in range(offset,L1.shape[0],stride):
        L3[L2[j]]+=L1[j]
    return L3



def multiinterp(x : np.ndarray,xp : np.ndarray,fp : np.ndarray,axis = 0):
    """
    Interpolates a tensorial quantity in 1D at the points `x`, given the sampling points `xp` and sampled values `fp`.
    Parameters:
    -----------
    x : np.ndarray
        An array-like object corresponding to the points where interpolation is desired.
    xp : np.ndarray
        A list or array of grid points where the function values are sampled.
    fp : np.ndarray
        An array of shape `(len(xp[0]), len(xp[1]), ...)` representing the function values at the grid points.
    axis : int, optional
        The axis along which to interpolate. Default is 0.
    Returns:
    --------
    np.ndarray
        The interpolated values at the points `x`.
    Notes:
    ------
    - This function performs linear interpolation along the specified axis.
    - The input `xp` is expected to be sorted in ascending order.
    - The function uses broadcasting to handle multidimensional arrays.
    Example:
    --------
    >>> import numpy as np
    >>> x = np.array([2.5, 3.5])
    >>> xp = np.array([1, 2, 3, 4])
    >>> fp = np.array([10, 20, 30, 40])
    >>> multiinterp(x, xp, fp)
    array([25., 35.])
    """
    # print(x)
    inds = np.searchsorted(xp,x,side='right')
    inds = np.clip(inds, 1, len(xp)-1).astype(int)
    #finding the left and right indices.
    xp=np.array(xp)
    xr = xp[inds]
    xl = xp[inds-1]
    yl,yr = np.take(fp, [inds-1, inds], axis=axis)
    #permute the axes, so that the multiplication broadcasts correctly. 
    yl,yr = np.moveaxis(yl,axis,-1), np.moveaxis(yr,axis,-1) 
    #compute the linear interpolation
    y = (xr-x)/(xr-xl) * yl + (x-xl)/(xr-xl) * yr
    #and put the axes back
    y = np.moveaxis(y,axis,-1)
    return y
