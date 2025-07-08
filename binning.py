import numpy as np 
from dklib.plothelp import binCenters

def average(x,y,bins = None, prune_empty = False):
    """
    Compute the binned average and standard deviation of y with respect to x.
    Parameters
    ----------
    x : array_like
        1D array of independent variable values.
    y : array_like
        1D array of dependent variable values; must be the same shape as x.
    bins : int or sequence of scalars, optional
        If an integer, defines the number of equal-width bins between min(x) and max(x).
        If a sequence, defines the bin edges directly.
        If None, defaults to `np.linspace(np.min(x), np.max(x))`.
        Default is None.
    prune_empty : bool, optional
        If True, remove any bins for which the computed average is zero (empty bins).
        Default is False.
    Returns
    -------
    binc : ndarray
        Array of bin center values.
    avg : ndarray
        Array of mean y-values within each bin.
    std : ndarray
        Array of standard deviations of y within each bin.
    Examples
    --------
    >>> x = np.array([0, 1, 2, 3, 4, 5])
    >>> y = np.array([1, 2, 1, 2, 1, 2])
    >>> binc, avg, std = average(x, y, bins=3)
    """

    if(bins is None):
        bins = np.linspace(np.min(x),np.max(x))
    elif(type(bins) is int):
        bins = np.linspace(np.min(x),np.max(x),bins)
    binc = binCenters(bins)
    avg,std = binnedavg(bins,x,y)
    if(prune_empty):
        filt = avg>0
        binc = binc[filt]
        avg = avg[filt]
        std = std[filt]
    return binc,avg,std

    

def binnedavg(tbins,times,quantity,geomean=False):
    counts,wcounts,w2counts = binnedavg_accumulate(tbins,times,quantity,geomean=geomean)
    return binnedavg_compute(counts,wcounts,w2counts,geomean=geomean)

def binnedavg_accumulate(tbins,times,quantity,counts=None,wcounts=None,w2counts=None,geomean:bool =False ):
    q_use = quantity 
    if(geomean):
        np.log(quantity)
    counts_temp,_ = np.histogram(times,bins=tbins)
    wcounts_temp,_ = np.histogram(times,bins=tbins,weights=q_use)
    w2counts_temp,_ = np.histogram(times,bins=tbins,weights=q_use**2)
    
    if(not(counts is None)):
        counts+=counts_temp 
        wcounts+=wcounts_temp 
        w2counts+=w2counts_temp 
        return counts,wcounts,w2counts
    else:
        return counts_temp,wcounts_temp,w2counts_temp

def binnedavg_compute(counts,wcounts,w2counts,geomean:bool=False):    
    filt = counts>0

    wcounts[filt]=wcounts[filt]/counts[filt]
    w2counts[filt] = w2counts[filt]/counts[filt]
    std = np.sqrt(w2counts-wcounts**2)
    if(geomean):
        wcounts = np.exp(wcounts)
        std = np.exp(std)

    return wcounts,std


#given a list of measurements (say from different replicates) of q1s and q2 datas, average q2 in terms of q1s. 
def binnedavg_list(q1list,q2list,q1bins=None):
    if(q1bins is None):
        mmax = np.max([np.max(q1) for q1 in q1list])
        mmin = np.min([np.min(q1) for q1 in q1list])
        q1bins = np.linspace(mmin,mmax) 
    q1binc= binCenters(q1bins)#just to get the size right. 
    q1_counts,q1_wcounts,q1_w2counts = np.zeros(shape = q1binc.shape),np.zeros(shape = q1binc.shape),np.zeros(shape = q1binc.shape)
    for q1,q2 in zip(q1list,q2list):
        binnedavg_accumulate(q1bins,q1,q2,q1_counts,q1_wcounts,q1_w2counts)
    q2_q1,q2_q1_std = binnedavg_compute(q1_counts,q1_wcounts,q1_w2counts)
    return q1bins,q2_q1,q2_q1_std

#from a pair of lists of dataset, compute averages in terms of each other? 
def paired_binnedavg(q1list,q2list,q1bins=None,q2bins=None):
    return binnedavg_list(q1list,q2list,q1bins),binnedavg_list(q2list,q1list,q2bins)#(q1bins,q2_q1,q2_q1_std), (q2bins,q1_q2,q1_q2_std) # first, returns q2 as a function of q1, then returns q1 as a function of q2
