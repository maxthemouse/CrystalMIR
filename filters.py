from __future__ import division
from __future__ import absolute_import
import numpy as np
import scipy.ndimage


def clean_img(ws, limits=None, offset=0.0):
    u"""Clean an image to remove 'bad' points.
    Remove things like NAN and infinity. If limits is given use those
    values to trim the data to the threshold.

    Parameters
    ----------
    ws     : 2D array (float)
             image

    limits : tuple (float), None
             (min, max) values to clip the data
             None uses no limits

    offset : scalar (float)
             add an offset value to ws
    """
    idx = np.isnan(ws)
    ws[idx] = 0.0
    idx = np.isinf(ws)
    ws[idx] = 0.0
    idx = np.isneginf(ws)
    ws[idx] = 0.0
    ws += offset
    if limits is not None:
        min, max = limits
        np.clip(ws, min, max, ws)
    return ws


def reject_outliers(data, m=2., size=9):
    u"""Reject outliers
    Remove outliers based on the median and the abolute distance to the
    median. Parameter m (default = 2) controls the range to keep.

    Parameters
    ----------
    data : array (number)

    m    : scalar (number)
           distance from median to use as the cutoff

    size : scalar (integer)
           size of region to use for median

    Returns
    -------
    output : array (number)
    """
    data_median = scipy.ndimage.median(data, size)
    d = np.abs(data - data_median)
    mdev = np.median(d)
    s = d / mdev if mdev else 0
    mask = s > m
    data[mask] = 0
    return data
