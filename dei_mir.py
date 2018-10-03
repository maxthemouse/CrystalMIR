from __future__ import division
from __future__ import absolute_import
import numpy as np
import numexpr as ne
import os
import os.path
from scipy.optimize import brentq
from scipy.optimize import curve_fit
from numpy import exp, log
from numpy import polyfit
import fnmatch
import re
import warnings
import pyprind
import io_files as io
from itertools import izip
import xrt.backends.raycing.materials as rm
np.seterr(all=u'ignore')  # don't print warnings
warnings.simplefilter(u'ignore', np.RankWarning)


def bkg_subtract(image, flat, dark):
    u""" Subtract background from image.
    (image - dark) / (flat - dark)

    Parameters
    ----------

    image : 2D array (float)
           object image

    flat  : 2D array (float)
            flat field image

    dark  : 2D array (float)
            dark field image

    Returns
    -------
    output : 2D array (float)
             final image
    """
    return ne.evaluate(u"(image - dark) / (flat - dark)")


def calc_dei_2(images, rc, rc_d):
    u"""Calculate the images based on 2 inputs.

    Parameters
    ----------
    images : List (2D array (float))
             input images

    rc     : List (float)
             Rocking curve reflectivity values

    rc_d   : List (float)
             First derivative of rocking curve

    Returns
    -------
    IR     : 2D array (float)
             Absorption image

    deltaR : 2D array (float)
             Refraction image
    """
    I1 = images[0]
    R1, R1p = rc[0], rc_d[0]
    I3 = images[1]
    R3, R3p = rc[1], rc_d[1]
    IR = ne.evaluate(u"(I3*R1p - I1*R3p) / (R3*R1p - R1*R3p)")
    deltaR = ne.evaluate(u"(I1*R3 - I3*R1) / (I3*R1p - I1*R3p)")
    del I1, I3, R1, R1p, R3, R3p  # delete to make linter happy
    return IR, deltaR


def calc_dei_3(images, rc, rc_d, rc_dd):
    u"""Calculate the images based on 3 inputs.

    Parameters
    ----------
    images : List (2D array (float))
             input images

    rc     : List (float)
             Rocking curve reflectivity values

    rc_d   : List (float)
             First derivative of rocking curve

    rc_dd  : List (float)
             Second derivative of rocking curve

    Returns
    -------
    IR     : 2D array (float)
             Absorption image

    deltaR : 2D array (float)
             Refraction image

    sigma2 : 2D array (float)
             ultra-small angle scattering image
    """
    I1 = images[0]
    R1, R1p, R1pp = rc[0], rc_d[0], rc_dd[0]
    I2 = images[1]
    R2, R2p, R2pp = rc[1], rc_d[1], rc_dd[1]
    I3 = images[2]
    R3, R3p, R3pp = rc[2], rc_d[2], rc_dd[2]
    IR = ne.evaluate(u"(I1*(R2p*R3pp-R2pp*R3p) - I2*(R1p*R3pp-R1pp*R3p) + I3*(R1p*R2pp-R1pp*R2p))"
                     u"*(R1*(R2p*R3pp-R2pp*R3p) - R2*(R1p*R3pp-R1pp*R3p) + R3*(R1p*R2pp-R1pp*R2p))**-1")
    deltaR = ne.evaluate(u"-(I1*(R2*R3pp-R2pp*R3) - I2*(R1*R3pp-R1pp*R3) + I3*(R1*R2pp-R1pp*R2))"
                         u"*(I1*(R2p*R3pp-R2pp*R3p) - I2*(R1p*R3pp-R1pp*R3p) + I3*(R1p*R2pp-R1pp*R2p))**-1")
    sigma2 = ne.evaluate(u"2*(I1*(R2*R3p-R2p*R3) - I2*(R1*R3p-R1p*R3) + I3*(R1*R2p-R1p*R2))"
                         u"*(I1*(R2p*R3pp-R2pp*R3p) - I2*(R1p*R3pp-R1pp*R3p) + I3*(R1p*R2pp-R1pp*R2p))**-1"
                         u"- deltaR**2")
    del I1, R1, R1p, R1pp, I2, R2, R2p, R2pp, I3, R3, R3p, R3pp  # delete for linter
    return IR, deltaR, sigma2


def r2(data1, data2):
    u"""Return the r-squared difference between data1 and data2.

    Parameters
    ----------
    data1 : 1D array

    data2 : 1D array

    Returns:
    output: scalar (float)
            difference in the input data
    """
    ss_res = 0.0
    ss_tot = 0.0
    mean = sum(data1) / len(data1)
    for i in xrange(len(data1)):
        ss_res += (data1[i] - data2[i]) ** 2
        ss_tot += (data1[i] - mean) ** 2
    return 1 - ss_res / ss_tot


def search(vals, param, limits, f_min, pts_t, verbose=False):
    u"""search for the points on the rocking curve

    Parameters
    ----------
    vals    : list (float)
              values to search for

    param   : list (float)
              input parameters for fitting function used

    limits  : tuple (float)
              (min, max) limits of region to search

    f_min   : function
              Function to to find the zero of.

    pts_t   : tuple (float)
              (x, y) values of the top position of the curve

    verbose : bool
              Flag for verbose mode

    Returns
    -------
    output : list (float)
             X-values coresponding to the input values
    """
    result = []
    x_t, y_t = pts_t
    for val in vals:
        if val == 1:
            pos = u'top'
            result.append(x_t)
        elif val < 0:
            pos = u"LA"
            # low angle
            a = limits[0]
            b = x_t
            f = f_min(abs(val) * y_t, param)
            if verbose:
                print u"Search limits: a = %s, b = %s" % (a, b)
                print u"F(a) = %s, F(b) = %s" % (f(a), f(b))
            result.append(brentq(f, a, b))
        else:
            pos = u"HA"
            # High angle
            a = x_t
            b = limits[1]
            f = f_min(val * y_t, param)
            if verbose:
                print u"Search limits: a = %s, b = %s" % (a, b)
                print u"F(a) = %s, F(b) = %s" % (f(a), f(b))
            result.append(brentq(f, a, b))
        if verbose:
            print u"val = %s, position = %s" % (unicode(val), pos)
            print u"top = %s, val = %s, search is for %s" %  \
                  (unicode(y_t), unicode(val), unicode(abs(val) * y_t))
    return result


def calc_rc(hkl_val=[2, 2, 0], energy=20000):
    u"""Calculate the rocking curve.

    Parmeters
    ---------
    hkl_val    : list (integer)
                 Three numbers which represent values of h,k,l crystal indices

    energy     : float
                 Energy of the X-rays in eV

    Returns
    -------
    dtheta, rc : tuple (dtheta, rc)

    dtheta     : array (float)
                 angle values

    rc         : array (float)
                 reflectivity values

    The algorithm used is the same as IDL code written by Dean Chapman
    so that the results here are comparible.
    """
    crystal = rm.CrystalSi(hkl=hkl_val)
    dtheta = np.linspace(-60, 60, 801)
    #dt = dtheta[1] - dtheta[0]
    theta = crystal.get_Bragg_angle(energy) + dtheta * 1e-6
    refl = np.abs(crystal.get_amplitude(
        energy, np.sin(theta))[0])**2  # s-polarization
    #rc = np.convolve(refl, refl, 'same') / (refl.sum()*dt) * dt
    refl_sq = refl * refl
    rc = np.convolve(refl_sq, refl, 'same') / refl_sq.sum()
    return dtheta, rc


def get_rc_values(path_name, points, func=u'pearson', verbose=False):
    u"""Get the values from the rocking curve.

        Parameters
        ----------
        path_name : string (os.path)
                    Full path to the rocking curve file

        points    : list (float)
                    list of the rocking curve points, where 1 is the top,
                    negative is low angle and positive is the high angle.
                    example [-0.5, 1, -0.5] are half on each side plus top

        func      : function
                    fitting function to use
                    options are pearson and gauss.
                    default = pearson
                    

        Returns
        -------
        angles : list (float)
                 list of the x-axis values at the points
                 this is the angle if the rocking curve is calculated

        rc     : list (float)
                 list of rocking curve values at those points

        rc_d   : list (float)
                 list of first derivatives at the points

        rc_dd  : list (float)
                 list of second derivatives at the points

        Note that the output rc values are calculated based on the fitting
        function which may be slightly different than the data.
    """
    if func == u'gauss':  # use gaussian
        import gauss as g
        curve = g.gaus
        curve_d = g.gaus_d
        curve_dd = g.gaus_dd
        curve_diff = g.gaus_diff
        curve_d_func = g.gaus_d_func
    else:  # use pearson by default
        import pearson as p
        curve = p.pearson
        curve_d = p.pearson_d
        curve_dd = p.pearson_dd
        curve_diff = p.pearson_diff
        curve_d_func = p.pearson_d_func

    px, py = io.read_rc_file(path_name)
    px = np.array(px)
    py = np.array(py)
    # make a guess of the parameters
    n = len(px)
    mean = sum(px * py) / n
    sigma = sum(py * (px - mean) ** 2) / n
    if func == u'gauss':
        p0 = (max(py), mean, sigma)
    else:  # assume shape is close to Gaussian so use large m
        p0 = (max(py), 4, sigma, mean)
    # do the fit
    popt1, pcov1 = curve_fit(curve, px, py, p0)
    # estimate the center
    p_y_d = curve_d(px, *popt1)
    a = max(list(izip(p_y_d, px)))[1]
    b = min(list(izip(p_y_d, px)))[1]
    x_t = brentq(curve_d_func(popt1), a, b)
    y_t = curve(x_t, *popt1)
    # get the points
    angles = search(points, popt1, (px[0], px[-1]),
                    curve_diff, (x_t, y_t), verbose)
    angles = np.array(angles)
    rc = curve(angles, *popt1)
    rc_d = curve_d(angles, *popt1)
    rc_dd = curve_dd(angles, *popt1)

    return angles, rc, rc_d, rc_dd


def all_files(root, patterns=u'*.tif;*.log', single_level=False, yield_folders=False):
    u"""Walk the directories and return files that match the pattern.
    Used to do the first pass to find the data folders by selecting tif and log files.
    Credit: Robin Parmer, Alex Martelli
    Python Cookbook, 2nd Edition, pg 88

    Parameters
    ----------
    root          : string (os.path)
                    Root directory to start

    patterns      : string (search pattern)
                    patterns to look for using fnmatch
                    ; between values
                    default = '*.tif;*.log'

    single_level  : bool
                    flag to limit to single folder
                    default = False

    yield_folders : bool
                    flag to return folders that match
                    default = False

    Returns
    -------
    output : Generator object
             Yields string (os.path)
    """
    # Expand patterns from semicolon-sparated string to list
    patterns = patterns.split(u';')
    for path, subdirs, files in os.walk(root):
        if yield_folders:
            files.extend(subdirs)
        files.sort
        for name in files:
            for pattern in patterns:
                if fnmatch.fnmatch(name, pattern):
                    yield os.path.join(path, name)
                    break
        if single_level:
            break


def group_files(root, patterns=u'*.tif;*.log', single_level=False, yield_folders=False):
    u"""Group the files according to the directories they were found in.

    Parameters
    ----------
    root          : string (os.path)
                    Root directory to start

    patterns      : string (search pattern)
                    patterns to look for using fnmatch
                    ; between values
                    default = '*.tif;*.log'

    single_level  : bool
                    flag to limit to single folder
                    default = False

    yield_folders : bool
                    flag to return folders that match
                    default = False

    Returns
    -------
    output : dictionary
             key   : string (os.path)
                     directory path
             value : string (os.path)
                     full path to file
    """
    folders = {}
    for path in all_files(root, patterns, single_level, yield_folders):
        folders.setdefault(os.path.dirname(path), []).append(path)
    return folders


re_digits = re.compile(ur'(\d+)')
re_value = re.compile(ur'(0_\d+)')


def under2point(text):
    u"""Replace underscore with point in text.
    This is a simple wrapper to do str.replace. This used to be a more
    general code but was replaced during the transition to python 3.5.

    Parameters
    ----------
    text   : string


    Returns
    -------
    output : string
    """
    return text.replace(u'_', u'.')


def check_folders(folders):
    u"""Read file names and check for input and output folders.

    Parameters
    ----------
    folders : dictionary
              key   : string (os.path)
                      directory path
              value : string (os.path)
                      full path to file

    Returns
    -------
    output : dictionary
             key   : string (os.path)
                     directory path
             value : string (os.path)
                     full path to file
             Return the folders that likely contain data.
             Folders with a .log file are likely output and can be skipped.
             Data folders should have files that contain
             'dark' + 'flat' + 'LA' + 'HA'
    """
    # check for .log files
    # these are in worked data folders and should be skipped
    # look for at least one of 'dark' + 'flat' + 'LA' + 'HA'
    pop_set = set()
    for folder in folders:
        files = folders[folder]
        d = f = l = h = False
        for file_ in files:
            name = os.path.basename(file_).lower()
            if fnmatch.fnmatch(name, u"*.log"):
                pop_set.add(folder)
            if fnmatch.fnmatch(name, u"*dark*"):
                d = True
            if fnmatch.fnmatch(name, u"*flat*"):
                f = True
            if fnmatch.fnmatch(name, u"*la*"):
                l = True
            if fnmatch.fnmatch(name, u"*ha*"):
                h = True
        if not all([d, f, l, h]):
            pop_set.add(folder)
    new_folders = {}
    for folder in folders:
        if folder not in pop_set:
            new_folders[folder] = folders[folder]
    return new_folders


def get_basename(files):
    u"""return the basenames of a list of files

    Parameters
    ----------
    files : list (os.path)
            list of path to file

    Returns
    -------
    output : list (os.path.basename)
             list of basenames of files
    """
    output = []
    for file_ in files:
        output.append(os.path.basename(file_))
    return output


def parse_names(files, basename=True):
    u"""Parse a list of file names and return the separated lists.

    Parameters
    ----------
    files : list (os.path)
            list of path names to data files

    basename : bool
               flag to indicate to return the basenames rather than full paths

    Returns
    -------
    darks     : list (os.path or os.path.basename)
                list of dark field images

    flats     : list (os.path or os.path.basename)
                list of flat field field images

    images    : list (os.path or os.path.basename)
                list of sample images

    positions : list (float)
                list of positions read from the file names

    Note that the output is sorted according to the poistions which does
    not corespond to sorting according to angle on the rocking curve.
    """
    darks = []
    flats = []
    images = []
    positions1 = []
    positions2 = []
    positions = []
    flats_f = []
    images_f = []
    for file_ in files:
        name = os.path.basename(file_).lower()
        m = re.search(re_value, name)
        if name.startswith(u'dark'):
            darks.append(file_)
        elif name.startswith(u'flat'):
            if fnmatch.fnmatch(name, u"*top*"):
                flats.append((1, file_))
                positions1.append(1)
            elif m is not None:
                v = float(under2point(m.group(0)))
                if fnmatch.fnmatch(name, u"*la*"):
                    v = -v
                if fnmatch.fnmatch(name, u"*ha*"):
                    v = abs(v)
                flats.append((v, file_))
                positions1.append(v)
        elif fnmatch.fnmatch(name, u"*top*"):
            images.append((1, file_))
            positions2.append(1)
        elif m is not None:
            v = float(under2point(m.group(0)))
            if fnmatch.fnmatch(name, u"*la*"):
                v = -v
            images.append((v, file_))
            positions2.append(v)
    positions1.sort()
    positions2.sort()
    if positions1 == positions2:
        flats.sort()
        images.sort()
        for flat in flats:
            positions.append(flat[0])
            flats_f.append(flat[1])
        for image in images:
            images_f.append(image[1])
    if basename:
        darks = get_basename(darks)
        flats_f = get_basename(flats_f)
        images_f = get_basename(images_f)
    return darks, flats_f, images_f, positions


def calc_dei_fit(images, angles, rc, PBar=None, Qt=None, Stop=None):
    u"""Calculate the images based on all inputs.

    Parameters
    ----------
    images : list (2D array (float))
             list of input images

    angles : list (float)
             list of angles

    rc     : list (float)
             list of rocking curve reflextivity values

    PBar   : progress bar object
             based on needed methods of qprogressbar

    Returns
    -------
    IR      : 2D array (float32)
              absorption image

    deltaR  : 2D array (float32)
              refraction image

    sigma2  : 2D array (float32)
              ultra-small angle scattering image

    a_img   : 2D array (float)
              image of the A fitting parameter

    b_img   : 2D array (float)
              image of the B fitting parameter

    c_img   : 2D array (float)
              image of the C fitting parameter

    abs_img : 2D array (float)
              absorbtion image (intensity has log scale)

    Print is used for output which will not show up in a Qt GUI.
    """
    # sort the input in order of angles so x-axis is increasing as expected
    z = list(izip(angles, images, rc))
    z.sort()
    angles, images, rc = list(izip(*z))
    x = np.array(angles)
    # print("size of image: " + str(images[0].shape))
    xsize, ysize = images[0].shape
    IR = np.zeros(images[0].shape, dtype=np.float)
    deltaR = np.zeros(images[0].shape, dtype=np.float)
    sigma2 = np.zeros(images[0].shape, dtype=np.float)
    a_img = np.zeros(images[0].shape, dtype=np.float)
    b_img = np.zeros(images[0].shape, dtype=np.float)
    c_img = np.zeros(images[0].shape, dtype=np.float)
    abs_img = np.zeros(images[0].shape, dtype=np.float)
    if PBar is None:
        PBar = ProgressBar(xsize, True)
    PBar.reset()
    for i in xrange(xsize):
        if Stop is not None:  # Stop signal in Qt interface
            if Stop.isChecked():
                break
        # if not i % 10:
        # print "row: " + str(i)
        # sys.stdout.flush()
        for j in xrange(ysize):
            y = []
            for k in xrange(len(images)):
                # fetch the point from the images list and scale by the rocking
                # curve
                y.append(images[k][i, j] * rc[k])
            y = np.array(y)
            popt = polyfit(x, log(y), 2, w=y * y)
            c, b, a = popt
            IR[i, j] = exp(a - (b ** 2 / (4.0 * c)))
            deltaR[i, j] = b / (2.0 * c)
            sigma2[i, j] = -1.0 / (2.0 * c)
            a_img[i, j] = a
            b_img[i, j] = b
            c_img[i, j] = c
            abs_img[i, j] = (b ** 2 / (4.0 * c)) - a
        PBar.setValue(i / float(xsize) * 100.0)
        if Qt is not None:
            Qt.processEvents()
    if Stop is not None:
        Stop.setCheckState(False)
    PBar.setValue(100.0)
    # print(str(PBar))
    # PBar.reset()

    return IR, deltaR, sigma2, a_img, b_img, c_img, abs_img


class ProgressBar(object):
    u"""Progress bar object to provide common method wrapping for
    using qa widget in a gui and pyprind elsewhere."""

    def __init__(self, limit, monitor):
        self.bar = pyprind.ProgBar(limit, monitor)

    def maximum(self, val):
        pass

    def setValue(self, val):
        self.bar.update()

    def reset(self):
        pass

    def show(self):
        pass



    