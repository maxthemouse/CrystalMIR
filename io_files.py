from __future__ import with_statement
from __future__ import absolute_import
import os
import os.path
import tifffile as tf
from skimage import exposure
from io import open
import codecs


def add_to_config(config, section, items):
    u"""Add a section to a configuration.

    Parameters
    ----------
    config  : ConfigParser.ConfigParser()

    section : String

    items   : Dictionary

              key   : String
                      parameter name

              value : String
                      value of parameter
    Returns
    -------
    output: config

    Input is a section name and a dict of key:values to add.
    Keys are sorted before being added.
    Output is the configuration.
    """
    config.add_section(section)
    keys = list(items.keys())
    keys.sort()
    for key in keys:
        config.set(section, key, items[key])
    return config


def write_ini_file(path, config):
    u"""Write ini file

    Parameters
    ----------
    path    : String (os.path)
              full path and name to save file

    config  : ConfigParser.ConfigParser()

    Returns
    -------
    None
    """

    config.write(codecs.open(path, u'w', u'utf-8'))


def load_img(path_name, crop=None):
    u""" Load image from file.
        Option to crop the image.

    Parameters
    ----------
    path_name : String (os.path)
                Full path to file.

    crop      : list or tuple or None
                [LEFT, RIGHT, TOP, BOTTOM]
                -1 indicates to use limit
                if None no cropping is done

    Returns
    -------
    output : 2D array (float32)
             image
    """
    # im = tf.imread(os.path.join(root_dir,name + ".tif")).astype('float32')
    im = tf.imread(path_name).astype(u'float32')
    height, width = im.shape
    if crop is None:
        return im
    l, r, t, b = crop
    if l < 0:
        l = 0
    if r < 0:
        r = width - 1
    if t < 0:
        t = 0
    if b < 0:
        b = height - 1
    im = im[t:b, l:r]
    return im


def load_ave_img(root_path, names, error, crop=None):
    u"""Load files from a list and return the average.

    Parameters
    ----------
    root_path : String (os.path)
                Full path to folder.

    names     : List (String)
                File names

    crop      : list (int), tuple (Int), None
                [LEFT, RIGHT, TOP, BOTTOM]
                -1 indicates to use limit
                if None no cropping is done

    Returns
    -------
    output : 2D array (float32)
             image
    """
    result = None
    num = len(names)
    for name in names:
        im = load_img(os.path.join(root_path, name), crop)
        if result is None:
            result = im
        else:
            result += im
    if result is None:
        error[u"Dark file error"] = u"Can't open dark files. Check name."
    else:
        result /= float(num)
    return result, error


def save_img(path_name, im, dtype=u'float'):
    u""" Save image as 32 bit float tiff.

    Parameters
    ----------
    path_name : string (os.path)
                full path to file location

    im        : 2D array (float)
                image

    dtype     : string
                type to save image values
                possible values = float, ubyte, uint
                output types = float32, uint8, uint16
                default = float

    Returns
    -------
    None

    Note. Files are converted to float32 as float64 might be the
    returned type from math functions.
    """
    if dtype == u'ubyte':
        im += abs(im.min())
        im = exposure.rescale_intensity(im, out_range=u'uint8')
        im = im.astype(u'uint8')
    elif dtype == u'uint':
        im += abs(im.min())
        im = exposure.rescale_intensity(im, out_range=u'uint16')
        im = im.astype(u'uint16')
    else:
        im = im.astype(u'float32')
    tf.imsave(path_name, im)


def mkdir(root_dir, output_dir):
    u""" make a directory if it does already exist

    Parameters
    ----------
    root_dir   : String (os.path)
                 Full path to folder.

    output_dir : String (os.path)
                 Full path to folder.

    Returns
    -------
    None
    """
    if not os.path.isdir(os.path.join(root_dir, output_dir)):
        os.makedirs(os.path.join(root_dir, output_dir))


def read_rc_file(name):
    u""" Read rocking curve file.
    Data is assumed to be in columns with tab or space.

    Parameters
    ----------
    name : String (os.path)
           Full path to input file

    Returns
    -------
    x : List (float)
        X-values

    y : List (float)
        Y-values
    """
    x = []
    y = []
    with open(name, u'r') as f:
        for line in f:
            line = line.strip()
            if line:
                pair = line.split()
                x.append(float(pair[0]))
                y.append(float(pair[1]))
    return x, y


def write_rc_file(path_name, X, Y):
    u"""Write rocking curve file.

    Parameters
    ----------
    path_name : String (os.path)
                Full path to output file

    X         : array (float)
                x values

    Y         : array (float)
                y values

    Returns
    -------
    None
    """

    with open(path_name, u'w') as f:
        for x, y in zip(X, Y):
            f.write(u"%15.9f     %15.9f\n" % (x, y))


class Text_Output(object):
    u"""Output object to use for printing. Uses same methods as Qstatusbar
    so it can be a substiute."""

    def clearMessage(self):
        pass

    def showMessage(self, text, timeout=0):
        print text
