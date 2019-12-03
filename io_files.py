import os
import os.path
import codecs
import glob
import tifffile as tf
from skimage import exposure
from io import open
from numpy import min, sqrt, tanh, arctanh
from functools import reduce


def add_to_config(config, section, items):
    """Add a section to a configuration.

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
    keys = sorted(items.keys())
    for key in keys:
        config.set(section, key, str(items[key]))
    return config


def write_ini_file(path, config):
    """Write ini file

    Parameters
    ----------
    path    : String (os.path)
              full path and name to save file

    config  : ConfigParser.ConfigParser()

    Returns
    -------
    None
    """

    config.write(codecs.open(path, "w", "utf-8"))


def load_img(path_name, crop=None):
    """ Load image from file.
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
    im = tf.imread(path_name).astype("float32")
    height, width = im.shape
    if crop is None:
        return im
    l, r, t, b = crop
    if l < 0:
        l = 0
    if r < 0:
        r = width
    if t < 0:
        t = 0
    if b < 0:
        b = height
    im = im[t:b, l:r]
    return im


def load_ave_img(root_path, names, error, crop=None):
    """Load files from a list and return the average.

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
        error["Dark file error"] = "Can't open dark files. Check name."
    else:
        result /= float(num)
    return result, error


def save_img(path_name, im, dtype="float"):
    """ Save image as 32 bit float tiff.

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
    if dtype == "ubyte":
        im += abs(im.min())
        im = exposure.rescale_intensity(im, out_range="uint8")
        im = im.astype("uint8")
    elif dtype == "uint":
        im += abs(im.min())
        im = exposure.rescale_intensity(im, out_range="uint16")
        im = im.astype("uint16")
    else:
        im = im.astype("float32")
    tf.imsave(path_name, im)


def mkdir(output_dir):
    """ make a directory if it does not already exist

    Parameters
    ----------
    output_dir   : String (os.path)
                   Full path to folder.

    Returns
    -------
    None
    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)


def read_rc_file(name):
    """ Read rocking curve file.
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
    with open(name, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                pair = line.split()
                x.append(float(pair[0]))
                y.append(float(pair[1]))
    return x, y


def write_rc_file(path_name, X, Y):
    """Write rocking curve file.

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

    with open(path_name, "w") as f:
        for x, y in zip(X, Y):
            f.write("%15.9f     %15.9f\n" % (x, y))


class Text_Output(object):
    """Output object to use for printing. Uses same methods as Qstatusbar
    so it can be a substiute."""

    def clearMessage(self):
        pass

    def showMessage(self, text, timeout=0):
        print(text)


def sub_dark(image, dark, mode=0):
    """Subtract dark signal from image and shift to avoid negative values.
    mode 0: adjust zero level
    mode 1: subtract dark and adjust zero level
    mode 2: do nothing

    Parameters
    ----------
    image  : 2D array (float)
             image

    dark   : 2D array (float)
             image

    mod    : scalar (int)
             set mode of action

    Returns
    -------
    result  : 2D array (float)
              image
    """
    if mode == 2:
        return image
    if mode == 0:
        result = image
    if mode == 1:
        result = image - dark
    m = min(result)
    if m < 0:
        result = result + abs(m) + 1
    return result


def mibeta(r2_values):
    """Multiple imputation estimate. Combine R-squraed values of various
    inputs that are combined into one model. Without going into detail,
    the MI estimate of a parameter (e.g. a regression coefficient) is
    the average of the estimated coefficients from the MI datasets. The
    MI estimate of the standard error of a parameter is calculated based
    on the standard error of the coefficient in the individual
    imputations (sometimes called the within imputation variance) and
    the degree to which the coefficient estimates vary across the
    imputations (the between imputation variance)

    R2 is (among other things) the squared correlation (denoted r) between
    the observed and expect values of the dependent variable, in equation
    form: r = sqrt(R2). As mentioned above, the MI estimate of a parameter
    is typically the mean value across the imputations, and this method can
    be used to estimate the R2 for an MI model. However, because of the way
    values of R2 are distributed, directly averaging the values may not be
    the most appropriate method of calculating the central tendency (i.e.
    mean) of the distribution. It is possible to transform correlation
    coefficients so that the mean becomes a more reasonable estimate of
    central tendency.

    Harel (2009) suggests using Fisher’s r to z transformation when
    calculating MI estimates of R2 and adjusted R2. Harel’s method is to
    first estimate the model and calculate the R2 and/or adjusted R2 in each
    of the imputed datasets. Each model R2 is then transformed into a
    correlation (r) by taking its square-root. Fisher’s r to z
    transformation is then used to transform each of the r values into a z
    value. The average z across the imputations can then be calculated.
    Finally, the mean of the z values is transformed back into an R2. Harel
    writes that the technique works best when the number of imputations is
    large. Harel also notes that as with any number of statistical
    procedures, this method works best in large samples.

    https://stats.idre.ucla.edu/stata/faq/how-can-i-estimate-r-squared-for-a-model-estimated-with-multiply-imputed-data/
    Harel, O. (2009). The estimation of R2 and adjusted R2 in incomplete data sets using multiple imputation.
    Journal of Applied Statistics, 36(10), 1109-1118.

    z = atanh(r)
    r = tanh(z)

    Parameters
    ----------
    r2_values  :  list of 2DArray (float)
                  image of r2 values

    Result
    ------
    mibeta     :  2DArray (float)
                  image

    """
    N = len(r2_values)
    r = (sqrt(a) for a in r2_values)
    z = (arctanh(a) for a in r)
    ave_z = reduce(lambda a, b: a + b, z) / N
    ave_r = tanh(ave_z)
    return ave_r ** 2


def create_paths(param, out):
    """Create a number of folders and/or set paths for input and output.

    Parameters
    ----------
    param : Dictionary
            Program parameters
    out   : output class (Qstatusbar, Text_Output)

    Returns
    ------
    None  : Input dictionary is mutable
    """
    # create output paths
    param["result_path"] = os.path.join(param["root_dir"], param["data_dir"], "result")
    mkdir(param["result_path"])
    param["root_result_path"] = os.path.join(param["root_dir"], "result")
    mkdir(param["root_result_path"])      
    param["abs_path"] = os.path.join(param["root_result_path"], "Abs")
    mkdir(param["abs_path"])
    param["abs_log_path"] = os.path.join(param["root_result_path"], "Abs_log")
    mkdir(param["abs_log_path"])
    param["refract_path"] = os.path.join(param["root_result_path"], "Refraction")
    mkdir(param["refract_path"])
    param["scatter_path"] = os.path.join(param["root_result_path"], "USAXS")
    mkdir(param["scatter_path"])
    param["resid_path"] = os.path.join(param["root_result_path"], "R2")
    mkdir(param["resid_path"])
    # paths for input
    param["darks_path"] = os.path.join(param["root_dir"], param["dark_dir"], "darks")
    out.showMessage(param["darks_path"])
    out.showMessage(os.path.isdir(param["darks_path"]))
    param["flat_path"] = os.path.join(param["root_dir"], param["flat_dir"], "flat")
    out.showMessage(param["flat_path"])
    out.showMessage(os.path.isdir(param["flat_path"]))
    if param["flat2_dir"]:
        param["flat2"] = True
        param["flat2_path"] = os.path.join(
            param["root_dir"], param["flat2_dir"], "flat"
        )
        out.showMessage(param["flat2_path"])
        out.showMessage(os.path.isdir(param["flat2_path"]))
    else:
        param["flat2"] = False
    param["image_path"] = os.path.join(param["root_dir"], param["data_dir"], "image")
    out.showMessage(param["image_path"])
    out.showMessage(os.path.isdir(param["image_path"]))


def load_data(param, crop):
    """Load dark, flat and image files. May return empty list.

    Parameters
    ----------
    param  : Dictionary
             Program parameters

    crop   : scalars (list of int)

    Returns
    ------
    image       : list of 2D array (float)
                  image

    flat_image  : list of 2D array (float)
                  image

    flat2_image : list 2D array (float)
                  image

    """
    # create dark image
    darks = sorted(glob.glob(os.path.join(param["darks_path"], "*.tif")))
    d_names = [os.path.basename(n) for n in darks]
    dark_image, err = load_ave_img(param["darks_path"], d_names, {}, crop)
    # create flat images
    mode = 1
    flats = sorted(glob.glob(os.path.join(param["flat_path"], "*.tif")))
    flat_image = [sub_dark(load_img(f, crop), dark_image, mode) for f in flats]
    # create images
    images = sorted(glob.glob(os.path.join(param["image_path"], "*.tif")))
    image = [sub_dark(load_img(f, crop), dark_image, mode) for f in images]
    # create flat2 images
    if param["flat2"]:
        flats2 = sorted(glob.glob(os.path.join(param["flat2_path"], "*.tif")))
        flat2_image = [sub_dark(load_img(f, crop), dark_image, mode) for f in flats2]
    else:
        flat2_image = []
    return image, flat_image, flat2_image


def save_results(param, result1, result2):
    """Save all of the results as images.

    Parameters
    ----------
    param  : Dictionary
             Program parameters

    result1 : list of 2D array (float)
              results of calc_dei_fit function

    result2 : list of 2D array (float)
              results of calc_dei_fit function

    Returns
    -------
    None
    """
    dtype = param["dtype"]
    _filter = param["image_filter"]
    m = param["filter_width"]
    if param["save_images"]:
        for i in range(8):
            save_img(
                os.path.join(param["result_path"], "flat_{:02d}.tif".format(i)),
                _filter(result1[i], m),
                dtype,
            )
            save_img(
                os.path.join(param["result_path"], "image_{:02d}.tif".format(i)),
                _filter(result2[i], m),
                dtype,
            )
    save_img(
        os.path.join(param["abs_path"], "Abs_{}.tif".format(param["data_dir"])),
        _filter(result2[0] / result1[0], m),
        dtype,
    )
    save_img(
        os.path.join(param["abs_log_path"], "Abs_log_{}.tif".format(param["data_dir"])),
        _filter(result2[6] - result1[6], m),
        dtype,
    )
    save_img(
        os.path.join(param["refract_path"], "DPC_{}.tif".format(param["data_dir"])),
        _filter(result2[1] - result1[1], m),
        dtype,
    )
    save_img(
        os.path.join(param["scatter_path"], "USAXS_{}.tif".format(param["data_dir"])),
        _filter(result2[2] - result1[2], m),
        dtype,
    )
    save_img(
        os.path.join(param["resid_path"], "R2_{}.tif".format(param["data_dir"])),
        _filter(mibeta([result2[7], result1[7]]), m),
        dtype,
    )
