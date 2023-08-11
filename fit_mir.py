import os.path
import warnings

import numpy as np
import pyprind
import scipy.ndimage
from numba import jit
from numpy import exp, linspace, load, log, pi, polyfit, savez_compressed, sqrt

np.seterr(all="ignore")  # don't print warnings
warnings.simplefilter("ignore", np.RankWarning)

# set some program defaults


mir_defaults = {
    "crop": [-1, -1, -1, -1],
    "threshold": 0.1,  # fitting threshold
    "dtype": "float",  # type for saving images
    # filter for images, use no_filter if no filter is desired
    "image_filter": scipy.ndimage.median_filter,
    "filter_width": 3,  # filter width
    "Calc_res": True,  # flag to calculate r-squared, output zeros if false
    "save_images": False,  # save the results as images with each data set
    "save_result_npz": True,  # save the fit results into a npz file
}


def no_filter(img, m, *args):
    """filter that does nothing

    Parameters
    ----------
    img : 2D array (float)
          image

    m   : scalar (float)
          filter width

    Returns
    -------
    output : 2D array (float)
             same image as input
    """
    return img


def calc_dei_fit(images, angles, PBar=None, Qt=None, Stop=None, Tr=0.0, Calc_res=False):
    """Calculate the images based on all inputs.

    Parameters
    ----------
    images : list (2D array (float))
             list of input images

    angles : list (float)
             list of angles

    PBar   : ProgressBar object
             based on needed methods of qprogressbar

    Qt     : PyQt5.QtWidgets.QApplication

    Stop   : PyQt5.QtWidgets.QCheckBox

    Tr     : Scalar (float)
             Threshold for including data point

    Calc_res: flag (Boolean)
              Select if r-squared values should be calculated.

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
              absorption image (intensity has log scale)

    res_img : 2D array (float)
              r-squared image

    area_img: 2D array (float)
              area image (Area = sqrt(2pi) × amplitude × width)

    radio_img: 2D array (float)
               radiograph image (radio = -log(area))

    Print is used for output which will not show up in a Qt GUI.
    """

    x = np.array(angles)
    xsize, ysize = images[0].shape
    IR = np.zeros(images[0].shape, dtype=np.float)
    deltaR = np.zeros(images[0].shape, dtype=np.float)
    sigma2 = np.zeros(images[0].shape, dtype=np.float)
    a_img = np.zeros(images[0].shape, dtype=np.float)
    b_img = np.zeros(images[0].shape, dtype=np.float)
    c_img = np.zeros(images[0].shape, dtype=np.float)
    abs_img = np.zeros(images[0].shape, dtype=np.float)
    res_img = np.zeros(images[0].shape, dtype=np.float)
    area_img = np.zeros(images[0].shape, dtype=np.float)
    radio_img = np.zeros(images[0].shape, dtype=np.float)
    if PBar is None:
        PBar = ProgressBar(xsize, True)
    PBar.reset()
    for i in range(xsize):
        if Stop is not None:  # Stop signal in Qt interface
            if Stop.isChecked():
                break
        # if not i % 10:
        # print "row: " + str(i)
        # sys.stdout.flush()
        for j in range(ysize):
            y = []
            for k in range(len(images)):
                # fetch the point from the images list
                y.append(images[k][i, j])
            y = np.array(y)
            # only fit where signal is larger than given threshold
            Itr = Tr * np.max(y)
            tmp = np.where(y > Itr)
            x1 = x[tmp]
            y1 = y[tmp]
            # polynomial fit
            popt = polyfit(x1, log(y1), 2, w=y1 * y1)
            c, b, a = popt
            IR[i, j] = exp(a - (b ** 2 / (4.0 * c)))
            deltaR[i, j] = -b / (2.0 * c)
            sigma2[i, j] = -1.0 / (2.0 * c)
            a_img[i, j] = a
            b_img[i, j] = b
            c_img[i, j] = c
            abs_img[i, j] = (b ** 2 / (4.0 * c)) - a
            if Calc_res:
                res_img[i, j] = r2(y, np.exp(np.poly1d(popt)(x)))
            area_img[i, j] = IR[i, j] * sqrt(sigma2[i, j] * 2 * pi)
            radio_img[i, j] = -log(area_img[i, j])
        PBar.setValue(i / float(xsize) * 100.0)
        if Qt is not None:
            Qt.processEvents()
    if Stop is not None:
        Stop.setCheckState(False)
    PBar.setValue(100.0)
    # print(str(PBar))
    # PBar.reset()

    return (
        IR,
        deltaR,
        sigma2,
        a_img,
        b_img,
        c_img,
        abs_img,
        res_img,
        area_img,
        radio_img,
    )


class ProgressBar:
    """Progress bar object to provide common method wrapping for
    using qt widget in a gui and pyprind elsewhere."""

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


@jit
def r2(data1, data2):
    """Return the r-squared difference between data1 and data2.

    Parameters
    ----------
    data1 : 1D array

    data2 : 1D array

    Returns
    -------
    output: scalar (float)
            difference in the input data
    """
    ss_res = 0.0
    ss_tot = 0.0
    mean = sum(data1) / len(data1)
    for i in range(len(data1)):
        ss_res += (data1[i] - data2[i]) ** 2
        ss_tot += (data1[i] - mean) ** 2
    return 1 - ss_res / ss_tot


def merge_images(img1, img2, overlap, blend=False):
    """Merge two images vertically.

    Parameters
    ----------
    img1    : 2D array (float32)
              input image 1

    img2    : 2D array (float32)
              input image 2

    overlap : scalar (integer)

    blend   : flag (bool)
              select if blending should be used

    Returns
    -------
    output  : 2D array (float32)
              output image

    """
    (height1, width1) = img1.shape
    (height2, width2) = img2.shape
    # overlap region has to merged somehow
    c1 = img1[height1 - overlap : height1, 0:width1]
    c2 = img2[0:overlap, 0:width2]
    if blend:
        # linear blending
        scale = linspace(0, 1, overlap + 2)
        scale = scale[1:-1]
        c1_t = c1.transpose()
        c1_t *= 1.0 - scale
        c1 = c1_t.transpose()
        c2_t = c2.transpose()
        c2_t *= scale
        c2 = c2_t.transpose()
        c3 = c1 + c2
    else:
        # use average values
        c3 = (c1 + c2) / 2.0
    result = np.vstack(
        (img1[0 : height1 - overlap, 0:width1], c3, img2[overlap:height2, 0:width2])
    )
    return result


def fit_dirs(param, out, image, flat1, flat2, threshold=0.1):
    """Fit flats and images."""
    xaxis = param.get("xaxis", (-10, 10))
    start, stop = xaxis
    pos = linspace(start, stop, len(image))
    param["x-axis"] = pos

    def run_calc(d_path, data, save=True):
        """Run the calculation.

        Parameters
        ----------
        d_path  :  String (os.path)
                   Full path to file
        data    :  list of 2DArray (float)
                   images
        save    :  Flag (Boolean)
                   Save the result.
                   default = True

        Returns
        ------
        result  : list of 2DArray (float)
                  images
        """
        if os.path.exists(d_path):
            results = load(d_path, mmap_mode="r")
            out.showMessage(list(key for key in results.keys()))
            result = [results[key] for key in results.keys()]
        else:
            result = calc_dei_fit(data, pos, Tr=threshold, Calc_res=True)
            if save:
                savez_compressed(d_path, *result)
        return result

    # fit flat1
    result1 = run_calc(os.path.join(param["root_result_path"], "flat_fit.npz"), flat1)
    # fit image
    result2 = run_calc(
        os.path.join(
            param["root_result_path"], "image_fit_{}.npz".format(param["data_dir"])
        ),
        image,
        param["save_result_npz"],
    )
    # fit flat2 if present
    result3 = []
    if param["flat2"]:
        result3 = run_calc(
            os.path.join(param["root_result_path"], "flat2_fit.npz"), flat2
        )
    # combine results1 and results3
    if param["flat2"]:
        result4 = list(map(lambda x, y: (x + y) / 2, result1, result3))
        return result4, result2
    else:
        return result1, result2
