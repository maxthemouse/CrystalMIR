"""
This was an experimental attempt to use a reference area to adjust
parameters between views. This actually caused greater banding.
The method used or having params in the calculation might be useful.
"""

import os
from collections import namedtuple

import numpy as np
from numpy import exp, linspace, load, log, median, pi, polyfit, savez_compressed, sqrt

from fit_mir import ProgressBar, r2

Adjustment = namedtuple("Adjustment", "A B C")


def fit_dirs_ref(param, out, image, flat1, flat2, threshold=0.1):
    """Fit flats and images."""
    start = -10
    stop = 10
    pos = linspace(start, stop, len(image))
    param["x-axis"] = pos
    zero = Adjustment(0, 0, 0)
    if param["set_ref"]:
        param["adjustment"] = zero
        param["apply_adj"] = True
    else:
        param["apply_adj"] = True

    def run_calc(d_path, data, param, save=True):
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
            result = calc_dei_fit_ref(data, pos, param, Tr=threshold, Calc_res=True)
            if save:
                savez_compressed(d_path, *result)
        return result

    # fit flat1
    result1 = run_calc(
        os.path.join(param["root_result_path"], "flat_fit.npz"), flat1, param
    )
    # fit image
    result2 = run_calc(
        os.path.join(
            param["root_result_path"], "image_fit_{}.npz".format(param["data_dir"])
        ),
        image,
        param,
        param["save_result_npz"],
    )
    # fit flat2 if present
    result3 = []
    if param["flat2"]:
        result3 = run_calc(
            os.path.join(param["root_result_path"], "flat2_fit.npz"), flat2, param
        )
    # combine results1 and results3
    if param["flat2"]:
        result4 = list(map(lambda x, y: (x + y) / 2, result1, result3))
        return result4, result2
    else:
        return result1, result2


def calc_dei_fit_ref(
    images, angles, param, PBar=None, Qt=None, Stop=None, Tr=0.0, Calc_res=False
):
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
    IR = np.zeros(images[0].shape, dtype=float)
    deltaR = np.zeros(images[0].shape, dtype=float)
    sigma2 = np.zeros(images[0].shape, dtype=float)
    a_img = np.zeros(images[0].shape, dtype=float)
    b_img = np.zeros(images[0].shape, dtype=float)
    c_img = np.zeros(images[0].shape, dtype=float)
    abs_img = np.zeros(images[0].shape, dtype=float)
    res_img = np.zeros(images[0].shape, dtype=float)
    area_img = np.zeros(images[0].shape, dtype=float)
    radio_img = np.zeros(images[0].shape, dtype=float)
    if PBar is None:
        PBar = ProgressBar(xsize, True)
    PBar.reset()
    two_pi = 2 * pi
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
            b_c_term = b**2 / (4.0 * c)
            IR[i, j] = exp(a - b_c_term)
            deltaR[i, j] = -b / (2.0 * c)
            sigma2[i, j] = -1.0 / (2.0 * c)
            a_img[i, j] = a
            b_img[i, j] = b
            c_img[i, j] = c
            abs_img[i, j] = b_c_term - a
            if Calc_res:
                res_img[i, j] = r2(y, np.exp(np.poly1d(popt)(x)))
            area_img[i, j] = IR[i, j] * sqrt(sigma2[i, j] * two_pi)
            radio_img[i, j] = -log(area_img[i, j])
        PBar.setValue(i / float(xsize) * 100.0)
        if Qt is not None:
            Qt.processEvents()
    if Stop is not None:
        Stop.setCheckState(False)
    PBar.setValue(100.0)
    # work on reference
    img_list = [a_img, b_img, c_img]
    if param["set_ref"]:
        result = [0, 0, 0]
        for index, img in enumerate(img_list):
            ref_img = extract_ref(param, img)
            value = extract_adjustment(ref_img)
            result[index] = value
        param["adjustment"] = Adjustment(result[0], result[1], result[2])
    if param["apply_adj"]:
        ref_value = param["adjustment"]
        result = [0, 0, 0]
        for index, img in enumerate(img_list):
            ref_img = extract_ref(param, img)
            value = extract_adjustment(ref_img)
            result[index] = value
        delta_a = result[0] / ref_value.A
        delta_b = ref_value.B - result[1]
        # delta_c = ref_value.C - result[2]
        # adjust images
        for i in range(xsize):
            for j in range(ysize):
                a = a_img[i, j] * delta_a
                b = b_img[i, j] + delta_b
                # c = c_img[i,j] + delta_c
                b_c_term = b**2 / (4.0 * c)
                IR[i, j] = exp(a - b_c_term)
                deltaR[i, j] = -b / (2.0 * c)
                sigma2[i, j] = -1.0 / (2.0 * c)
                a_img[i, j] = a
                b_img[i, j] = b
                c_img[i, j] = c
                abs_img[i, j] = b_c_term - a
                if Calc_res:
                    res_img[i, j] = r2(y, np.exp(np.poly1d(popt)(x)))
                area_img[i, j] = IR[i, j] * sqrt(sigma2[i, j] * two_pi)
                radio_img[i, j] = -log(area_img[i, j])
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


def extract_ref(param, im):
    """
    Extract the reference from the image.
    """
    height, width = im.shape
    crop = param.get("ref")
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


def extract_adjustment(im):
    """
    Find values of fits to make adjustment
    """
    return median(im)
