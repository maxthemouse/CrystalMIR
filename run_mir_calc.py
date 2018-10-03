from __future__ import division
from __future__ import absolute_import
import dei_mir as mir
import os
import os.path
import time
import scipy.ndimage
from numpy import log
import ConfigParser
import io_files as io
import filters
from itertools import izip


# set some default values for parameters
hkl_choice = {
    0: (2, 2, 0),
    1: (4, 4, 0),
    }
default_param = {
    u"root_dir": ur"Not/a/Valid/Path",
    u"current_dir": ur"Not/a/Valid/Path",
    u"paths": None,  # should be a dictionary of dir:[file list]
    u"output_dir": u"worked_data",
    u"output_bkg": 0,
    u"crop": [-1, -1, -1, -1],
    u"verbose": 0,
    u"plot_rc": 0,
    u"rc_file": ur"Not/a/Valid/File.txt",
    u"fit_func": u"pearson",
    u"clean_filter": 2,
    u"clean_filter_limits": (0.00001, 10),
    u"clean_offset": 0.25,
    u"median_filter": 2,
    u"median_filter_m": 3,
    u"abs_log": 2,
    u"calc_2": 0,
    u"index_2": [],
    u"calc_3": 0,
    u"index_3": [],
    u"calc_3_gauss": 0,
    u"index_3_gauss": [],
    u"calc_gauss": 0,
    u"save_param_map": 0,
    u"save_dtype": u'float',
    u"save_threshold": 0.0,
    u"output_text": None,
    u"output_bar": None,
    u"output_QtGui.qApp": None,
    u"stop_calc": None,
    u"hkl_index": 0,
    u"rc_energy_kev": 20.0,
    u"calc_rocking_curve": 2,
}
default_param[u"hkl"] = hkl_choice[default_param[u"hkl_index"]]


def no_filter(img, m):
    u"""filter that does nothing

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


def run_calc(param):
    u"""Run the calculations.

    Parameters
    ----------
    param : dictionary
            key  : string
                   parameter name

           value : varies
                   parameter value

    Returns
    -------
    None
    """

    # set the output for text if needed
    if param[u"output_text"] is None:
        param[u"output_text"] = io.Text_Output()
    t_output = param[u"output_text"]
    #  set the root path
    root_dir = param[u"root_dir"]
    if not os.path.isdir(root_dir):
        t_output.showMessage(
            u"Input root folder, %s, is not valid." % root_dir)
        return
    # get the file names automagically
    if param[u"paths"] is not None:
        paths = param[u"paths"]
    else:
        first = mir.group_files(root_dir)
        paths = mir.check_folders(first)
        param[u"paths"] = paths
        if not paths:
            t_output.showMessage(
                u"No data found. Check root path: %s." % param[u"root_dir"])
    # figure out if rc file is present or to be calculated
    # if calculated then do so and save file and set path to load it
    if param[u"calc_rocking_curve"]:
        val_x, val_y = mir.calc_rc(param[u"hkl"], param[u"rc_energy_kev"] * 1000)
        h, k, l = param[u"hkl"]
        e_t1 = "%4.2f" % param[u"rc_energy_kev"]
        e_t2 = e_t1[:2] + 'p' + e_t1[3:]
        rc_out = os.path.join(root_dir, "Rocking_Curve_[%d,%d,%d]@%skeV.txt" % (
            h, k, l, e_t2))
        io.write_rc_file(rc_out, val_x, val_y)
        param[u"rc_file"] = rc_out
    if not os.path.isfile(param[u"rc_file"]):
        t_output.showMessage(
            u"Rocking curve file given, %s, is not valid." % param[u"rc_file"])
        return
    t_output.showMessage(u"Number of data directories: %s" %
                         len(list(paths.keys())))
    for path in list(paths.keys()):
        try:
            param[u"current_dir"] = path
            run_each_path(param)
        except Exception, e:
            t_output.showMessage(unicode(e))
            raise e


def run_each_path(param):
    u"""Run each path

    Parameters
    ----------
    param : dictionary
            key   : string
                    parameter name
            value : varies
                    parameter value

    Returns
    -------
    None
    """
    Qt = param[u"output_QtGui.qApp"]
    if Qt is not None:
        Qt.processEvents()
    # set the output for text if needed
    if param[u"output_text"] is None:
        param[u"output_text"] = io.Text_Output()
    t_output = param[u"output_text"]
    if os.path.isdir(param[u"current_dir"]):
        t_output.showMessage(u"Now working in folder: %s." %
                             param[u"current_dir"])
    else:
        t_output.showMessage(
            u"Input data folder, %s, is not valid." % param[u"current_dir"])
        return
    root_dir = param[u"current_dir"]
    output_dir = param[u"output_dir"]
    error = {}  # dict for error messages
    result_msg = {}  # dict for output from the analyis
    names = param[u"paths"][param[u"current_dir"]]
    darks, flats, images, position = mir.parse_names(names, basename=True)
    # load the data and do background correction
    corrs = []
    d_im, error = io.load_ave_img(root_dir, darks, error, param[u"crop"])
    for image, flat in izip(images, flats):
        i_im = io.load_img(os.path.join(root_dir, image), param[u"crop"])
        f_im = io.load_img(os.path.join(root_dir, flat), param[u"crop"])
        if param[u"clean_filter"]:
            corrs.append(filters.clean_img(mir.bkg_subtract(i_im, f_im, d_im),
                                           param[u"clean_filter_limits"],
                                           param[u"clean_offset"]))
        else:
            corrs.append(mir.bkg_subtract(i_im, f_im, d_im))
    io.mkdir(root_dir, output_dir)
    if param[u"output_bkg"]:
        for i, corr in enumerate(corrs):
            io.save_img(os.path.join(
                root_dir, output_dir, os.path.splitext(images[i])[0] + u"_c" +
                os.path.splitext(images[i])[1]), corr,
                param[u"save_dtype"])
    # get the rocking curve and make the images
    angles, rc, rc_d, rc_dd = mir.get_rc_values(
        param[u"rc_file"], position, param[u"fit_func"],
        param[u"verbose"])
    # sort the input in order of angles
    z = list(izip(angles, rc, rc_d, rc_dd, images, flats))
    z.sort()
    angles, rc, rc_d, rc_dd, images, flats = list(izip(*z))
    # do calculations if the flags are set
    if param[u"calc_2"]:
        l = 0
        h = 1
        fatal = False
        if param[u"verbose"]:
            t_output.showMessage(u"num images: %s" % len(images))
            t_output.showMessage(u"num corrs: %s" % len(corrs))
        if param[u"index_2"]:
            if len(param[u"index_2"]) != 2:
                error[u"Index 2 error"] = \
                    u"Two point calcution requires two points. %s given." \
                    % len(param[u"index_2"])
                fatal = True
            else:
                l, h = param[u"index_2"]
        elif len(images) == 2:
            l = 0
            h = 1
            result_msg[u"Index 2"] = [l, h]
        else:
            found = False
            for i in xrange(len(images)):
                if u"top" in images[i].lower():
                    if param[u"verbose"]:
                        t_output.showMessage(u"top found")
                    found = True
                    l = i - 1
                    h = i + 1
                    result_msg[u"Index 2"] = [l, h]
            if not found:
                error[u"Index 2 Error"] = \
                    u"Can't decide on which images are the side. "  \
                    u"Check file names."
                fatal = True
        if not fatal:
            if param[u"verbose"]:
                t_output.showMessage(u"l=%s, h=%s" % (l, h))
            result_msg[u"calc 2 images"] = [images[l], images[h]]
            error, result_msg = run_dei_2(param,
                                          [corrs[l], corrs[h]],
                                          [rc[l], rc[h]],
                                          [rc_d[l], rc_d[h]],
                                          error, result_msg)
    if param[u"calc_3"]:
        l = 0
        t = 1
        h = 2
        fatal = False
        if param[u"verbose"]:
            t_output.showMessage(u"num images: %s" % len(images))
            t_output.showMessage(u"num corrs: %s" % len(corrs))
        if param[u"index_3"]:
            if len(param[u"index_3"]) != 3:
                error[u"Index 3 error"] = \
                    u"Three point calcution requires two points. %s given." \
                    % len(param[u"index_3"])
                fatal = True
            else:
                l, h = param[u"index_3"]
        elif len(images) == 3:
            l = 0
            t = 1
            h = 2
            result_msg[u"Index 3"] = [l, t, h]
        else:
            found = False
            for i in xrange(len(images)):
                if u"top" in images[i].lower():
                    if param[u"verbose"]:
                        t_output.showMessage(u"top found")
                    found = True
                    l = i - 1
                    t = i
                    h = i + 1
                    result_msg[u"Index 3"] = [l, t, h]
            if not found:
                error[u"Index 3 Error"] = \
                    u"Can't decide on which images are the side and top. "  \
                    u"Check file names."
                fatal = True
        if not fatal:
            if param[u"verbose"]:
                t_output.showMessage(u"l=%s, t=%s, h=%s" % (l, t, h))
            result_msg[u"calc 3 images"] = [images[l], images[t], images[h]]
            error, result_msg = run_dei_3(param,
                                          [corrs[l], corrs[t], corrs[h]],
                                          [rc[l], rc[t], rc[h]],
                                          [rc_d[l], rc_d[t], rc_d[h]],
                                          [rc_dd[l], rc_dd[t], rc_dd[h]],
                                          error, result_msg)

    if param[u"calc_gauss"]:
        if len(images) < 5:
            error[u"calc gauss number error"] = \
                u"Fitting only %s images. Use 5 or more." % len(images)
        else:
            result_msg[u"calc gauss images"] = [images]
            error, result_msg = run_mir_gauss(
                param, corrs, angles, rc, error, result_msg)
    # create a log file and save everything
    config = ConfigParser.ConfigParser()
    config = io.add_to_config(config, u"Parameters", param)
    config = io.add_to_config(config, u"Messages", result_msg)
    config = io.add_to_config(config, u"Errors", error)
    io.write_ini_file(os.path.join(root_dir, output_dir,
                                   u"MIR_calculation.log"), config)


def run_dei_2(param, corrs, rc, rc_d, error, result_msg):
    u"""Run the 2 point calculation

    Parameters
    ----------
    param      : dictionary
                 key   : string
                         parameter name
                 value : varies
                         parameter value
    corrs      : list (2D array (float))
                 list of background corrected images

    rc         : List (float)
                 Rocking curve reflectivity values

    rc_d       : List (float)
                 First derivative of rocking curve
    error      : dictionary
                 key   : string
                         error name
                 value : varies
                         error value
    result_msg : dictionary
                 key   : string
                         message name
                 value : varies
                         message value

    Returns
    -------
    error      : dictionary
                 key   : string
                         error name
                 value : varies
                         error value
    result_msg : dictionary
                 key   : string
                         message name
                 value : varies
                         message value
    """
    # set the output for text if needed
    if param[u"output_text"] is None:
        param[u"output_text"] = io.Text_Output()
    Qt = param[u"output_QtGui.qApp"]
    if Qt is not None:
        Qt.processEvents()
    t_output = param[u"output_text"]
    try:
        t_output.showMessage(u"Beginning 2-point DEI calculation.")
        root_dir = param[u"current_dir"]
        output_dir = param[u"output_dir"]
        start = time.time()
        IR, deltaR = mir.calc_dei_2(corrs, rc, rc_d)
        end = time.time()
        delta = end - start
        rate = corrs[0].size / delta
        if delta < 1:
            delta *= 1000
            result_msg[u"Two point calc time"] = \
                (u"Calculation took %s milliseconds at a rate of %s pixels per second"
                 % (int(delta), int(rate)))
        else:
            result_msg[u"Two point calc time"] = \
                (u"Calculation took %s seconds at a rate of %s pixels per second"
                 % (int(delta), int(rate)))
        m = param[u"median_filter_m"]
        if not param[u"median_filter"]:
            median_filter = no_filter
        else:
            median_filter = scipy.ndimage.median_filter
        if param[u"abs_log"]:
            IR_log = -log(IR)
            io.save_img(os.path.join(root_dir, output_dir,
                                     u"abs_log_2" + u".tif"),
                        median_filter(IR_log, m),
                        param[u"save_dtype"])
        io.save_img(os.path.join(root_dir, output_dir,
                                 u"absorption_2" + u".tif"),
                    median_filter(IR, m),
                    param[u"save_dtype"])
        io.save_img(os.path.join(root_dir, output_dir,
                                 u"refraction_2" + u".tif"),
                    median_filter(deltaR, m),
                    param[u"save_dtype"])

        t_output.showMessage(u"......finished")
    except Exception, e:
        error[u"Error in 2 point calculation"] = unicode(e)
    if Qt is not None:
        Qt.processEvents()
    return error, result_msg


def run_dei_3(param, corrs, rc, rc_d, rc_dd, error, result_msg):
    u"""Run the 3 point calculation

    Parameters
    ----------
    param      : dictionary
                 key   : string
                         parameter name
                 value : varies
                         parameter value
    corrs      : list (2D array (float))
                 list of background corrected images

    rc         : List (float)
                 Rocking curve reflectivity values

    rc_d       : List (float)
                 First derivative of rocking curve

    rc_dd      : List (float)
                 Second derivative of rocking curve

    error      : dictionary
                 key   : string
                         error name
                 value : varies
                         error value
    result_msg : dictionary
                 key   : string
                         message name
                 value : varies
                         message value

    Returns
    -------
    error      : dictionary
                 key   : string
                         error name
                 value : varies
                         error value
    result_msg : dictionary
                 key   : string
                         message name
                 value : varies
                         message value
    """
    # set the output for text if needed
    if param[u"output_text"] is None:
        param[u"output_text"] = io.Text_Output()
    t_output = param[u"output_text"]
    Qt = param[u"output_QtGui.qApp"]
    if Qt is not None:
        Qt.processEvents()
    try:
        t_output.showMessage(u"Beginning 3-point DEI calculation.")
        root_dir = param[u"current_dir"]
        output_dir = param[u"output_dir"]
        start = time.time()
        IR, deltaR, sigma2 = mir.calc_dei_3(corrs, rc, rc_d, rc_dd)
        end = time.time()
        delta = end - start
        rate = corrs[0].size / delta
        if delta < 1:
            delta *= 1000
            result_msg[u"Three point calc time"] = \
                (u"Calculation took %s milliseconds at a rate of %s pixels per second"
                 % (int(delta), int(rate)))
        else:
            result_msg[u"three point calc time"] = \
                (u"Calculation took %s seconds at a rate of %s pixels per second"
                 % (int(delta), int(rate)))
        m = param[u"median_filter_m"]
        if not param[u"median_filter"]:
            median_filter = no_filter
        else:
            median_filter = scipy.ndimage.median_filter
        if param[u"abs_log"]:
            IR_log = -log(IR)
            io.save_img(os.path.join(root_dir, output_dir,
                                     u"abs_log_3" + u".tif"),
                        median_filter(IR_log, m),
                        param[u"save_dtype"])
        io.save_img(os.path.join(root_dir, output_dir,
                                 u"absorption_3" + u".tif"),
                    median_filter(IR, m),
                    param[u"save_dtype"])
        io.save_img(os.path.join(root_dir, output_dir,
                                 u"refraction_3" + u".tif"),
                    median_filter(deltaR, m),
                    param[u"save_dtype"])
        io.save_img(os.path.join(root_dir, output_dir,
                                 u"scatter_3" + u".tif"),
                    median_filter(sigma2, m),
                    param[u"save_dtype"])

        t_output.showMessage(u"......finished")
    except Exception, e:
        error[u"Error in 3 point calculation"] = unicode(e)
    if Qt is not None:
        Qt.processEvents()
    return error, result_msg


def run_mir_gauss(param, corrs, angles, rc, error, result_msg):
    u"""Run the Gaussian fitting calculation

    Parameters
    ----------
    param      : dictionary
                 key   : string
                         parameter name
                 value : varies
                         parameter value
    corrs      : list (2D array (float))
                 list of background corrected images

    angles     : list (float)
                 list of angles

    rc         : List (float)
                 Rocking curve reflectivity values

    rc_d       : List (float)
                 First derivative of rocking curve
    error      : dictionary
                 key   : string
                         error name
                 value : varies
                         error value
    result_msg : dictionary
                 key   : string
                         message name
                 value : varies
                         message value

    Returns
    -------
    error      : dictionary
                 key   : string
                         error name
                 value : varies
                         error value
    result_msg : dictionary
                 key   : string
                         message name
                 value : varies
                         message value
    """
    # set the output for text if needed
    if param[u"output_text"] is None:
        param[u"output_text"] = io.Text_Output()
    Qt = param[u"output_QtGui.qApp"]
    if Qt is not None:
        Qt.processEvents()
    t_output = param[u"output_text"]
    try:
        t_output.showMessage(u"Beginning multi-point DEI calculation.")
        root_dir = param[u"current_dir"]
        output_dir = param[u"output_dir"]
        start = time.time()
        IR, deltaR, sigma2, a, b, c, IR_log = mir.calc_dei_fit(
            corrs, angles, rc, param[u"output_bar"],
            param[u"output_QtGui.qApp"], param[u"stop_calc"])
        end = time.time()
        delta = end - start
        rate = corrs[0].size / delta
        delta = delta / 60.
        result_msg[u"Gaussian fit calc time"] = \
            (u"Calculation took %s minutes at a rate of %s pixels per second"
             % (delta, int(rate)))
        m = param[u"median_filter_m"]
        if not param[u"median_filter"]:
            median_filter = no_filter
        else:
            median_filter = scipy.ndimage.median_filter

        io.save_img(os.path.join(root_dir, output_dir,
                                 u"absorption_fit" + u".tif"),
                    median_filter(IR, m),
                    param[u"save_dtype"])
        io.save_img(os.path.join(root_dir, output_dir,
                                 u"refraction_fit" + u".tif"),
                    median_filter(deltaR, m),
                    param[u"save_dtype"])
        io.save_img(os.path.join(root_dir, output_dir,
                                 u"scatter_fit" + u".tif"),
                    median_filter(sigma2, m),
                    param[u"save_dtype"])
        if param[u"save_param_map"]:
            io.save_img(os.path.join(root_dir, output_dir,
                                     u"a_fit" + u".tif"),
                        median_filter(a, m),
                        param[u"save_dtype"])
            io.save_img(os.path.join(root_dir, output_dir,
                                     u"b_fit" + u".tif"),
                        median_filter(b, m),
                        param[u"save_dtype"])
            io.save_img(os.path.join(root_dir, output_dir,
                                     u"c_fit" + u".tif"),
                        median_filter(c, m),
                        param[u"save_dtype"])
        if param[u"abs_log"]:
            io.save_img(os.path.join(root_dir, output_dir,
                                     u"abs_log_fit" + u".tif"),
                        median_filter(IR_log, m),
                        param[u"save_dtype"])
        t_output.showMessage(u"......finished")
    except Exception, e:
        error[u"Error in Gaussian fitting calculation"] = unicode(e)
    return error, result_msg


def test_run():
    # assume data is a sub-folder and the rocking curve is here
    param = default_param
    param[u"root_dir"] = os.getcwdu()
    param[u"rc_file"] = os.path.join(
        os.getcwdu(),
        u"Rocking_Curve_[2,2,0]@33p39keV_Relative_Angle(Microradians).txt")
    param[u"output_bkg"] = False
    param[u"crop"] = [-1, -1, 0, 900]
    param[u"calc_2"] = True
    param[u"calc_3"] = True
    param[u"verbose"] = False
    param[u"calc_gauss"] = False
    param[u"save_param_map"] = False
    param[u"clean_offset"] = 0.2
    param[u"save_dtype"] = u'uint'
    param[u"save_threshold"] = 0.0
    param[u"abs_log"] = True
    param[u"hkl_index"] =  0
    param[u"rc_energy_kev"] = 33.39
    param[u"calc_rocking_curve"] = True
    run_calc(param)

if __name__ == u"__main__":
    test_run()
