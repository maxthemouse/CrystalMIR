import matplotlib.pyplot as plt
import dei_mir as mir
from scipy.optimize import brentq
from scipy.optimize import curve_fit
import numpy as np
import io_files as io


def get_rc_values(path_name, points, func='pearson', verbose=False, plot_rc=False):
    """Get the values from the rocking curve.

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
    if func == 'gauss':  # use gaussian
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
    if plot_rc:
        plt.plot(px, py, 'k-')
        plt.show()
    # make a guess of the parameters
    n = len(px)
    mean = sum(px * py) / n
    sigma = sum(py * (px - mean) ** 2) / n
    if func == 'gauss':
        p0 = (max(py), mean, sigma)
    else:  # assume shape is close to Gaussian so use large m
        p0 = (max(py), 4, sigma, mean)
    # do the fit
    popt1, pcov1 = curve_fit(curve, px, py, p0)
    if verbose:
        print("RC file = %s" % path_name)
        print("The initial fitting parameters are:")
        print(p0)
        print("The final fitting parameters are:")
        print(popt1)
    if plot_rc:
        p_fit = curve(px, *popt1)
        plt.plot(px, p_fit, 'b-')
        plt.show()
    # estimate the center
    p_y_d = curve_d(px, *popt1)
    a = max(list(zip(p_y_d, px)))[1]
    b = min(list(zip(p_y_d, px)))[1]
    x_t = brentq(curve_d_func(popt1), a, b)
    y_t = curve(x_t, *popt1)
    if verbose:
        print("centre = %s, rc value = %s" % (x_t, y_t))
    # get the points
    angles = mir.search(points, popt1, (px[0], px[-1]),
                        curve_diff, (x_t, y_t), verbose)
    angles = np.array(angles)
    rc = curve(angles, *popt1)
    rc_d = curve_d(angles, *popt1)
    rc_dd = curve_dd(angles, *popt1)

    return angles, rc, rc_d, rc_dd
