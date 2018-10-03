u"""
Gaussian Function
f(x) == a*e^(-1/2*(x - x0)^2/sigma^2)

first derivative
f'(x) == -a*(x - x0)*e^(-1/2*(x - x0)^2/sigma^2)/sigma^2

second derivative
f''(x) == -a*e^(-1/2*(x - x0)^2/sigma^2)/sigma^2 + a*(x - x0)^2*e^(-1/2*(x - x0)^2/sigma^2)/sigma^4
"""

from __future__ import division
from __future__ import absolute_import
from numpy import exp


def gaus(x, a, x0, sigma):
    u"""Calculate Gaussian function.

    Parameters
    ----------
    a     : scalar (float)
            amplitude

    x0    : scalar (float)
            centroid

    sigma : scalar (float)
            width

    Returns
    -------
    output : scalar (float)
             Function value
    """
    return a * exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def gaus_d(x, a, x0, sigma):
    u"""Calculate the first derivative of Gaussian function.

    Parameters
    ----------
    a     : scalar (float)
            amplitude

    x0    : scalar (float)
            centroid

    sigma : scalar (float)
            width

    Returns
    -------
    output : scalar (float)
             First derivative value
    """
    return (-a * (x - x0) *
            exp((-1 / 2 * (x - x0) ** 2 / sigma ** 2) / sigma ** 2))


def gaus_dd(x, a, x0, sigma):
    u"""Return the second derivative of Gaussian function

    Parameters
    ----------
    a     : scalar (float)
            amplitude

    x0    : scalar (float)
            centroid

    sigma : scalar (float)
            width

    Returns
    -------
    output : scalar (float)
             Second Derivative value
    """
    return (-a * exp((-1 / 2 * (x - x0) ** 2 / sigma ** 2) / sigma ** 2) +
            a * (x - x0) ** 2 *
            exp((-1 / 2 * (x - x0) ** 2 / sigma ** 2) / sigma ** 4))


def gaus_diff(value, param):
    u""" Return a function, G(x), G(x) = F(x) - value
    F(x) is a Gaussian function

    Parameters
    ----------
    value : scalar (float)

    param : tuple (float)
            Three values of (a, x0, sigma)

            a     : scalar (float)
                    amplitude

            x0    : scalar (float)
                    centroid

            sigma : scalar (float)
                    width

    Returns
    -------
    output : function
    """
    a, x0, sigma = param

    def f(x):
        return gaus(x, a, x0, sigma) - value
    return f


def gaus_d_func(param):
    u"""Return a function, f(x), that returns the derivative of the gaussian
    function at x.

    Parameters
    ----------
    param : tuple (float)
            Three values of (a, x0, sigma)

            a     : scalar (float)
                    amplitude

            x0    : scalar (float)
                    centroid

            sigma : scalar (float)
                    width

    Returns
    -------
    output : function
    """
    a, x0, sigma = param

    def f(x):
        return gaus_d(x, a, x0, sigma)
    return f
