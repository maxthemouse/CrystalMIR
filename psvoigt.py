u"""

Pseudo-Voigt function, linear combination of Gaussian and Lorentzian with different FWHM.

y = 


"""

from __future__ import division
from __future__ import absolute_import
from numpy import exp, pi


def psvoigt(x,y0,A,x0,WL,WG,mu):

    return y0 + A*((mu*WL**2)/((x-x0)**2+WL**2)
                   + (1-mu)*exp(-(x - x0) ** 2 / (2 * WG ** 2)))
