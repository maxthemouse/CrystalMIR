u"""
Pearson VII function
f(x) == c*((u - x)^2/(a^2*m) + 1)^(-m)

first derivative
f'(x) == 2*c*(u - x)*((u - x)^2/(a^2*m) + 1)^(-m - 1)/a^2

second derivative
f''(x) == -2*c*((u - x)^2/(a^2*m) + 1)^(-m - 1)/a^2 + 4*c*(m + 1)*(u - x)^2*((u - x)^2/(a^2*m) + 1)^(-m - 2)/(a^4*m)
"""


from __future__ import division
def pearson(x, c, m, a, mu):
    u"""Calculate Pearson VII function

    Parameters
    ----------
    c  : scalar (float)
         0 < c <= 1

    m  : scalar (float)
         m > 0

    a  : scalar (float)
         a > 0

    mu : scalar (float)
         centroid

    Returns
    -------
    output : scalar (float)
             Function value
    """
    return c * (1 + (x - mu) ** 2 / (m * a ** 2)) ** -m


def pearson_d(x, c, m, a, mu):
    u"""Return the first derivative of Pearson VII function

       Parameters
       ----------
       c  : scalar (float)
            0 < c <= 1

       m  : scalar (float)
            m > 0

       a  : scalar (float)
            a > 0

       mu : scalar (float)
            centroid

       Returns
       -------
       output : scalar (float)
                First derivative value
    """
    return (2 * c * (mu - x) *
            ((mu - x) ** 2 / (a ** 2 * m) + 1) ** (-m - 1) / a ** 2)


def pearson_dd(x, c, m, a, mu):
    u"""Return the second derivative of Pearson VII function

       Parameters
       ----------
       c  : scalar (float)
            0 < c <= 1

       m  : scalar (float)
            m > 0

       a  : scalar (float)
            a > 0

       mu : scalar (float)
            centroid

       Returns
       -------
       output : scalar (float)
                Second derivative value
    """

    return (-2 * c * ((mu - x) ** 2 / (a ** 2 * m) + 1) ** (-m - 1) / a ** 2 +
            4 * c * (m + 1) * (mu - x) ** 2 *
            ((mu - x) ** 2 / (a ** 2 * m) + 1) ** (-m - 2) / (a ** 4 * m))


def pearson_diff(value, param):
    u""" Return a function, G(x), G(x) = F(x) - value
    F(x) is a Pearson VII function

    Parameters
    ----------
    value : scalar (float)

    param : tuple (float)
            Four values of (c, m, a, mu)

            c  : scalar (float)
                 0 < c <= 1

            m  : scalar (float)
                 m > 0

            a  : scalar (float)
                 a > 0

            mu : scalar (float)
                 centroid

    Returns
    -------
    output : function
    """
    c, m, a, mu = param

    def f(x):
        return pearson(x, c, m, a, mu) - value
    return f


def pearson_d_func(param):
    u"""Return a function, f(x), that returns the derivative of the pearson
    function at x.

    Parameters
    ----------
    value : scalar (float)

    param : tuple (float)
            Four values of (c, m, a, mu)

            c  : scalar (float)
                 0 < c <= 1

            m  : scalar (float)
                 m > 0

            a  : scalar (float)
                 a > 0

            mu : scalar (float)
                 centroid

    Returns
    -------
    output : function
    """
    c, m, a, mu = param

    def f(x):
        return pearson_d(x, c, m, a, mu)
    return f
