""" Some testing for upgrading etc."""

import numpy as np
from numpy.polynomial import polynomial, Polynomial
from numpy import polyfit

def func(a):
    return 0.5 + 2.0*a + 0.2 * a**2

x = range(-10, 10)
x = np.array(x)
y = func(x)
print(y)

# old way
print('Old way')
popt1 = polyfit(x, y, 2, w=y * y)
print(f'coeff:{popt1}')
p1 = np.poly1d(popt1)
print(p1)


# new way
print('New way')
popt2 = polynomial.polyfit(x, y, 2, w=y*y)
print(f'coeff:{popt2}')
p2 = Polynomial(popt2)
print(p2)