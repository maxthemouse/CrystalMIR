# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import xrt.backends.raycing.materials_crystals as rmc
from scipy.interpolate import UnivariateSpline

# path to xrt:
import os, sys

sys.path.append(os.path.join("..", "..", ".."))  # analysis:ignore

crystal = rmc.Si(hkl=(2, 2, 0))
E = 33300
# crystal, E = rm.CrystalFromCell(xtl), 33300
dtheta = np.linspace(-30, 40, 601)
dt = dtheta[1] - dtheta[0]
theta = crystal.get_Bragg_angle(E) + dtheta * 1e-6
refl = np.abs(crystal.get_amplitude(E, np.sin(theta))[0]) ** 2  # s-polarization
# rc = np.convolve(refl, refl, 'same') / (refl.sum()*dt) * dt
refl_sq = refl * refl
rc = np.convolve(refl_sq, refl, "same") / refl_sq.sum()
spline = UnivariateSpline(dtheta, rc - rc.max() / 2, s=0)
r1, r2 = spline.roots()  # find the roots

plt.plot(
    dtheta,
    refl,
    "r",
    label="one crystal\nFWHM = {0:.1f} µrad".format(crystal.get_Darwin_width(E) * 1e6),
)
plt.plot(
    dtheta,
    rc,
    "b",
    label="two crystal\n(convolution)" "\nFWHM = {0:.1f} µrad".format(r2 - r1),
)
plt.gca().set_xlabel("$\\theta - \\theta_{B}$ (µrad)")
plt.gca().set_ylabel(r"reflectivity")
plt.axvspan(r1, r2, facecolor="g", alpha=0.2)
plt.legend(loc="upper right", fontsize=12)

text = "Rocking curve of {0}{1[0]}{1[1]}{1[2]} at E={2:.0f} eV".format(
    crystal.name, crystal.hkl, E
)
plt.text(0.5, 1.02, text, transform=plt.gca().transAxes, size=15, ha="center")
plt.show()

with open("Rocking_curve_XRT_33_3.txt", "w") as f:
    for i in range(len(rc)):
        f.write(str(dtheta[i]) + "     " + str(rc[i]) + "\n")
