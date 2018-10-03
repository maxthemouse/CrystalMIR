from __future__ import absolute_import
from cx_Freeze import setup, Executable
import scipy
import numpy
import os.path

scipy_path = os.path.dirname(scipy.__file__)
numpy_path = os.path.dirname(numpy.__file__)
# Dependencies are automatically detected, but it might need
# fine tuning.
buildOptions = dict(
    packages=[],
    excludes=["collections.abc", "Tkinter", "matplotlib", "pandas", "h5py"],
    includes=[],
    include_files=["LICENSE.md", "README.md", "main_window.ui",
                   scipy_path, "mir_icon.ico", "MIR manual.pdf",
                   numpy_path],
    #icon = "mir_icon.ico"
)

import sys
base = 'Win32GUI' if sys.platform == 'win32' else None

executables = [
    Executable('mir_qt.py', base=base, icon = "mir_icon.ico")
]

setup(name='MIR',
      version='0.3.0',
      author='M. Adam Webb',
      author_email='adam.webb@lightsource.ca',
      description='Multiple Image Radiography',
      options=dict(build_exe=buildOptions),
      executables=executables)
