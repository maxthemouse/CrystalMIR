# MultipleImageRadiography
Analysis of data for Multiple Image Radiography

This project consists of files related to diffraction enhanced imaging. They are
developed at BMIT at the Canadian Light Source [link](http://www.lightsource.ca/).

This is a new repository based on older code. The old repo has large data files and other things that are not appropriate for this code. In part this is to split out this specific program. 

Install
-------

The simplest installation method is to use a .msi installer which is available. Double-click
on the file and follow the installation windows. The program does not create any icons or links
in the start menu.

If installing from source the required python modules, tifffile and pyprind, can be installed using pip.
Installing psutil from pip is optional.


 Running the program
 -------------------

If using the installer the program is run using mir_qt.exe. This opens the Qt-based interface.
See the manual for details.

 If Jupyter is installed than a notebook can be used. An example notebook is included but others can be created.

 It is also possible to use the run_mir_calc.py script directly.
 Edit the script to input the parameters and run the script. Most of the default values can usually
 be used. Primarily the user needs to input the location of the root data directory, the location
 of the rocking curve file and inout the True/False values to select which calculations to run.

