# CrystalMIR
Analysis of data for Multiple Image Radiography or Analyzer Based Images and data 
using an analyzer crystal.

This project consists of files related to diffraction enhanced imaging. They are
developed at BMIT at the Canadian Light Source [link](http://www.lightsource.ca/). The previous name was 
MultipleImageRadiography. 

## Install

All files are simply installed into one folder and does not need to be installed as a package. 
Jupyter notebooks are given for several modes of operation and include some instructions. 

1. Need python 3 and poetry
2. Clone this repository
3. cd to repo and use command
```sh
poetry install
```
 Poetry will create a virtual environment and install all dependencies. The included poetry.lock file will set the versions. 

 Initial setting. To avoid having some settings in the files the scripts are using dotenv. This requires an intial environment file, .env.
 Create a .env file with the following content where S_folder is the folder where the settings folder will be. Change this value as needed. A default may be the folder where the program is installed. A good value is the location of the data or the results.
 ```
s_name='settings.json'
s_folder='E:\WPy64-31241\notebooks\CrystalMIR'
```

4. To start notebooks
```sh
poetry run jupyter lab
```
