# glyP 

## Description

Development of this package was done in order to generate a quantitative metric in how similar IR plots of carbohydrate conformers. In addition to this there are graphical functionalities to display molecular data.

This package is to utilizes data of carbohydrate molecules in the form of .log and .xyz files generated through Gaussian. Given a directory of such files the package can generate IR plots for each molecule, superimpose IR plots of two different molecules, display and sort a table of energy calculations, calculate the RMSD and Pendry R factor, identify and modify dihedral angles of molecules and also provide simple graphical displays of this data such as tables, bar plots and heatmaps. 

Furthermore, there is also a simple genetic algorithm that generates and updates the a set of molecules in order to optimize the structure in search of the lowest energy level. Further developments are being made to incorporate changes in the position of other features such as linker groups. 

## Dependencies

Required Python 3.8.3

The following packages are also required:
  
  - numpy
  - networkx
  - copy
  - math
  - matplotlib
  - seaborn
  - texttable

## Installation

Clone this project directory

Within a python environment install all the dependencies

```pip install numpy,networkx,copy,math,matplotlib,seaborn,texttable```

## Contact

srahim457@gmail.com
