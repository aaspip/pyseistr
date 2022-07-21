**Pyseistr**
======

## Description

**Pyseistr** is a python package for structural denoising and interpolation of multi-channel seismic data. This package has a variety of applications in both exploration and earthquake seismology.

## Reference
Chen et al., 2022, Pyseistr: a python package for structural denoising and interpolation of multi-channel seismic data, under review. 

BibTeX:

	@article{pyseistr,
	  title={Pyseistr: a python package for structural denoising and interpolation of multi-channel seismic data},
	  author={Yangkang Chen and Alexandros Savvaidis and Sergey Fomel and Yunfeng Chen and Omar M. Saad and Yapo Abol{\'e} Serge Innocent Obou{\'e} and Quan Zhang and Wei Chen},
	  journal={TBD},
	  volume={1},
	  number={1},
	  pages={1-10},
	  year={2022}
	}

-----------
## Copyright
    Initial version: Yangkang Chen (chenyk2016@gmail.com), 2021-2022
	Later version: pyseistr developing team, 2022-present
-----------

## License
    GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)   

-----------

## Install
Using the latest version

    git clone https://github.com/aaspip/pyseistr
    cd pyseistr
    pip install -v -e .
or using Pypi

    pip install pyseistr

-----------
## Examples
    The "demo" directory contains all runable scripts to demonstrate different applications of pyseistr. 

-----------
## Gallery
The gallery figures of the pydrr package can be found at
    https://github.com/aaspip/gallery/tree/main/pyseistr
Each figure in the gallery directory corresponds to a DEMO script in the "demo" directory with the exactly the same file name.

-----------
## Dependence Packages
* scipy 
* numpy 
* matplotlib

-----------
## Modules
    dip2d.py  -> 2D local slope estimation
    dip3d.py  -> 3D local slope estimation
    divne.py  -> element-wise division constrained by shaping regularization
    somean2d.py -> 2D structure-oriented mean filter 
    somean3d.py -> 3D structure-oriented mean filter 
    somf2d.py 	-> 2D structure-oriented median filter 
    somf3d.py 	-> 3D structure-oriented median filter 
    soint2d.py  -> 2D structural interpolation
    soint3d.py  -> 3D structural interpolation
    ricker.py	-> Ricker wavelet
    bp.py		-> Butterworth bandpass filter
    fk.py		-> FK dip filter
    plot.py		-> seismic plotting functions
    
-----------
## Development
    The development team welcomes voluntary contributions from any open-source enthusiast. 
    If you want to make contribution to this project, feel free to contact the development team. 

-----------
## Contact
    Regarding any questions, bugs, developments, collaborations, please contact  
    Yangkang Chen
    chenyk2016@gmail.com

