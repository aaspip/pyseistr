**Pyseistr**
======

## Description

**Pyseistr** is a python package for structural denoising and interpolation of multi-channel seismic data. The latest version has incorporated both Python and C (hundreds of times faster) implementations of the embedded functions. We keep both implementations for both educational and production purposes. This package has a variety of applications in both exploration and earthquake seismology.

## Reference
Chen et al., 2023, Pyseistr: a python package for structural denoising and interpolation of multi-channel seismic data, Seismological Research Letters, 94(3), 1703-1714. 

BibTeX:

	@article{pyseistr,
	  title={Pyseistr: a python package for structural denoising and interpolation of multi-channel seismic data},
	  author={Yangkang Chen and Alexandros Savvaidis and Sergey Fomel and Yunfeng Chen and Omar M. Saad and Yapo Abol{\'e} Serge Innocent Obou{\'e} and Quan Zhang and Wei Chen},
	  journal={Seismological Research Letters},
	  volume={94},
	  number={3},
	  pages={1703–1714},
	  year={2023}
	}

-----------
## Copyright
    The pyseistr developing team, 2021-present
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
## DEMO scripts
    The "demo" directory contains all runable scripts to demonstrate different applications of pyseistr. 

-----------
## Gallery
The gallery figures of the pyseistr package can be found at
    https://github.com/aaspip/gallery/tree/main/pyseistr
Each figure in the gallery directory corresponds to a DEMO script in the "demo" directory with the exactly the same file name.

-----------
## Dependence Packages
* scipy 
* numpy 
* matplotlib

-----------
## Modules
    dip2d.py  	-> 2D local slope estimation (including both python and C implementations)
    dip3d.py  	-> 3D local slope estimation (including both python and C implementations)
    divne.py  	-> element-wise division constrained by shaping regularization (python implementation)
    somean2d.py 	-> 2D structure-oriented mean filter  (including both python and C implementations)
    somean3d.py 	-> 3D structure-oriented mean filter  (including both python and C implementations)
    somf2d.py 	-> 2D structure-oriented median filter  (including both python and C implementations)
    somf3d.py 	-> 3D structure-oriented median filter  (including both python and C implementations)
    soint2d.py  	-> 2D structural interpolation  (including both python and C implementations)
    soint3d.py  	-> 3D structural interpolation  (including both python and C implementations)
    ricker.py	-> Ricker wavelet
    bp.py		-> Butterworth bandpass filter (including both python and C implementations)
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

-----------
## Examples
# Example 1 (2D structure-oriented mean/smoothing filter) 
Generated by [demos/test_pyseistr_somean2d.py](https://github.com/aaspip/pyseistr/tree/main/demos/test_pyseistr_somean2d.py)

<img src='https://github.com/aaspip/gallery/blob/main/pyseistr/test_pyseistr_somean2d.png' alt='Slicing' width=960/>

# Example 2 (3D structure-oriented mean/smoothing filter) 
Generated by [demos/test_pyseistr_somean3d.py](https://github.com/aaspip/pyseistr/tree/main/demos/test_pyseistr_somean3d.py)

<img src='https://github.com/aaspip/gallery/blob/main/pyseistr/test_pyseistr_somean3d.png' alt='Slicing' width=960/>

# Example 3 (2D structure-oriented median filter) 
Generated by [demos/test_pyseistr_somf2d.py](https://github.com/aaspip/pyseistr/tree/main/demos/test_pyseistr_somf2d.py)

<img src='https://github.com/aaspip/gallery/blob/main/pyseistr/test_pyseistr_somf2d.png' alt='Slicing' width=960/>

# Example 4 (3D structure-oriented median filter) 
Generated by [demos/test_pyseistr_somf3d.py](https://github.com/aaspip/pyseistr/tree/main/demos/test_pyseistr_somf3d.py)

<img src='https://github.com/aaspip/gallery/blob/main/pyseistr/test_pyseistr_somf3d.png' alt='Slicing' width=960/>

# Example 5 (3D structure-oriented interpolation) 
Generated by [demos/test_pyseistr_passive_recon3d.py](https://github.com/aaspip/pyseistr/tree/main/demos/test_pyseistr_passive_recon3d.py)

<img src='https://github.com/aaspip/gallery/blob/main/pyseistr/test_pyseistr_passive_recon3d.png' alt='Slicing' width=960/>

# Example 6 (SS precursor data enhancement) 
Generated by [demos/test_pyseistr_ssprecursor.py](https://github.com/aaspip/pyseistr/tree/main/demos/test_pyseistr_ssprecursor.py)

<img src='https://github.com/aaspip/gallery/blob/main/pyseistr/test_pyseistr_ssprecursor.png' alt='Slicing' width=960/>

# Example 7 (receiver function data enhancement) 
Generated by [demos/test_pyseistr_rf.py](https://github.com/aaspip/pyseistr/tree/main/demos/test_pyseistr_rf.py)

<img src='https://github.com/aaspip/gallery/blob/main/pyseistr/test_pyseistr_rf.png' alt='Slicing' width=960/>

# Example 8 (structure-oriented distributed acoustic sensing (DAS) data processing) 
Generated by [demos/test_pyseistr_das.py](https://github.com/aaspip/pyseistr/tree/main/demos/test_pyseistr_das.py)

<img src='https://github.com/aaspip/gallery/blob/main/pyseistr/test_pyseistr_das.png' alt='Slicing' width=960/>

# Below are new examples in addition to the results in the original paper

# Below is an example for 2D structure-oriented interpolation of a multi-channel synthetic seismic data
Generated by [demos/test_pyseistr_soint2d.py](https://github.com/aaspip/pyseistr/tree/main/demos/test_pyseistr_soint2d.py)
<img src='https://github.com/aaspip/gallery/blob/main/pyseistr/test_pyseistr_soint2d.png' alt='comp' width=960/>

# Below is an example for 2D structure-oriented interpolation of a seafloor dataset
Generated by [demos/test_pyseistr_soint2d_seafloor.py](https://github.com/aaspip/pyseistr/tree/main/demos/test_pyseistr_soint2d_seafloor.py)
<img src='https://github.com/aaspip/gallery/blob/main/pyseistr/test_pyseistr_soint2d_seafloor.png' alt='comp' width=960/>

# Below is an another example for 2D structure-oriented interpolation of a sparse seismic data
Generated by [demos/test_pyseistr_soint2d2.py](https://github.com/aaspip/pyseistr/tree/main/demos/test_pyseistr_soint2d2.py)
<img src='https://github.com/aaspip/gallery/blob/main/pyseistr/test_pyseistr_soint2dnew.png' alt='comp' width=960/>

# Below is an example for calculating relative geological time (RGT) from 2D seismic data
Generated by [demos/test_pyseistr_rgt2d.py](https://github.com/aaspip/pyseistr/tree/main/demos/test_pyseistr_rgt2d.py)
<img src='https://github.com/aaspip/gallery/blob/main/pyseistr/test_pyseistr_rgt2d.png' alt='comp' width=960/>




