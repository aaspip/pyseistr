def somf3d(dn,dipi,dipx,r1,r2,eps,order,option=1):
	# somf3d: 3D structure-oriented median filter
	# 
	# by Yangkang Chen, 2022
	#
	# INPUT:
	# dn: model  noisy data
	# dipi: inline slope
	# dipx: xline slope
	# r1,r2:    spray radius
	# order:    PWD order
	# eps: regularization (default:0.01);
	# 
	# OUTPUT:
	# ds: filtered data
	#  
	# References
	# Huang et al., 2021, Erratic noise suppression using iterative structure-oriented space-varying median filtering with sparsity constraint, Geophysical Prospecting, 69, 101-121.
	# Chen et al., 2020, Deblending of simultaneous-source data using a structure-oriented space-varying median filter, Geophysical Journal International, 222, 1805–1823.
	# Gan et al., 2016, Separation of simultaneous sources using a structural-oriented median filter in the flattened dimension, Computers & Geosciences, 86, 46-54.
	# Chen, Y., 2015, Deblending using a space-varying median filter, Exploration Geophysics, 46, 332-341.
	#
	# Demo
	# demos/test_xxx_somf3d.py
	import numpy as np
	nnp=(2*r1+1)*(2*r2+1);
	
	#flattening
	from .pwspray3d import pwspray3d
	from .mf import mf
	from .mf import svmf
	
	u = pwspray3d(dn,dipi,dipx,r1,r2,order,eps);

	n3=dn.shape[2];
	n2=dn.shape[1];
	ns2=2*r1*r2+1;
	for i3 in range(0,n3):
		for i2 in range(0,n2):
			if option==1:
				u[:,:,i2,i3]=mf(u[:,:,i2,i3],ns2,1,2);
			else:
				u[:,:,i2,i3],win_tmp=svmf(u[:,:,i2,i3],ns2,1,2);
	
	ds=u[:,int(nnp/2),:,:];

	return ds




