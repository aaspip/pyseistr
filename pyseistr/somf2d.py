from sofcfun import *

def somf2d(dn,dip,ns,order,eps,option=1):
	# somf2d: structure-oriented median filter
	#
	# INPUT:
	# dn: model   noisy data
	# dip: slope (2D array)
	# ns:       spray radius
	# order:    PWD order
	# eps: regularization (default:0.01);
	# option=1 or 2: (1 for MF; 2 for SVMF)
	#
	# OUTPUT:
	# ds:  filtered data
	#  
	# References
	# Huang et al., 2021, Erratic noise suppression using iterative structure-oriented space-varying median filtering with sparsity constraint, Geophysical Prospecting, 69, 101-121.
	# Chen et al., 2020, Deblending of simultaneous-source data using a structure-oriented space-varying median filter, Geophysical Journal International, 222, 1805–1823.
	# Gan et al., 2016, Separation of simultaneous sources using a structural-oriented median filter in the flattened dimension, Computers & Geosciences, 86, 46-54.
	# Chen, Y., 2015, Deblending using a space-varying median filter, Exploration Geophysics, 46, 332-341.
	#
	# Demo
	# demos/test_xxx_somf2d.py
	
	import numpy as np
	from .pwspray2d import pwspray2d
	from .mf import mf
	from .mf import svmf
	
	n1=dn.shape[0];
	n2=dn.shape[1];
	ns2=2*ns+1;	#spray diameter

	#flattening
	utmp=pwspray2d(dn,dip,ns,order,eps);
	u=utmp.reshape(n1,ns2,n2,order='F')


	for i2 in range(0,n2):
		if option==1:
			u[:,:,i2]=mf(u[:,:,i2],ns2,1,2);
		else:
			u[:,:,i2],win_tmp=svmf(u[:,:,i2],ns2,1,2);
	
	ds=u[:,ns,:];
	
	return ds

def somf2dc(dn,dip,ns,order,eps,option=1,verb=1):
	# somf2dc: structure-oriented median filter in C
	#
	# INPUT:
	# dn: model   noisy data
	# dip: slope (2D array)
	# ns:       spray radius
	# order:    PWD order
	# eps: regularization (default:0.01);
	# option=1 or 2: (1 for MF; 2 for SVMF)
	#
	# OUTPUT:
	# ds:  filtered data
	#  
	# References
	# Huang et al., 2021, Erratic noise suppression using iterative structure-oriented space-varying median filtering with sparsity constraint, Geophysical Prospecting, 69, 101-121.
	# Chen et al., 2020, Deblending of simultaneous-source data using a structure-oriented space-varying median filter, Geophysical Journal International, 222, 1805–1823.
	# Gan et al., 2016, Separation of simultaneous sources using a structural-oriented median filter in the flattened dimension, Computers & Geosciences, 86, 46-54.
	# Chen, Y., 2015, Deblending using a space-varying median filter, Exploration Geophysics, 46, 332-341.
	#
	# Demo
	# demos/test_xxx_somf2d.py
	
	import numpy as np
	ns2=2*ns+1;	#spray diameter
	
	if dn.ndim==2:
		[n1,n2]=dn.shape;
		n3=1;
	else: #assuming ndim=3;
		[n1,n2,n3]=dn.shape;
	
	dn=np.float32(dn).flatten(order='F');
	dip=np.float32(dip).flatten(order='F');
	
	ds=csomf2d(dn,dip,n1,n2,n3,ns,2*ns+1,option,order,eps,verb);
	ds=ds.reshape(n1,n2,n3,order='F')
	
	if n3==1:	#for 2D problems
		ds=np.squeeze(ds)
	return ds
	

	
	
	