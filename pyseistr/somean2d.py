def somean2d(dn,dip,ns,order,eps):
	# somean2d: plane-wave smoothing 
	#
	# INPUT:
	# dn: model   noisy data
	# dip: slope (2D array)
	# ns:       spray radius
	# order:    PWD order
	# eps: regularization (default:0.01);

	# OUTPUT:
	# ds:  smoothed data
	#
	import numpy as np
	n1=dn.shape[0];
	n2=dn.shape[1];

	ns2=2*ns+1;	#spray diameter

	#flattening
	from .pwspray2d import pwspray2d
	
	utmp=pwspray2d(dn,dip,ns,order,eps);
	u=utmp.reshape(n1,ns2,n2,order='F')
	ds=np.sum(u,1)/ns2;

	return ds

	