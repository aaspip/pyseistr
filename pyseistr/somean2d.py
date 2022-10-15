from sofcfun import *

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

def somean2dc(dn,dip,ns,order,eps,adj=0,verb=1):
	# somean2dc: plane-wave smoothing implemented in C
	#
	# INPUT:
	# dn: model   noisy data
	# dip: slope (2D array)
	# ns:       spray radius
	# order:    PWD order
	# eps: regularization (default:0.01);
	# adj: adjoint flat (default: 0, forward)
	# verb: verbosity
	# 
	# OUTPUT:
	# ds:  smoothed data
	#
	import numpy as np

	ns2=2*ns+1;	#spray diameter
	
	if dn.ndim==2:
		[n1,n2]=dn.shape;
		n3=1;
	else: #assuming ndim=3;
		[n1,n2,n3]=dn.shape;
	
	dn=np.float32(dn).flatten(order='F');
	dip=np.float32(dip).flatten(order='F');
	
	ds=csomean2d(dn,dip,n1,n2,n3,ns,order,adj,eps,verb);
	ds=ds.reshape(n1,n2,n3,order='F')
	
	if n3==1:	#for 2D problems
		ds=np.squeeze(ds)
	return ds
	
	
	