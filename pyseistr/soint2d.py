from soint2dcfun import *

def soint2d(din,mask,dip,order=1,niter=100,njs=[1,1],drift=0,verb=1):
	# soint2d: 2D structure-oriented interpolation
	# 
	# by Yangkang Chen, 2022
	#
	# INPUT:
	# dn: model  noisy data
	# dip: slope
	# order:    PWD order
	# eps: regularization (default:0.01);
	# 
	# OUTPUT:
	# ds: filtered data 
	#
	# Demo
	# demos/test_pyseistr_soint2d.py
	
	from .solvers import solver
	from .solvers import cgstep
	from .operators import allpass3_lop
	import numpy as np
	
	nw=order;
	nj1=njs[0];
	nj2=njs[1];

	[n1,n2]=din.shape;
	n12=n1*n2;
	
	mm=din;
	mm=mm.flatten(order='F');
	known=np.zeros([n12,1]);
	
	if mask  is not None:
		dd=mask.flatten(order='F');
		for ii in range(0,n12):
			known[ii] = (dd[ii] !=0) ;
			dd[ii]=0;
	else:
		for ii in range(0,n12):
			known[ii] = (mm[ii] !=0) ;
			dd[ii]=0;

	pp=dipi.flatten(order='F');
	
	dout=din

	return dout
	
	
def soint2dc(din,mask,dip,order=1,niter=100,njs=[1,1],drift=0,hasmask=1,twoplane=0,prec=0,verb=1):
	# soint2d: 3D structure-oriented interpolation
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
	# hasmask: if 1, using the provided mask; if 0, using the data itself to determine
	# 
	# OUTPUT:
	# ds: filtered data 
	#
	# Demo
	# demos/test_pyseistr_soint3d.py
	
	from .solvers import solver
	from .solvers import cgstep
	from .operators import allpass3_lop
	import numpy as np
	
	nw=order;
	nj1=njs[0];
	nj2=njs[1];
	[n1,n2]=din.shape;

	if twoplane==0:
		dip1=np.float32(dip).flatten(order='F');
		dip2=dip1;
	else:
		dip1=np.float32(dip[:,:,0]).flatten(order='F');
		dip2=np.float32(dip[:,:,1]).flatten(order='F');
	mask=np.float32(mask).flatten(order='F');
	din=np.float32(din).flatten(order='F');
	
	dout=csoint2d(din,mask,dip1,dip2,n1,n2,nw,nj1,nj2,niter,drift,hasmask,twoplane,prec,verb);
	dout=dout.reshape(n1,n2,order='F');


	
	
	return dout



