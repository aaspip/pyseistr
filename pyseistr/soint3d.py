def soint3d(din,mask,dipi,dipx,order=1,niter=100,njs=[1,1],drift=0,verb=1):
	# soint3d: 3D structure-oriented interpolation
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
	# Demo
	# demos/test_xxx_soint3d.py
	
	from .solvers import solver
	from .solvers import cgstep
	from .operators import allpass3_lop
	import numpy as np
	
	nw=order;
	nj1=njs[0];
	nj2=njs[1];

	[n1,n2,n3]=din.shape;
	n123=n1*n2*n3;
	
	mm=din;
	mm=mm.flatten(order='F');
	known=np.zeros([n123,1]);
	
	if mask  is not None:
		dd=mask.flatten(order='F');
		for ii in range(0,n123):
			known[ii] = (dd[ii] !=0) ;
			dd[ii]=0;
	else:
		for ii in range(0,n123):
			known[ii] = (mm[ii] !=0) ;
			dd[ii]=0;

	pp=dipi.flatten(order='F');
	qq=dipx.flatten(order='F');
	
	ap1={'nw':nw,'nj':nj1,'nx':n1,'ny':n2,'nz':n3,'drift':drift,'pp':pp};
	ap2={'nw':nw,'nj':nj2,'nx':n1,'ny':n2,'nz':n3,'drift':drift,'pp':qq};
	
	par_L={'ap1':ap1,'ap2':ap2,'nm':n123,'nd':2*n123};
	
	par_sol={'known':known,'x0':mm,'verb':1};
	
	dd=np.zeros([2*n123,1]);
	mm2,tmp=solver(allpass3_lop,cgstep,n123,2*n123,mm,dd,niter,par_L,par_sol);
	
	dout=mm2.reshape(n1,n2,n3,order='F');

	return dout
	
	




