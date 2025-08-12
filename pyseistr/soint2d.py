from soint2dcfun import *

def soint2d(din,mask,dip,order=1,niter=100,njs=[1,1],drift=0,verb=1, prec=False):
	'''
	soint2d: 2D structure-oriented interpolation (correct)
	
	by Yangkang Chen, 2022
	
	INPUT
	dn: model  noisy data
	dip: slope
	order:    PWD order
	eps: regularization (default:0.01);
	prec: if apply preconditioning operator
	
	OUTPUT
	ds: filtered data 
	
	EXAMPLE
	demos/test_pyseistr_soint2d.py
	'''
	from .solvers import solver
	from .solvers import cgstep
	from .operators import allpass3_lop
	import numpy as np
	
	nw=order;
	nj1=njs[0];
	nj2=njs[1];

	[n1,n2]=din.shape;
	n12=n1*n2;
	
	mm=din.flatten(order='F'); 			  #data vector flattened to 1D
	known=np.zeros(n12, dtype=np.bool_);  #mask vector flattened to 1D
	pp=dip.flatten(order='F'); #slope vector flattened to 1D
	
	if mask  is not None:
		dd=mask.flatten(order='F');
		for ii in range(0,n12):
			known[ii] = (dd[ii] !=0) ;
			dd[ii]=0;
	else:
		for ii in range(0,n12):
			known[ii] = (mm[ii] !=0) ;
			dd[ii]=0;

	
	if np.ndim(din)>2:
		qq=np.zeros(n1,n2, dtype='float')
	else:
		qq=None;
	
	if qq is not None:
		if prec:
			par_predict = {'nm': n12, 'nd': n12, 'n1': n1, 'n2': n2, "dip": dip, "tt": np.zeros(n1, dtype=np.float_), 'nw': nw, 'e': 0.0001}
		else:
			par_allpass={}
	else:
		if prec:
			par_predict = {'nm': n12, 'nd': n12, 'n1': n1, 'n2': n2, "dip": dip, "tt": np.zeros(n1, dtype=np.float_), 'nw': nw, 'e': 0.0001}
		else:
# 			allpass22_init(allpass2_init(nw, nj1, n1,n2, drift, pp));
			par_allpass={}
			

	from .solvers import cgstep, solver, solver_prec
	from .operators import mask_lop, predict_lop
		
	if qq is not None:
		pass
		
	else:
		if prec:
			par_L={'nm': n12, 'nd': n12, 'mask': known}
			par_P=par_predict
			par_sol={'verb': 1}
			mm2,tmp=solver_prec(mask_lop, cgstep, predict_lop, n12, n12, n12, mm, mm, niter, 0, par_L, par_P, par_sol)
			
		else:
			par_L={'ap1': ap1}
			par_sol={'known': known, 'x0': mm, 'verb': 1}
			mm2,tmp=solver(allpass21_lop, cgstep, n12, n12, mm, dd, niter, par_L, par_sol)
			
	dout=mm2.reshape(n1,n2,order='F')


# 	dout=din;
	return dout
	
	
def soint2dc(din,mask,dip,order=1,niter=100,njs=[1,1],drift=0,hasmask=1,twoplane=0,prec=0,verb=1):
	'''
	soint2d: 3D structure-oriented interpolation
	
	by Yangkang Chen, 2022
	
	INPUT
	dn: model  noisy data
	dipi: inline slope
	dipx: xline slope
	r1,r2:    spray radius
	order:    PWD order
	eps: regularization (default:0.01);
	hasmask: if 1, using the provided mask; if 0, using the data itself to determine
	
	OUTPUT
	ds: filtered data 
	
	EXAMPLE
	demos/test_pyseistr_soint3d.py
	'''
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



