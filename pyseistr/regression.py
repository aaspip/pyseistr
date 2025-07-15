import numpy as np
def lpf(data,basis,rect=[5,5,1],niter=100,verb=1,aafilt=None):
	'''
	LPF: Local prediction filter (n-dimensional) 
	
	INPUT   
	data:     data to fit [n1,n2,n3] 
	basis:    basis functions to fit [n1,n2,n3,nw] [basis size is always ndata*nw]
	rect:     smoothing radius [r1,r2,r3]
	niter:  number of CG iterations
	aafilt:   helicon preconditioning filter (default: None)
	verb:   verbosity flag (default: 0)
	
	OUTPUT
	data_pred: 	predicted data  hat{d} = sum_{n} b_n(t)*a_n(t)" 
	pef:	predictive error filter


	EXAMPLE
	neqdemos/test_17_lpf.py
	'''
	
	import numpy as np
	from .divne import divne
	
	ndim=np.ndim(data)
	ndat=list(data.shape)
	ndat.append(1)
	ndat.append(1)
	
	n=data.size	
	nw=basis.shape[-1]
	
	print("niter,nw,n,ndat,rect=",niter,nw,n,ndat,rect);
	print('basis.shape',basis.shape)
	

	mean=(basis.flatten()*basis.flatten()).sum()
	mean = np.sqrt (mean/n/nw);
	
	print('mean=',mean)
	
	data=data/mean;
	basis=basis/mean
	
	data=data.reshape(n,order='F')
	basis=basis.reshape(n,nw,order='F')
	
	pef=multidivn(data, basis, niter, nw, n, ndat, rect, ndim, aa=aafilt, verb=True)
	
	from .operators import weight2_lop
	par={'w':basis, 'nw':nw, 'nm':n*nw, 'nd':n}
	
# 	basis=basis*mean;
	data_pre=weight2_lop(pef.reshape(n*nw,order='F'),par,False,False);
	data_pre=data_pre*mean;

	if ndat[1]==1: #1D problem
		data_pre=data_pre.reshape(ndat[0],order='F')
		
	elif ndim==2: #2D problem
		data_pre=data_pre.reshape(ndat[0],ndat[1],order='F')
		pef=pef.reshape(ndat[0],ndat[1],nw,order='F')
	
	return data_pre, pef
	
def multidivn(num, den, niter, nw, n, ndat, nbox, ndim, aa=None, verb=True):
	'''
	
	INPUT
	num:	numerator [nd*1]
	den:	denominator [nd*nw]
	niter:  number of iterations
	nw: 	number of components
	n:  	data size (i.e., nd)
	ndat: 	data dimensions [ndim] (e.g., [nd,1])
	nbox: 	smoothing radius [ndim] (e.g., nbox=[4,4,1])
	ndim:   dimension (e.g., 1 if input is 1D; 2 if input is 2D; 3 if input is 3D)
	aa:		data filter [sf_filter type FYI], helix preconditioning filter a dictionary like  	
			{'flt':np.array([-1]),'lag':np.array([251]),'nh':1, 'mis': False, 'h0': 0}
	verb: 	verbosity flag
	
	OUTPUT
	rat:	division ratio
	
	EXAMPLE
	
	
	'''

	if num.size != n:
		Exception("Sorry, num.size [nd] must be n")
	
	n2 = n*nw
	p = np.zeros (n2);
	prec =  (None != aa);
	if prec:
		#helicon_init(aa); 
		pass
	
	from .solvers import conjgrad
	from .divne import trianglen_lop
	eps_cg=1;
	tol_cg=1.e-6;
	
	par_P={'nm':n,'nd':n,'aa': aa}
	par_L={'nm':n2,'nd':n,'w':den, 'nw': nw} 							# parameters of weight2_lop
	par_S={'nm':n2,'nd':n2,'n1':n,'n2':nw,'oper':trianglen_lop,'par_op':{'nm':n,'nd':n,'nbox':nbox,'ndat':ndat,'ndim':ndim}}	# parameters of repeat_lop
	ifhasp0=prec;
	
	print('ifhasp0 (if preconditioning) = ',ifhasp0)
	
	from .operators import weight2_lop, repeat_lop, helicon_lop
	from .bp import ifnot
	rat = conjgrad(ifnot(prec, helicon_lop, None), weight2_lop, repeat_lop, p, None, num, eps_cg, tol_cg, niter, ifhasp0, par_P, par_L, par_S, verb);
	
	rat=rat.reshape(n,nw,order='F')
	
	return rat
	
	
	
	