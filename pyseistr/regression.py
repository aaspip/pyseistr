import numpy as np
def lpf(data,basis,rect=[5,5,1],niter=100,verb=1):
	'''
	LPFC: Local prediction filter (n-dimensional) (not started)
	
	INPUT   
	data:     data to fit [n1,n2,n3] 
	basis:    basis functions to fit [n1,n2,n3,nw]
	rect:     smoothing radius [r1,r2,r3]
	niter:  number of CG iterations
	verb:   verbosity flag (default: 0)
	
	OUTPUT
	data_pred: 	predicted data  hat{d} = sum_{n} b_n(t)*a_n(t)" 
	pef:	predictive error filter

	
	
	EXAMPLE
	demos/test_pyortho_localortho2d.py
	'''
	
	import numpy as np
	from .divne import divne
	
	ndim=np.ndim(data)
	ndat=list(data.shape)
	ndat.append(1)
	ndat.append(1)
	n=data.size
# 	if signal.ndim==2:	#for 2D problems
# 		signal=np.expand_dims(signal, axis=2)
# 	if noise.ndim==2:	#for 2D problems
# 		noise=np.expand_dims(noise, axis=2)
# 	[n1,n2,n3]=signal.shape
# 	
# 	nd=n1*n2*n3;
# 	ndat=[n1,n2,n3];
# 	
# 	eps_dv=eps;
# 	eps_cg=0.1; 
# 	tol_cg=0.000001;
# 	ratio = divne(noise, signal, niter, rect, ndat, eps_dv, eps_cg, tol_cg,verb);
	
	nw=basis.shape[1]
	
	print("niter,nw,n,ndat,rect=",niter,nw,n,ndat,rect);
	print('basis.shape',basis.shape)
	

	mean=(basis*basis).sum()
	mean = np.sqrt (mean/n/nw);
	
	data=data/mean;
	basis=basis/mean

	
	pef=multidivn(data, basis, niter, nw, n, ndat, rect, ndim, aa=None, verb=True)
	

	
	from .operators import weight2_lop
	par={'w':pef, 'nw':nw, 'nm':n*nw, 'nd':n}
	
	basis=basis*mean;
	data_pre=weight2_lop(basis.reshape(n*nw,order='F'),par,False,False);
	
	

	return data_pre, pef
	
# def lpfc(signal,noise,rect,niter=50,eps=0.0,verb=1):
# 	'''
# 	LPFC: Local prediction filter (n-dimensional) (not started)
# 	
# 	'''
# 
# 	if signal.ndim==2:	#for 2D problems
# 		signal=np.expand_dims(signal, axis=2)
# 	
# 	[n1,n2,n3]=signal.shape
# 	
# 	r1=rect[0];
# 	r2=rect[1];
# 	r3=rect[2];
# 	
# 	signal=np.float32(signal.flatten(order='F'));
# 	noise=np.float32(noise.flatten(order='F'));
# 	
# 	print(n1,n2,n3,r1,r2,r3,niter,eps,verb)
# 	tmp=Clocalortho(signal,noise,n1,n2,n3,r1,r2,r3,niter,eps,verb);
# 	
# 	signal2=tmp[0:n1*n2*n3];
# 	noise2=tmp[n1*n2*n3:n1*n2*n3*2];
# 	low=tmp[n1*n2*n3*2:n1*n2*n3*3];
# 
# 	signal2=signal2.reshape(n1,n2,n3,order='F')
# 	noise2=noise2.reshape(n1,n2,n3,order='F')
# 	low=low.reshape(n1,n2,n3,order='F')
# 	
# 	if n3==1:	#for 1D/2D problems
# 		signal2=np.squeeze(signal2)
# 		noise2=np.squeeze(noise2)
# 		low=np.squeeze(low)
# 	
# 	return signal2,noise2,low


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
	aa:		data filter [sf_filter type FYI]
	verb: 	verbosity flag
	
	OUTPUT
	rat:	division ratio
	
	EXAMPLE
	
	
	'''

# 	num=num.reshape(n,order='F')
# 	den=den.reshape(n*nw,order='F')
# 	if np.ndim(ndat) <=2:
# 		ndat=np.expand_dims(ndat, axis=2) #ndat needs to be [n1,n2,n3]
# 	print('ndat',ndat)
# 	print(ndat[0],ndat[1],ndat[2])

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
# 	eps_cg=1.e-6;
# 	tol_cg=1;
	
	par_P=[]
	par_L={'nm':n2,'nd':n,'w':den, 'nw': nw} 							# parameters of weight2_lop
	par_S={'nm':n2,'nd':n2,'n1':n,'n2':nw,'oper':trianglen_lop,'par_op':{'nm':n,'nd':n,'nbox':nbox,'ndat':ndat,'ndim':ndim}}	# parameters of repeat_lop
	ifhasp0=prec;
	
	print('ifhasp0 (if preconditioning) = ',ifhasp0)
	
	from .operators import weight2_lop, repeat_lop, helicon_lop
	from .bp import ifnot
	rat = conjgrad(ifnot(prec, helicon_lop, None), weight2_lop, repeat_lop, p, None, num, eps_cg, tol_cg, niter, ifhasp0, par_P, par_L, par_S, verb);
	
	rat=rat.reshape(n,nw,order='F')
	
	return rat
	
	
	
	