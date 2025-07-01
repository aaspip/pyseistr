
import numpy as np
def lpf(basis,data,niter=100,verb=1):
	'''
	LPFC: Local prediction filter (n-dimensional) (not started)
	
	INPUT   
	basis:    basis functions to fit
	data:     data to fit
	niter:  number of CG iterations
	verb:   verbosity flag (default: 0)
	
	OUTPUT
	pef:	predictive error filter
	pred: predicted data  \hat{d} = \sum_{n} b_n(t)*a_n(t)
	
	
	EXAMPLE
	demos/test_pyortho_localortho2d.py
	'''
	
	import numpy as np
	from .divne import divne
	
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
	
	pef=[]
	pred=[]
	return pef,pred
	
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

