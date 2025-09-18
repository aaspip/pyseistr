import numpy as np
from dipcfun import *

def dip2d(din,niter=5,liter=20,order=2,eps_dv=0.01, eps_cg=1, tol_cg=0.000001,rect=[10,10,1],verb=1):
	'''
	dip2d: dip estimation based on shaping regularized PWD algorithm 
	(independent implementation)
	
	INPUT
	din: input data (nt*nx)
	niter: number of nonlinear iterations
	liter: number of linear iterations (in divn)
	order: accuracy order
	eps_dv: eps for divn  (default: 0.01)
	eps_cg: eps for CG    (default: 1)
	tol_cg: tolerence for CG (default: 0.000001)
	rect:  smoothing radius (ndim*1)
	verb: verbosity flag
	
	OUTPUT
	dip:  2D slope
	
	NOTE: mask function is not enabled yet (will do in the future)
	
	EXAMPLE 1
	import numpy as np
	import matplotlib.pyplot as plt
	import pyseistr as ps

	## Generate synthetic data
	from pyseistr import gensyn,smooth
	data=gensyn();
	data=data[:,0::10];#or data[:,0:-1:10];
	data=data/np.max(np.max(data));
	np.random.seed(202122);
	scnoi=(np.random.rand(data.shape[0],data.shape[1])*2-1)*0.2;

	#add erratic noise
	nerr=scnoi*0;inds=[10,20,50];
	for i in range(len(inds)):
		nerr[:,inds[i]]=scnoi[:,inds[i]]*10;

	dn=data+scnoi+nerr;

	dtemp=smooth(dn,rect=[1,5,1]);
	
	print('size of data is (%d,%d)'%data.shape)
	print(data.flatten().max(),data.flatten().min())

	dip=ps.dip2d(dtemp,rect=[10,20,1]); 
	print(dn.shape)
	print(dip.flatten().max(),dip.flatten().min())

	## Structural smoothing
	r=2;
	eps=0.01;
	order=2;
	d1=ps.somean2dc(dn,dip,r,order,eps);
	d2=ps.somf2dc(dn,dip,r,order,eps,1);

	## plot results
	fig = plt.figure(figsize=(10, 8))
	ax=plt.subplot(2,4,1)
	plt.imshow(data,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Clean data');
	ax=plt.subplot(2,4,2)
	plt.imshow(dn,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Noisy data');
	ax=plt.subplot(2,4,3)
	plt.imshow(d1,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Filtered (SOMEAN)');
	ax=plt.subplot(2,4,4)
	plt.imshow(dn-d1,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Noise (SOMEAN)');
	ax=plt.subplot(2,4,6)
	plt.imshow(dip,cmap='jet',clim=(-2, 2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Slope');
	ax=plt.subplot(2,4,7)
	plt.imshow(d2,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Filtered (SOMF)');
	ax=plt.subplot(2,4,8)
	plt.imshow(dn-d2,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Noise (SOMF)');
	plt.show()
	'''
	
	from .divne import divne
	
	p0 = 0.0;
	nj1 = 1;
	nj2 = 1;
	dim = 3;
	
	n = np.zeros(dim,dtype='int');
	n1 = din.shape[0];
	n2 = din.shape[1];
	n[0] = n1;
	n[1] = n2;
	n[2] = 1;

	n123 = din.size;

	ratio=np.zeros([n1,n2,1]);
	dip=np.zeros([n1,n2,1]);

	for iter in range(0,niter):
		"""corresponding to the eq.21 in the paper  (Wang et al., 2022)"""
		[u1,u2] = conv_allpass(din,dip,order); 
		ratio = divne(-u2, u1, liter, rect, n, eps_dv, eps_cg, tol_cg,verb);
		dip=dip+ratio;

	dip=dip.squeeze()
	return dip

def dip2dc(din,niter=5,liter=20,order=2,eps_dv=0.01, eps_cg=1, tol_cg=0.000001,rect=[10,10,1],verb=1,mask=None):
	'''
	dip2dc: dip estimation based on shaping regularized PWD algorithm 
	(C implementation)
	
	INPUT
	din: input data (nt*nx)
	niter: number of nonlinear iterations
	liter: number of linear iterations (in divn)
	order: accuracy order
	eps_dv: eps for divn  (default: 0.01)
	eps_cg: eps for CG    (default: 1)
	tol_cg: tolerence for CG (default: 0.000001)
	rect:  smoothing radius (ndim*1)
	verb: verbosity flag
	
	OUTPUT
	dip:  2D slope
	
	EXAMPLE 1
	import numpy as np
	import matplotlib.pyplot as plt
	import pyseistr as ps

	## Generate synthetic data
	from pyseistr import gensyn,smooth
	data=gensyn();
	data=data[:,0::10];#or data[:,0:-1:10];
	data=data/np.max(np.max(data));
	np.random.seed(202122);
	scnoi=(np.random.rand(data.shape[0],data.shape[1])*2-1)*0.2;

	#add erratic noise
	nerr=scnoi*0;inds=[10,20,50];
	for i in range(len(inds)):
		nerr[:,inds[i]]=scnoi[:,inds[i]]*10;

	dn=data+scnoi+nerr;

	dtemp=smooth(dn,rect=[1,5,1]);
	
	print('size of data is (%d,%d)'%data.shape)
	print(data.flatten().max(),data.flatten().min())

	dip=ps.dip2dc(dtemp,rect=[10,20,1]); 
	print(dn.shape)
	print(dip.flatten().max(),dip.flatten().min())

	## Structural smoothing
	r=2;
	eps=0.01;
	order=2;
	d1=ps.somean2dc(dn,dip,r,order,eps);
	d2=ps.somf2dc(dn,dip,r,order,eps,1);

	## plot results
	fig = plt.figure(figsize=(10, 8))
	ax=plt.subplot(2,4,1)
	plt.imshow(data,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Clean data');
	ax=plt.subplot(2,4,2)
	plt.imshow(dn,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Noisy data');
	ax=plt.subplot(2,4,3)
	plt.imshow(d1,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Filtered (SOMEAN)');
	ax=plt.subplot(2,4,4)
	plt.imshow(dn-d1,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Noise (SOMEAN)');
	ax=plt.subplot(2,4,6)
	plt.imshow(dip,cmap='jet',clim=(-2, 2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Slope');
	ax=plt.subplot(2,4,7)
	plt.imshow(d2,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Filtered (SOMF)');
	ax=plt.subplot(2,4,8)
	plt.imshow(dn-d2,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Noise (SOMF)');
	plt.show()
	'''
	
	p0 = 0.0;
	nj1 = 1;
	nj2 = 1;
	dim = 3;
	
	n1 = din.shape[0];
	n2 = din.shape[1];
	n3 = 1;
	
	r1=rect[0]
	r2=rect[1]
	r3=rect[2]

	if mask is None:
		din=np.float32(din.flatten(order='F'));
		hasmask=0
	else:
		if din.size != mask.size:
			Exception("Mask and Data should have the same dimension")
		else:
			din=np.float32(np.concatenate([din.flatten(order='F'),mask.flatten(order='F')],axis=0));
			hasmask=1;
	dip=dipc(din,n1,n2,n3,niter,liter,order,eps_dv,eps_cg,tol_cg,r1,r2,r3,hasmask,verb);
	dip=dip.reshape(n1,n2,order='F');
		
	return dip
	
def conv_allpass(din,dip,order):
	'''
	conv_allpass: Convolutional operator implemented by an allpass filter
	
	Linearized inverse problem
	C'(\sigma)d\Delta \sigma =  C(\sigma)d
	
	IPNUT:
	din: input data
	dip: 3D dip
	order: accuracy order
	
	OUTPUT:
	u1: C'(\sigma)d (denominator)
	u2: C(\sigma)d  (numerator)
	
	'''
	[n1,n2]=din.shape;
	u1=np.zeros([n1,n2]);
	u2=np.zeros([n1,n2]);

	if order==1:
		nw=1;
	else:
		nw=2;

	filt1=np.zeros(2*nw+1);
	filt2=np.zeros(2*nw+1);

	#cprresponding to eqs.17-19
	#destruct the data
	for i1 in range(nw,n1-nw):
		for i2 in range(0,n2-1):
			if order==1:
				filt1=B3d(dip[i1,i2]);
				filt2=B3(dip[i1,i2]);
			else:
				filt1=B5d(dip[i1,i2]);
				filt2=B5(dip[i1,i2]);

			for iw in range(-nw,nw+1):
				u1[i1,i2]=u1[i1,i2]+(din[i1+iw,i2+1]-din[i1-iw,i2])*filt1[iw+nw];
				u2[i1,i2]=u2[i1,i2]+(din[i1+iw,i2+1]-din[i1-iw,i2])*filt2[iw+nw];

	return u1,u2

#cprresponding to eqs.13-16
#form the filters
def B3(sigma):
	'''
	B3 coefficient
	sigma: slope
	'''
	b3=np.zeros[2];
	b3[0]=(1-sigma)*(2-sigma)/12;
	b3[1]=(2+sigma)*(2-sigma)/6;
	b3[2]=(1+sigma)*(2+sigma)/12;
	
	return b3

def B3d(sigma):
	'''
	B3 coefficient derivative
	sigma: slope
	'''
	b3d=np.zeros[2];
	b3d[0]=-(2-sigma)/12-(1-sigma)/12;
	b3d[1]=(2-sigma)/6-(2+sigma)/6;
	b3d[2]=(2+sigma)/12+(1+sigma)/12;

	return b3d

def B5(sigma):
	'''
	B5 coefficient
	sigma: slope
	'''
	b5=np.zeros(5);
	b5[0]=(1-sigma)*(2-sigma)*(3-sigma)*(4-sigma)/1680;
	b5[1]=(4-sigma)*(2-sigma)*(3-sigma)*(4+sigma)/420;
	b5[2]=(4-sigma)*(3-sigma)*(3+sigma)*(4+sigma)/280;
	b5[3]=(4-sigma)*(2+sigma)*(3+sigma)*(4+sigma)/420;
	b5[4]=(1+sigma)*(2+sigma)*(3+sigma)*(4+sigma)/1680;

	return b5

def B5d(sigma):
	'''
	B5 coefficient derivative
	sigma: slope
	'''
	b5d=np.zeros(5);
	b5d[0]=-(2-sigma)*(3-sigma)*(4-sigma)/1680-\
	(1-sigma)*(3-sigma)*(4-sigma)/1680-\
	(1-sigma)*(2-sigma)*(4-sigma)/1680-\
	(1-sigma)*(2-sigma)*(3-sigma)/1680;

	b5d[1]=-(2-sigma)*(3-sigma)*(4+sigma)/420-\
	(4-sigma)*(3-sigma)*(4+sigma)/420-\
	(4-sigma)*(2-sigma)*(4+sigma)/420+\
	(4-sigma)*(2-sigma)*(3-sigma)/420;

	b5d[2]=-(3-sigma)*(3+sigma)*(4+sigma)/280-\
	(4-sigma)*(3+sigma)*(4+sigma)/280+\
	(4-sigma)*(3-sigma)*(4+sigma)/280+\
	(4-sigma)*(3-sigma)*(3+sigma)/280;

	b5d[3]=-(2+sigma)*(3+sigma)*(4+sigma)/420+\
	(4-sigma)*(3+sigma)*(4+sigma)/420+\
	(4-sigma)*(2+sigma)*(4+sigma)/420+\
	(4-sigma)*(2+sigma)*(3+sigma)/420;

	b5d[4]=(2+sigma)*(3+sigma)*(4+sigma)/1680+\
	(1+sigma)*(3+sigma)*(4+sigma)/1680+\
	(1+sigma)*(2+sigma)*(4+sigma)/1680+\
	(1+sigma)*(2+sigma)*(3+sigma)/1680;

	return b5d
