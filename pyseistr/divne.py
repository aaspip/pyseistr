import numpy as np
from .operators import adjnull

def divne(num, den, Niter, rect, ndat, eps_dv, eps_cg, tol_cg,verb):
	'''
	divne: N-dimensional smooth division rat=num/den		  
	This is a subroutine from the seistr package (https://github.com/chenyk1990/seistr)
	
	by Yangkang Chen, 2022, verified to be exactly the same as the Matlab version
	
	INPUT
	num: numerator
	den: denominator
	Niter: number of iterations
	rect: triangle radius [ndim], e.g., [5,5,1]
	ndat: data dimensions [ndim], e.g., [n1,n2,1]
	eps_dv: eps for divn  (default: 0.01)
	eps_cg: eps for CG	(default: 1)
	tol_cg: tolerence for CG (default: 0.000001)
	verb: verbosity flag
	
	OUTPUT
	rat: output ratio
	 
	REFERENCE
	H. Wang, Y. Chen, O. Saad, W. Chen, Y. Oboue, L. Yang, S. Fomel, and Y. Chen, 2022, A Matlab code package for 2D/3D local slope estimation and structural filtering. Geophysics, 87(3), F1–F14, doi: 10.1190/geo2021-0266.1. 
	
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

	dip=ps.dip2d(dtemp,rect=[10,20,1]); #dip2d uses divne
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
	
	n=num.size
	
	ifhasp0=0
	p=np.zeros(n)
	
	num=num.reshape(n,order='F')
	den=den.reshape(n,order='F')
	
	if eps_dv > 0.0:
		for i in range(0,n):
			norm = 1.0 / np.hypot(den[i], eps_dv);
			num[i] = num[i] * norm;
			den[i] = den[i] * norm;
	norm=np.sum(den*den);
	if norm == 0.0:
		rat=np.zeros(n);
		return rat
	norm = np.sqrt(n / norm);
	num=num*norm;
	den=den*norm;
		
	par_L={'nm':n,'nd':n,'w':den}
	par_S={'nm':n,'nd':n,'nbox':rect,'ndat':ndat,'ndim':3}
	
	from .solvers import conjgrad
	rat = conjgrad(None, weight_lop, trianglen_lop, p, None, num, eps_cg, tol_cg, Niter,ifhasp0,[],par_L,par_S,verb);
	rat=rat.reshape(ndat[0],ndat[1],ndat[2],order='F')

	return rat


def weight_lop(din,par,adj,add):
	'''
	weight_lop: Weighting operator (verified)
	
	Ported to Python by Yangkang Chen, 2022
	
	INPUT
	din: model/data
	par: parameter
	adj: adj flag
	add: add flag
	
	OUTPUT
	dout: data/model
	'''
	nm=par['nm'];
	nd=par['nd'];
	w=par['w'];

	if adj==1:
		d=din;
		if 'm' in par and add==1:
			m=par['m'];
		else:
			m=np.zeros(par['nm']);
	else:
		m=din;
		if 'd' in par and add==1:
			d=par['d'];
		else:
			d=np.zeros(par['nd']);
	m,d  = adjnull( adj,add,nm,nd,m,d );
	if adj==1:
		m=m+d*w; #dot product
	else: #forward
		d=d+m*w; #d becomes model, m becomes data


	if adj==1:
		dout=m;
	else:
		dout=d;

	return dout
	
def trianglen_lop(din,par,adj,add ):
	'''
	trianglen_lop: N-D triangle smoothing operator (verified)
	
	Ported to Python by Yangkang Chen, 2022
	
	INPUT
	din: model/data
	par: parameter
	adj: adj flag
	add: add flag
	
	OUTPUT
	dout: data/model
	'''
	if adj==1:
		d=din;
		if 'm' in par and add==1:
			m=par['m'];
		else:
			m=np.zeros(par['nm']);
	else:
		m=din;
		if 'd' in par and add==1:
			d=par['d'];
		else:
			d=np.zeros(par['nd']);


	nm=par['nm'];	 #int
	nd=par['nd'];	 #int
	ndim=par['ndim']; #int
	nbox=par['nbox']; #vector[ndim]
	ndat=par['ndat']; #vector[ndim]

	[ m,d ] = adjnull( adj,add,nm,nd,m,d );

	tr = [];

	s =[1,ndat[0],ndat[0]*ndat[1]];

	for i in range(0,ndim):
		if (nbox[i] > 1):
			nnp = ndat[i] + 2*nbox[i];
			wt = 1.0 / (nbox[i]*nbox[i]);
			tr.append({'nx':ndat[i], 'nb':nbox[i], 'box':0, 'np':nnp, 'wt':wt, 'tmp':np.zeros(nnp)});
		else:
			tr.append('NULL');

	if adj==1:
		tmp=d;
	else:
		tmp=m;

	for i in range(0,ndim):
		if tr[i] != 'NULL':
			for j in range(0,int(nd/ndat[i])):
				i0=first_index(i,j,ndim,ndat,s);
				[tmp,tr[i]]=smooth2(tr[i],i0,s[i],0,tmp);
	
	if adj==1:
		m=m+tmp;
	else:
		d=d+tmp;
		
	if adj==1:
		dout=m;
	else:
		dout=d;

	return dout


def first_index( i, j, dim, n, s ):
	'''
	first_index: Find first index for multidimensional transforms
	Ported to Python by Yangkang Chen, 2022
	
	INPUT
	i:	dimension [0...dim-1]
	j:	line coordinate
	dim:  number of dimensions
	n:	box size [dim], vector
	s:	step [dim], vector
	
	OUTPUT
	i0:   first index
	'''

	n123 = 1;
	i0 = 0;
	for k in range(0,dim):
		if (k == i):
			continue;
		ii = np.floor(np.mod((j/n123), n[k]));
		n123 = n123 * n[k];
		i0 = i0 + ii * s[k];

	return int(i0)

def triangle_init(nbox,ndat,box):
	'''
	triangle smoother initialization
	Ported to Python by Yangkang Chen, 2023
	
	INPUT
	nbox:	triangle length
	ndat:	data length
	box:	if box instead of triangle
	
	OUTPUT
	tr:		triangle struct
	
	EXAMPLE
	from pyseistr import smooth
	
	'''
	if box:
		wt=1.0/(2*nbox-1);
	else:
		wt=1.0/(nbox*nbox);
	
	tr={'nx':ndat, 'nb':nbox, 'box':box, 'np':ndat+2*nbox, 'wt':wt, 'tmp':np.zeros(ndat+2*nbox)}
	
	return tr

def smooth2( tr, o, d, der, x):
	'''
	smooth2: apply adjoint triangle smoothing
	
	Ported to Python by Yangkang Chen, 2022
	
	INPUT
	tr:   smoothing object
	o:	trace sampling
	d:	trace sampling
	x:	data (smoothed in place)
	der:  if derivative
	
	OUTPUT
	x: smoothed result
	tr: triangle struct
	'''

	tr['tmp'] = triple2(o, d, tr['nx'], tr['nb'], x, tr['tmp'], tr['box'], tr['wt']);
	tr['tmp'] = doubint2(tr['np'], tr['tmp'], (tr['box'] or der));
	x = fold2(o, d, tr['nx'], tr['nb'], tr['np'], x, tr['tmp']);

	return x,tr

def smooth( tr, o, d, der, x):
	'''
	smooth: apply triangle smoothing
	
	Ported to Python by Yangkang Chen, 2023
	
	INPUT
	tr:   smoothing object
	o:	trace sampling
	d:	trace sampling
	x:	data (smoothed in place)
	der:  if derivative
	
	OUTPUT
	x: smoothed result
	tr: triangle struct
	'''
	tr['tmp'] = fold(o, d, tr['nx'], tr['nb'], tr['np'], x, tr['tmp']);
	tr['tmp'] = doubint(tr['np'], tr['tmp'], (tr['box'] or der));
	x = triple(o, d, tr['nx'], tr['nb'], x, tr['tmp'], tr['box'], tr['wt']);
	
	return x,tr

def triple2( o, d, nx, nb, x, tmp, box, wt ):
	'''
	BY Yangkang Chen, Nov, 04, 2019
	'''
	for i in range(0,nx+2*nb):
		tmp[i] = 0;

	if box:
		tmp[1:]	 = cblas_saxpy(nx,  +wt,x[o:],d,tmp[1:],   1); 	#y += a*x
		tmp[2*nb:]  = cblas_saxpy(nx,  -wt,x[o:],d,tmp[2*nb:],1);
	else:
		tmp		 = cblas_saxpy(nx,  -wt,x[o:],d,tmp,	   1); 	#y += a*x
		tmp[nb:]	= cblas_saxpy(nx,2.*wt,x[o:],d,tmp[nb:],  1);
		tmp[2*nb:]  = cblas_saxpy(nx,  -wt,x[o:],d,tmp[2*nb:],1);

	return tmp

def triple( o, d, nx, nb, x, tmp, box, wt ):
	'''
	BY Yangkang Chen, Jun, 06, 2023
	'''

	if box:
		for i in range(0,nx):
			x[o+i*d]=(tmp[i+1]-tmp[i+2*nb])*wt;
	else:
		for i in range(0,nx):
			x[o+i*d]=(2.0*tmp[i+nb]-tmp[i]-tmp[i+2*nb])*wt;

	return x
	
def doubint2( nx, xx, der ):
	'''
	Modified by Yangkang Chen, Nov, 04, 2019
	'''
	#integrate forward
	t = 0.0;
	for i in range(0,nx):
		t = t + xx[i];
		xx[i] = t;

	if der:
		return xx

	#integrate backward
	t = 0.0;
	for i in range(nx-1,-1,-1):
		t = t + xx[i];
		xx[i] = t

	return xx

def doubint( nx, xx, der ):
	'''
	Modified by Yangkang Chen, Jun, 06, 2023
	'''
	
	#integrate backward
	t = 0.0;
	for i in range(nx-1,-1,-1):
		t = t + xx[i];
		xx[i] = t

	if der:
		return xx
		
	#integrate forward
	t = 0.0;
	for i in range(0,nx):
		t = t + xx[i];
		xx[i] = t;
	
	return xx

def cblas_saxpy( n, a, x, sx, y, sy ):
	'''
	y += a*x
	Modified by Yangkang Chen, Nov, 04, 2019
	'''
	for i in range(0,n):
		ix = i * sx;
		iy = i * sy;
		y[iy] = y[iy] + a * x[ix];

	return y

def fold2(o, d, nx, nb, np, x, tmp):
	'''
	Modified by Yangkang Chen, Nov, 04, 2019
	'''
	#copy middle
	for i in range(0,nx):
		x[o+i*d] = tmp[i+nb];

	#reflections from the right side
	for j in range(nb+nx,np+1,nx):
		if (nx <= np-j):
			for i in range(0,nx):
				x[o+(nx-1-i)*d] = x[o+(nx-1-i)*d] + tmp[j+i];
		else:
			for i in range(0,np-j):
				x[o+(nx-1-i)*d] = x[o+(nx-1-i)*d] + tmp[j+i];
		j = j + nx;
		if (nx <= np-j):
			for i in range(0,nx): 
				x[o+i*d] = x[o+i*d] + tmp[j+i];
		else:
			for i in range(0,np-j):
				x[o+i*d] = x[o+i*d] + tmp[j+i];

	#reflections from the left side
	for j in range(nb,-1,-nx):
		if (nx <= j):
			for i in range(0,nx):
				x[o+i*d] = x[o+i*d] + tmp[j-1-i];
		else:
			for i in range(0,j):
				x[o+i*d] = x[o+i*d] + tmp[j-1-i];
		j = j - nx;
		if (nx <= j):
			for i in range(0,nx):
				x[o+(nx-1-i)*d] = x[o+(nx-1-i)*d] + tmp[j-1-i];
		else:
			for i in range(0,j):
				x[o+(nx-1-i)*d] = x[o+(nx-1-i)*d] + tmp[j-1-i];
	return x

def fold(o, d, nx, nb, np, x, tmp):
	'''
	Modified by Yangkang Chen, Jun, 06, 2023
	'''
	#copy middle
	for i in range(0,nx):
		tmp[i+nb] = x[o+i*d];

	#reflections from the right side
	for j in range(nb+nx,np+1,nx):
		if (nx <= np-j):
			for i in range(0,nx):
				tmp[j+i] = x[o+(nx-1-i)*d];
		else:
			for i in range(0,np-j):
				tmp[j+i] = x[o+(nx-1-i)*d];
		j = j + nx;
		if (nx <= np-j):
			for i in range(0,nx): 
				tmp[j+i] = x[o+i*d];
		else:
			for i in range(0,np-j):
				tmp[j+i] = x[o+i*d];

	#reflections from the left side
	for j in range(nb,-1,-nx):
		if (nx <= j):
			for i in range(0,nx):
				tmp[j-1-i] = x[o+i*d];
		else:
			for i in range(0,j):
				tmp[j-1-i] = x[o+i*d];
		j = j - nx;
		if (nx <= j):
			for i in range(0,nx):
				tmp[j-1-i] = x[o+(nx-1-i)*d];
		else:
			for i in range(0,j):
				tmp[j-1-i] = x[o+(nx-1-i)*d];
	return tmp

	
