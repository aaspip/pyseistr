import numpy as np
EPSILON=1.e-12;

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
	multidivn: multidimensional division regression
	
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
	
	
def npef(din, filt=None, filt_pch=None, filt_lag=None, pch=None, epsilon=0.01, a=3, center=None, gap=None, niter=100, ifmaskout=False):
	'''
	npef: Estimate Non-stationary PEF in N dimensions.
	
	INPUT
	din: input data
	filt: double-helix filter,			an float array  (must input)
	filt_pch: double-helix filter pch, 	an float array  (optional input)
	filt_lag: double-helix filter lags,	an int array 	(must input)
	pch: patch file (?), 				an int array    (optional input)
	epsilon: regularization parameter (default: 0.01)
	a:	  [dimension], e.g., 3,3,3   (integer array of size dim)
	center: [dimension], e.g., 3,3,3 (integer array of size dim)
	gap:	[dimension], e.g., 0,0,0 (integer array of size dim)
	niter: number of iterations 
	ifmaskout: if output mask (currently disabled)
	
	OUTPUT
	dout: output data
	lag: 	lag file (int array)
	
	EXAMPLE
	neqdemos/test_19_npef.py
	
	
	'''
	
	if filt_lag is None:
		Exception("filt_lag is necessary")
	else:
		filt_lag=filt_lag.astype(np.int_)
	
	
	from .bp import ifnot
	dim=np.ndim(din)  #e.g., 1D/2D
	n=din.shape
	
	if center is None:
		center=np.zeros(dim,dtype=np.int_)
		for i in range(dim):
			center[i]=ifnot(i+1 < dim and a[i+1]>1, a[i]/2, 0)
	else:
		center=np.array(center).astype(np.int_)
	
	if gap is None:
		gap=np.zeros(dim,dtype=np.int_)
	else:
		gap=np.array(gap).astype(np.int_)
	
	n123=din.size
	
	dd=np.zeros(n123, dtype=np.float_)
	kk=np.zeros(n123, dtype=np.int_)
	
	dd=din.flatten(order='F') #input data, 1D array
	dabs=np.abs(dd).max()
	
	dd=dd/(dabs+100*EPSILON)
	
	pp=np.zeros(n123,dtype=np.int_)
	if pch is not None:
		pp=pch.flatten(order='F')
		
		if pch is not None:
			pp=pch;
			nnp=pp.max()
		else:
			pp=np.array(range(n123))
			nnp=n123;
	else:
		nnp=n123;
		pp=np.array(range(n123), dtype=np.int_)
		
	aa=createnhelix(dim, n, center, gap, a, pp)
# 	print(aa.keys())
	nf= aa['hlx'][0]['nh']

	print("dim=%d"%dim);
	print("n=%d"%n[0]);
	print("center=%d"%center[0]);
	print("gap=%d"%gap[0]);
	print("a=%d"%a[0]);
	print("pp=%d"%pp[0]);
#   
# 	print("main: type(aa['hlx'][0]['flt'])",type(aa['hlx'][0]['flt']))
	aa=nfind_mask(n123, kk, aa);
	print('aa.keys',aa.keys())
	####### output lag
	lag=[]
# 	for ip in range(nnp):
# 		lag.append(aa['hlx'][ip]['lag'])
# 	lag=np.array(lag, dtype=np.int_)
	####### output lag
	
	
	eps=epsilon;eps=0.01;
	print("niter=",niter,"epsilon",eps);
	

	nbf=filt.shape[0]
	if np.ndim(filt)==1:
		nbp=1
		filt=filt[:,np.newaxis]
	
	if filt_pch is not None:
		pp=filt_pch.flatten(order='F')
		pp=np.zeros(nnp, dtype=np.int_)
	else:
		pp=None
	
	pch=np.zeros(nf*nnp, dtype=np.int_);
	nh=np.zeros(nbp, dtype=np.int_)
	
	for i in range(nbp):
		nh[i]=nbf;
	
	iid=0;
	for ig in range(nf):
		for ip in range(nnp):
			pch[iid]=ifnot(pp is not None,pp[ip],ip);
			iid=iid+1;
	
	print('before nallocate, nbp, nf, nnp, nh',nbp, nf, nnp, nh);
	print('len(pch)',len(pch))
	print('pch',pch)
	bb = nallocate (nbp, nf*nnp, nh, pch);
	print('npef:',type(bb['hlx'][0]['flt']),bb['hlx'][0]['flt'])
	for ip in range(nbp):
		kk[0:nbf]=filt_lag[0:nbf]
		for i in range(nbf):
			bb['hlx'][ip]['lag'][i]=kk[i]*nf;
	print('nf=',nf)
	
	for ip in range(nbp):
		bb['hlx'][ip]['flt']=filt[:,ip]

	print('n123=',n123)
	print('dd=',dd)
	print('aa',aa)
	print('bb',bb)
	print('niter=',niter)
	print('eps=',eps)
	print('nf=',nf)
	
	
	##define aa
	dim=1;n=[400];center=[0];gap=[0];a=[3];pp=list(range(400));
	aa = createnhelix(dim, n, center, gap, a, pp);
	
	##define bb
	nbp=1; nf=2; nnp=400; nh=[1];
	pch=np.zeros(800);
	rr=nallocate (nbp, nf*nnp, nh, pch);

	rr['hlx'][0]['flt'][0]=-1.0
	rr['hlx'][0]['lag'][0]=2
	bb=rr;
	
	aa=nfind_pef (n123, dd, aa, bb, niter, eps, nf);
	
	print('nf:',nf, 'np:', nnp)
	dout=[]
	for ip in range(nnp):
		dout.append(aa['hlx'][ip]['flt'])
# 	dout=din;
# 	lag=dout;
	dout=np.array(dout)
	dout=dout.T
	
	#dout: [nf,nnp]
	#lag:  [nf,nnp]
	
	return dout,lag,aa,bb
	
def createnhelix(dim, nd, center, gap, na, pch):
	'''
	createnhelix:  allocate and output a non-stationary filter (Correct!)
	
	INPUT
	dim:	number of dimension (INT number)
	nd:	  	data size (INT numpy array of size ndim)
	center: filter center (INT numpy array of size ndim)
	gap:	filter gap (INT numpy array of size ndim)
	na:		filter size (INT numpy array of size ndim)
	pch:	patching [product(nd)] (INT numpy array of size product[nd])
	
	OUTPUT
	nsaa:		the helix filter
	
	
	EXAMPLE
	from pyseistr import createnhelix
	aa=createnhelix(1,[400], [0], [0], [3], list(range(400)))
	'''
	from .struct import nhelix
	aa=createhelix(dim, nd, center, gap, na);
	
	n123=1;
	for i in range(dim):
		n123=n123*nd[i];
	
	nnp=max(pch)+1;
	
	nh=np.zeros(nnp, dtype=np.int_)
	for ip in range(nnp):
		nh[ip]=aa['nh']

	nsaa=nallocate(nnp, n123, nh, pch);

	for ip in range(0,nnp):
		for i in range(aa['nh']):
			nsaa['hlx'][ip]['lag'][i]=aa['lag'][i];
		nsaa=nbound(ip, dim, nd, na, nsaa);

# 	print("createnhelix: type(aa['hlx'][0]['flt'])",type(nsaa['hlx'][0]['flt']))
	
	return nsaa

def nallocate(nnp, nd, nh, pch):
	'''
	allocate a non-stationary helix filter
	
	INPUT
	nnp: 	number of patches
	nd:	    data size, INT array
	nh:		filter size [np], INT array
	pch:	patching [nd], INT array
	
	OUTPUT
	aa
	'''
	from .struct import nhelix, helix
	
	aa=nhelix();
	aa['np']=nnp;
	
	aa['hlx']=[]; #aa['hlx'] is a list of helix filters
	for ii in range(nnp):
		aa['hlx'].append(helix(nh[ii]))

	print("nh",nh)
# 	print("nallocate: type(aa['hlx'][0]['flt'])",type(aa['hlx'][0]['flt']))
	
	print('nd',nd)
	aa['pch']=np.zeros(nd, dtype=np.int_)
	for iid in range(nd):
		aa['pch'][iid] = pch[iid]
	
	aa['mis']=None
	
	return aa

def nbound(ip, dim, nd, na, aa):
	'''
	define boundaries
	
	INPUT
	ip:		patch number
	dim: 	number of dimensions
	nd:		data size [ndim]
	na:		filter size [dim]
	aa:		non-stationary helix filteer
	
	OUTPUT
	aa:		output filter
	'''
	
	n=1;
	for i in range(dim):
		n=n*nd[i]
		
	aa['mis']=np.zeros(n, dtype=np.int_)
	bb=aa['hlx'][ip]
	
	bb = bound(dim, False, nd, nd, na, bb);
	
	for i in range(n):
		aa['mis'][i] = bb['mis'][i]
		
# 	del bb
	bb['mis']=None;
	
	return aa

def bound(dim, both, nold, nd, na, aa):
	'''
	bound: Mark helix filter outputs where input is off data.
	
	INPUT
	dim: 	number of dimensions
	both:	[bool] if both input and output
	nold:	old data coordinate [dim]
	nd:		data size [ndim]
	na:		filter size [dim]
	aa:		non-stationary helix filteer
	
	OUTPUT
	aa:		output filter
	'''
	from .struct import helix
	from .coords import cart2line, line2cart
	from .operators import helicon_lop
	
	nb=np.ones(3, dtype=np.int_)
	my=1;mb=1;
	for i in range(dim):
		nb[i]=nd[i]+2*na[i];
		mb=mb*nb[i]
		my=my*nb[i]
		
	xx=np.zeros(mb, dtype=np.float_)
	yy=np.zeros(mb, dtype=np.float_)
	
	for ib in range(mb):
		ii=line2cart(dim, nb, ib);  
		xx[ib]=0.0;
		for i in range(dim):
			if ii[i]+1 <= na[i] or ii[i]+1 > nb[i]-na[i]:
				xx[ib]=1.0;
				break;
	
	par={'nm':mb,'nd':mb,'aa': aa} #parameter for helicon filtering operator
	aa=regrid(dim, nold, nb, aa); 
	
	for i in range(aa['nh']):
		aa['flt'][i]=1.0;
	
	yy=helicon_lop(xx, par, 0, 0);
	aa=regrid(dim, nb, nd, aa);
	
	for i in range(aa['nh']):
		aa['flt'][i] = 0.0;
		
	aa['mis']=np.zeros(my, dtype=np.bool_)
	
	for iy in range(my):
		ii=line2cart(dim, nd, iy);
		for i in range(dim):
			ii[i]=ii[i]+na[i];
		ib=cart2line(dim, nb, ii);
		if both:
			aa['mis'][iy] = bool (yy[ib] > 0. or xx[ib] > 0.);
		else:
			aa['mis'][iy] = bool (yy[ib] > 0.);
	
	return aa

def createhelix(ndim, nd, center, gap, na):
	'''
	createhelix: allocate and output a helix filter
	
	INPUT
	ndim:	number of dimension (INT number)
	nd:	  	data size (INT numpy array of size ndim)
	center: filter center (INT numpy array of size ndim)
	gap:	filter gap (INT numpy array of size ndim)
	na:		filter size (INT numpy array of size ndim)
	
	OUTPUT
	aa:		the helix filter
	
	
	EXAMPLE
	from pyseistr import createnhelix
	aa=createnhelix(2,[20,5], [5,1], [2,2], [2,2])
	
	'''
	from .coords import cart2line, line2cart
	from .struct import helix
	
	na123=1;
	for i in range(ndim):
		na123=na123*na[i]
	
	lag=np.zeros(na123, dtype=np.int_) #this lag is a 2D array?
	
	# index pointing to the "1.0"
	lag0a = cart2line(ndim, na, center);
	
	nh=0;
	# loop over linear index
	for ia in range(1+lag0a, na123, 1):
		ii=line2cart(ndim, na, ia)
		skip=False;
		
		for i in range(ndim):
			if ii[i] < gap[i]:
				skip=True;
				break;
		if skip:
			continue;
		lag[nh] = cart2line(ndim, nd, ii);
		nh=nh+1;
	
	lag0d = cart2line(ndim,  nd, center);
	
	aa=helix(nh)
	for ia in range(nh):
		aa['lag'][ia] = lag[ia] - lag0d;
		aa['flt'][ia] = 0.;
	
	
	
	return aa
	

def regrid(dim, nold, nnew, aa):
	'''
	regrid: change data size
	
	INPUT
	dim:	number of dimensions
	nold:	old data size [dim], INT array
	nnew:	new data size [dim], INT array
	aa:		modified filter
	
	OUTPUT
	aa:		helix filter
	
	'''
	from .coords import cart2line, line2cart
	ii=np.zeros(3, dtype=np.int_)
	for i in range(dim):
		ii[i] = nold[i]/2-1;
	
	h0=cart2line( dim, nold, ii); # midpoint lag on nold 
	h1=cart2line( dim, nnew, ii); #             on nnew 
	
	for i in range(aa['nh']):
		h=aa['lag'][i]+h0;
		ii=line2cart( dim, nold, h);
		aa['lag'][i] = cart2line(dim,nnew,ii) - h1;
	
	return aa

def nfind_pef(nd, dd, aa, rr, niter, eps, nh):
	'''
	nfind_pef: estimate non-stationary PEF
	
	INPUT
	nd:		data size, 	INT
	dd:		data, 		FLOAT array
	aa:		estimated filter, nhelix filter
	rr:		regularization filter, nhelix filter
	niter:	number of iterations, INT
	eps:	regularization parameter, FLOAT
	nh:		filter size,			INT
	
	
	OUTPUT
	aa:		estimated filter, nhelix filter
	
	EXAMPLE
	TBD
	
	'''
	nnp=aa['np']
	nr=nnp*nh;
	flt=np.zeros(nr, dtype=np.float_)
	
	from .solvers import solver_prec, cgstep
	from .operators import nhconest_lop, npolydiv2_lop
	
	opL=nhconest_lop;opP=npolydiv2_lop;		#define forward and preconditioning operator
	par_L={'nm': nr, 'nd': nd, 'x': dd, 'aa': aa, 'nhmax': nh};	#parameter file of forward operator
	par_P={'nm': nr, 'nd': nr, 'aa':rr, 'tt': np.zeros(nr)}		#parameter dic of preconditioning operator
	par_sol={'verb': 1}		#parameter dic of the solver
	
	flt,tmp=solver_prec(opL, cgstep, opP, nr, nr, nd, flt, dd, niter, eps, par_L, par_P, par_sol);
# 	print('flt.shape',flt.shape)
# 	print("aa['hlx'][ip]['flt'].shape",aa['hlx'][0]['flt'].shape)
	for ip in range(nnp):
		na=aa['hlx'][ip]['nh']
		for ih in range(na):
			aa['hlx'][ip]['flt'][ih] = -flt[ip*nh+ih];
	
	return aa

def find_mask(n, known, aa):
	'''
	find_mask: create a filter mask
	
	INPUT
	n:		data size, INT
	known:	mask for known data, INT array
	aa:		helical filter
	
	OUTPUT
	aa:		helical filter
	'''
	from .bp import ifnot
	from .operators import helicon_lop
	
	rr=np.zeros(n, dtype=np.float_)
	dfre=np.zeros(n, dtype=np.float_)
	
	for i in range(n):
		dfre[i] = ifnot(known[i], 0, 1);
	
	par_helicon={'nm': n, 'nd': n, 'aa': aa}
	print(aa.keys())
	
	rr=helicon_lop(dfre,par_helicon,0,0);
	for ih in range(aa['nh']):
		aa['flt'][ih] = 0.;
		
	for i in range(n):
		if rr[i]>0:
			aa['mis'][i] = True;
	
	return 

def nfind_mask(nd, known, aa):
	'''
	find_mask: create a filter mask
	
	INPUT
	nd:		data size, INT
	known:	mask for known data, INT array
	aa:		helical filter
	
	OUTPUT
	aa:		helical filter
	'''
	from .bp import ifnot
	from .operators import nhelicon_lop
	
	rr=np.zeros(nd, dtype=np.float_)
	dfre=np.zeros(nd, dtype=np.float_)
	
	for i in range(nd):
		dfre[i] = ifnot(known[i], 0, 1);
	
	par_nhelicon={'nm': nd, 'nd': nd, 'aa': aa}
# 	print(aa.keys())
	
	for ip in range(aa['np']):
		for i in range(aa['hlx'][ip]['nh']):
# 			print(aa.keys())
# 			print("type(aa['hlx'][ip]['flt'])",type(aa['hlx'][ip]['flt']))
			aa['hlx'][ip]['flt'][i]=1.0;
	
	rr=nhelicon_lop(dfre,par_nhelicon,0,0);
		
	for i in range(nd):
		if rr[i]>0:
			aa['mis'][i] = True;

	for ip in range(aa['np']):
		for i in range(aa['hlx'][ip]['nh']):
			aa['hlx'][ip]['flt'][i] = 0.;
		
	return aa








