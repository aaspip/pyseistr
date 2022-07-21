import numpy as np
def pwspray2d(u1,dip,nr,order,eps):
	# pwspray2d: 2D plane-wave spray operator
	#
	# INPUT:
	# u1: noisy data  
	# dip: slope 
	# nr: spray radius
	# order: PWD order
	# eps: regularization (default:0.01);
	#
	# OUTPUT:
	# u: smooth data
   
	[n1,n2]=u1.shape;
	ns=nr;	 	#spray radius
	ns2=2*ns+1;	#spray diameter

	n=n1*n2;nu=n1*n2*ns2;

	u=np.zeros(n1*ns2*n2);

	p=dip;
	trace=np.zeros(n1);
	e=eps*eps;
	nw=order;

	if n!=n1*n2:
		print('Wrong size 	%d != 	%d*	%d'%(n,n1,n2)); 


	if nu!=n*ns2:
		print('Wrong size 	%d != 	%d*	%d'%(nu,n,ns2)); 

	u1=u1.flatten(order='F')
	for i in range(0,n2):
		for i1 in range(0,n1):
			trace[i1]=u1[i*n1+i1];
			u[(i*ns2+ns)*n1+i1] = u[(i*ns2+ns)*n1+i1] + trace[i1];
	
		#predict forward
		#corresponding to the eq.24
		for iis in range(0,ns):
			ip=i-iis-1;
			if ip<0:
				break;
			j=ip*ns2+ns-iis-1;
			[w,diag,offd,trace] = predict_step(e,nw,0,p[:,ip],trace);
			for i1 in range(0,n1):
				u[j*n1+i1] = u[j*n1+i1] + trace[i1];
	
		for i1 in range(0,n1):
			trace[i1]=u1[i*n1+i1];

	
		# predict backward
		# 	#corresponding to the eq.25
		for iis in range(0,ns):
			ip=i+iis+1;
			if ip>=n2:
				break;
			j=ip*ns2+ns+iis+1;
			[w,diag,offd,trace] = predict_step(e,nw,1,p[:,ip-1],trace);
			for i1 in range(0,n1):
				u[j*n1+i1]= u[j*n1+i1]+trace[i1];

	return u

def predict_step(e,nw,forw,pp,trace1):
	# predict_step: prediction step
	#
	# INPUT:
	# e: regularization parameter (default, 0.01*0.01);
	# nw: accuracy order
	# forw: forward or backward
	# n1: trace length
	# pp: slope
	# trace: input trace
	# 
	# OUTPUT:
	# w: PWD object
	# diag,offd: diagonal/offdiagonals of the banded matrix
	# trace: output trace
	#
	n1=trace1.shape[0];
	nb=2*nw;
	eps=e;
	eps2=e;

	diag=np.zeros(n1);
	offd=np.zeros([n1,nb]);


	diag,offd = regularization(diag,offd,nw,eps,eps2);


	#only define diagonal,offdiagonal

	w,diag,offd = pwd_define(forw,diag,offd,n1,nw,pp);
   

	t0=trace1[0];
	t1=trace1[1];
	t2=trace1[n1-2];
	t3=trace1[n1-1];

	trace = pwd_set(0,w,diag,offd,pp,trace1);


	trace[0]=trace[0]+eps2*t0;
	trace[1]=trace[1]+eps2*t1;

	trace[n1-2]=trace[n1-2]+eps2*t2;
	trace[n1-1]=trace[n1-1]+eps2*t3;


	trace = banded_solve(n1,nb,diag,offd,trace);   

	return w,diag,offd,trace


def regularization(diag,offd,nw,eps,eps2):
	# fill diag and offd using regularization 
	# 
	# INPUT:
	# diag: defined diagonal .   (1D array)
	# offd: defined off-diagonal (2D array)
	# eps: regularization parameter (default: e*e, e=0.01);
	# eps2: second regularization parameter (default, same as eps)
	# nw: accuracy order (nb=2*nw)
	#
	# OUTPUT:
	# diag: defined diagonal .   (1D array)
	# offd: defined off-diagonal (2D array)
	#

	nb=2*nw;
	n1=diag.size;

	for i1 in range(0,n1):
		diag[i1]=6*eps;
		offd[i1,0]=-4*eps;
		offd[i1,1]=eps;
	
		for ib in range(2,nb):
			offd[i1,ib]=0.0;

		diag[0]=eps2+eps;
		diag[1]=eps2+5*eps;
	
		diag[n1-1]=eps2+eps;
		diag[n1-2]=eps2+5*eps;  
	
		offd[0,0]=-2*eps;
		offd[n1-2,0]=-2*eps;

	return diag,offd

def pwd_define(forw,diag,offd,n1,nw,pp):
	# pwd_define: matrix multiplication
	# 
	# INPUT:
	# forw:forward or backward
	# diag: defined diagonal .   (1D array)
	# offd: defined off-diagonal (2D array)
	# n1: trace length
	# nw: PWD filter(accuracy) order (default nw=1)
	# pp: slope				  (1D array)
	#
	# OUTPUT:
	# diag: defined diagonal .   (1D array)
	# offd: defined off-diagonal (2D array)
	# w: PWD object(struct)
	#
	#
	
	#define PWD object(struct)
	w={'n':n1,'na':2*nw+1,'a':np.zeros([n1,2*nw+1]),'b':np.zeros(2*nw+1)}
	
	n=w['n'];
	nb=2*nw;

	for i in range(0,n):
		w['b']=passfilter(pp[i],nw)[0];
		for j in range(0,w['na']):
			if forw:
				w['a'][i,j]=w['b'][w['na']-j-1];
			else:
				w['a'][i,j]=w['b'][j];

	for i in range(0,n):
		for j in range(0,w['na']):
			k=i+j-nw;
			if k>=nw and k<n-nw:
				aj=w['a'][k,j];
				diag[i]=diag[i]+aj*aj;
		for m in range(0,2*nw):
			for j in range(m+1,w['na']):
				k=i+j-nw;
				if k>=nw and k<n-nw:
					aj=w['a'][k,j];
					am=w['a'][k,j-m-1];
					offd[i,m]=offd[i,m]+am*aj;
	return w,diag,offd


def passfilter(p,nw):
	# passfilter: find filter coefficients 
	# All-pass plane-wave destruction filter coefficients
	# 
	# INPUT:
	# p: slope
	# nw: PWD filter(accuracy) order
	#
	# OUTPUT:
	# a: output filter (n+1) (1D array)
	# b: temp variable of a
	#

	n=nw*2;
	b=np.zeros(n+1);
	a=np.zeros(n+1);
	for k in range(0,n+1):
		bk=1;
		for j in range(0,n):
			if j<n-k:
				bk=bk*(k+j+1.0)/(2*(2*j+1)*(j+1));
			else:
				bk=bk*1.0/(2*(2*j+1));
		b[k]=bk;
	for k in range(0,n+1):
		ak=b[k];
		for j in range(0,n):
			if j<n-k:
				ak=ak*(n-j-p);
			else:
				ak=ak*(p+j+1);
		a[k]=ak;
	return a,b

def pwd_set(adj,w,diag,offd,pp,inp):
	# pwd_set: matrix multiplication
	# 
	# INPUT:
	# adj:adjoint flag
	# w: PWD object(struct)
	# diag: defined diagonal .   (1D array)
	# offd: defined off-diagonal (2D array)
	# pp: slope				  (1D array)
	# inp: model
	#
	# OUTPUT:
	# out: data 
	#

	n=w['n'];
	nw=np.int((w['na']-1)/2);

	#	# pwd_set
	tmp=np.zeros(n);
	out=np.zeros(n);
	if adj:
		for i in range(0,n):
			tmp[i]=0.0;

		for i in range(0,n):
			for j in range(0,w['na']):
				k=i+j-nw;
				if k>=nw and k<n-nw:
					tmp[k]=tmp[k]+w['a'][k,j]*inp[i];
		for i in range(0,n):
			out[i]=0.0;
		for i in range(nw,n-nw):
			for j in range(0,w['na']):
				k=i+j-nw;
				out[k]=out[k]+w['a'][i,j]*tmp[i];
	else:
		for i in range(0,n):
			tmp[i]=0.0;

		for i in range(nw,n-nw):
			for j in range(0,w['na']):
				k=i+j-nw;
				tmp[i]=tmp[i]+w['a'][i,j]*inp[k];
		for i in range(0,n):		
			out[i]=0.0;
			for j in range(0,w['na']):
				k=i+j-nw;
				if k>=nw and k<n-nw:
					out[i]=out[i]+w['a'][k,j]*tmp[k];
	return out

def banded_solve(n,band,diag,offd,b):
	# banded_solve: Banded matrix solver
	# 
	# INPUT:
	# n:	matrix size
	# band: band size
	# diag: defined diagonal .   (1D array)
	# offd: defined off-diagonal (2D array)
	# b: input trace
	#
	# OUTPUT:
	# b: trace solution
	#

	#define Band object(struct)
	slv={'n':n,'band':band,'d':np.zeros(n),'o':np.zeros([n-1,band])}

	# define the banded matrix
	for k in range(0,slv['n']):
		t=diag[k];
		m1=np.min([k,slv['band']]);
		for m in range(0,m1):
			t=t-slv['o'][k-m-1,m]*slv['o'][k-m-1,m]*slv['d'][k-m-1];

		slv['d'][k]=t;
		n1=np.min([slv['n']-k-1,slv['band']]);
		for n in range(0,n1):
			t=offd[k,n];
			m1=np.min([k,slv['band']-n-1]);
			for m in range(0,m1):
				t=t-slv['o'][k-m-1,m]*slv['o'][k-m-1,n+m+1]*slv['d'][k-m-1];

			slv['o'][k,n]=t/slv['d'][k];


	# the solver 
	for k in range(1,slv['n']):
		t=b[k];
		m1=np.min([k,slv['band']]);
		for m in range(0,m1):
			t=t-slv['o'][k-m-1,m]*b[k-m-1];
		b[k]=t;

	for k in range(slv['n']-1,-1,-1):
		t=b[k]/slv['d'][k];
		m1=np.min([slv['n']-k-1,slv['band']]);
		for m in range(0,m1):
			t=t-slv['o'][k,m]*b[k+m+1];

		b[k]=t;


	return b