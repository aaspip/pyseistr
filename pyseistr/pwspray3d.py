import numpy as np
def pwspray3d(din,dipi,dipx,ns2,ns3,order,eps):
	#pwspray3d: 3D plane-wave spray operator 
	# 
	# by Yangkang Chen, 2022
	#
	# INPUT:
	# din: input
	# dipi: inline slope
	# dipx: xline slope
	# ns2/3: smoothing radius
	# order: PWD order
	# eps: regularization (default:0.01);
	# 
	# OUTPUT:
	# dout: result
	# 
	[n1,n2,n3]=din.shape;
	n23=n2*n3;

	np2=2*ns2+1;	#spray diameter
	np3=2*ns3+1;	#spray diameter
	nnp=np2*np3;

	trace=np.zeros(n1);
	e=eps*eps;
	nw=order;

	# traveltime table
	t=np.zeros([np2,np3]);
	for i3 in range(0,np3):
		for i2 in range(0,np2):
			t[i2,i3] = np.hypot(i2-ns2,i3-ns3); 	#square root of the sum of squares

	visit=np.argsort(t.flatten(order='F'));
	u=np.zeros([n1,nnp,n23]);
	din=din.reshape(n1,n23,order='F')
	p1=dipi.reshape(n1,n23,order='F')
	p2=dipx.reshape(n1,n23,order='F')

	for ii in range(0,n23):
	
		u[:,ns3*np2+ns2,ii]=din[:,ii]; 	#central trace in the predicted dimension
	
		i2=np.mod(ii,n2);
		i3=np.int64(np.floor(ii/n2));

		for ip in range(0,nnp):

			[up, jp, up2, up3 ] = get_update(ip, np2, np3, t.flatten(order='F'), visit);
# 			print(ii,i2,i3,ip,up,jp,up2,up3)
			# from jp to j
			k2=np.mod(jp,np2);
			k3=np.int64(np.floor(jp/np2));
		
			j2=i2+k2-ns2;
			j3=i3+k3-ns3;
				
			if j2<0 or j2>=n2 or j3<0 or j3>=n3:
				continue;
		
			j=j2+j3*n2;
			if up & 1:
				if up2:
					if j2==0:
						continue;
					j2=j-1;
					q2=p1[:,j2];
					k2=jp-1;
				else:
					if j2==n2-1:
						continue;

					j2=j+1;
					q2=p1[:,j]; ###### error?
					k2=jp+1;
		
			if up & 2:
				if up3:
					if j3==0:
						continue;

					j3=j-n2;
					q3=p2[:,j3];
					k3=jp-np2;
				else:
					if j3==n3-1:
						continue;

					j3=j+n2;
					q3=p2[:,j];###### error?
					k3=jp+np2;

			#corresponding to the eqs.24/25
			#prediction forward or backward
			if up==0:
					continue;
			elif up==1:
					u[:,jp,j] = predict1_step(e,nw,up2,q2,u[:,k2,j2]);
			elif up==2:
					u[:,jp,j] = predict1_step(e,nw,up3,q3,u[:,k3,j3]);
			elif up==3:
					u[:,jp,j] = predict2_step(e,nw,up2,up3,q2,q3,u[:,k2,j2],u[:,k3,j3]);
		
	dout=u.reshape(n1,nnp,n2,n3,order='F');

	return dout

def get_update(ii, n1,n2, t, visit):
	# forward or backward options
	# 
	# INPUT
	# ii: loop number
	# n1:   [n1,n2]=size(t)
	# n2:   [n1,n2]=size(t)
	# t:	traveltime
	# visit: sorted sequence of t
	# 
	# OUTPUT
	# jj: local linear index
	# up: prediction type
	# up1: forward or backward
	# up2: forward or backward


	n12=n1*n2;
	jj=visit[ii];
	t1=t[jj];

	i1=np.mod(jj,n1);
	i2=np.floor(jj/n1);

	up=0;
	if n1>1:
		a1=jj-1;
		b1=jj+1;
		
# 		print(ii)
# 		print(visit.shape)
# 		print(jj)
# 		print(a1)
# 		print((t[a1]>t[b1]))
		up1= (i1 and ((i1==n1-1) or (1!= (t[a1]>t[b1]))));
		if up1:
			c1=a1;
		else:
			c1=b1;

		if t1>t[c1]:
			up = up | 1; 	#Bit-wise OR.

	if n2>1:
		a2=jj-n1;
		b2=jj+n1;
	
		up2= (i2 and (i2==n2-1 or 1!= (t[a2]>t[b2])));
		if up2:
			c2=a2;
		else:
			c2=b2;

		if t1>t[c2]:
			up = up | 2; 	#Bit-wise OR.

	return up, jj, up1, up2

def predict1_step(e,nw,forw,pp,trace1):
	# predict1_step: prediction step from one trace
	# 
	# INPUT:
	# e: regularization parameter (default, 0.01*0.01);
	# nw: accuracy order
	# forw: forward or backward
	# pp: slope
	# trace1: input trace
	# 
	# OUTPUT:
	# trace: output trace
	#
	n1=trace1.shape[0];
	nb=2*nw;
	eps=e;
	eps2=e;

	diag=np.zeros(n1);
	offd=np.zeros([n1,nb]);

	diag,offd = regularization(diag,offd,nw,eps,eps2);

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

	return trace


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


def predict2_step(e,nw,forw1,forw2,pp1,pp2,trace1,trace2):
	# predict2_step: prediction step from two trace
	# 
	# INPUT:
	# e: regularization parameter (default, 0.01*0.01);
	# nw: accuracy order
	# forw1: forward or backward
	# forw2: forward or backward
	# pp1: slope
	# pp2: slope
	# trace1: input trace
	# trace2: input trace
	# 
	# OUTPUT:
	# trace: output trace
	#
	n1=trace1.shape[0];
	nb=2*nw;
	eps=e;
	eps2=e;

	diag=np.zeros(n1);
	offd=np.zeros([n1,nb]);
	diag,off = regularization(diag,offd,nw,eps,eps2);

	w1,diag,offd = pwd_define(forw1,diag,offd,n1,nw,pp1);
	w2,diag,offd = pwd_define(forw2,diag,offd,n1,nw,pp2);

	t0=0.5*(trace1[0]+trace2[0]);
	t1=0.5*(trace1[1]+trace2[1]);
	t2=0.5*(trace1[n1-2]+trace2[n1-2]);
	t3=0.5*(trace1[n1-1]+trace2[n1-1]);

	tmp1 = pwd_set(0,w1,diag,offd,pp1,trace1);
	tmp2 = pwd_set(0,w2,diag,offd,pp2,trace2);

	trace=tmp1+tmp2;

	trace[0]=trace[0]+eps2*t0;
	trace[1]=trace[1]+eps2*t1;
	trace[n1-2]=trace[n1-2]+eps2*t2;
	trace[n1-1]=trace[n1-1]+eps2*t3;

	trace = banded_solve(n1,nb,diag,offd,trace);   

	return trace