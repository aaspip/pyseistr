import numpy as np

def adjnull( adj,add,nm,nd,m,d ):
	'''
	Claerbout-style adjoint zeroing Zeros out the output (unless add is true). 
	Useful first step for and linear operator.
	'''
	if add:
		return m,d

	if adj:
		m=np.zeros(nm);
		for i in range(0,nm):
			m[i] = 0.0;
	else:
		d=np.zeros(nd);
		for i in range(0,nd):
			d[i] = 0.0;

	return m,d

def copy_lop( adj,add,nm,nd,m,d ):
	'''
	copy operator
	'''
	
	if nm != nd:
		ValueError('Size mismatch, nm should be nd')

	adjnull (adj, add, nm, nd, m, d);
	
	for i in range(nm):
		if adj==1:
			m[i] = m[i] + d[i]
		else:
			d[i] = d[i] + m[i]

	return m,d
	
def allpass3_lop(din,par,adj,add):
	#3-D Plane-wave destruction filter
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
	ap1=par['ap1'];  #int
	ap2=par['ap2'];  #int
	n1=nm;

	[ xx,yy ] = adjnull( adj,add,nm,nd,m,d );

	nx = ap1['nx'];
	ny = ap1['ny'];
	nz = ap1['nz'];
	nw = ap1['nw'];
	nj = ap1['nj'];

	for iz in range(0,ap1['nz']):
		for iy in range(0,ap1['ny']-1):
			for ix in range(ap1['nw']*ap1['nj'],ap1['nx']-ap1['nw']*ap1['nj']):
				ii=ix+ap1['nx']*(iy+ap1['ny']*iz);
			
				if ap1['drift']:
					id=SF_NINT(ap1['pp'][ii]);
					if ix-ap1['nw']*ap1['nj']-id<0 or ix+ap1['nw']*ap1['nj']-id>=nx:
						continue;
					ap1['flt'],tmp=passfilter(ap1['pp'][ii]-id, ap1['nw'] );
				
					for iw in range(0,2*ap1['nw']+1):
						iss = (iw-ap1['nw'])*ap1['nj'];
					
						if adj:
							xx[ii+iss+nx] = xx[ii+iss+nx]+yy[ii]*ap1['flt'][iw];
							xx[ii-iss-id] = xx[ii-iss-id]-yy[ii]*ap1['flt'][iw];
						else:
							yy[ii] = yy[ii] + (xx[ii+iss+nx] - xx[ii-iss-id])*ap1['flt'][iw];
				else:
					ap1['flt'],tmp=passfilter(ap1['pp'][ii],ap1['nw']);
				
					for iw in range(0,2*ap1['nw']+1):
						iss = (iw-ap1['nw'])*ap1['nj'];
						if adj:
							xx[ii+iss+nx] = xx[ii+iss+nx] + yy[ii]*ap1['flt'][iw];
							xx[ii-iss]	= xx[ii-iss]	- yy[ii]*ap1['flt'][iw];
						else:
							yy[ii] = yy[ii]+(xx[ii+iss+nx] - xx[ii-iss])*ap1['flt'][iw];
							
	nx = ap2['nx'];
	ny = ap2['ny'];
	nz = ap2['nz'];
	nw = ap2['nw'];
	nj = ap2['nj'];

	if nx*ny*nz !=n1:
		print('size mismatch');


	for iz in range(0,ap2['nz']-1):
		for iy in range(0,ap2['ny']):
			for ix in range(ap2['nw']*ap2['nj'],ap2['nx']-ap2['nw']*ap2['nj']):
				ii=ix+ap2['nx']*(iy+ap2['ny']*iz);
			
				if ap2['drift']:
					id=SF_NINT(ap2['pp'][ii]);
					if ix-ap2['nw']*ap2['nj']-id<0 or ix+ap2['nw']*ap2['nj']-id>=nx:
						continue;

					ap2['flt'],tmp=passfilter(ap2['pp'][ii]-id, ap2['nw'] );
				
					for iw in range(0,2*ap2['nw']+1):
						iss = (iw-ap2['nw'])*ap2['nj'];
					
						if adj:
							xx[ii+iss+nx*ny] = xx[ii+iss+nx]+yy[ii+n1]*ap2['flt'][iw];
							xx[ii-iss-id] = xx[ii-iss-id]-yy[ii+n1]*ap2['flt'][iw];
						else:
							yy[ii+n1] = yy[ii+n1] + (xx[ii+iss+nx*ny] - xx[ii-iss-id])*ap2['flt'][iw];

				else:
					ap2['flt'],tmp=passfilter(ap2['pp'][ii],ap2['nw']);
				
					for iw in range(0,2*ap2['nw']+1):
						iss = (iw-ap2['nw'])*ap2['nj'];
						if adj:
							xx[ii+iss+nx*ny] = xx[ii+iss+nx*ny] + yy[ii+n1]*ap2['flt'][iw];
							xx[ii-iss]	= xx[ii-iss]	- yy[ii+n1]*ap2['flt'][iw];
						else:
							yy[ii+n1] = yy[ii+n1]+(xx[ii+iss+nx*ny] - xx[ii-iss])*ap2['flt'][iw];



	if adj==1:
		dout=xx;
	else:
		dout=yy;

	return dout


def passfilter(p,nw):
	'''
	passfilter: find filter coefficients£®verfied)
	All-pass plane-wave destruction filter coefficients
	
	INPUT
	p: slope
	nw: filter order
	
	OUTPUT
	a: output filter (n+1) (1D array)
	
	'''
	n=nw*2;
	b=np.zeros([n+1,1]);
	a=np.zeros([n+1,1]);
	
	for k in range(0,n+1):
		bk=1;
		for j in range(0,n):
			if (j<n-k):
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



def mask_lop(din,par,adj,add):

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
	mask=par['mask']

	[ m,d ] = adjnull( adj,add,nm,nd,m,d );

	for im in range(nm):
		if mask[im]:
			if adj:
				m[im]=m[im]+d[im];
			else:
				d[im]=d[im]+m[im];
	
	if adj==1:
		dout=m;
	else:
		dout=d;
		
	return dout	
	

def weight2_lop(din,par,adj,add):
	'''
	weight2_lop: Weighting operator (verified)
	
	Ported to Python by Yangkang Chen, 2025
	
	INPUT
	din: model/data
	par: parameter (par['w'],par['nw'])
	adj: adj flag
	add: add flag
	
	OUTPUT
	dout: data/model
	
	EXPLANATION
	The only difference between weight2_lop and weight_lop is that
	the parameter 'w' is 1D in weight_lop but 2D in weight2_lop
	
	Therefore, in 1D version, nm=nd
	in 2D version, nm=nd*nw 
	
	'''
	nm=par['nm'];
	nd=par['nd'];
	w=par['w']; #2D in weight2_lop while 1D in weight_lop
	nw=par['nw']; #number of components in multidivn
	
	if nd*nw != nm:
		ValueError('Size mismatch, nd*nw should be nm')
	
	if w.shape[1]!=nw:
		ValueError('Shape mismatch, w should be of [nd,nw]')

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
		for iw in range(nw):
			for i in range(nd):
# 				print('nw,nd=',nw,nd)
# 				print('w.shape',w.shape)
# 				print('i,iw',i,iw)
# 				print('m[i+iw*nd]',m[i+iw*nd])
# 				print('d[i]',d[i])
# 				print('w[i,iw]',w[i,iw])
				m[i+iw*nd]=m[i+iw*nd]+d[i]*w[i,iw]; #dot product
	else: #forward
		for iw in range(nw):
			for i in range(nd):
				d[i]=d[i]+m[i+iw*nd]*w[i,iw]; #d becomes model, m becomes data

	if adj==1:
		dout=m;
	else:
		dout=d;

	return dout

def repeat_lop(din,par,adj,add):
	'''
	repeat_lop: combined linear operator
	
	Ported to Python by Yangkang Chen, 2025
	
	INPUT
	din: model/data
	par: parameter
	adj: adj flag
	add: add flag
	
	OUTPUT
	dout: data/model
	
	EXPLANATION
	for two vectors of the same size nm=nd, 
	repeatedly apply an operator (input parameter) to vector m on n1 samples by n2 times
	
	'''
	nm=par['nm'];
	nd=par['nd'];
	n1=par['n1'];
	n2=par['n2'];
	
# 	w=par['w']; #2D in weight2_lop while 1D in weight_lop
	oper=par['oper']; #operator to be combined
	
	if nm != nd:
		ValueError('Size mismatch, nm should be nd')
	
	if nm != n1*n2:
		ValueError('Shape mismatch, nm should be n1*n2')

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
	
	par_op=par['par_op']
# 	print(par_op)
	for i2 in range(n2):
		if adj==1:
			par_op['m']=m[i2*n1:(i2+1)*n1];
			m[i2*n1:(i2+1)*n1] = oper(d[i2*n1:(i2+1)*n1],par_op,1,1) #m=m+oper is different from m=oper (WHY?)
		else:
# 			d[i2*n1:(i2+1)*n1]=d[i2*n1:(i2+1)*n1]+oper(m[i2*n1:(i2+1)*n1],par_op,0,1) #d=d+oper = d=oper (WHY?)
			par_op['d']=d[i2*n1:(i2+1)*n1]
			d[i2*n1:(i2+1)*n1]=oper(m[i2*n1:(i2+1)*n1],par_op,0,1) #d=d+oper = d=oper (WHY?)

	if adj==1:
		dout=m;
	else:
		dout=d;

	return dout


def helicon_lop(din,par,adj,add):
	'''
	helicon_lop: helicon filtering
	
	by Yangkang Chen, July 14, 2025
	
	INPUT
	din: model/data
	par: parameter (par['w'],par['nw'])
	adj: adj flag
	add: add flag
	
	OUTPUT
	dout: data/model
	
	EXPLANATION
	TBD
	
	EXAMPLE
	TBD
	
	'''
	nm=par['nm'];
	nd=par['nd'];
	
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
	
	copy_lop(adj, add, nm, nd, m, d);
	
	aa=par['aa'] #helix preconditioning filter
	for ia in range(aa['nh']):
		for iy in range(aa['lag'][ia],nm,1):
			if aa['mis'] !=None and aa['mis'][iy]:
					continue;
			ix = iy-aa['lag'][ia]
			if adj==1:
				m[ix] = m[ix] + d[iy]*aa['flt'][ia]
			else:
# 				print('ix=',ix,'nm=',nm,'iy=',iy,'nm',nm,'nd',nd)
				d[iy] = d[iy] + m[ix]*aa['flt'][ia]
	
	if adj==1:
		dout=m;
	else:
		dout=d;

	return dout

def nhconest_lop(din,par,adj,add):
	'''
	nhconest_lop: Nonstationary Helical convolution operator
	
	by Yangkang Chen, Aug 7, 2025
	
	INPUT
	din: model/data
	par: parameter (par['w'],par['nw'])
	adj: adj flag
	add: add flag
	
	OUTPUT
	dout: data/model
	
	EXPLANATION
	TBD
	
	EXAMPLE
	TBD
	
	'''
	nm=par['nm'];
	nd=par['nd'];
	
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
	
	aa=par['aa'] #helix filter
	x=par['x']
	nhmax=par['nhmax']
	
	for iy in range(nd):
		if aa['mis'][iy]:
			continue;
		ip=aa['pch'][iy];
		lag=aa['hlx'][ip]['lag']
		na=aa['hlx'][ip]['nh']
		
		for ia in range(na):
			ix=iy-lag[ia];
			if ix<0:
				continue;
			
			if adj==1:
				m[ia+nhmax+ip] = m[ia+nhmax+ip] + d[iy]*x[ix];
			else:
				d[iy] = d[iy] + 		 m[ia+nhmax+ip]*x[ix];
	
	
	if adj==1:
		dout=m;
	else:
		dout=d;

	return dout
	
	
	
def npolydiv2_lop(din,par,adj,add):
	'''
	npolydiv2_lop: Double polynomial division with a non-stationary helical filter
	
	by Yangkang Chen, Aug 7, 2025
	
	INPUT
	din: model/data
	par: parameter (par['w'],par['nw'])
	adj: adj flag
	add: add flag
	
	OUTPUT
	dout: data/model
	
	EXPLANATION
	TBD
	
	EXAMPLE
	TBD
	
	'''
	nm=par['nm'];
	nd=par['nd'];
	
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
	
# 	aa=par['aa'] #helix filter
# 	x=par['x']
	tt=par['tt']
	
	if adj==1:
		tt=npolydiv_lop(d,par,0,0);
		par['m']=m;
		m=npolydiv_lop(tt,par,1,1);
	else:
		tt=npolydiv_lop(m,par,0,0);
		par['m']=d
		d=npolydiv_lop(tt,par,1,1);
	
	if adj==1:
		dout=m;
	else:
		dout=d;

	return dout

def npolydiv_lop(din,par,adj,add):
	'''
	npolydiv_lop: Inverse filtering with a non-stationary helix filter
	
	by Yangkang Chen, Aug 7, 2025
	
	INPUT
	din: model/data
	par: parameter (par['w'],par['nw'])
	adj: adj flag
	add: add flag
	
	OUTPUT
	dout: data/model
	
	EXPLANATION
	TBD
	
	EXAMPLE
	TBD
	
	'''
	nm=par['nm'];
	nd=par['nd'];
	
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
	
	aa=par['aa']; #helix filter
	tt=par['tt']
	
	from .bp import ifnot
	for iid in range(nd):
		tt[iid]=ifnot(adj, d[iid], m[iid]);
	
	if adj==1:
		for iy in range(nd-1,-1,-1):
			ip=aa['pch'][iy];
			lag=aa['hlx'][ip]['lag']
			flt=aa['hlx'][ip]['flt']
			na=aa['hlx'][ip]['nh']
			for ia in range(na):
				ix=iy-lag[ia];
				if ix<0:
					continue;
				tt[ix]=tt[ix]-flt[ia]*tt[iy];
		for iid in range(nd):
			m[iid] = m[iid] + tt[iid];
	else:
		for iy in range(nd):
			ip=aa['pch'][iy]
			lag=aa['hlx'][ip]['lag']
			flt=aa['hlx'][ip]['flt']
			na=aa['hlx'][ip]['nh']
			for ia in range(na):
				ix=iy-lag[ia];
				if ix<0:
					continue;
				tt[ix]=tt[ix]-flt[ia]*tt[iy];
		
		for iid in range(nd):
			d[iid] = d[iid] + tt[iid]

		
	if adj==1:
		dout=m;
	else:
		dout=d;

	return dout

