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
	par: parameter
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

# void sf_repeat_init(int m1            /* trace length */, 
# 		 int m2            /* number of traces */, 
# 		 sf_operator oper1 /* operator */)
# /*< initialize >*/
# {
#     n1 = m1;
#     n2 = m2;
#     oper = oper1;
# }
# 
# void sf_repeat_lop (bool adj, bool add, int nx, int ny, float *xx, float *yy)
# /*< combined linear operator >*/
# {
#     int i2;       
#     
#     if (nx != ny || nx != n1*n2) 
# 	sf_error("%s: Wrong size (nx=%d ny=%d n1=%d n2=%d)",
# 		 __FILE__,nx,ny,n1,n2);
# 
#     sf_adjnull (adj, add, nx, ny, xx, yy);
# 
#     for (i2=0; i2 < n2; i2++) {
# 	oper(adj,true,n1,n1,xx+i2*n1,yy+i2*n1);
#     }
# }

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
	
	par_op={'nm':n1,'nd':n1}
	for i2 in range(n2):
		if adj==1:
			m[i2*n1+(i2+1)*n1]=oper(d[i2*n1+(i2+1)*n1],par_op,1,1)
		else:
			d[i2*n1+(i2+1)*n1]=oper(m[i2*n1+(i2+1)*n1],par_op,0,1)

	if adj==1:
		dout=m;
	else:
		dout=d;

	return dout
