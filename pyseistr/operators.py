import numpy as np
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
	# passfilter: find filter coefficients£®verfied)
	# All-pass plane-wave destruction filter coefficients
	#
	# INPUT
	# p: slope
	# nw: filter order
	#
	# OUTPUT
	# a: output filter (n+1) (1D array)
	#
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


def adjnull( adj,add,nm,nd,m,d ):
	#Claerbout-style adjoint zeroing Zeros out the output (unless add is true). 
	#Useful first step for and linear operator.
	# 
	#  This program is free software; you can redistribute it and/or modify
	#  it under the terms of the GNU General Public License as published by
	#  the Free Software Foundation; either version 2 of the License, or
	#  (at your option) and later version.
	#  
	#  This program is distributed in the hope that it will be useful,
	#  but WITHOUT ANY WARRANTY; without even the implied warranty of
	#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	#  GNU General Public License for more details.
	#  
	#  You should have received a copy of the GNU General Public License
	#  along with this program; if not, write to the Free Software
	#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
	#adj : adjoint flag; add: addition flag; nm: size of m; nd: size of d

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
