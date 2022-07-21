import numpy as np
def bandpass(din,dt,flo=0,fhi=0.5,nplo=6,nphi=6,phase=0,verb=1):
	# bandpass: Bandpass filtering.
	#
	# Aug, 05, 2020
	# by Yangkang Chen
	#
	# INPUT
	# din:	  input data
	# dt:       sampling
	# flo:      Low frequency in band, default is 0
	# fhi:      High frequency in band, default is Nyquist
	# nplo=6:   number of poles for low cutoff
	# nphi=6:   number of poles for high cutoff
	# phase=0:  y: minimum phase, n: zero phase
	# verb=0:   verbosity flag
	#
	# OUTPUT
	# dout:     output data
	#
	# RFERENCE
	# November 2012 program of the month: http://ahay.org/blog/2012/11/03/program-of-the-month-sfbandpass/
	#
	# Example
	# demos/test_xxx_das.py
	#
	
	if din.ndim==2:	#for 2D problems
		din=np.expand_dims(din, axis=2)
		
	if din.shape[0]==1 and din.shape[1]>1 and din.shape[2]==1: #row vector
		din=din.flatten();
	
	[n1,n22,n33]=din.shape;
	
	n2=n22*n33;
	dout=np.zeros([n1,n2]);
	
	d1=dt;
	eps=0.0001;

	if flo<0:
		print('Negative flo');
	else:
		flo=flo*d1;


	if fhi<0:
		print('Negative flo');
	else:
		fhi=fhi*d1;
		if flo>fhi:
			print('Need flo < fhi\n');

		if 0.5<fhi:
			print('Need fhi < Nyquist\n');


	if nplo<1:
		nplo=1;

	if nplo>1 and not phase:
		nplo=nplo/2;

	if nphi<1:
		nphi=1;

	if nphi>1 and not phase:
		nphi=nphi/2;


	if verb:
		print("flo=%g fhi=%g nplo=%d nphi=%d\n",flo,fhi,nplo,nphi);
	

	if flo>eps:
		blo=butter_init(0,flo,nplo);
	else:
		blo='NULL';

	if fhi<0.5-eps:
		bhi=butter_init(1,fhi,nphi);
	else:
		bhi='NULL';

	for i2 in range(0,n2):
		trace=din[:,i2].copy(); #.copy is necessary
		if blo != 'NULL':
			trace=butter_apply(blo,n1,trace);
			if not phase:
				trace=reverse (n1, trace);
				trace=butter_apply (blo, n1, trace);
				trace=reverse (n1, trace);
	
		if bhi != 'NULL':
			trace=butter_apply(bhi,n1,trace);
			if not phase:
				trace=reverse (n1, trace);
				trace=butter_apply (bhi, n1, trace);
				trace=reverse (n1, trace);

		dout[:,i2]=trace[:,0];

	dout=dout.reshape(n1,n22,n33);

	dout=np.squeeze(dout);
	return dout


def butter_init(low,cutoff,nn):
	# butter_init: initialize
	# Aug, 5, 2020
	# Yangkang Chen
	# 
	# INPUT
	# low:	  low-pass (or high-pass)
	# cutoff:   cut off frequency
	# nn:	   number of poles
	# 
	# OUTPUT
	# bw:	   butterworth struct
	# 
# 	bw=struct;
	arg=2*np.pi*cutoff;
	sinw=np.sin(arg);
	cosw=np.cos(arg);

	bw={'nn':nn,'low':low,'den':np.zeros([2,int(np.floor((nn+1)/2))])};
	
	if np.mod(nn,2)>0:
		if low:
			fact=(1+cosw)/sinw;
			bw['den'][0,int(np.floor(nn/2))]=1./(1.+fact);
			bw['den'][1,int(np.floor(nn/2))]=1.-fact;
		else:
			fact=sinw/(1.+cosw);
			bw['den'][0,int(np.floor(nn/2))]=1./(fact+1.0);
			bw['den'][1,int(np.floor(nn/2))]=fact-1.0;


	fact=ifnot(low,np.sin(0.5*arg),np.cos(0.5*arg));
	fact=fact*fact;

	for j in range(0,int(np.floor(nn/2))):
		ss=np.sin(np.pi*(2*j+1)/(2*nn))*sinw;
		bw['den'][0,j]=fact/(1.+ss);
		bw['den'][1,j]=(1-ss)/fact;

	bw['mid']=-2.*cosw/fact;
	return bw


def butter_apply(bw,nx,x):
	# butter_apply: filter the data (in place)
	# 
	#Implementation is inspired by D. Hale and J.F. Claerbout, 1983, Butterworth
	#dip filters: Geophysics, 48, 1033-1038.
	#
	# Aug, 5, 2020
	# Yangkang Chen
	# 
	# INPUT
	# bw: butterworth struct
	# nx: size of x
	# x: input data
	# 
	# OUTPUT
	# x: output data
	d1=bw['mid'];
	nn=bw['nn'];
	if np.mod(nn,2)>0:
		d0=bw['den'][0,int(np.floor(nn/2))];
		d2=bw['den'][1,int(np.floor(nn/2))];
		x0=0;
		y1=0;
		for ix in range(0,nx):
			x1=x0;x0=x[ix].copy(); #This bug is striking, .copy is necessary because otherwise x0 will automatically change according to y0.
			y0=ifnot(bw['low'],(x0 + x1 - d2 * y1)*d0,(x0 - x1 - d2 * y1)*d0);
			x[ix]=y0;
			y1=y0;
	for j in range(0,int(np.floor(nn/2))):
		d0=bw['den'][0,j];
		d2=bw['den'][1,j];
		x1=0;x0=0;y1=0;y2=0;
		for ix in range(0,nx):
			x2=x1;x1=x0;x0=x[ix].copy();#This bug is striking, .copy is necessary because otherwise x0 will automatically change according to y0.
			y0=ifnot(bw['low'],(x0 + 2*x1 + x2 - d1 * y1 - d2 * y2)*d0,(x0 - 2*x1 + x2 - d1 * y1 - d2 * y2)*d0);
			y2=y1;x[ix]=y0;y1=y0;
	return x

def reverse(n1,trace):
	# reverse a trace (in place)
	for i1 in range(0,int(np.floor(n1/2))):
		t=trace[i1].copy();#This bug is striking, .copy is necessary because otherwise x0 will automatically change according to y0.
		trace[i1]=trace[n1-i1-1];
		trace[n1-i1-1]=t;
	return trace


def ifnot(yes, v1, v2):
	# ifnot: equivalent to C grammar (v=yes?v1:v2)
	# Yangkang Chen
	# July, 22, 2020

	if yes:
		v=v1;
	else:
		v=v2;

	return v