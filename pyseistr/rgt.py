from paint2dcfun import *
import numpy as np

def rgt(dip,o1=0,d1=0.004,order=1,i0=0,eps=0.01,verb=False):
	'''
	rgt: Generate relative geological time
	
	INPUT
	dip: 	slope
	o1: 	starting time in axis 0
	d1: 	time interval in axis 0
	
	OUTPUT
	time: 	relative geological time
	
	EXAMPLE
	from pyseistr import rgt
	from pylib.io import binread
	import matplotlib.pyplot as plt;
	dip=binread('/Users/chenyk/data/datapath/tccs/flat/flat/sdip-pad.rsf@',n1=300,n2=210).copy(order='F')
	
	time=rgt(dip,o1=-0.2,d1=0.004,order=2,i0=105,eps=0.1);
	
	plt.imshow(dip);plt.show();
	plt.imshow(time,cmap='jet');plt.colorbar();plt.show();
	
	'''
	[n1,n2]=dip.shape;
	trace=np.linspace(0,d1*(n1-1),n1)+o1;
	time=pwpaintc(dip,trace,order=order,i0=i0,eps=eps,verb=verb);
	
	return time
	
	
def pwpaintc(dip,trace,order=1,i0=0,eps=0.01,verb=False):
	'''
	pwpaint: plane-wave painting (C version)
	
	INPUT
	dip:		slope
	trace:		seed trace (this can also be a time trace for RGT)
	order:		accuracy order 
	eps:		regularization 
	io:			reference trace
	
	OUTPUT
	dout:		painted gather
	
	EXAMPLE
	from pyseistr import pwpaintc
	from pylib.io import binread
	import matplotlib.pyplot as plt;
	dip=binread('/Users/chenyk/data/datapath/tccs/flat/flat/sdip-pad.rsf@',n1=300,n2=210).copy(order='F')
	trace=binread('/Users/chenyk/data/datapath/tccs/flat/flat/trace.rsf@',n1=300,n2=1)
	dout=pwpaintc(dip,trace,order=2,i0=105,eps=0.1);
	plt.imshow(dip);plt.show();
	plt.plot(trace);plt.show();
	plt.imshow(dout);plt.show();
	'''

	[n1,n2]=dip.shape;
	dip=np.float32(dip).flatten(order='F');
	trace=np.float32(trace).flatten(order='F');
	dout=cpaint2d(dip,trace,n1,n2,order,i0,eps,verb);
	dout=dout.reshape(n1,n2,order='F');
	
	return dout
	
	