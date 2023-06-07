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
	
	EXAMPLE 1
	from pyseistr import smooth
	from pyseistr import dip2dc
	from pyseistr import rgt
	import matplotlib.pyplot as plt;
	from pyseistr import sigmoid
	sig=sigmoid(n1=200,n2=210);
	sig=smooth(sig,rect=[3,1,1],diff=[1,0,0],adj=0);
	sig=smooth(sig,rect=[3,1,1])
	sdip=dip2dc(sig,order=2,niter=10,rect=[4,4,1],verb=0)
	time=rgt(sdip,o1=0,d1=0.004,order=2,i0=50,eps=0.1);

	fig = plt.figure(figsize=(12, 6))
	ax=plt.subplot(1,3,1)
	plt.imshow(sig,cmap='gray',clim=(-0.02, 0.02),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);plt.title('Seismic');
	plt.colorbar(orientation='horizontal');
	ax=plt.subplot(1,3,2)
	plt.imshow(sdip,cmap='jet',clim=(-2, 2),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);plt.title('Slope');
	plt.colorbar(orientation='horizontal');
	ax=plt.subplot(1,3,3)
	plt.imshow(time,cmap='jet',clim=(0, 0.7),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);plt.title('RGT');
	plt.colorbar(orientation='horizontal');
	plt.show()
	
	EXAMPLE 2
	from pyseistr import rgt
	from pylib.io import binread
	import matplotlib.pyplot as plt;
	dip=binread('/Users/chenyk/data/datapath/tccs/flat/flat/sdip-pad.rsf@',n1=300,n2=210).copy(order='F')
	time=rgt(dip,o1=-0.2,d1=0.004,order=2,i0=105,eps=0.1);
	plt.imshow(dip,cmap='jet');plt.colorbar();plt.show();
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
	
	