def gensyn():
	'''
	quickgen: quickly generate the representative synthetic data used in the paper
	
	INPUT
	None
	
	OUTPUT
	data: synthetic data
	
	EXAMPLE
	from pyseistr import gensyn
	data=gensyn();
	import matplotlib.pyplot as plt;
	plt.imshow(data);plt.ylabel('Time sample');plt.xlabel('Trace');plt.show();

	'''
	import numpy as np
	from .ricker import ricker
	[w,tw]=ricker(30,0.001,0.1);
	t=np.zeros([300,1000]);sigma=300;A=100;B=200;
	data=np.zeros([400,1000]);

	[m,n]=t.shape;
	for i in range(1,n+1):
		k=np.floor(-A*np.exp(-np.power(i-n/2,2)/np.power(sigma,2))+B);k=int(k);
		if k>1 and k<=m:
			t[k-1,i-1]=1;

	for i in range(1,n+1):
		data[:,i-1]=np.convolve(t[:,i-1],w);
	
	return data
	
	
def genflat(nt=100,nx=60,t=[30,50,80],amp=[1,1,1],freq=30,dt=0.004):
	'''
	quickgen: quickly generate the representative synthetic data used in the paper
	
	INPUT
	None
	
	OUTPUT
	data: synthetic data
	
	EXAMPLE
	from pyseistr import genflat
	data=genflat();
	import matplotlib.pyplot as plt;
	plt.imshow(data);plt.ylabel('Time sample');plt.xlabel('Trace');plt.show();

	'''
	import numpy as np
	from .ricker import ricker
	[w,tw]=ricker(freq,dt,60*dt);
	data=np.zeros([nt,nx]);

	trace=np.zeros(nt)
	for ii in range(len(t)):
		trace[t[ii]]=amp[ii];

	for ii in range(nx):
		data[:,ii]=np.convolve(trace,w,mode='same');
	
	return data