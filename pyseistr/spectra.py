import numpy as np
import matplotlib.pyplot as plt

def spectrafx(din):
	'''
	spectrafx: return FX spectra
	
	INPUT
	din: input data (numpy array)
	
	OUTPUT
	fx: FX spectra
	
	HISTORY
	By Yangkang Chen
	06/18/2025
	
	EXAMPLE
	from pyseistr import gensyn,cseis,spectrafx
	import numpy as np

	data=gensyn()
	data_fx=spectrafx(data)

	dt=0.004;dx=1;
	nf=512; #from "spectra print value"
	df=1/dt/nf
	[nt,nx]=data.shape
	import matplotlib.pyplot as plt;
	plt.imshow(np.abs(data_fx),cmap=plt.jet(),aspect='auto',extent=[0,(nx-1)*dx,(nf/2-1)*df,0]);
	plt.title("Spectra of the data");
	plt.xlabel("Location (m)"); plt.ylabel("Frequency (Hz)");
	plt.show()	
	'''
	
	[n1,n2]=din.shape
	
	l1=1;s1=1; #l1,s1 in patch2d


	nf=nextpow2(n1);
	nf=int(nf)

	fx=np.fft.fft(din,nf,0);
	fx=fx[0:int(nf/2)+1,:]
	
	print('nf',nf)
	print('fx.shape',fx.shape)
	
	return fx
	
def nextpow2(N):
    """ Function for finding the next power of 2 """
    n = 1
    while n < N: n *= 2
    return n
	