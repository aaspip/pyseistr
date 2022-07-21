def ricker(f,dt,tlength=None):
	# ricker: Ricker wavelet of central frequency f.
	#
	# INPUT:
	# f : central freq. in Hz (f <<1/(2dt) )
	# dt: sampling interval in sec
	# tlength : the duration of wavelet in sec
	#
	# OUTPUT: 
	# w:  the Ricker wavelet
	# tw: time axis
	#
	# Example
	#
	#   [w,tw] = str_ricker(10,0.004,0.2);
	#    plot(tw,w);
	import numpy as np
	
	if tlength!=None:
		nw=np.floor(tlength/dt)+1;
	else:
		nw=2.2/f/dt;
		nw=2*np.floor(nw/2)+1;
	nc=np.floor(nw/2);
	nw=int(nw)
	w =np.zeros(nw);
	
	k=np.arange(1,nw+1,1);
	alpha = (nc-k+1)*f*dt*np.pi;
	beta=np.power(alpha,2);
	w = (1.-beta*2)*np.exp(-beta);
	tw = -(nc+1-k)*dt;
	return w,tw