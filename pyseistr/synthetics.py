import numpy as np
def gensyn(noise=False,seed=202122,var=0.2):
	'''
	gensyn: quickly generate the representative synthetic data used in the paper
	
	INPUT
	noise: if add noise
	seed: random number seed
	var: noise variance (actually the maximum amplitude of noise)
	
	OUTPUT
	case 1. data: clean synthetic data
	case 2. data,noisy: noisy case
		
	EXAMPLE 1
	from pyseistr import gensyn
	data=gensyn();
	import matplotlib.pyplot as plt;
	plt.imshow(data);plt.ylabel('Time sample');plt.xlabel('Trace');plt.show();

	EXAMPLE 2
	from pyseistr import gensyn
	data,noisy=gensyn(noise=True);
	import matplotlib.pyplot as plt;
	plt.subplot(1,2,1);plt.imshow(data,clim=[-0.2,0.2],aspect='auto');plt.xlabel('Trace');plt.ylabel('Time sample');
	plt.subplot(1,2,2);plt.imshow(noisy,clim=[-0.2,0.2],aspect='auto');plt.xlabel('Trace');
	plt.show();
	
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
	
	if noise:
		data=data/np.max(np.max(data));
		np.random.seed(seed);
		noise=(np.random.rand(data.shape[0],data.shape[1])*2-1)*var;
		noisy=data+noise
		return data,noisy
	else:
		return data
	
	
def genflat(nt=100,nx=60,t=[30,50,80],amp=[1,1,1],freq=30,dt=0.004):
	'''
	genflat: quickly generate the representative synthetic data used in the paper
	
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
	
	
def sigmoid(n1=400,n2=100,d1=0.004,d2=0.032,o1=0,o2=0,large=None,reflectivity=True,taper=True,sd=19921995):
	'''
	sigmoid: generate the famous Sigmoid synthetic data
	
	INPUT
	n1,n2: 			dimension
	large:			length of reflectivity series
	reflectivity: 	if output reflectivity (otherwise output impedance model)
	taper: 			if taper the edges
	
	sd: random number seed
	
	o1,o2,d1,d2: actually no use (just for retaining the previous parameters)
	
	OUTPUT
	refl (reflectivity) or earth (model)

	RFERENCE
	October 2014 program of the month:
	https://reproducibility.org/blog/2014/10/08/program-of-the-month-sfsigmoid/
	
	EXAMPLE 1
	from pyseistr import sigmoid
	data=sigmoid();
	import matplotlib.pyplot as plt;
	plt.imshow(data,aspect='auto');plt.ylabel('Time sample');plt.xlabel('Trace');plt.show();

	EXAMPLE 2
	from pyseistr import sigmoid
	data=sigmoid(sd=2024);
	import matplotlib.pyplot as plt;
	plt.imshow(data,aspect='auto');plt.ylabel('Time sample');plt.xlabel('Trace');plt.show();
	
	EXAMPLE 3 (set "static long int seed = 1996;" in "RSFSRC/user/gee/random.c")
	from pyseistr import sigmoid
	data=sigmoid(n1=200,n2=210);
	import m8r,os,numpy as np
	os.system('sfsigmoid n1=200 n2=210 d2=0.008>tt.rsf')
	data2 = m8r.File('tt.rsf')[:].transpose()
	print('Difference between pyseistr and Madagascar:',np.linalg.norm(data-data2))
	
	import matplotlib.pyplot as plt;
	plt.figure(figsize=(12,8))
	plt.subplot(1,3,1);
	plt.imshow(data,clim=(-0.05,0.05),aspect='auto');plt.xlabel('Trace');plt.ylabel('Time sample');plt.title('pyseistr');
	plt.subplot(1,3,2);
	plt.imshow(data2,clim=(-0.05,0.05),aspect='auto');plt.xlabel('Trace');plt.title('Madagascar');
	plt.subplot(1,3,3);
	plt.imshow(data-data2,clim=(-0.05,0.05),aspect='auto');plt.xlabel('Trace');plt.title('Difference');
	plt.show();

	'''
	if large is None:
		large=5*n1;
	
	imp1=np.zeros(large);
	imp2=np.zeros(large);
	dipper=np.zeros([n1,n2]);
	earth=np.zeros([n1,n2]);
	refl=np.zeros([n1,n2]);
	sig1=np.zeros([n1,n2]);
	sig2=np.zeros([n1,n2]);
	fault=np.zeros([n1,n2]);
	
	#seed for the random number generator
# 	sd=19921995;
	
	imp1[0]=1.0;
	for i1 in range(1,large,1):
		rand,sd=random0(seed=sd);
		if rand>0.2:
			imp1[i1]=imp1[i1-1];
		else:
			rand,sd=random0(seed=sd);
			imp1[i1]=1.0+0.1*rand;
		
	imp2[0]=1.0;
	for i1 in range(1,large,1):
		rand,sd=random0(seed=sd);
		if rand>0.3:
			imp2[i1]=imp2[i1-1];
		else:
			rand,sd=random0(seed=sd);
			imp2[i1]=1.0+0.1*rand;
			
	for i2 in range(n2):
		for i1 in range(n1):
			t = i1 + (i2+1.) * 0.3 * (1.0*n1)/(1.0*n2);
			it=int(t);
			frac=t-it;
			
			if it>=0 and it+1<large:
				dipper[i1,i2]=imp1[it]*(1-frac)+imp1[it+1]*frac;
			else:
				dipper[i1,i2]=0;
				
	
	for i2 in range(n2):
		for i1 in range(n1):
			t = i1 + n1 *0.15 * ( 1.1 + np.cos( ( 3 *3.14 * (i2+1.))/n2) );
			it=int(t);
			iss=it+10;
			frac=t-it;
			
			if it>=0 and it+1<large and iss>=0 and iss+1<large:
				sig1[i1,i2]=imp2[it]*(1-frac)+imp2[it+1]*frac;
				sig2[i1,i2]=imp2[iss]*(1-frac)+imp2[iss+1]*frac;
			else:
				sig1[i1,i2]=0;
				sig2[i1,i2]=0;
			
	for i2 in range(n2):
		for i1 in range(n1):
			if (i2+1.0)/n2-0.9 < -0.75*np.power((i1+2.0)/n1,2):
				fault[i1,i2]=sig1[i1,i2];
			else:
				fault[i1,i2]=sig2[i1,i2];
		
	for i2 in range(n2):
		for i1 in range(n1):
			it = int(i1 + 1. + (i2 + 1.) * .3 * (1.*n1)/(1.*n2));
			if it>n1/2:
				earth[i1,i2]=fault[i1,i2];
			else:
				earth[i1,i2]=dipper[i1,i2];
	
	if reflectivity:
		for i2 in range(n2):
			refl[0,i2]=0.0;
			for i1 in range(1,n1,1):
				refl[i1,i2] = (earth[i1-1,i2]-earth[i1,i2])/(earth[i1-1,i2]+earth[i1,i2])
		
		if taper:
			for i2 in range(10):
				for i1 in range(n1):
					refl[i1,i2]=refl[i1,i2]*(i2/10.0);
					refl[i1,n2-i2-1]=refl[i1,n2-i2-1]*(i2/10.0);
		
			for i2 in range(5):
				for i1 in range(n1):
					refl[i1,i2]=refl[i1,i2]*(i2/5.0);
					refl[i1,n2-i2-1]=refl[i1,n2-i2-1]*(i2/5.0);
					
		return refl
	else:
		return earth
	
def genplane3d(noise=False,seed=202324,var=0.1):
	'''
	genplane3d: quickly generate the widely used 3D plane-wave synthetic data
	
	INPUT
	noise: if add noise
	seed: random number seed
	var: noise variance relative to the maximum amplitude of clean data
	
	OUTPUT
	dout (or dclean,dnoisy)
		
	EXAMPLE 1
	from pyseistr import genplane3d,plot3d
	data=genplane3d();
	plot3d(data)
	
	EXAMPLE 2 (2D example)
	from pyseistr import genplane3d
	data,datan=genplane3d(noise=True,seed=202425,var=0.1);
	dn=datan[:,:,10]
	import pydrr as pd
	d1=pd.drr3d(dn,0,120,0.004,3,3);	#DRR
	noi1=dn-d1;
	import numpy as np
	import matplotlib.pyplot as plt
	plt.imshow(np.concatenate([dn,d1,noi1],axis=1),aspect='auto');plt.show()
	
	REFERENCES
	This "simple" example has been extensively used by dozens of papers, just to name a few
	[1] Chen, Y., W. Huang, D. Zhang, W. Chen, 2016, An open-source matlab code package for improved rank-reduction 3D seismic data denoising and reconstruction, Computers & Geosciences, 95, 59-66.
	[2] Huang, W., R. Wang, Y. Chen, H. Li, and S. Gan, 2016, Damped multichannel singular spectrum analysis for 3D random noise attenuation, Geophysics, 81, V261-V270.
	[2] Chen, et al., 2023, DRR: an open-source multi-platform package for the damped rank-reduction method and its applications in seismology, Computers & Geosciences, 180, 105440.

	'''
	
	a1=np.zeros([300,20])
	[n,m]=a1.shape
	a3=np.zeros([300,20])
	a4=np.zeros([300,20])

	k=-1;
	a=0.1;
	b=1;
	pi=np.pi

	ts=np.arange(-0.055,0.055+0.002,0.002)
	b1=np.zeros([len(ts)])
	b2=np.zeros([len(ts)])
	b3=np.zeros([len(ts)])
	b4=np.zeros([len(ts)])

	for t in ts:
		k=k+1;
		b1[k]=(1-2*(pi*30*t)*(pi*30*t))*np.exp(-(pi*30*t)*(pi*30*t));
		b2[k]=(1-2*(pi*40*t)*(pi*40*t))*np.exp(-(pi*40*t)*(pi*40*t));
		b3[k]=(1-2*(pi*40*t)*(pi*40*t))*np.exp(-(pi*40*t)*(pi*40*t));
		b4[k]=(1-2*(pi*30*t)*(pi*30*t))*np.exp(-(pi*30*t)*(pi*30*t));

	t1=np.zeros([m],dtype='int')
	t3=np.zeros([m],dtype='int')
	t4=np.zeros([m],dtype='int')
	for i in range(m):
		t1[i]=np.round(140);
		t3[i]=np.round(-6*i+180);
		t4[i]=np.round(6*i+10);
		a1[t1[i]:t1[i]+k+1,i]=b1; 
		a3[t3[i]:t3[i]+k+1,i]=b1; 
		a4[t4[i]:t4[i]+k+1,i]=b1; 

	temp=a1[0:300,:]+a3[0:300,:]+a4[0:300,:];

	shot=np.zeros([300,20,20])
	for j in range(20):
		a4=np.zeros([300,20]);
		for i in range(m):
			t4[i]=np.round(6*i+10+3*j); 
			a4[t4[i]:t4[i]+k+1,i]=b1;

			t1[i]=np.round(140-2*j);
			a1[t1[i]:t1[i]+k+1,i]=b1;

		shot[:,:,j]=a1[0:300,:]+a3[0:300,:]+a4[0:300,:];


	if noise:
		## add noise
		[n1,n2,n3]=shot.shape
		np.random.seed(seed)
		var=var*np.abs(shot).max()
		n=var*np.random.randn(n1,n2,n3); #np.random.randn()'s variance is around 1, mean is 0
		shotn=shot+n;
		return shot,shotn
	else:
		return shot
	

def random0(seed=1996,ia=727,im=524287):
	'''
	random0: Simple pseudo-random number generator
	'''
	seed = np.mod(int(seed*ia),im)
	rand = (float(seed)-0.5)/(float(im-1));
	return rand,seed
	




