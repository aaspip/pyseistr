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
	
	REFERENCE
	This data was originally from
	Wang, H., Chen, Y., Saad, O.M., Chen, W., Oboué, Y.A.S.I., Yang, L., Fomel, S. and Chen, Y., 2022. A MATLAB code package for 2D/3D local slope estimation and structural filtering. Geophysics, 87(3), pp.F1-F14.
	
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
	

def genplane2d(noise=False,seed=202324,var=0.1):
	'''
	genplane2d: quickly generate the widely used 2D plane-wave synthetic data
	You can generate 2D using genplane3d, but this one is bigger (80 traces, while the geneplane3d one is 20 traces)
	
	INPUT
	noise: if add noise
	seed: random number seed
	var: noise variance relative to the maximum amplitude of clean data
	
	OUTPUT
	dout (or dclean,dnoisy)
		
	EXAMPLE 1
	from pyseistr import genplane2d
	import matplotlib.pyplot as plt
	data=genplane2d();
	plt.imshow(data,aspect='auto');plt.show()
	
	EXAMPLE 2 (2D example)
	from pyseistr import genplane2d
	dc,dn=genplane2d(noise=True,seed=202425,var=0.1);
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
	
	a1=np.zeros([300,80])
	[n,m]=a1.shape
	a3=a1.copy();
	a4=a1.copy();

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
		t3[i]=np.round(-2*i+220);
		t4[i]=np.round(2*i+10);
		a1[t1[i]:t1[i]+k+1,i]=b1; 
		a3[t3[i]:t3[i]+k+1,i]=b1; 
		a4[t4[i]:t4[i]+k+1,i]=b1; 

	dc=a1[0:300,:]+a3[0:300,:]+a4[0:300,:];
	dc=dc/np.abs(dc).max()

	if noise:
		## add noise
		[n1,n2]=dc.shape
		np.random.seed(seed)
		var=var*np.abs(dc).max()
		n=var*np.random.randn(n1,n2); #np.random.randn()'s variance is around 1, mean is 0
		dn=dc+n;
		return dc,dn
	else:
		return dc
		
def genmask(u, r, type, seed):
	"""
	GENMASK:Generate Random Sampling Mask
	
	INPUT
	u: 		image
	r: 		data KNOWN ratio
	type: 	data lose type
	   		'r': random lose rows
	   		'c': random lose columns
	   		'p': random lose pixel
	seed: 	seed of random number generator
	
	OUTPUT
	mask: 	sampling mask
	
	EXAMPLE 1 (2D example)
	
	## decimate traces
	from pyseistr import genplane2d,plot3d,genmask,snr
	from pydrr import drr3drecon
	import numpy as np
	import matplotlib.pyplot as plt
	
	dc,dn=genplane2d(noise=True,seed=202425,var=0.1);
	[n1,n2]=dn.shape
	ratio=0.7;
	mask=genmask(dn,ratio,'c',201415);
	d0=dn*mask;
	
	## Recon
	flow=0;fhigh=125;dt=0.004;N=3;NN=3;Niter=10;mode=1;a=np.linspace(1,0,10);verb=1;eps=0.00001;
	d1=drr3drecon(d0,mask,flow,fhigh,dt,N,100,Niter,eps,mode,a,verb);
	d2=drr3drecon(d0,mask,flow,fhigh,dt,N,NN,Niter,eps,mode,a,verb);
	noi1=dc-d1;
	noi2=dc-d2;

	print('SNR of RR is %g'%snr(dc,d1));
	print('SNR of DRR is %g'%snr(dc,d2));

	## plot results
	fig = plt.figure(figsize=(8, 8.5))
	ax = fig.add_subplot(3,2,1)
	plt.imshow(dc,cmap='jet',clim=(-0.1, 0.1),aspect=0.2);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Clean data');
	ax = fig.add_subplot(3,2,2)
	plt.imshow(d0,cmap='jet',clim=(-0.1, 0.1),aspect=0.2);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Incomplete noisy data');
	ax = fig.add_subplot(3,2,3)
	plt.imshow(d1,cmap='jet',clim=(-0.1, 0.1),aspect=0.2);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Reconstructed (RR, SNR=%.4g dB)'%snr(dc,d1));
	ax = fig.add_subplot(3,2,4)
	plt.imshow(noi1,cmap='jet',clim=(-0.1, 0.1),aspect=0.2);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Error (RR)');
	ax = fig.add_subplot(3,2,5)
	plt.imshow(d2,cmap='jet',clim=(-0.1, 0.1),aspect=0.2);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Reconstructed (DRR, SNR=%.4g dB)'%snr(dc,d2));
	ax = fig.add_subplot(3,2,6)
	plt.imshow(noi2,cmap='jet',clim=(-0.1, 0.1),aspect=0.2);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Error (DRR)');
	plt.savefig('test_pydrr_drr2drecon.png',format='png',dpi=300);
	plt.show()

	EXAMPLE 2 (3D example, taking quite a bit of time)
	## decimate traces
	from pyseistr import genplane3d,plot3d,genmask,snr
	from pydrr import drr3drecon
	import numpy as np
	import matplotlib.pyplot as plt
	
	dc,dn=genplane3d(noise=True,seed=202425,var=0.1);
	[n1,n2,n3]=dn.shape
	ratio=0.7;
	mask=genmask(dn.reshape(n1,n2*n3),ratio,'c',201415).reshape(n1,n2,n3);
	d0=dn*mask;
	
	## Recon
	flow=0;fhigh=125;dt=0.004;N=3;NN=3;Niter=10;mode=1;a=np.linspace(1,0,10);verb=1;eps=0.00001;
	d1=drr3drecon(d0,mask,flow,fhigh,dt,N,100,Niter,eps,mode,a,verb);
	d2=drr3drecon(d0,mask,flow,fhigh,dt,N,NN,Niter,eps,mode,a,verb);
	noi1=dc-d1;
	noi2=dc-d2;

	print('SNR of RR is %g'%snr(dc,d1,mode=2));
	print('SNR of DRR is %g'%snr(dc,d2,mode=2));

	## plot results
	fig = plt.figure(figsize=(8, 7))
	ax=fig.add_subplot(3, 2, 1)
	plt.imshow(dc.transpose(0,2,1).reshape(n1,n2*n3),cmap='jet',clim=(-0.1, 0.1),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Clean data');
	ax=fig.add_subplot(3, 2, 2)
	plt.imshow(d0.transpose(0,2,1).reshape(n1,n2*n3),cmap='jet',clim=(-0.1, 0.1),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
plt.title('Incomplete noisy data');
	ax=fig.add_subplot(3, 2, 3)
	plt.imshow(d1.reshape(n1,n2*n3,order='F'),cmap='jet',clim=(-0.1, 0.1),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Reconstructed (RR, SNR=%.4g dB)'%snr(dc,d1,2));
	ax=fig.add_subplot(3, 2, 4)
	plt.imshow(noi1.transpose(0,2,1).reshape(n1,n2*n3),cmap='jet',clim=(-0.1, 0.1),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Error (RR)');
	ax=fig.add_subplot(3, 2, 5)
	plt.imshow(d2.reshape(n1,n2*n3,order='F'),cmap='jet',clim=(-0.1, 0.1),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Reconstructed (DRR, SNR=%.4g dB)'%snr(dc,d2,2));
	ax=fig.add_subplot(3, 2, 6)
	plt.imshow(noi2.transpose(0,2,1).reshape(n1,n2*n3),cmap='jet',clim=(-0.1, 0.1),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Error (DRR)');
	plt.savefig('test_pydrr_drr3drecon.png',format='png',dpi=300);
	plt.show()
	
	"""
	
	m=u.shape[0];
	n=u.shape[1];
	

	mask = np.zeros([m,n]);
	
	if type=='r':
		row = rperm(m,seed);
		k = np.fix(r*m);k=int(k);
		row = row[0:k-1];
		mask[row,:] = 1;
		
	elif type=='c':
		column = rperm(n,seed);
		k = np.fix(r*n);k=int(k);
		column = column[0:k-1];
		mask[:, column] = 1;
	elif type=='p':
		pix = rperm(m*n,seed);
		k = np.fix(r*m*n);k=int(k);
		pix = pix[0:k-1];
		mask[pix]= 1;
	else:
		print("mask type not found");
		
	return mask

def rperm(n,seed):
	"""
	RPERM: Random permutation of my version.
	
	RPERM(n) is a random permutation of the integers from 1 to n.
	For example, RANDPERM(6) might be [2 4 5 6 1 3].

	"""
	
	np.random.seed(seed);
	p = np.argsort(np.random.rand(n));

	return p
	
def generr(n1,n2,n3=1,seed=202122,seed2=201415,var=10,ratio=0.1):
	'''
	generr: quickly generate sparse but erratic noise
	
	INPUT
	n1/n2/n3: dimension
	seed: random number seed
	seed: random number seed for the mask
	var: noise variance (actually the maximum amplitude of noise), suppose the target signal has a maximum amplitude of 1 (normalized)
	ratio: ratio for including erratic noise (default: 10%; the more the dense)
	
	OUTPUT
	noise: returned (sparse) erratic noise
		
	EXAMPLE 1
	from pyseistr import generr,plot3d
	import matplotlib.pyplot as plt;
	
	data=generr(300,20,20);
	plot3d(data);
	plt.show()
	
	data=generr(300,20,20, ratio=0.5);
	plot3d(data);
	plt.show()

	EXAMPLE 2
	import numpy as np
	import matplotlib.pyplot as plt
	import pyseistr as ps

	## Generate synthetic data
	from pyseistr import gensyn,generr,smoothc,dip2dc
	dc,dn=gensyn(noise=True,var=0.2);
	dc=dc[:,0::10];dn=dn[:,0::10];
	[n1,n2]=dc.shape
	nerr=generr(n1,n2,ratio=0.1,var=1).squeeze()
	dn=dn+nerr;
	
	dtemp=smoothc(dn,rect=[1,5,1]); #for a better slope estimation
	dip=ps.dip2dc(dtemp,rect=[10,20,1]);
	
	## Structural smoothing
	r=2;
	eps=0.01;
	order=2;
	d1=ps.somean2dc(dn,dip,r,order,eps);
	d2=ps.somf2dc(dn,dip,r,order,eps,1);

	## plot results
	fig = plt.figure(figsize=(10, 8))
	ax=plt.subplot(2,4,1)
	plt.imshow(dc,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Clean data');
	ax=plt.subplot(2,4,2)
	plt.imshow(dn,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Noisy data');
	ax=plt.subplot(2,4,3)
	plt.imshow(d1,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Filtered (SOMEAN)');
	ax=plt.subplot(2,4,4)
	plt.imshow(dn-d1,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Noise (SOMEAN)');
	ax=plt.subplot(2,4,6)
	plt.imshow(dip,cmap='jet',clim=(-2, 2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Slope');
	ax=plt.subplot(2,4,7)
	plt.imshow(d2,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Filtered (SOMF)');
	ax=plt.subplot(2,4,8)
	plt.imshow(dn-d2,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
	plt.title('Noise (SOMF)');
	plt.savefig('test_pyseistr_somf2d.png',format='png',dpi=300)
	plt.show()
	
	'''
	import numpy as np
	from .ricker import ricker
	
	np.random.seed(seed);
	noise=(np.random.rand(n1,n2,n3)*2-1)*var;
	
	mask=genmask(noise.reshape(n1,n2*n3),ratio,'c',seed2).reshape(n1,n2,n3);
	
	noise=noise*mask;
	
	return noise

def dither(din, shift):
	'''
	dither: Make a dithering to each trace of the input data
	
	INPUT
	din: input data (2D)
	shift: random time shift (in samples) for each trace (shift >0, downward)
	
	OUTPUT
	dout: shifted data
	
	EXAMPLE 1
	from pyseistr import gensyn, dither
	import matplotlib.pyplot as plt;
	
	data=gensyn()
	data2=dither(data,-50)
	plt.imshow(data);plt.show()
	plt.imshow(data2);plt.show()

	data=gensyn()
	data2=dither(data,50)
	plt.imshow(data);plt.show()
	plt.imshow(data2);plt.show()

	EXAMPLE 2
	from pyseistr import gensyn, dither
	import matplotlib.pyplot as plt;
	import numpy as np
	
	data=gensyn(); nx=data.shape[1]
	data2=dither(data,np.random.permutation(nx))
	plt.imshow(data);plt.show()
	plt.imshow(data2);plt.show()

	data=gensyn()
	data2=dither(data,-np.random.permutation(nx))
	plt.imshow(data);plt.show()
	plt.imshow(data2);plt.show()
	
	'''
	
	nt,nx = din.shape
	
	dout=np.zeros([nt,nx])
	
	if type(shift)==int or type(shift)==float or len(shift) == 1:
		shift=np.ones(nx, dtype=np.int_)*shift
	else:
		shift=shift.astype(np.int_)
		if din.shape[1] != len(shift):
			Exception("Sorry, n2 must be the ssame")

	for ix in range(nx):
		if shift[ix]>0:
			dout[shift[ix]:,ix]=din[:-shift[ix],ix]
		elif shift[ix]<0:
			dout[:shift[ix],ix]=din[-shift[ix]:,ix]
		elif shift[ix]==0:
			dout[:,ix]=din[:,ix].copy()
	
	return dout



def random0(seed=1996,ia=727,im=524287):
	'''
	random0: Simple pseudo-random number generator
	'''
	seed = np.mod(int(seed*ia),im)
	rand = (float(seed)-0.5)/(float(im-1));
	return rand,seed
	




