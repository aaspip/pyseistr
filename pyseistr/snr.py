def snr(g,f,mode=1):
	'''
	snr: calculate the SNR
	
	by Yangkang Chen, 2022
	
	INPUT
	g:		ground truth image
	f:		noisy/restored image
	mode: 	1->2D SNR, 2->3D SNR
	
	OUTPUT
	snr: calculated SNR
	 
	REFERENCE
	Chen and Fomel, 2015, Random noise attenuation using local signal-and-noise orthogonalization, Geophysics.
	
	EXAMPLE
	from pyseistr import gensyn,dip2dc,somean2dc,smooth,snr
	data,noisy=gensyn(noise=True);
	data1=smooth(noisy,rect=[1,5,1]);
	dip=dip2dc(data1);
	d1=somean2dc(noisy,dip,2,2,0.01);
	
	import matplotlib.pyplot as plt;
	fig = plt.figure(figsize=(16, 8))
	plt.subplot(2,2,1);plt.imshow(data,clim=[-0.2,0.2],aspect='auto');plt.ylabel('Time sample');plt.title('Clean')
	plt.subplot(2,2,2);plt.imshow(noisy,clim=[-0.2,0.2],aspect='auto');plt.title('SNR=%g'%snr(data,noisy))
	plt.subplot(2,2,3);plt.imshow(d1,clim=[-0.2,0.2],aspect='auto');plt.xlabel('Trace');plt.title('SNR=%g'%snr(data,d1))
	plt.subplot(2,2,4);plt.imshow(noisy-d1,clim=[-0.2,0.2],aspect='auto');plt.xlabel('Trace');plt.ylabel('Time sample');plt.title('Noisy')
	plt.show();
	'''
	
	import numpy as np

	if g.ndim==2:
		g=np.expand_dims(g, axis=2)

	if f.ndim==2:
		f=np.expand_dims(f, axis=2)
		
	g = np.double(g); #in case of data format is unit8,12,16
	f = np.double(f);

	if f.size != g.size:
		print('Dimesion of two images don''t match!');

	if mode ==1:
		s = g.shape[2];
		if s==1: #single channel	
			psnr = 20.*np.log10(np.linalg.norm(g[:,:,0],'fro')/np.linalg.norm(g[:,:,0]-f[:,:,0],'fro'));   
		else: #multi-channel
			psnr = np.zeros(s);
			for i in range(0,s):
				psnr[i] = 20.*np.log10(np.linalg.norm(g[:,:,i],'fro')/np.linalg.norm(g[:,:,i]-f[:,:,i],'fro'));

	else:
		[n1,n2,n3]=g.shape;
		psnr = 20.*np.log10(np.linalg.norm(g.reshape(n1,n2*n3,order='F'),'fro')/np.linalg.norm(g.reshape(n1,n2*n3,order='F')-f.reshape(n1,n2*n3,order='F'),'fro'));   

	return psnr