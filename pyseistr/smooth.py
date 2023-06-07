import numpy as np

def smooth(din,rect=[1,1,1],diff=[0,0,0],repeat=1,adj=False,box=[0,0,0]):
	'''
	
	INPUT
	din:	input data
	rect: 	smoothing radius on #-th axis
	diff:	differentiation on #-th axis (default: 0)
	repeat:	repeat filtering several times
	adj:	run in the adjoint mode 
	
	OUTPUT
	dout:	
	
	EXAMPLE 1
	from pyseistr import smooth
	from pyseistr import gensyn
	import matplotlib.pyplot as plt;
	
	data,noisy=gensyn(noise=True);
	data=data[:,0::10];noisy=noisy[:,0::10];
	data1=smooth(noisy,rect=[1,5,1]);
	
	fig = plt.figure(figsize=(10, 8))
	ax=plt.subplot(1,4,1)
	plt.imshow(data,cmap='jet',clim=(-0.2, 0.2),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);plt.title('Clean data');
	ax=plt.subplot(1,4,2)
	plt.imshow(noisy,cmap='jet',clim=(-0.2, 0.2),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);plt.title('Noisy data');
	ax=plt.subplot(1,4,3)
	plt.imshow(data1,cmap='jet',clim=(-0.2, 0.2),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);plt.title('Smoothed data');
	ax=plt.subplot(1,4,4)
	plt.imshow(noisy-data1,cmap='jet',clim=(-0.2, 0.2),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);plt.title('Noise');
	plt.show()
	'''
	from .divne import smooth as smooth1
	from .divne import triangle_init,first_index,smooth2

	if din.ndim==2:	#for 2D problems
		din=np.expand_dims(din, axis=2)
	
	[n1,n2,n3]=din.shape;
	nrep=repeat;

	if n3>1:
		dim=3;
	else:
		dim=2;

	if rect[2]>1:
		dim1=2;
	else:
		if rect[1]>1:
			dim1=1;
		else:
			dim1=0;

	n=[n1,n2,n3];
	s=[1,n1,n1*n2];

	if dim1==0:
		n2=n2*n3;
	else:
		if dim1==1:
			n1=n1*n2;n2=n3;
		else:
			n1=n1*n2*n3;n2=1;

	din=din.flatten(order='F');
	dout=[]
	for i2 in range(0,n2):
		x=din[i2*n1:(i2+1)*n1];
		for i in range(dim1+1):
			if (rect[i] <= 1):
				continue;
			tr = triangle_init (rect[i],n[i],box[i]);
			for j in range(0,int(n1/n[i])):
				i0 = first_index (i,j,dim1+1,n,s);
				for irep in range(nrep):
					if adj:
						x,tr=smooth1 (tr,i0,s[i],diff[i],x);
					else:
						x,tr=smooth2 (tr,i0,s[i],diff[i],x);
		dout.append(x)
	dout=np.concatenate(dout,axis=0).reshape(n[0],n[1],n[2],order='F')
	dout=np.squeeze(dout);
	return dout
	
	
def smoothc(din,rect=[1,1,1],diff=[0,0,0],repeat=1,adj=True):
	'''
	
	'''
	





	return dout