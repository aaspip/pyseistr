import numpy as np
from dipcfun import *

def smooth(din,rect=[1,1,1],diff=[0,0,0],box=[0,0,0],repeat=1,adj=1):
	'''
	
	INPUT
	din:	input noisy data
	rect: 	smoothing radius on #-th axis
	diff:	differentiation on #-th axis (default: 0)
	box:	box (rather than triangle) on #-th axis )
	repeat:	repeat filtering several times
	adj:	run in the adjoint mode 
	
	OUTPUT
	dout:	smoothed data
	
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
	
	EXAMPLE 2 (set "static long int seed = 1996;" in "RSFSRC/user/gee/random.c")
	from pyseistr import sigmoid
	from pyseistr import smooth
	data=sigmoid(n1=200,n2=210);
	data1=smooth(data,rect=[3,3,1],diff=[1,0,0],adj=1);
	
	import m8r,os,numpy as np
	os.system('sfsigmoid n1=200 n2=210 d2=0.008>tt.rsf')
	os.system('sfsmooth<tt.rsf rect1=3 rect2=3 diff1=1 adj=y >tt2.rsf')
	data2 = m8r.File('tt2.rsf')[:].transpose()
	print('Difference between pyseistr and Madagascar:',np.linalg.norm(data1-data2))
	
	import matplotlib.pyplot as plt;
	plt.figure(figsize=(12,8))
	plt.subplot(1,3,1);
	plt.imshow(data1,clim=(-0.05,0.05),aspect='auto');plt.xlabel('Trace');plt.ylabel('Time sample');plt.title('pyseistr');
	plt.subplot(1,3,2);
	plt.imshow(data2,clim=(-0.05,0.05),aspect='auto');plt.xlabel('Trace');plt.title('Madagascar');
	plt.subplot(1,3,3);
	plt.imshow(data1-data2,clim=(-0.05,0.05),aspect='auto');plt.xlabel('Trace');plt.title('Difference');
	plt.show();
	
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
	
	
def smoothc(din,rect=[1,1,1],diff=[0,0,0],box=[0,0,0],repeat=1,adj=1):
	'''
	
	INPUT
	din:	input noisy data
	rect: 	smoothing radius on #-th axis
	diff:	differentiation on #-th axis (default: 0)
	box:	box (rather than triangle) on #-th axis )
	repeat:	repeat filtering several times
	adj:	run in the adjoint mode 
	
	OUTPUT
	dout:	smoothed data
	
	EXAMPLE 1
	from pyseistr import smoothc
	from pyseistr import gensyn
	import matplotlib.pyplot as plt;
	
	data,noisy=gensyn(noise=True);
	data=data[:,0::10];noisy=noisy[:,0::10];
	data1=smoothc(noisy,rect=[1,5,1]);
	
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
	
	EXAMPLE 2 (set "static long int seed = 1996;" in "RSFSRC/user/gee/random.c")
	from pyseistr import sigmoid
	from pyseistr import smoothc
	data=sigmoid(n1=200,n2=210);
	data1=smoothc(data,rect=[3,3,1],diff=[1,0,0],adj=1);
	
	import m8r,os,numpy as np
	os.system('sfsigmoid n1=200 n2=210 d2=0.008>tt.rsf')
	os.system('sfsmooth<tt.rsf rect1=3 rect2=3 diff1=1 adj=y >tt2.rsf')
	data2 = m8r.File('tt2.rsf')[:].transpose()
	print('Difference between pyseistr and Madagascar:',np.linalg.norm(data1-data2))
	
	import matplotlib.pyplot as plt;
	plt.figure(figsize=(12,8))
	plt.subplot(1,3,1);
	plt.imshow(data1,clim=(-0.05,0.05),aspect='auto');plt.xlabel('Trace');plt.ylabel('Time sample');plt.title('pyseistr');
	plt.subplot(1,3,2);
	plt.imshow(data2,clim=(-0.05,0.05),aspect='auto');plt.xlabel('Trace');plt.title('Madagascar');
	plt.subplot(1,3,3);
	plt.imshow(data1-data2,clim=(-0.05,0.05),aspect='auto');plt.xlabel('Trace');plt.title('Difference');
	plt.show();
	'''
	
	if din.ndim==2:	#for 2D problems
		din=np.expand_dims(din, axis=2)
	
	[n1,n2,n3]=din.shape;
	r1,r2,r3=rect;
	diff1,diff2,diff3=diff;
	box1,box2,box3=box;
	
	din=np.float32(din.flatten(order='F'));
	dout=smoothcf(din,n1,n2,n3,repeat,adj,r1,r2,r3,diff1,diff2,diff3,box1,box2,box3);
	dout=np.squeeze(dout.reshape(n1,n2,n3,order='F'));
	
	return dout