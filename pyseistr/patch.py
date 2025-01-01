import numpy as np
def patch2d(A,l1=8,l2=8,s1=4,s2=4,mode=1):
	"""
	patch2d: decompose the image into patches:
	
	INPUT
	D: input image
	l1: first patch size
	l2: second patch size
	s1: first shifting size
	s2: second shifting size
	mode: patching mode
	
	OUTPUT
	X: patches
	
	HISTORY
	by Yangkang Chen
	Oct, 2017
	Modified on Dec 12, 2018 (the edge issue, arbitrary size for the matrix)
			 Dec 31, 2018 (tmp1=mod(n1,l1) -> tmp1=mod(n1-l1,s1))
	
	EXAMPLE 1
	
	#generate data
	from pyseistr import gensyn
	data,noisy=gensyn(noise=True);[n1,n2]=data.shape;
	import matplotlib.pyplot as plt;
	plt.subplot(1,2,1);plt.imshow(data,clim=[-0.2,0.2],aspect='auto');plt.xlabel('Trace');plt.ylabel('Time sample');
	plt.subplot(1,2,2);plt.imshow(noisy,clim=[-0.2,0.2],aspect='auto');plt.xlabel('Trace');
	plt.show();
	
	from pyseistr import patch2d
	X=patch2d(data,l1=16,l2=16,s1=8,s2=8);

	#visualize the patches
	from pyseistr import cseis
	plt.imshow(X,aspect='auto',cmap=cseis());plt.ylabel('Patch NO');plt.xlabel('Patch Pixel');plt.show()
	plt.figure(figsize=(8,8))
	for ii in range(64):
		ax=plt.subplot(8,8,ii+1)
		plt.imshow(X[3200+ii,:].reshape(16,16,order='F'),cmap=cseis(),clim=(-0.5,0.5),aspect='auto');
		plt.setp(ax.get_xticklabels(), visible=False);plt.setp(ax.get_yticklabels(), visible=False);
	plt.show()

	#reconstruct
	from pyseistr import patch2d_inv
	import numpy as np
	data2=patch2d_inv(X,n1,n2,l1=16,l2=16,s1=8,s2=8);
	print('Error=',np.linalg.norm(data.flatten()-data2.flatten()))
	
	plt.figure(figsize=(16,8));
	plt.imshow(np.concatenate([data,data2,data-data2],axis=1),aspect='auto');
	plt.show()

	EXAMPLE 2
	https://github.com/chenyk1990/mlnb/blob/main/DL_denoise_simple2D.ipynb
	
	EXAMPLE 3
	sgk_denoise() in pyseisdl/denoise.py
	"""
	[n1,n2]=A.shape;

	if mode==1: 	#possible for other patching options
		
		tmp=np.mod(n1-l1,s1);
		if tmp!=0:
			A=np.concatenate((A,np.zeros([s1-tmp,n2])),axis=0); 
		tmp=np.mod(n2-l2,s2);
		if tmp!=0:
			A=np.concatenate((A,np.zeros([A.shape[0],s2-tmp])),axis=1); 
		
		[N1,N2]=A.shape;
		X=[]
		for i1 in range(0,N1-l1+1,s1):
			for i2 in range(0,N2-l2+1,s2):
						tmp=np.reshape(A[i1:i1+l1,i2:i2+l2],(l1*l2,1),order='F');
						X.append(tmp)
		X = np.array(X)
	else:
		#not written yet
		pass;
	return X[:,:,0]


def patch3d(A,l1=4,l2=4,l3=4,s1=2,s2=2,s3=2,mode=1):
	"""
	patch3d: decompose 3D data into patches:
	
	INPUT
	D: input image
	mode: patching mode
	l1: first patch size
	l2: second patch size
	l3: third patch size
	s1: first shifting size
	s2: second shifting size
	s3: third shifting size
	
	OUTPUT
	X: patches
	
	HISTORY
	by Yangkang Chen
	Oct, 2017
	Modified on Dec 12, 2018 (the edge issue, arbitrary size for the matrix)
			Dec 31, 2018 (tmp1=mod(n1,l1) -> tmp1=mod(n1-l1,s1))
	
	EXAMPLE 1
	#generate data
	import numpy as np
	from pyseistr import genplane3d,plot3d
	import matplotlib.pyplot as plt
	data,noisy=genplane3d(noise=True,seed=202425,var=0.1);

	#visualize the data
	dz=1;dx=1;dy=1;
	[nz,nx,ny]=data.shape;
	plot3d(data,vmin=-1,vmax=1,figsize=(14,10),z=np.arange(nz)*dz,x=np.arange(nx)*dx,y=np.arange(ny)*dy,barlabel='Amplitude',showf=False,close=False)
	plt.gca().set_xlabel("X (sample)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y (sample)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Z (sample)",fontsize='large', fontweight='normal')
	plt.title('3D seismic data (Clean)')
	plt.savefig(fname='data3d-clean.png',format='png',dpi=300)
	plt.show()

	plot3d(noisy,vmin=-1,vmax=1,figsize=(14,10),z=np.arange(nz)*dz,x=np.arange(nx)*dx,y=np.arange(ny)*dy,barlabel='Amplitude',showf=False,close=False)
	plt.gca().set_xlabel("X (sample)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y (sample)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Z (sample)",fontsize='large', fontweight='normal')
	plt.title('3D seismic data (Noisy)')
	plt.savefig(fname='data3d-noisy.png',format='png',dpi=300)
	plt.show()

	from pyseistr import patch3d,patch3d_inv,snr
	X=patch3d(data,l1=16,l2=10,l3=10,s1=8,s2=2,s3=2);
	Xnoisy=patch3d(noisy,l1=16,l2=10,l3=10,s1=8,s2=2,s3=2);

	from pyseistr import cseis
	plt.imshow(X,aspect='auto',cmap=cseis());plt.ylabel('Patch NO');plt.xlabel('Patch Pixel');plt.show()

	#visualize the patches
	plt.figure(figsize=(8,8))
	for ii in range(16):
            ax=plt.subplot(4,4,ii+1,projection='3d')
            plot3d(X[600+ii,:].reshape(16,10,10,order='F'),ifnewfig=False,showf=False,close=False)
            plt.gca().axis('off')
	plt.show()
	plt.figure(figsize=(8,8))
	for ii in range(16):
            ax=plt.subplot(4,4,ii+1,projection='3d')
            plot3d(Xnoisy[600+ii,:].reshape(16,10,10,order='F'),ifnewfig=False,showf=False,close=False)
            plt.gca().axis('off')
	plt.show()


	#reconstruct
	data2=patch3d_inv(X,nz,nx,ny,l1=16,l2=10,l3=10,s1=8,s2=2,s3=2);
	print('Error=',np.linalg.norm(data.flatten()-data2.flatten()))

	plot3d(np.concatenate([data,data2,data-data2],axis=1),vmin=-1,vmax=1,figsize=(14,10),z=np.arange(nz)*dz,x=np.arange(nx*3)*dx,y=np.arange(ny)*dy,barlabel='Amplitude',showf=False,close=False)
	plt.gca().set_xlabel("X (sample)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y (sample)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Z (sample)",fontsize='large', fontweight='normal')
	plt.title('3D seismic data (Noisy)')
	plt.savefig(fname='data3d-recon.png',format='png',dpi=300)
	plt.show()

	EXAMPLE 2
	https://github.com/chenyk1990/mlnb/blob/main/DL_denoise_simple3D.ipynb
	
	EXAMPLE 3
	sgk_denoise() in pyseisdl/denoise.py
	"""

	[n1,n2,n3]=A.shape;

	if mode==1: 	#possible for other patching options
	
		tmp=np.mod(n1-l1,s1);
		if tmp!=0:
			A=np.concatenate((A,np.zeros([s1-tmp,n2,n3])),axis=0);

		tmp=np.mod(n2-l2,s2);
		if tmp!=0:
			A=np.concatenate((A,np.zeros([A.shape[0],s2-tmp,n3])),axis=1);

		tmp=np.mod(n3-l3,s3);
		if tmp!=0:
			A=np.concatenate((A,np.zeros([A.shape[0],A.shape[1],s3-tmp])),axis=2);	#concatenate along the third dimension

		[N1,N2,N3]=A.shape;
		X=[]
		for i1 in range(0,N1-l1+1,s1):
			for i2 in range(0,N2-l2+1,s2):
				for i3 in range(0,N3-l3+1,s3):
						tmp=np.reshape(A[i1:i1+l1,i2:i2+l2,i3:i3+l3],[l1*l2*l3,1],order='F');
						X.append(tmp)
		X = np.array(X)				
						
	else:
		#not written yet
		pass;
		
	return X


def patch2d_inv(X,n1,n2,l1=8,l2=8,s1=4,s2=4,mode=1):
	"""
	patch2d_inv: insert patches into the image
	
	INPUT
	D: input patches (sample,patchsize)
	mode: patching mode
	l1: first patch size
	l2: second patch size
	s1: first shifting size
	s2: second shifting size
	
	OUTPUT
	X: patches
	
	HISTORY
	by Yangkang Chen
	Oct, 2017
	Modified on Dec 12, 2018 (the edge issue, arbitrary size for the matrix)
				Dec 31, 2018 (tmp1=mod(n1,l1) -> tmp1=mod(n1-l1,s1))
	
	EXAMPLE 1
	
	#generate data
	from pyseistr import gensyn
	data,noisy=gensyn(noise=True);[n1,n2]=data.shape;
	import matplotlib.pyplot as plt;
	plt.subplot(1,2,1);plt.imshow(data,clim=[-0.2,0.2],aspect='auto');plt.xlabel('Trace');plt.ylabel('Time sample');
	plt.subplot(1,2,2);plt.imshow(noisy,clim=[-0.2,0.2],aspect='auto');plt.xlabel('Trace');
	plt.show();
	
	from pyseistr import patch2d
	X=patch2d(data,l1=16,l2=16,s1=8,s2=8);

	#visualize the patches
	from pyseistr import cseis
	plt.imshow(X,aspect='auto',cmap=cseis());plt.ylabel('Patch NO');plt.xlabel('Patch Pixel');plt.show()
	plt.figure(figsize=(8,8))
	for ii in range(64):
		ax=plt.subplot(8,8,ii+1)
		plt.imshow(X[3200+ii,:].reshape(16,16,order='F'),cmap=cseis(),clim=(-0.5,0.5),aspect='auto');
		plt.setp(ax.get_xticklabels(), visible=False);plt.setp(ax.get_yticklabels(), visible=False);
	plt.show()

	#reconstruct
	from pyseistr import patch2d_inv
	import numpy as np
	data2=patch2d_inv(X,n1,n2,l1=16,l2=16,s1=8,s2=8);
	print('Error=',np.linalg.norm(data.flatten()-data2.flatten()))
	
	plt.figure(figsize=(16,8));
	plt.imshow(np.concatenate([data,data2,data-data2],axis=1),aspect='auto');
	plt.show()

	EXAMPLE 2
	https://github.com/chenyk1990/mlnb/blob/main/DL_denoise_simple3D.ipynb
	
	EXAMPLE 3
	sgk_denoise() in pyseisdl/denoise.py

	"""

	if mode==1: 	#possible for other patching options

		tmp1=np.mod(n1-l1,s1);
		tmp2=np.mod(n2-l2,s2);
		if tmp1!=0 and tmp2!=0:
			A=np.zeros([n1+s1-tmp1,n2+s2-tmp2]); 
			mask=np.zeros([n1+s1-tmp1,n2+s2-tmp2]); 

		if tmp1!=0 and tmp2==0:
			A=np.zeros([n1+s1-tmp1,n2]); 
			mask=np.zeros([n1+s1-tmp1,n2]);

		if tmp1==0 and tmp2!=0:
			A=np.zeros([n1,n2+s2-tmp2]);   
			mask=np.zeros([n1,n2+s2-tmp2]);   


		if tmp1==0 and tmp2==0:
			A=np.zeros([n1,n2]); 
			mask=np.zeros([n1,n2]);

		[N1,N2]=A.shape;
		id=-1;
		for i1 in range(0,N1-l1+1,s1):
			for i2 in range(0,N2-l2+1,s2):
				id=id+1;
				A[i1:i1+l1,i2:i2+l2]=A[i1:i1+l1,i2:i2+l2]+np.reshape(X[id,:],[l1,l2],order='F');
				mask[i1:i1+l1,i2:i2+l2]=mask[i1:i1+l1,i2:i2+l2]+np.ones([l1,l2]);

		A=A/mask; 
		A=A[0:n1,0:n2];
	else:
		#not written yet
		pass;
	return A


def patch3d_inv( X,n1,n2,n3,l1=4,l2=4,l3=4,s1=2,s2=2,s3=2,mode=1):
	"""
	patch3d_inv: insert patches into the 3D data
	
	INPUT
	D: input image
	mode: patching mode
	n1: first dimension size
	n1: second dimension size
	n3: third dimension size
	l1: first patch size
	l2: second patch size
	l3: third patch size
	s1: first shifting size
	s2: second shifting size
	s3: third shifting size
	
	OUTPUT
	X: patches
	
	HISTORY
	by Yangkang Chen
	Oct, 2017
	Modified on Dec 12, 2018 (the edge issue, arbitrary size for the matrix)
			Dec 31, 2018 (tmp1=mod(n1,l1) -> tmp1=mod(n1-l1,s1))
	Marich, 31, 2020, 2D->3D
	
	EXAMPLE 1
	#generate data
	import numpy as np
	from pyseistr import genplane3d,plot3d
	import matplotlib.pyplot as plt
	data,noisy=genplane3d(noise=True,seed=202425,var=0.1);

	#visualize the data
	dz=1;dx=1;dy=1;
	[nz,nx,ny]=data.shape;
	plot3d(data,vmin=-1,vmax=1,figsize=(14,10),z=np.arange(nz)*dz,x=np.arange(nx)*dx,y=np.arange(ny)*dy,barlabel='Amplitude',showf=False,close=False)
	plt.gca().set_xlabel("X (sample)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y (sample)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Z (sample)",fontsize='large', fontweight='normal')
	plt.title('3D seismic data (Clean)')
	plt.savefig(fname='data3d-clean.png',format='png',dpi=300)
	plt.show()

	plot3d(noisy,vmin=-1,vmax=1,figsize=(14,10),z=np.arange(nz)*dz,x=np.arange(nx)*dx,y=np.arange(ny)*dy,barlabel='Amplitude',showf=False,close=False)
	plt.gca().set_xlabel("X (sample)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y (sample)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Z (sample)",fontsize='large', fontweight='normal')
	plt.title('3D seismic data (Noisy)')
	plt.savefig(fname='data3d-noisy.png',format='png',dpi=300)
	plt.show()

	from pyseistr import patch3d,patch3d_inv,snr
	X=patch3d(data,l1=16,l2=10,l3=10,s1=8,s2=2,s3=2);
	Xnoisy=patch3d(noisy,l1=16,l2=10,l3=10,s1=8,s2=2,s3=2);

	from pyseistr import cseis
	plt.imshow(X,aspect='auto',cmap=cseis());plt.ylabel('Patch NO');plt.xlabel('Patch Pixel');plt.show()

	#visualize the patches
	plt.figure(figsize=(8,8))
	for ii in range(16):
            ax=plt.subplot(4,4,ii+1,projection='3d')
            plot3d(X[600+ii,:].reshape(16,10,10,order='F'),ifnewfig=False,showf=False,close=False)
            plt.gca().axis('off')
	plt.show()
	plt.figure(figsize=(8,8))
	for ii in range(16):
            ax=plt.subplot(4,4,ii+1,projection='3d')
            plot3d(Xnoisy[600+ii,:].reshape(16,10,10,order='F'),ifnewfig=False,showf=False,close=False)
            plt.gca().axis('off')
	plt.show()


	#reconstruct
	data2=patch3d_inv(X,nz,nx,ny,l1=16,l2=10,l3=10,s1=8,s2=2,s3=2);
	print('Error=',np.linalg.norm(data.flatten()-data2.flatten()))

	plot3d(np.concatenate([data,data2,data-data2],axis=1),vmin=-1,vmax=1,figsize=(14,10),z=np.arange(nz)*dz,x=np.arange(nx*3)*dx,y=np.arange(ny)*dy,barlabel='Amplitude',showf=False,close=False)
	plt.gca().set_xlabel("X (sample)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y (sample)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Z (sample)",fontsize='large', fontweight='normal')
	plt.title('3D seismic data (Noisy)')
	plt.savefig(fname='data3d-recon.png',format='png',dpi=300)
	plt.show()

	EXAMPLE 2
	https://github.com/chenyk1990/mlnb/blob/main/DL_denoise_simple3D.ipynb
	
	EXAMPLE 3
	sgk_denoise() in pyseisdl/denoise.py

	"""

	if mode==1: 	#possible for other patching options
	
		tmp1=np.mod(n1-l1,s1);
		tmp2=np.mod(n2-l2,s2);
		tmp3=np.mod(n3-l3,s3);
		if tmp1!=0 and tmp2!=0 and tmp3!=0:
			A=np.zeros([n1+s1-tmp1,n2+s2-tmp2,n3+s3-tmp3]);
			mask=np.zeros([n1+s1-tmp1,n2+s2-tmp2,n3+s3-tmp3]);

		if tmp1!=0 and tmp2!=0 and tmp3==0:
			A=np.zeros([n1+s1-tmp1,n2+s2-tmp2,n3]);
			mask=np.zeros([n1+s1-tmp1,n2+s2-tmp2,n3]);
	
		if tmp1!=0 and tmp2==0 and tmp3==0:
			A=np.zeros([n1+s1-tmp1,n2,n3]);
			mask=np.zeros([n1+s1-tmp1,n2,n3]);
	
		if tmp1==0 and tmp2!=0 and tmp3==0:
			A=np.zeros([n1,n2+s2-tmp2,n3]);
			mask=np.zeros([n1,n2+s2-tmp2,n3]);
	
		if tmp1==0 and tmp2==0 and tmp3!=0:
			A=np.zeros([n1,n2,n3+s3-tmp3]);
			mask=np.zeros([n1,n2,n3+s3-tmp3]);
	
		if tmp1==0 and tmp2==0  and tmp3==0:
			A=np.zeros([n1,n2,n3]);
			mask=np.zeros([n1,n2,n3]);
	
		[N1,N2,N3]=A.shape;
		id=-1;
		for i1 in range(0,N1-l1+1,s1):
			for i2 in range(0,N2-l2+1,s2):
				for i3 in range(0,N3-l3+1,s3):
					id=id+1;
					A[i1:i1+l1,i2:i2+l2,i3:i3+l3]=A[i1:i1+l1,i2:i2+l2,i3:i3+l3]+np.reshape(X[id,:],[l1,l2,l3],order='F');
					mask[i1:i1+l1,i2:i2+l2,i3:i3+l3]=mask[i1:i1+l1,i2:i2+l2,i3:i3+l3]+np.ones([l1,l2,l3]);
		A=A/mask;
	
		A=A[0:n1,0:n2,0:n3];
	else:
		#not written yet
		pass;
	return A


def patch5d(A,l1=4,l2=4,l3=4,l4=4,l5=4,s1=2,s2=2,s3=2,s4=2,s5=2,mode=1):
	"""
	patch5d: decompose 4D/5D data into patches:
	
	INPUT
	D: input image
	mode: patching mode
	l1: first patch size
	l2: second patch size
	l3: third patch size
	l4: fourth patch size
	l5: fifth patch size	(when n5=1, l5=1, s5=0)
	s1: first shifting size
	s2: second shifting size
	s3: third shifting size
	s4: fourth shifting size
	s5: fifth shifting size (when n5=1, l5=1, s5=0)
	
	OUTPUT
	X: patches
	
	HISTORY
	by Yangkang Chen
	April, 2020
	Modified on Dec 12, 2018 (the edge issue, arbitrary size for the matrix)
				Dec 31, 2018 (tmp1=mod(n1,l1) -> tmp1=mod(n1-l1,s1))
	April 2, 2020 (3D-5D)
	
	EXAMPLE
	sgk_denoise() in pyseisdl/denoise.py

	"""

	[n1,n2,n3,n4,n5]=A.shape;

	if mode==1: 	#possible for other patching options
	
		tmp=np.mod(n1-l1,s1);
		if tmp!=0:
			A=np.concatenate((A,zeros(s1-tmp,n2,n3,n4,n5)),axis=0);
	
		tmp=np.mod(n2-l2,s2);
		if tmp!=0:
			A=np.concatenate((A,np.zeros(A.shape[0],s2-tmp,n3,n4,n5)),axis=1);
	
		tmp=np.mod(n3-l3,s3);
		if tmp!=0:
			A=np.concatenate((A,np.zeros(A.shape[0],A.shape[1],s3-tmp,n4,n5)),axis=2);	#concatenate along the third dimension
	
		tmp=np.mod(n4-l4,s4);
		if tmp!=0:
			A=np.concatenate((A,np.zeros(A.shape[0],A.shape[1],A.shape[2],s4-tmp,n5)),axis=3);	#concatenate along the forth dimension

		tmp=np.mod(n5-l5,s5);
		if tmp!=0:
			A=np.concatenate((A,np.zeros(A.shape[0],A.shape[1],A.shape[2],A.shape[3],s5-tmp)),axis=4);	#concatenate along the fifth dimension  
	
		[N1,N2,N3,N4,N5]=A.shape;
		X=np.array([]);
		for i1 in range(0,N1-l1+1,s1):
			for i2 in range(0,N2-l2+1,s2):
				for i3 in range(0,N3-l3+1,s3):
					for i4 in range(0,N4-l4+1,s4):
						for i5 in range(0,N5-l5+1,s5):
							tmp=np.reshape(A[i1:i1+l1,i2:i2+l2,i3:i3+l3,i4:i4+l4,i5:i5+l5],[l1*l2*l3*l4*l5,1],order='F');
							X=np.concatenate((X,tmp),axis=1); #extremely slow, change it to X.append(tmp)
	else:
		#not written yet
		pass;
	return X

def patch5d_inv( X,n1,n2,n3,n4,n5,l1=4,l2=4,l3=4,l4=4,l5=4,s1=2,s2=2,s3=2,s4=2,s5=2,mode=1):
	"""
	patch5d_inv: insert patches into the 4D/5D data
	
	INPUT
	D: input image
	n1: first dimension size
	n1: second dimension size
	n3: third dimension size
	n4: forth dimension size
	n5: fifth dimension size
	l1: first patch size
	l2: second patch size
	l3: third patch size
	l4: fourth patch size
	l5: fifth patch size	(when n5=1, l5=1, s5=0)
	s1: first shifting size
	s2: second shifting size
	s3: third shifting size
	s4: fourth shifting size
	s5: fifth shifting size (when n5=1, l5=1, s5=0)
	mode: patching mode
	
	OUTPUT
	X: patches
	
	HISTORY
	by Yangkang Chen
	April, 2020
	Modified on Dec 12, 2018 (the edge issue, arbitrary size for the matrix)
				Dec 31, 2018 (tmp1=mod(n1,l1) -> tmp1=mod(n1-l1,s1))
				March, 31, 2020, 2D->3D
				April 2, 2020 (3D-5D)
	
	EXAMPLE
	sgk_denoise() in pyseisdl/denoise.py

	"""

	if mode==1: 	#possible for other patching options
		tmp1=np.mod(n1-l1,s1);
		tmp2=np.mod(n2-l2,s2);
		tmp3=np.mod(n3-l3,s3);
		tmp4=np.mod(n4-l4,s4);
		tmp5=np.mod(n5-l5,s5);
	
		if tmp1!=0 and tmp2!=0 and tmp3!=0 and tmp4!=0 and tmp5!=0:
			A=zeros(n1+s1-tmp1,n2+s2-tmp2,n3+s3-tmp3,n4+s4-tmp4,n5+s5-tmp5);
			mask=zeros(n1+s1-tmp1,n2+s2-tmp2,n3+s3-tmp3,n4+s4-tmp4,n5+s5-tmp5);

	
		if tmp1!=0 and tmp2==0 and tmp3==0 and tmp4==0 and tmp5==0:
			A=zeros(n1+s1-tmp1,n2,n3,n4,n5);
			mask=zeros(n1+s1-tmp1,n2,n3,n4,n5);

	
		if tmp1==0 and tmp2!=0 and tmp3==0 and tmp4==0 and tmp5==0:
			A=zeros(n1,n2+s2-tmp2,n3,n4,n5);
			mask=zeros(n1,n2+s2-tmp2,n3,n4,n5);

	
		if tmp1==0 and tmp2==0 and tmp3!=0 and tmp4==0 and tmp5==0:
			A=zeros(n1,n2,n3+s3-tmp3,n4,n5);
			mask=zeros(n1,n2,n3+s3-tmp3,n4,n5);


		if tmp1==0 and tmp2==0 and tmp3==0 and tmp4!=0 and tmp5==0:
			A=zeros(n1,n2,n3,n4+s4-tmp4,n5);
			mask=zeros(n1,n2,n3,n4+s4-tmp4,n5);

	
		if tmp1==0 and tmp2==0 and tmp3==0 and tmp4==0 and tmp5!=0:
			A=zeros(n1,n2,n3,n4,n5+s5-tmp5);
			mask=zeros(n1,n2,n3,n4,n5+s5-tmp5);

	
		if tmp1==0 and tmp2==0  and tmp3==0 and tmp4==0 and tmp5==0:
			A=zeros(n1,n2,n3,n4,n5);
			mask=zeros(n1,n2,n3,n4,n5);


		[N1,N2,N3,N4,N5]=A.shape;
		id=-1;
		for i1 in range(0,N1-l1+1,s1):
			for i2 in range(0,N2-l2+1,s2):
				for i3 in range(0,N3-l3+1,s3):
					for i4 in range(0,N4-l4+1,s4):
						for i5 in range(0,N5-l5+1,s5):
							A[i1:i1+l1,i2:i2+l2,i3:i3+l3,i4:i4+l4,i5:i5+l5]=A[i1:i1+l1,i2:i2+l2,i3:i3+l3,i4:i4+l4,i5:i5+l5]+np.reshape(X[:,id],[l1,l2,l3,l4,l5],order='F');
							mask[i1:i1+l1,i2:i2+l2,i3:i3+l3,i4:i4+l4,i5:i5+l5]=mask[i1:i1+l1,i2:i2+l2,i3:i3+l3,i4:i4+l4,i5:i5+l5]+np.ones([l1,l2,l3,l4,l5]);
		A=A/mask;
		A=A[0:n1,0:n2,0:n3,0:n4,0:n5];
	else:
		#not written yet
		pass;
	return A