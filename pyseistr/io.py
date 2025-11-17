def xyzwrite(fname,vel,xx,yy,zz,withnewline=False):
	'''
	xyzwrite: write ASCII file
	
	INPUT
	fname: file name
	vel:   velocity model in numpy array (nx,ny,nz)
	
	Example:
	from pyseistr import xyzwrite
	import numpy as np
	
	lon1,lon2=-104.7,-103.7
	lat1,lat2=31.3,31.9
	dlon=(lon2-lon1)/(nx-1)
	dlat=(lat2-lat1)/(ny-1)
	lons=np.linspace(0,nx-1,nx)*dlon+lon1
	lats=np.linspace(0,ny-1,ny)*dlat+lat1
	Vcmez3d=np.load("v_est_topo_10813.npy") #x,y,z
	xyzwrite('CMEZ3D_xyz.txt',Vcmez3d,lons,lats,zz)

	'''
	
	
	nx,ny,nz=vel.shape
	
	din=[]
	for ix in range(nx):
		for iy in range(ny):
			for iz in range(nz):
				din.append(" ".join([str(xx[ix]),str(yy[iy]),str(zz[iz]),str(vel[ix,iy,iz])]))
	
	
	f=open(fname,'w')

	for ii in range(len(din)):
			f.write(str(din[ii])+"\n")
			
	
			
def asciiwrite(fname,din,withnewline=False):
	'''
	asciiwrite: write ASCII file
	
	INPUT
	fname: file name
	din:   a list of lines
	withnewline: if with the newline symbol '\n': True: with; False: without
	
	Example:
	
	from pyseistr import asciiwrite
	import os
	
	f=open(os.getenv('HOME')+'/chenyk.data2/various/cyksmall/texnet_stations_2022_1019.csv')
	lines=f.readlines();
	
	asciiwrite('stations.txt',lines,withnewline=True);
	'''
	
	f=open(fname,'w')
	if withnewline:
		for ii in range(len(din)):
			f.write(str(din[ii]))
	else:
		for ii in range(len(din)):
			f.write(str(din[ii])+"\n")
	
def asciiread(fname):
	'''
	asciiread: read ASCII file
	
	INPUT
	fname: file name
	din:   a list of lines
	withnewline: if with the newline symbol '\n': True: with; False: without
	
	Example:
	
	from pyseistr import asciiread
	import os
	
	lines=asciiread(os.getenv('HOME')+'/chenyk.data2/various/cyksmall/texnet_stations_2022_1019.csv');
	'''
	
	f=open(fname,'r')
	lines=f.readlines()
	lines=[ii.strip() for ii in lines]
	
	return lines
	
def binwrite(fname,din):
	'''
	binwrite: write Binary file
	
	INPUT
	fname: file name
	din:   matrix data [n1*n2*n3]	
	
	EXAMPLE
	from pyseistr import binwrite
	from pyseistr import gensyn
	din=gensyn();
	binwrite('data_400_1000.bin',din)
	'''
	import numpy as np
	fid=open(fname,"wb")
	data=np.float32(din.flatten(order='F')) #.copy(order='C'))
	fid.write(data)
	
	
def binread(fname,n1,n2=1,n3=1):
	'''
	binread: read Binary file
	
	INPUT
	fname: file name
	n1,n2,n3: dimension
	
	EXAMPLE
	from pyseistr import binwrite,binread
	from pyseistr import gensyn
	din=gensyn();
	binwrite('data_400_1000.bin',din)
	
	data=binread('data_400_1000.bin',n1=400,n2=1000)
	import matplotlib.pyplot as plt
	plt.imshow(data);
	plt.show()
	'''
	import numpy as np
	fid=open(fname,"rb")
	data=np.fromfile(fid, dtype = np.float32, count = n1*n2*n3) ### remember double precision
	
	if n2==1 and n3==1:
		data=np.reshape(data,[n1],order='F')
	elif n3==1:
		data=np.reshape(data,[n1,n2],order='F')
	else:
		data=np.reshape(data,[n1,n2,n3],order='F')
	
	return data

def binwritecmpx(fname,din):
	'''
	binwritecmpx: write Binary file of complex values
	the values are saved in an interleaved way (e.g., real,image,real,image,...)
	
	INPUT
	fname: file name
	din:   matrix data [n1*n2*n3], type: np.complex_
	
	EXAMPLE
	from pyseistr import binwritecmpx
	from pyseistr import gensyn
	import numpy as np
	din=gensyn();
	dinc=np.empty(din.shape,dtype=np.complex_) #complex din
	dinc.real=din;
	dinc.imag=-din;
	binwritecmpx('data_400_1000.bin',dinc)
	'''
	import numpy as np
	
	data = np.empty(din.size * 2, dtype=np.float32)
	data[0::2] = din.real.flatten(order='F').copy(order='C')
	data[1::2] = din.imag.flatten(order='F').copy(order='C')

	fid=open(fname,"wb")
	fid.write(data)
	
def binreadcmpx(fname,n1,n2=1,n3=1):
	'''
	binreadcmpx: read Binary file of complex values
	the values are saved in an interleaved way (e.g., real,image,real,image,...)
	
	INPUT
	fname: file name
	n1,n2,n3: dimension
	
	OUTPUT
	data: data in numpy array format, type: np.complex_	
	
	EXAMPLE

	from pyseistr import binwritecmpx,binreadcmpx
	from pyseistr import gensyn
	import numpy as np
	din=gensyn();
	dinc=np.empty(din.shape,dtype=np.complex_) #complex din
	dinc.real=din;
	dinc.imag=-din;
	binwritecmpx('data_400_1000.bin',dinc)
	
	data=binreadcmpx('data_400_1000.bin',n1=400,n2=1000) #type: np.complex_
	import matplotlib.pyplot as plt
	plt.imshow(np.concatenate([data.real,data.imag,data.real-data.imag],axis=1),aspect='auto');
	plt.show()
	
	import matplotlib.pyplot as plt
	plt.imshow(np.concatenate([data.real,-data.imag,data.real+data.imag],axis=1),aspect='auto');
	plt.show()
	'''
	import numpy as np
	fid=open(fname,"rb")
	tmp=np.fromfile(fid, dtype = np.float32, count = 2*n1*n2*n3) ### remember double precision
	
	data=np.zeros(n1*n2*n3,np.complex_)
	data.real=tmp[0::2]
	data.imag=tmp[1::2]
	
	if n2==1 and n3==1:
		data=np.reshape(data,[n1],order='F')
	elif n3==1:
		data=np.reshape(data,[n1,n2],order='F')
	else:
		data=np.reshape(data,[n1,n2,n3],order='F')
		
	return data
	
def binwriteint(fname,din):
	'''
	binwrite: write Binary file in integer format
	
	INPUT
	fname: file name
	din:   matrix data [n1*n2*n3]	
	
	EXAMPLE
	from pyseistr import binwriteint
	from pyseistr import gensyn
	din=gensyn();
	binwriteint('data_400_1000.bin',din)
	'''
	import numpy as np
	fid=open(fname,"wb")
	data=np.int32(din.flatten(order='F').copy(order='C'))
	fid.write(data)
	
def binreadint(fname,n1,n2=1,n3=1):
	'''
	binreadint: read Binary file in integer format
	
	INPUT
	fname: file name
	n1,n2,n3: dimension
	
	EXAMPLE
	from pyseistr import binwriteint,binreadint
	from pyseistr import gensyn
	din=gensyn();
	binwriteint('data_400_1000.bin',din)
	
	data=binreadint('data_400_1000.bin',n1=400,n2=1000)
	import matplotlib.pyplot as plt
	plt.imshow(data);
	plt.show()
	'''
	import numpy as np
	fid=open(fname,"rb")
	data=np.fromfile(fid, dtype = np.int32, count = n1*n2*n3) ### remember double precision
	
	if n2==1 and n3==1:
		data=np.reshape(data,[n1],order='F')
	elif n3==1:
		data=np.reshape(data,[n1,n2],order='F')
	else:
		data=np.reshape(data,[n1,n2,n3],order='F')
	
	return data