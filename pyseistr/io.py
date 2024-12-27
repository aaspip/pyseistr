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
	data=np.float32(din.flatten(order='F').copy(order='C'))
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