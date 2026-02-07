def xyzwrite(fname,vel,xx,yy,zz,withnewline=False):
	'''
	xyzwrite: write ASCII file
	
	INPUT
	fname: file name
	vel:   velocity model in numpy array (nx,ny,nz)
	
	Example 1:
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
	
	Example 2
	# https://github.com/aaspip/data/blob/main/CMEZ3D-20260202.npy
	
	from pyseistr import xyzwrite	
	import os
	import numpy as np
	vel1=np.load(os.getenv('HOME')+'/pylib/notebooks/CMEZ3D-20260202.npy') #on 3D grid, X,Y,Z
	# Grid 1
	dx,dy,dz=(95.01-0)/100.0,95.01/100.0,95.01/100.0
	x0,y0,z0=0,0,-1.5
	nx,ny,nz=101,72,26
	x=np.linspace(0,nx-1,nx)*dx+x0;
	y=np.linspace(0,ny-1,ny)*dy+y0;
	z=np.linspace(0,nz-1,nz)*dz+z0;
	# Grid 2
	nz=101;z0=0;dz=0.1;
	xx=x;
	yy=y;
	zz=np.linspace(0,nz-1,nz)*dz+z0;
	
	from pyseistr import rsf3to3
	vel2=rsf3to3(vel1,x,y,z,xx,yy,zz)
	
	from pyseistr import plot3d 
	import matplotlib.pyplot as plt
	plot3d(np.transpose(vel1,(2,0,1)),vmin=4.3,vmax=6.3,z=z,x=x,y=y,cmap=plt.cm.jet,barlabel='P-wave velocity (km/s)',showf=False,close=False);
	plt.gca().set_xlabel("X (km)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y (km)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Depth (km)",fontsize='large', fontweight='normal')
	plt.title("CMEZ3D on the original grid",fontsize='large', fontweight='normal')
	plt.savefig('rsf3to3_0.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
	plt.show()

	plot3d(np.transpose(vel2,(2,0,1)),vmin=4.3,vmax=6.3,z=zz,x=xx,y=yy,cmap=plt.cm.jet,barlabel='P-wave velocity (km/s)',showf=False,close=False);
	plt.gca().set_xlabel("X (km)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y (km)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Depth (km)",fontsize='large', fontweight='normal')
	plt.title("CMEZ3D on an interpolated grid",fontsize='large', fontweight='normal')
	plt.savefig('rsf3to3_1.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
	plt.show()
	
	#### in lon/lat coordinate
	# Grid 1
	nx,ny,nz=101,72,26
	lon1,lon2=-104.7,-103.7
	lat1,lat2=31.3,31.9
	dlon=(lon2-lon1)/(nx-1)
	dlat=(lat2-lat1)/(ny-1)
	lons=np.linspace(0,nx-1,nx)*dlon+lon1
	lats=np.linspace(0,ny-1,ny)*dlat+lat1
	
	from pyseistr import rsf3to3
	plot3d(np.transpose(vel1,(2,0,1)),vmin=4.3,vmax=6.3,z=z,x=lons,y=lats,cmap=plt.cm.jet,barlabel='P-wave velocity (km/s)',showf=False,close=False);
	plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Depth (km)",fontsize='large', fontweight='normal')
	plt.title("CMEZ3D on the original grid",fontsize='large', fontweight='normal')
	plt.savefig('rsf3to3_2.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
	plt.show()
	
	plot3d(np.transpose(vel2,(2,0,1)),vmin=4.3,vmax=6.3,z=zz,x=lons,y=lats,cmap=plt.cm.jet,barlabel='P-wave velocity (km/s)',showf=False,close=False);
	plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Depth (km)",fontsize='large', fontweight='normal')
	plt.title("CMEZ3D on an interpolated grid",fontsize='large', fontweight='normal')
	plt.savefig('rsf3to3_3.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
	plt.show()

	xyzwrite('CMEZ3D-20260202_xyz.txt',vel2,lons,lats,zz)


	## Below is for Well-log model
	from pyseistr import asciiread
	file=os.getenv('HOME')+'/DATALIB/Jake/VPWS_Export' 
	lines=asciiread(file);
	lines=lines[10:]

	from pyproj import Proj, Transformer #pip install pyproj

	# Define UTM Zone 13N in US survey feet
	utm_proj = Proj(proj='utm', zone=13, datum='NAD83', units='us-ft', north=True)
	# WGS84 lat/lon
	wgs84 = Proj(proj='latlong', datum='WGS84')

	# Example coordinates in US survey feet
	E_ft = float(lines[0].split()[0])
	N_ft = float(lines[0].split()[1])
	print('E_ft,N_ft',E_ft,N_ft)

	# Convert
	transformer = Transformer.from_proj(utm_proj, wgs84)
	lon, lat = transformer.transform(E_ft, N_ft)
	print(lat, lon)

	lons=[]
	lats=[]
	deps=[]
	vps=[]

	for ii in range(len(lines)):
		E_ft = float(lines[ii].split()[0])
		N_ft = float(lines[ii].split()[1])
		lon, lat = transformer.transform(E_ft, N_ft)
		dep=-float(lines[ii].split()[2])*0.3048006096
		vp=float(lines[ii].split()[3])
		lons.append(lon)
		lats.append(lat)
		deps.append(dep)
		vps.append(vp)


	print('max,min of lons:',max(lons),min(lons))
	print('max,min of lats:',max(lats),min(lats))
	print('max,min of deps:',max(deps),min(deps))
	print('max,min of vps:',max(vps),min(vps))

	from pyekfmm import deg_to_km_factors, lonlat_to_local_km_equirectangular, local_km_to_lonlat_equirectangular

	lon1=-104.7
	lon2=-103.7

	lat1=31.3
	lat2=31.9

	# Define the CMEZ area of interest
	lon_0, lat_0 = lon1, lat1

	xkms=[]
	ykms=[]
	zkms=[]
	for ii in range(len(lons)):
		x_km, y_km = lonlat_to_local_km_equirectangular(lon_0, lat_0, lons[ii], lats[ii])
		z_km=deps[ii]/1000.0
		xkms.append(x_km)
		ykms.append(y_km)
		zkms.append(z_km)

	print('max,min of xkms:',max(xkms),min(xkms))
	print('max,min of ykms:',max(ykms),min(ykms))
	print('max,min of zkms:',max(zkms),min(zkms))


	import numpy as np
	nx=101 # horizontal ('east-west')
	ny=72 # horizontal ('north-south')
	nz=26 # vertical
	x0=0;xm=95.01;
	y0=0;
	z0=-1.5;
	dx=(xm-x0)/(nx-1); #lon
	dy=dx
	dz=dx

	x=np.linspace(0,nx-1,nx)*dx+x0;
	y=np.linspace(0,ny-1,ny)*dx+y0;
	z=np.linspace(0,nz-1,nz)*dx+z0;

	from pyseistr import xyz2rsf
	V=xyz2rsf(xkms,ykms,zkms,vps,x,y,z)
	
	# plot the 3D
	from pyseistr import plot3d 
	import matplotlib.pyplot as plt
	plot3d(V,z=z,x=x,y=y,cmap=plt.cm.jet,barlabel='P-wave velocity (km/s)',showf=False,close=False);
	plt.gca().set_xlabel("X (km)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y (km)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Depth (km)",fontsize='large', fontweight='normal')
	plt.title("Well log interpolated model on a sparse grid",fontsize='large', fontweight='normal')
	plt.savefig('xyz2rsf_0.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
	plt.show()
	
	
	#Or according to the data's area coverage
	
	dx,dy,dz=(95.01-0)/100.0,95.01/100.0,0.1
	x0,y0,z0=0,0,0
	nx,ny,nz=101,72,101
	xx=np.linspace(0,nx-1,nx)*dx+x0;
	yy=np.linspace(0,ny-1,ny)*dy+y0;
	zz=np.linspace(0,nz-1,nz)*dz+z0;
	V=xyz2rsf(xkms,ykms,zkms,vps,xx,yy,zz) #output ZXY

	plot3d(V,z=zz,x=xx,y=yy,cmap=plt.cm.jet,barlabel='P-wave velocity (km/s)',showf=False,close=False);
	plt.gca().set_xlabel("X (km)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y (km)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Depth (km)",fontsize='large', fontweight='normal')
	plt.title("Well log interpolated model",fontsize='large', fontweight='normal')
	plt.savefig('xyz2rsf.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
	plt.show()	

	# lon/lat coordinate
	lon1,lon2=-104.7,-103.7
	lat1,lat2=31.3,31.9
	dlon=(lon2-lon1)/(nx-1)
	dlat=(lat2-lat1)/(ny-1)
	lons=np.linspace(0,nx-1,nx)*dlon+lon1
	lats=np.linspace(0,ny-1,ny)*dlat+lat1

	plot3d(V,vmin=3.3,vmax=6.3,z=zz,x=lons,y=lats,cmap=plt.cm.jet,barlabel='P-wave velocity (km/s)',showf=False,close=False);
	plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Depth (km)",fontsize='large', fontweight='normal')
	plt.title("DB3D on an interpolated grid",fontsize='large', fontweight='normal')
	plt.savefig('rsf3to3_3.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
	plt.show()
	
	from pyseistr import xyzwrite
	xyzwrite('DB3D-20260202_xyz.txt',np.transpose(V,(1,2,0)),lons,lats,zz)
	
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