# Regularization has two general meanings in geophysics
# 1) regularization for inverse problems
# 2) regularization for data from irregular grids (what this subroutine is for)

import numpy as np
def xyz2rsf(xs,ys,zs,value,xx=None,yy=None,zz=None,nx=None,ny=None,nz=None):
	'''
	xyz2rsf: data regularization from unstructured xyz (ASCII) format to regularly sampled format (RSF)
	
	
	INPUT
	xs,ys,zs: 	lists of x,y,z coordinates
	value:		lists of values on the unstructured grid defined by x,y,z coordinates (same length as x,y,z)
	xx,yy,zz:	numpy arrays (1D for each) of regular coordinates in x,y,z axes (can be None for all)
	nx,ny,nz:	number of samples in each axis (can be None for all)
	
	OUTPUT
	dout: data on regular grids defined by xx,yy,zz (zxy)
	or
	dout,xx,yy,zz (if xx,yy,zz are not provided)
	
	By Yangkang Chen
	Feb, 3, 2026
	
	EXAMPLE 1
	#Download from: https://utexas.box.com/s/65r3hql5b1iqaggks722so4s57y9xad7 VPWS_Export? #try VS, RHOZED by yourself?
	from pyseistr import asciiread
	file='./VPWS_Export' 
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
	nx,ny,nz=101,101,101 #make it denser to leverage the resolution in the xyz data
	ox=min(xkms); mx=max(xkms);
	oy=min(ykms);my=max(ykms)
	oz=min(zkms);mz=max(zkms)

	x=np.linspace(ox,mx,nx)
	y=np.linspace(oy,my,ny)
	z=np.linspace(oz,mz,nz)

	from pyseistr import xyz2rsf
	V=xyz2rsf(xkms,ykms,zkms,vps,x,y,z)
	
	from pyseistr import plot3d 
	import matplotlib.pyplot as plt
	plot3d(V,z=z,x=x,y=y,cmap=plt.cm.jet,barlabel='P-wave velocity (km/s)',showf=False,close=False);
	plt.gca().set_xlabel("X (km)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y (km)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Depth (km)",fontsize='large', fontweight='normal')
	plt.title("Well log interpolated model",fontsize='large', fontweight='normal')
	plt.savefig('xyz2rsf.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
	plt.show()
	
	
	EXAMPLE 2
	#Using the following script to prepare the data
	#https://github.com/aaspip/notebook/blob/main/Texas/visualize_Texas_velocity_model.ipynb
	#or download data from
	#https://github.com/aaspip/data/blob/main/PB3D_Vp.csv

	from pyseistr import asciiread
	file='PB3D_Vp.csv' #Download from: https://github.com/aaspip/data/blob/main/PB3D_Vp.csv

	lines=asciiread(file);
	line1=lines[0]
	lines=lines[1:]

	lons=[float(ii.split(",")[0])  for ii in lines]
	lats=[float(ii.split(",")[1])  for ii in lines]
	deps=[float(ii.split(",")[2])  for ii in lines]
	vps=[float(ii.split(",")[3])  for ii in lines]

	## plot a grid figure
	import matplotlib.pyplot as plt
	fig = plt.figure(figsize=(8, 8))
	plt.subplot(221)
	plt.plot(lons,lats,'k.');
	plt.setp(plt.gca().get_xticklabels(), visible=False)
	plt.setp(plt.gca().get_yticklabels(), visible=False)
	plt.xlabel('Longitude (deg)');plt.ylabel('Latitude (deg)');

	plt.subplot(223)
	plt.plot(lons,deps,'k.');
	plt.setp(plt.gca().get_xticklabels(), visible=False)
	plt.setp(plt.gca().get_yticklabels(), visible=False)
	plt.xlabel('Longitude (deg)');plt.ylabel('Depth (km)');

	plt.subplot(224)
	plt.plot(lats,deps,'k.');
	plt.setp(plt.gca().get_xticklabels(), visible=False)
	plt.setp(plt.gca().get_yticklabels(), visible=False)
	plt.xlabel('Latitude (deg)');plt.ylabel('Depth (km)');
	plt.savefig('xyz2rsf_1.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
	plt.show()

	from pyseistr import xyz2rsf
	V2,x2,y2,z2=xyz2rsf(lons,lats,deps,vps,nx=41,ny=37,nz=21); #in Z,X,Y
	from pyseistr import plot3d
	plot3d(V2,z=z2,x=x2,y=y2,cmap=plt.cm.jet,barlabel='P-wave velocity (km/s)',showf=False,close=False);
	plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Depth (km)",fontsize='large', fontweight='normal')
	plt.savefig('xyz2rsf_2.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
	plt.show()
	
	'''
	
	
	from scipy.interpolate import griddata as gd
	
	
	if xx is not None and yy is not None and zz is not None:
		X, Y, Z=np.meshgrid(xx,yy,zz)
	else:
		print('nx,ny,nz',nx,ny,nz)
		
	# generate new grid
		ox=min(xs)
		mx=max(xs)


		oy=min(ys)
		my=max(ys)


		oz=min(zs)
		mz=max(zs)


		xx=np.linspace(ox,mx,nx)
		yy=np.linspace(oy,my,ny)
		zz=np.linspace(oz,mz,nz)

		X, Y, Z=np.meshgrid(xx,yy,zz)


	V = gd((xs,ys,zs), value, (X,Y,Z), method='nearest')
	dout=np.transpose(V, (2, 1, 0)) #from Y,X,Z to Z,X,Y dimensions
	
	if nx is not None:
		return dout, xx, yy, zz
	else:
		return dout
	
	
	
def rsf3to3(din,x,y,z,xx,yy,zz):
	'''
	rsf3to3: converting a RSF 3D data cube from grid 1 (x,y,z) to grid 2 (xx,yy,zz)
	
	INPUT
	din:	3D cube on grid 1
	x,y,z:  1D axis vector [nx,ny,nz] of grid 1
	xx,yy,zz:  1D axis vector [nx,ny,nz] of grid 2
	
	OUTPUT
	dout:	3D cube on grid 2
	
	EXAMPLE
	# https://github.com/aaspip/data/blob/main/CMEZ3D-20260202.npy
	
	import numpy as np
	vel1=np.load('CMEZ3D-20260202.npy') #on 3D grid, X,Y,Z
	# Grid 1
	dx,dy,dz=(95.01-0)/100.0,95.01/100.0,95.01/100.0
	z0,x0,y0=-1.5,0,0
	nx,ny,nz=101,72,26
	x=np.linspace(0,nx-1,nx)*dx+x0;
	y=np.linspace(0,ny-1,ny)*dy+y0;
	z=np.linspace(0,nz-1,nz)*dz+z0;
	# Grid 2
	dx,dy,dz=0.9501,0.9501,0.2
	z0,x0,y0=0,0,0
	nx,ny,nz=101,101,52
	xx=np.linspace(0,nx-1,nx)*dx+x0;
	yy=np.linspace(0,ny-1,ny)*dy+y0;
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
	'''
	
	from scipy.interpolate import RegularGridInterpolator
	interp = RegularGridInterpolator(
		(x, y, z),
		din,
		method='linear',		# or 'nearest'
		bounds_error=False,
		fill_value=None #np.nan
	)

	# build query points
	XX, YY, ZZ = np.meshgrid(xx, yy, zz, indexing='ij')
	points = np.stack([XX.ravel(), YY.ravel(), ZZ.ravel()], axis=-1)

	# interpolate
	dout = interp(points).reshape(len(xx), len(yy), len(zz))
	
	
	
	return dout
	
	
	
	
	
	
	