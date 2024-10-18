def xyz2rsf(lons,lats,deps,vps,nx,ny,nz):
	'''
	xyz2rsf: interpolate from xyz (unstructured data) to rsf (regularly sampled format)
	
	INPUT:
	lons: input X coordinates
	lats: input Y coordinates 
	deps: input Z coordinates
	vps: values at the XYZ points
	
	OUTPUT
	V: rsf data on [y,x,z] (ny,nx,nz)
	
	EXAMPLE
	
	Using the following script to prepare the data
	https://github.com/aaspip/notebook/blob/main/Texas/visualize_Texas_velocity_model.ipynb
	
	from pyseistr import xyz2rsf
	V2,x2,y2,z2=xyz2rsf(lons,lats,deps,vps,nx=41,ny=37,nz=21);
	V2=np.transpose(V2, (2, 1, 0))
	from pyseistr import plot3d
	plot3d(V2,z=z2,x=x2,y=y2,cmap=plt.cm.jet,barlabel='P-wave velocity (km/s)',showf=False,close=False);
	plt.gca().set_xlabel("Longitude (deg)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Latitude (deg)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Depth (km)",fontsize='large', fontweight='normal')
	plt.savefig('xyzdata_3.png')
	plt.show()

	'''
	from scipy.interpolate import griddata as gd
	import numpy as np
	
	# generate new grid
	ox=min(lons)
	mx=max(lons)
# 	nx=41

	oy=min(lats)
	my=max(lats)
# 	ny=37

	oz=min(deps)
	mz=max(deps)
# 	nz=21

	x=np.linspace(ox,mx,nx)
	y=np.linspace(oy,my,ny)
	z=np.linspace(oz,mz,nz)

	X, Y, Z=np.meshgrid(x,y,z)

	# V = gd((lons,lats,deps), vps, (X.flatten(),Y.flatten(),Z.flatten()), method='nearest').reshape(ny,nx,nz)
	
	V = gd((lons,lats,deps), vps, (X,Y,Z), method='nearest')

	return V,x,y,z
	
