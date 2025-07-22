from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def cseis():
	'''
	cseis: seismic colormap
	
	By Yangkang Chen
	June, 2022
	
	EXAMPLE
	from pyseistr import cseis
	import numpy as np
	from matplotlib import pyplot as plt
	plt.imshow(np.random.randn(100,100),cmap=cseis())
	plt.show()
	'''
	seis=np.concatenate(
(np.concatenate((0.5*np.ones([1,40]),np.expand_dims(np.linspace(0.5,1,88),axis=1).transpose(),np.expand_dims(np.linspace(1,0,88),axis=1).transpose(),np.zeros([1,40])),axis=1).transpose(),
np.concatenate((0.25*np.ones([1,40]),np.expand_dims(np.linspace(0.25,1,88),axis=1).transpose(),np.expand_dims(np.linspace(1,0,88),axis=1).transpose(),np.zeros([1,40])),axis=1).transpose(),
np.concatenate((np.zeros([1,40]),np.expand_dims(np.linspace(0,1,88),axis=1).transpose(),np.expand_dims(np.linspace(1,0,88),axis=1).transpose(),np.zeros([1,40])),axis=1).transpose()),axis=1)

	return ListedColormap(seis)
	

def plot3d(d3d,frames=None,z=None,x=None,y=None,dz=0.01,dx=0.01,dy=0.01,nlevel=100,figsize=(8, 6),ifnewfig=True,figname=None,showf=True,close=True,**kwargs):
	'''
	plot3d: plot beautiful 3D slices
	
	INPUT
	d3d: input 3D data (z in first-axis, x in second-axis, y in third-axis)
	frames: plotting slices on three sides (default: [nz/2,nx/2,ny/2])
	z,x,y: axis vectors  (default: 0.01*[np.arange(nz),np.arange(nx),np.arange(ny)])
	figname: figure name to be saved (default: None)
	showf: if show the figure (default: True)
	close: if not show a figure, if close the figure (default: True)
	kwargs: other specs for plotting (e.g., vmin=-0.1, vmax=0.1, and others accepted by contourf)
	dz,dx,dy: interval (default: 0.01)
	
	By Yangkang Chen
	June, 18, 2023
	
	EXAMPLE 1
	import numpy as np
	d3d=np.random.rand(100,100,100);
	from pyseistr import plot3d
	plot3d(d3d);
	
	EXAMPLE 2
	import scipy
	data=scipy.io.loadmat('/Users/chenyk/chenyk/matlibcyk/test/hyper3d.mat')['cmp']
	from pyseistr import plot3d
	plot3d(data);
	
	EXAMPLE 3
	import numpy as np
	import matplotlib.pyplot as plt
	from pyseistr import plot3d

	nz=81
	nx=81
	ny=81
	dz=20
	dx=20
	dy=20
	nt=1501
	dt=0.001

	v=np.arange(nz)*20*1.2+1500;
	vel=np.zeros([nz,nx,ny]);
	for ii in range(nx):
		for jj in range(ny):
			vel[:,ii,jj]=v;

	plot3d(vel,figsize=(16,10),cmap=plt.cm.jet,z=np.arange(nz)*dz,x=np.arange(nx)*dx,y=np.arange(ny)*dy,barlabel='Velocity (m/s)',showf=False,close=False)
	plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Z (m)",fontsize='large', fontweight='normal')
	plt.title('3D velocity model')
	plt.savefig(fname='vel3d.png',format='png',dpi=300)
	plt.show()

	EXAMPLE 4 (a spatially varying velocity model)
	import numpy as np
	import matplotlib.pyplot as plt
	from pyseistr import plot3d

	nz=81
	nx=81
	ny=81
	dz=20
	dx=20
	dy=20
	nt=1501
	dt=0.001
	
	# Velocity gradients
	x_gradient=0.05
	y_gradient=0.07
	z_gradient=0.10

	vel3d=np.ones([nx,ny,nz],dtype='float32')
	for x in range(nx):
		vel3d[x,:,:]+=(x*x_gradient)
	for y in range(ny):
		vel3d[:,y,:]+=(y*y_gradient)
	for z in range(nz):
		vel3d[:,:,z]+=(z*z_gradient)

	plot3d(np.transpose(vel3d,(2,0,1)),frames=[0,nx-1,0],figsize=(16,10),cmap=plt.cm.jet,z=np.arange(nz)*dz,x=np.arange(nx)*dx,y=np.arange(ny)*dy,barlabel='Velocity (m/s)',showf=False,close=False)
	plt.gca().set_xlabel("X (m)",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y (m)",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Z (m)",fontsize='large', fontweight='normal')
	plt.title('3D velocity model')
	plt.savefig(fname='vel3d.png',format='png',dpi=300)
	plt.show()
	'''
#,

	[nz,nx,ny] = d3d.shape;
	
	if frames is None:
		frames=[int(nz/2),int(nx/2),int(ny/2)]
		
	if z is None:
		z=np.arange(nz)*dz
	
	if x is None:
		x=np.arange(nx)*dx
		
	if y is None:
		y=np.arange(ny)*dy
	
	X, Y, Z = np.meshgrid(x, y, z)
	
	d3d=d3d.transpose([1,2,0])
	
	
	kw = {
	'vmin': d3d.min(),
	'vmax': d3d.max(),
	'levels': np.linspace(d3d.min(), d3d.max(), nlevel),
	'cmap':cseis()
	}
	
	kw.update(kwargs)
	
	if 'alpha' not in kw.keys():
		kw['alpha']=1.0
	
	if ifnewfig==False:
		ax=plt.gca()
	else:
		fig = plt.figure(figsize=figsize)
		ax = fig.add_subplot(111, aspect='auto',projection='3d')
		plt.jet()

	# Plot contour surfaces
	_ = ax.contourf(
	X[:, :, -1], Y[:, :, -1], d3d[:, :, frames[0]].transpose(), #x,y,z
	zdir='z', offset=0, **kw
	)

	_ = ax.contourf(
	X[0, :, :], d3d[:, frames[2], :], Z[0, :, :],
	zdir='y', offset=Y.min(), **kw
	)
	
	C = ax.contourf(
	d3d[frames[1], :, :], Y[:, -1, :], Z[:, -1, :],
	zdir='x', offset=X.max(), **kw
	)

	plt.gca().set_xlabel("X",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Z",fontsize='large', fontweight='normal')

	xmin, xmax = X.min(), X.max()
	ymin, ymax = Y.min(), Y.max()
	zmin, zmax = Z.min(), Z.max()
	ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
	plt.gca().invert_zaxis()

	# Colorbar
	if 'barlabel' in kw.keys():
		cbar=plt.gcf().colorbar(C, ax=ax, orientation='horizontal', fraction=0.02, pad=0.1, format= "%.2f", label=kw['barlabel'])
		cbar.ax.locator_params(nbins=5)
		kwargs.__delitem__('barlabel')

	if figname is not None:
		if 'cmap' in kwargs.keys():
			kwargs.__delitem__('cmap')
		plt.savefig(figname,**kwargs)
	
	if showf:
		plt.show()
	else:
		if close:
			plt.close() #or plt.clear() ?
		
def framebox(x1,x2,y1,y2,c=None,lw=None):
	'''
	framebox: for drawing a frame box
	
	By Yangkang Chen
	June, 2022
	
	INPUT
	x1,x2,y1,y2: intuitive
	
	EXAMPLE I
	from pyseistr.plot import framebox
	from pyseistr.synthetics import gensyn
	from matplotlib import pyplot as plt
	d=gensyn();
	plt.imshow(d);
	framebox(200,400,200,300);
	plt.show()

	EXAMPLE II
	from pyseistr.plot import framebox
	from pyseistr.synthetics import gensyn
	from matplotlib import pyplot as plt
	d=gensyn();
	plt.imshow(d);
	framebox(200,400,200,300,c='g',lw=4);
	plt.show()
	
	'''
	
	if c is None:
		c='r';
	if lw is None:
		lw=2;

	plt.plot([x1,x2],[y1,y1],linestyle='-',color=c,linewidth=lw);
	plt.plot([x1,x2],[y2,y2],linestyle='-',color=c,linewidth=lw);
	plt.plot([x1,x1],[y1,y2],linestyle='-',color=c,linewidth=lw);
	plt.plot([x2,x2],[y1,y2],linestyle='-',color=c,linewidth=lw);

	
	return
	
			
def gengif(inpath, outpath, duration=500):
	'''
	gengif: generate a GIF file from a list of PNG or other type of images.

	By Yangkang Chen
	Nov, 2024
	
	INPUT
	inpath:   input list of images (e.g., PNG)
	outpath:  output GIF path
	duration: Animation speed
	
	OUTPUT
	N/A
	
	EXAMPLE
	pywave/demos/test_second_wfd3ds.py
	
	'''
	from PIL import Image
	images = [Image.open(image_path) for image_path in inpath]

	images[0].save(
		outpath,
		save_all=True,
		append_images=images[1:],
		optimize=False,
		duration=duration,
		loop=0  # 0 means infinite loop
	)
	

def wiping(inpath, outpath, duration=500, number=30):
	'''
	wiping: generate a GIF file from a list of PNG or other type of images.

	By Yangkang Chen
	Nov, 2024
	
	INPUT
	inpath:   input list of images (e.g., PNG)
	outpath:  output GIF path
	duration: Animation speed
	number:	  number of temporary images
	
	OUTPUT
	N/A
	
	EXAMPLE
	
	from pyseistr import genplane3d,plot3d
	data,datan=genplane3d(noise=True,seed=202425,var=0.1);
	dn=datan[:,:,10]
	import pydrr as pd
	d1=pd.drr3d(dn,0,120,0.004,3,3);	#DRR
	noi1=dn-d1;
	import numpy as np
	import matplotlib.pyplot as plt	
	plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0); 
	plt.imshow(dn,clim=(-1, 1),aspect='auto');plt.axis('off');plt.margins(0, 0);plt.savefig('dn.png')
	plt.imshow(d1,clim=(-1, 1),aspect='auto');plt.margins(0, 0);plt.savefig('d1.png')
	from pyseistr import wiping
	wiping(['dn.png','d1.png'], 'test.gif', duration=500)
	
	'''
	from PIL import Image
	images = [Image.open(image_path) for image_path in inpath]
	nimages=len(inpath)
	I = [np.asarray(img).copy() for img in images]

	nx=I[0].shape[1]
	dx=int(nx/number)
	ind=0
	images=[]
	for ii in range(number):
		ind=ind+dx
# 		print(ii.shape,I[0].shape,'ind=',ind)
		Inew=I[0].copy()
		Inew[:,0:ind,:]=I[1][:,0:ind,:] #equal I[1] (e.g., denoised)
		Inew[:,ind-5:ind+5,:]=int(255/2)
		im = Image.fromarray(np.uint8(Inew))
		images.append(im)
	images[0].save(
		outpath,
		save_all=True,
		append_images=images[1:],
		optimize=False,
		duration=duration,
		loop=0  # 0 means infinite loop
	)	
	

