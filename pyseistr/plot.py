from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def cseis():
	seis=np.concatenate(
(np.concatenate((0.5*np.ones([1,40]),np.expand_dims(np.linspace(0.5,1,88),axis=1).transpose(),np.expand_dims(np.linspace(1,0,88),axis=1).transpose(),np.zeros([1,40])),axis=1).transpose(),
np.concatenate((0.25*np.ones([1,40]),np.expand_dims(np.linspace(0.25,1,88),axis=1).transpose(),np.expand_dims(np.linspace(1,0,88),axis=1).transpose(),np.zeros([1,40])),axis=1).transpose(),
np.concatenate((np.zeros([1,40]),np.expand_dims(np.linspace(0,1,88),axis=1).transpose(),np.expand_dims(np.linspace(1,0,88),axis=1).transpose(),np.zeros([1,40])),axis=1).transpose()),axis=1)

	return ListedColormap(seis)
	

def plot3d(d3d,frames=None,z=None,x=None,y=None,figname=None,showf=True,**kwargs):
	'''
	plot3d: plot beautiful 3D slices
	
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
	'''

	[nz,nx,ny] = d3d.shape;
	
	if frames is None:
		frames=[int(nz/2),int(nx/2),int(ny/2)]
		
	X, Y, Z = np.meshgrid(np.arange(nx)*0.01, np.arange(ny)*0.01, np.arange(nz)*0.01)

	d3d=d3d.transpose([1,2,0])
	
	kw = {
    'vmin': d3d.min(),
    'vmax': d3d.max(),
    'levels': np.linspace(d3d.min(), d3d.max(), 100),
    'cmap':cseis()
	}
	
	fig = plt.figure(figsize=(8, 8))
	
	
	ax = fig.add_subplot(111, aspect='auto',projection='3d')
	plt.jet()
	# Plot contour surfaces
	_ = ax.contourf(
	X[:, :, -1], Y[:, :, -1], d3d[:, :, frames[0]], #x,y,z
	zdir='z', offset=0, alpha=1, **kw
	)
	_ = ax.contourf(
	X[0, :, :], d3d[frames[2], :, :], Z[0, :, :],
	zdir='y', offset=0, alpha=1, **kw
	)
	C = ax.contourf(
	d3d[:, frames[1], :], Y[:, -1, :], Z[:, -1, :],
	zdir='x', offset=X.max(), alpha=1.0, **kw
	)

	plt.gca().set_xlabel("X",fontsize='large', fontweight='normal')
	plt.gca().set_ylabel("Y",fontsize='large', fontweight='normal')
	plt.gca().set_zlabel("Z",fontsize='large', fontweight='normal')

	xmin, xmax = X.min(), X.max()
	ymin, ymax = Y.min(), Y.max()
	zmin, zmax = Z.min(), Z.max()
	ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
	plt.gca().invert_zaxis()
	
	if figname is not None:
		plt.savefig(figname,**kwargs)
	
	if showf:
		plt.show()
	else:
		plt.close() #or plt.clear() ?
		
		

	
	