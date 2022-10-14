from sofcfun import *
from sof3dcfun import *

def somean3d(dn,dipi,dipx,r1,r2,eps,order):
	# somean3d: 3D structure-oriented mean filter
	# 
	# by Yangkang Chen, 2022
	#
	# INPUT:
	# dn: model  noisy data
	# dipi: inline slope
	# dipx: xline slope
	# r1,r2:    spray radius
	# order:    PWD order
	# eps: regularization (default:0.01);
	# 
	# OUTPUT:
	# ds: smoothed data
	# 
	import numpy as np
	nnp=(2*r1+1)*(2*r2+1);
	
	#flattening
	from .pwspray3d import pwspray3d
	u = pwspray3d(dn,dipi,dipx,r1,r2,order,eps);

	#smoothing
	ds=np.sum(u,1)/nnp;

	return ds

def somean3dc(dn,dipi,dipx,r1,r2,eps,order,verb=0):
	# somean3d: 3D structure-oriented mean filter
	# 
	# by Yangkang Chen, 2022
	#
	# INPUT:
	# dn: model  noisy data
	# dipi: inline slope
	# dipx: xline slope
	# r1,r2:    spray radius
	# order:    PWD order
	# eps: regularization (default:0.01);
	# 
	# OUTPUT:
	# ds: smoothed data
	# 
	import numpy as np
# 	nnp=(2*r1+1)*(2*r2+1);
	
	#flattening
# 	from .pwspray3d import pwspray3d
# 	u = pwspray3d(dn,dipi,dipx,r1,r2,order,eps);
# 	
# 	
# 	#smoothing
# 	ds=np.sum(u,1)/nnp;

	if dn.ndim==2:
		[n1,n2]=dn.shape;
		n3=1;
	else: #assuming ndim=3;
		[n1,n2,n3]=dn.shape;
		
	dn=np.float32(dn).flatten(order='F');
	dipi=np.float32(dipi).flatten(order='F');
	dipx=np.float32(dipx).flatten(order='F');
	
	print(dn.max(),dn.min(),dn.var())
	ds=csomean3d(dn,dipi,dipx,n1,n2,n3,r1,r2,order,eps,verb);
	ds=ds.reshape([n1,n2,n3],order='F');
	print(ds.max(),ds.min(),ds.var())
	
	return ds


