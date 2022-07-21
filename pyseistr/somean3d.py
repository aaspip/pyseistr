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




