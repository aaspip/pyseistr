import numpy as np
def dip3d(din,niter=5,liter=10,order=2,eps_dv=0.01, eps_cg=1, tol_cg=0.000001,rect=[5,5,5],verb=1):
	#dip3d: 3D dip estimation based on shaping regularized PWD algorithm
	#(independent implementation)
	#
	#Ported to Python by Yangkang Chen, 2022, verified to be exactly the same as the Matlab version
	#
	#INPUT
	#din: input data (nt*nx)
	#niter: number of nonlinear iterations
	#liter: number of linear iterations (in divn)
	#order: accuracy order
	#eps_dv: eps for divn  (default: 0.01)
	#eps_cg: eps for CG	(default: 1)
	#tol_cg: tolerence for CG (default: 0.000001)
	#rect:  smoothing radius (ndim*1)
	#verb: verbosity flag
	#
	#OUTPUT
	#dipi:  inline 3D slope
	#dipx:  xline 3D slope
	from .divne import divne
	
	dim = 3;
	n = np.zeros(dim,dtype='int');
	n1 = din.shape[0];
	n2 = din.shape[1];
	n3 = din.shape[2];
	n[0] = n1;
	n[1] = n2;
	n[2] = n3;

	n123 = din.size;

	dip_i=np.zeros([n1,n2,n3]);
	dip_x=np.zeros([n1,n2,n3]);

	for iter in range(0,niter):

		#corresponding to the eq.21 in the paper
		u1_i,u2_i = conv_allpass_i(din,dip_i,order); 	#inline linearization using the updated dip	
		ratio_i  = divne(-u2_i, u1_i, liter, rect, n, eps_dv, eps_cg, tol_cg,verb);

		#corresponding to the eq.21 in the paper
		u1_x,u2_x = conv_allpass_x(din,dip_x,order); 	#xline linearization using the updated dip
		ratio_x  = divne(-u2_x, u1_x, liter, rect, n, eps_dv, eps_cg, tol_cg,verb);
	
		dip_i=dip_i+ratio_i;
		dip_x=dip_x+ratio_x;

	return dip_i,dip_x

def conv_allpass_i(din,dip,order):
	# conv_allpass_i: Convolutional operator implemented by an allpass filter (iline direction)
	# 
	# Linearized inverse problem
	# C'(\sigma)d\Delta \sigma =  C(\sigma)d
	# 
	# IPNUT:
	# din: input data
	# dip: 3D dip
	# order: accuracy order
	#
	# OUTPUT:
	# u1: C'(\sigma)d (denominator)
	# u2: C(\sigma)d  (numerator)
	# 
	
	[n1,n2,n3]=din.shape;
	u1=np.zeros([n1,n2,n3]);
	u2=np.zeros([n1,n2,n3]);

	if order==1:
		nw=1;
	else:
		nw=2;


	filt1=np.zeros(2*nw+1);
	filt2=np.zeros(2*nw+1);

	# cprresponding to eqs.17-19
	# destruct the data
	for i1 in range(nw,n1-nw):
		for i2 in range(0,n2-1):
			for i3 in range(0,n3):
				if order==1:
					filt1=B3d(dip[i1,i2,i3]);
					filt2=B3(dip[i1,i2,i3]);
				else:
					filt1=B5d(dip[i1,i2,i3]);
					filt2=B5(dip[i1,i2,i3]);

				for iw in range(-nw,nw+1):
					u1[i1,i2,i3]=u1[i1,i2,i3]+(din[i1+iw,i2+1,i3]-din[i1-iw,i2,i3])*filt1[iw+nw];
					u2[i1,i2,i3]=u2[i1,i2,i3]+(din[i1+iw,i2+1,i3]-din[i1-iw,i2,i3])*filt2[iw+nw];

	return u1,u2

def conv_allpass_x(din,dip,order):
	# conv_allpass_x: Convolutional operator implemented by an allpass filter (xline direction)
	# 
	# Linearized inverse problem
	# C'(\sigma)d\Delta \sigma =  C(\sigma)d
	# 
	# IPNUT:
	# din: input data
	# dip: 3D dip
	# order: accuracy order
	#
	# OUTPUT:
	# u1: C'(\sigma)d (denominator)
	# u2: C(\sigma)d  (numerator)
	# 
	
	[n1,n2,n3]=din.shape;
	u1=np.zeros([n1,n2,n3]);
	u2=np.zeros([n1,n2,n3]);

	if order==1:
		nw=1;
	else:
		nw=2;


	filt1=np.zeros(2*nw+1);
	filt2=np.zeros(2*nw+1);

	# cprresponding to eqs.17-19
	# destruct the data
	for i1 in range(nw,n1-nw):
		for i2 in range(0,n2):
			for i3 in range(0,n3-1):
				if order==1:
					filt1=B3d(dip[i1,i2,i3]);
					filt2=B3(dip[i1,i2,i3]);
				else:
					filt1=B5d(dip[i1,i2,i3]);
					filt2=B5(dip[i1,i2,i3]);
				for iw in range(-nw,nw+1):
					u1[i1,i2,i3]=u1[i1,i2,i3]+(din[i1+iw,i2,i3+1]-din[i1-iw,i2,i3])*filt1[iw+nw];
					u2[i1,i2,i3]=u2[i1,i2,i3]+(din[i1+iw,i2,i3+1]-din[i1-iw,i2,i3])*filt2[iw+nw];
	return u1,u2


#cprresponding to eqs.13-16
#form the filters
def B3(sigma):
	#B3 coefficient
	#sigma: slope

	b3=np.zeros[2];
	b3[0]=(1-sigma)*(2-sigma)/12;
	b3[1]=(2+sigma)*(2-sigma)/6;
	b3[2]=(1+sigma)*(2+sigma)/12;
	
	return b3

def B3d(sigma):
	#B3 coefficient derivative
	#sigma: slope

	b3d=np.zeros[2];
	b3d[0]=-(2-sigma)/12-(1-sigma)/12;
	b3d[1]=(2-sigma)/6-(2+sigma)/6;
	b3d[2]=(2+sigma)/12+(1+sigma)/12;

	return b3d

def B5(sigma):
	#B5 coefficient
	#sigma: slope

	b5=np.zeros(5);
	b5[0]=(1-sigma)*(2-sigma)*(3-sigma)*(4-sigma)/1680;
	b5[1]=(4-sigma)*(2-sigma)*(3-sigma)*(4+sigma)/420;
	b5[2]=(4-sigma)*(3-sigma)*(3+sigma)*(4+sigma)/280;
	b5[3]=(4-sigma)*(2+sigma)*(3+sigma)*(4+sigma)/420;
	b5[4]=(1+sigma)*(2+sigma)*(3+sigma)*(4+sigma)/1680;

	return b5

def B5d(sigma):
	#B5 coefficient derivative
	#sigma: slope

	b5d=np.zeros(5);
	b5d[0]=-(2-sigma)*(3-sigma)*(4-sigma)/1680-\
	(1-sigma)*(3-sigma)*(4-sigma)/1680-\
	(1-sigma)*(2-sigma)*(4-sigma)/1680-\
	(1-sigma)*(2-sigma)*(3-sigma)/1680;

	b5d[1]=-(2-sigma)*(3-sigma)*(4+sigma)/420-\
	(4-sigma)*(3-sigma)*(4+sigma)/420-\
	(4-sigma)*(2-sigma)*(4+sigma)/420+\
	(4-sigma)*(2-sigma)*(3-sigma)/420;

	b5d[2]=-(3-sigma)*(3+sigma)*(4+sigma)/280-\
	(4-sigma)*(3+sigma)*(4+sigma)/280+\
	(4-sigma)*(3-sigma)*(4+sigma)/280+\
	(4-sigma)*(3-sigma)*(3+sigma)/280;

	b5d[3]=-(2+sigma)*(3+sigma)*(4+sigma)/420+\
	(4-sigma)*(3+sigma)*(4+sigma)/420+\
	(4-sigma)*(2+sigma)*(4+sigma)/420+\
	(4-sigma)*(2+sigma)*(3+sigma)/420;

	b5d[4]=(2+sigma)*(3+sigma)*(4+sigma)/1680+\
	(1+sigma)*(3+sigma)*(4+sigma)/1680+\
	(1+sigma)*(2+sigma)*(4+sigma)/1680+\
	(1+sigma)*(2+sigma)*(3+sigma)/1680;

	return b5d
	


