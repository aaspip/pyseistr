import numpy as np
def fkdip(d,w):
	# das_fk_dip: FK dip filter
	#
	#
	# INPUT
	# d: 	input data (2D)
	# w:  	half width (in percentage) of the cone filter (i.e., w*nk=nwidth)
	#
	# OUTPUT
	# d0: output data

	[n1,n2]=d.shape;
	nf=nextpow2(n1);
	nk=nextpow2(n2);
	nf2=int(nf/2);
	nk2=int(nk/2);
	Dfft1=np.fft.fft(d,nf,0);
	Dtmp=Dfft1[0:nf2+1,:];

	#processing area
	Dtmp2=np.fft.fft(Dtmp,nk,1);
	Dtmp2=np.fft.fftshift(Dtmp2,1);

	nw=w*nk;
	[nn1,nn2]=Dtmp2.shape;
	mask=np.zeros([nn1,nn2]);

	for i1 in range(1,nn1+1):
		for i2 in range(1,nn2+1):
			if i1> (nn1/nw)*(i2-(nk2)) and i1> (nn1/nw)*((nk2)-i2):
				mask[i1-1,i2-1]=1;

	Dtmp2=Dtmp2*mask;
	Dtmp=np.fft.ifft(np.fft.ifftshift(Dtmp2,1),nk,1);

	#honor symmetry for inverse fft
	Dfft2=np.zeros([nf,nk],dtype=np.complex_);
	Dfft2[0:nf2+1,:]=Dtmp;
	Dfft2[nf2+1:,:]=np.conj(np.flipud(Dtmp[1:-1,:]));
	d0=np.real(np.fft.ifft(Dfft2,nf,0));
	d0=d0[0:n1,0:n2];

	return d0
	
def nextpow2(N):
    """ Function for finding the next power of 2 """
    n = 1
    while n < N: n *= 2
    return n
    