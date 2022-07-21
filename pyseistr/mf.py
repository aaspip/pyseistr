def mf(D,nfw=7,ifb=1,axis=2):
	#MF: median filter along first or second axis for 2D profile
	#  IN   D:   	intput data 
	#       nfw:    window size 
	#       ifb:    if use padded boundary (if not, zero will be padded)
	#       axis:   along the vertical (1) or horizontal (2) axis
	#      
	#  OUT   D1:  	output data
	# 
	#  Copyright (C) 2014 The University of Texas at Austin
	#  Copyright (C) 2014 Yangkang Chen
	#  Ported to python in Apr, 17, 2022
	#
	#  This program is free software: you can redistribute it and/or modify
	#  it under the terms of the GNU General Public License as published
	#  by the Free Software Foundation, either version 3 of the License, or
	#  any later version.
	#
	#  This program is distributed in the hope that it will be useful,
	#  but WITHOUT ANY WARRANTY; without even the implied warranty of
	#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	#  GNU General Public License for more details: http://www.gnu.org/licenses/
	#  
	# References
	# Huang et al., 2021, Erratic noise suppression using iterative structure-oriented space-varying median filtering with sparsity constraint, Geophysical Prospecting, 69, 101-121.
	# Chen et al., 2020, Deblending of simultaneous-source data using a structure-oriented space-varying median filter, Geophysical Journal International, 222, 1805–1823.
	# Gan et al., 2016, Separation of simultaneous sources using a structural-oriented median filter in the flattened dimension, Computers & Geosciences, 86, 46-54.
	# Chen, Y., 2015, Deblending using a space-varying median filter, Exploration Geophysics, 46, 332-341.
	import numpy as np

	# nfw should be odd
	if np.mod(nfw,2)==0:
		nfw=nfw+1;

	if axis==2:
		D=D.transpose();
	n1=D.shape[0];
	n2=D.shape[1];
	
	nfw2=(nfw-1)/2;nfw2=int(nfw2);
	
	if ifb==1:
		D=np.concatenate((np.flipud(D[0:nfw2,:]),D,np.flipud(D[n1-nfw2:n1,:])),axis=0);
	else: 
		D=np.concatenate((np.zeros([nfw2,n2]),D,np.zeros([nfw2,n2])),axis=0);
	# output data
	D1=np.zeros([n1,n2]);
	for i2 in range(0,n2):
		for i1 in range(0,n1):
			D1[i1,i2]=np.median(D[i1:i1+nfw,i2]); 
	if axis==2:
		D1=D1.transpose();
		
	return D1


def svmf(D,nfw=7,ifb=1,axis=2,l1=2,l2=0,l3=2,l4=4):
	#SVMF: space-varying median filter along first or second axis for 2D profile
	#  IN   D:   	intput data 
	#       nfw:    window size
	#       ifb:    if use padded boundary (if not, zero will be padded)
	#       axis:   along the vertical (1) or horizontal (2) axis
	#      
	#  OUT   D1:  	output data
	# 		 win_len: window length distribution
	#  Copyright (C) 2019 Yangkang Chen
	#
	#  This program is free software: you can redistribute it and/or modify
	#  it under the terms of the GNU General Public License as published
	#  by the Free Software Foundation, either version 3 of the License, or
	#  any later version.
	#
	#  This program is distributed in the hope that it will be useful,
	#  but WITHOUT ANY WARRANTY; without even the implied warranty of
	#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	#  GNU General Public License for more details: http://www.gnu.org/licenses/
	#
	import numpy as np
	n1=D.shape[0];
	n2=D.shape[1];
	
	
	Dtmp=mf(D,nfw,ifb,axis);
	medianv=np.sum(np.abs(Dtmp.flatten()))/(n1*n2);

	# nfw should be odd
	if np.mod(nfw,2)==0:
		nfw=nfw+1;

	# calculate length
	win_len=np.zeros([n1,n2],dtype='int');
	for i2 in range(0,n2):
		for i1 in range(0,n1):
			if np.abs(Dtmp[i1,i2]) < medianv:
				if np.abs(Dtmp[i1,i2]) < medianv/2:
					win_len[i1,i2]=nfw+l1;
				else:
					win_len[i1,i2]=nfw+l2;
			else:
				if np.abs(Dtmp[i1,i2]) > medianv*2:
					win_len[i1,i2]=nfw-l4;
				else:
					win_len[i1,i2]=nfw-l3;
	if axis==2:
		D=D.transpose();
		win_len=win_len.transpose();
	n1=D.shape[0];
	n2=D.shape[1];
	win_len2=(win_len-1)/2;win_len2=win_len2.astype(int);

	nfw_b=(np.max([nfw+l1,nfw+l2])-1)/2;nfw_b=int(nfw_b);
	
	if ifb==1:
		D=np.concatenate((np.flipud(D[0:nfw_b,:]),D,np.flipud(D[n1-nfw_b:n1,:])),axis=0);
	else:
		D=np.concatenate((np.zeros([nfw_b,n2]),D,np.zeros([nfw_b,n2])),axis=0);
	
	# output data
	D1=np.zeros([n1,n2]);
	for i2 in range(0,n2):
		for i1 in range(0,n1):
# 			print(nfw_b,win_len2[i1,i2],i1+nfw_b-win_len2[i1,i2],i1+nfw_b+win_len2[i1,i2])
# 			print(D.shape)
# 			print(i1,i2,n1,n2)
			D1[i1,i2]=np.median(D[i1+nfw_b-win_len2[i1,i2]:i1+nfw_b+win_len2[i1,i2]+1,i2]); 
	win_len=win_len2*2+1;
	
	if axis==2:
		D1=D1.transpose();
		win_len=win_len.transpose();    

	return D1,win_len
