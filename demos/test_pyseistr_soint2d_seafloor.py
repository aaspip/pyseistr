## This is a DEMO script for 2D structure-oriented interpolation on a seafloor dataset
# Short description of the data:
# SeaBeam is an apparatus for measuring water depth both directly under a boat and somewhat off to the sides of the boat's track. This is a benchmark SeaBeam data from a single day of acquisition. 

import numpy as np
import matplotlib.pyplot as plt
import pyseistr as ps


## Download data from https://github.com/aaspip/data
# https://github.com/aaspip/data/blob/main/apr18_160_160.bin

## Load data
fid=open("/Users/chenyk/RSFSRC/book/sep/pwd/seab/apr18_160_160.bin","rb");
raw = np.fromfile(fid, dtype = np.float32, count = 160*160).reshape([160,160],order='F')


## Calculate local slope
dip=ps.dip2dc(raw,mask=raw,order=2,rect=[10,10,1],verb=0);

## SOINT2D
mask=0*dip;
recon=ps.soint2dc(raw,mask,dip,order=2,niter=100,njs=[1,1],drift=0,hasmask=0,twoplane=0,prec=0,verb=0);

fig = plt.figure(figsize=(6, 8))
ax=plt.subplot(3,1,1)
plt.imshow(raw,cmap='gray',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Input');

ax=plt.subplot(3,1,2)
plt.imshow(dip,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Slope');

ax=plt.subplot(3,1,3)
plt.imshow(recon,cmap='gray',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Reconstructed');

plt.savefig('test_pyseistr_soint2d_seafloor.png',format='png',dpi=300)
plt.show()
