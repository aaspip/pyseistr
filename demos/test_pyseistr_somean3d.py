## This is a DEMO script for 3D structure-oriented mean/smoothing filter
import numpy as np
import matplotlib.pyplot as plt
import pyseistr as ps

## load data
#The input 3D source data file "real3d.bin" can be downloaded from
#https://github.com/chenyk1990/reproducible_research/blob/master/drr3d/matfun/real3d.bin,
#and should be placed in the same folder as test_pyseistr_somean3d.py.

fid=open('real3d.bin','rb')
d = np.fromfile(fid, dtype = np.float32, count = 300*1000).reshape([300,1000],order='F')
d=d.reshape(300,100,10,order='F');
d=d[199:299,49:99,:]
# d=d(200:300,50:100,:);
cmp=d/d.flatten().max();
cmpn=cmp;
print(cmpn.flatten().sum())

## 3D slope calculation (inline and xline)
[dipi,dipx] = ps.dip3dc(cmpn);


## Structural smoothing
r1=2;
r2=2;
eps=0.01;
order=2;
cmpn_d1=ps.somean3dc(cmpn,dipi,dipx,r1,r2,eps,order);

## plot results
fig = plt.figure(figsize=(8, 8))
ax=plt.subplot(5,1,1)
plt.imshow(cmpn.reshape(100,500,order='F'),cmap='jet',clim=(-0.2, 0.2),aspect=0.8)
plt.title('Raw data',color='k');ax.set_xticks([]);ax.set_yticks([]);
ax=plt.subplot(5,1,2)
plt.imshow(dipi.reshape(100,500,order='F'),cmap='jet',clim=(-1,1),aspect=0.8)
plt.title('Iline slope',color='k');ax.set_xticks([]);ax.set_yticks([]);
ax=plt.subplot(5,1,3)
plt.imshow(dipx.reshape(100,500,order='F'),cmap='jet',clim=(-1,1),aspect=0.8)
plt.title('Xline slope',color='k');ax.set_xticks([]);ax.set_yticks([]);
ax=plt.subplot(5,1,4)
plt.imshow(cmpn_d1.reshape(100,500,order='F'),cmap='jet',clim=(-0.2, 0.2),aspect=0.8)
plt.title('Filtered (SOMEAN)',color='k');ax.set_xticks([]);ax.set_yticks([]);
ax=plt.subplot(5,1,5)
plt.imshow((cmpn-cmpn_d1).reshape(100,500,order='F'),cmap='jet',clim=(-0.2, 0.2),aspect=0.8)
plt.title('Noise (SOMEAN)',color='k');ax.set_xticks([]);ax.set_yticks([]);
plt.savefig('test_pyseistr_somean3d.png',format='png',dpi=300)
plt.show()



