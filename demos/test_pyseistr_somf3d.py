## This is a DEMO script for 3D structure-oriented median filter
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

np.random.seed(202122);
noi=(np.random.rand(cmp.shape[0],cmp.shape[1],cmp.shape[2])*2-1)*0.2;

#add erratic noise
nerr=noi*0;
inds1=[10,10,20,40,45];
inds2=[3,5,5,5,7];
nerr=np.zeros([cmp.shape[0],cmp.shape[1],cmp.shape[2]]);
for i2 in range(len(inds2)):
	for i1 in range(len(inds1)):
		nerr[:,inds1[i1],inds2[i2]]=noi[:,inds1[i1],inds2[i2]]*10;
cmpn=cmp+nerr;

## 3D slope calculation (inline and xline)
[dipi,dipx] = ps.dip3d(cmpn);

## Structural smoothing
r1=2;
r2=2;
eps=0.01;
order=2;
cmpn_d1=ps.somean3d(cmpn,dipi,dipx,r1,r2,eps,order);
cmpn_d2=ps.somf3d(cmpn,dipi,dipx,r1,r2,eps,order);

## plot results
fig = plt.figure(figsize=(16, 8))
ax=plt.subplot(5,2,1)
plt.imshow(cmpn.reshape(100,500,order='F'),cmap='jet',clim=(-0.2, 0.2),aspect=0.8)
plt.title('Raw data',color='k');ax.set_xticks([]);ax.set_yticks([]);
ax=plt.subplot(5,2,3)
plt.imshow(dipi.reshape(100,500,order='F'),cmap='jet',clim=(-1,1),aspect=0.8)
plt.title('Iline slope',color='k');ax.set_xticks([]);ax.set_yticks([]);
ax=plt.subplot(5,2,5)
plt.imshow(dipx.reshape(100,500,order='F'),cmap='jet',clim=(-1,1),aspect=0.8)
plt.title('Xline slope',color='k');ax.set_xticks([]);ax.set_yticks([]);
ax=plt.subplot(5,2,7)
plt.imshow(cmpn_d1.reshape(100,500,order='F'),cmap='jet',clim=(-0.2, 0.2),aspect=0.8)
plt.title('Filtered (SOMEAN)',color='k');ax.set_xticks([]);ax.set_yticks([]);
ax=plt.subplot(5,2,9)
plt.imshow((cmpn-cmpn_d1).reshape(100,500,order='F'),cmap='jet',clim=(-0.2, 0.2),aspect=0.8)
plt.title('Noise (SOMEAN)',color='k');ax.set_xticks([]);ax.set_yticks([]);


# ax=plt.subplot(5,2,2)
# plt.imshow(cmpn.reshape(100,500,order='F'),cmap='jet',clim=(-0.2, 0.2),aspect=0.8)
# plt.title('Raw data',color='k');ax.set_xticks([]);ax.set_yticks([]);
# ax=plt.subplot(5,2,4)
# plt.imshow(dipi.reshape(100,500,order='F'),cmap='jet',clim=(-1,1),aspect=0.8)
# plt.title('Iline slope',color='k');ax.set_xticks([]);ax.set_yticks([]);
# ax=plt.subplot(5,2,6)
# plt.imshow(dipx.reshape(100,500,order='F'),cmap='jet',clim=(-1,1),aspect=0.8)
# plt.title('Xline slope',color='k');ax.set_xticks([]);ax.set_yticks([]);
ax=plt.subplot(5,2,8)
plt.imshow(cmpn_d2.reshape(100,500,order='F'),cmap='jet',clim=(-0.2, 0.2),aspect=0.8)
plt.title('Filtered (SOMF)',color='k');ax.set_xticks([]);ax.set_yticks([]);
ax=plt.subplot(5,2,10)
plt.imshow((cmpn-cmpn_d2).reshape(100,500,order='F'),cmap='jet',clim=(-0.2, 0.2),aspect=0.8)
plt.title('Noise (SOMF)',color='k');ax.set_xticks([]);ax.set_yticks([]);
plt.savefig('test_pyseistr_somf3d.png',format='png',dpi=300)

plt.show()




