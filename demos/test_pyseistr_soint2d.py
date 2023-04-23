import numpy as np
import matplotlib.pyplot as plt
import pyseistr as ps

fid=open("/Users/chenyk/RSFSRC/book/sep/pwd/seab/dip.bin","rb");
dip = np.fromfile(fid, dtype = np.float32, count = 160*160).reshape([160,160],order='F')

fid=open("/Users/chenyk/RSFSRC/book/sep/pwd/seab/bin.bin","rb");
raw = np.fromfile(fid, dtype = np.float32, count = 160*160).reshape([160,160],order='F')

fid=open("/Users/chenyk/RSFSRC/book/sep/pwd/seab/mis.bin","rb");
mis = np.fromfile(fid, dtype = np.float32, count = 160*160).reshape([160,160],order='F')



## SOINT2D
mask=0*dip;
d1=ps.soint2dc(raw,mask,dip,order=2,niter=100,njs=[1,1],drift=0,hasmask=0,twoplane=0,prec=0,verb=1);

print('Benchmark dip:',dip.max(),dip.min(),dip.std())
print('Benchmark recon:',mis.max(),mis.min(),mis.std())
print('Pyseistr recon:',d1.max(),d1.min(),d1.std())


fig = plt.figure(figsize=(10, 10))
ax=plt.subplot(3,2,1)
plt.imshow(raw,cmap='gray',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Input');

ax=plt.subplot(3,2,2)
plt.imshow(dip,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Slope');

ax=plt.subplot(3,2,3)
plt.imshow(mis,cmap='gray',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Output 1');

ax=plt.subplot(3,2,4)
plt.imshow(d1,cmap='gray',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Output 2');

ax=plt.subplot(3,2,5)
plt.imshow(mis-d1,cmap='gray',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Error');
plt.show()