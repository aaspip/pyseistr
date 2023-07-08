## This is a DEMO script for 2D structure-oriented median filter
import numpy as np
import matplotlib.pyplot as plt
import pyseistr as ps

## Generate synthetic data
from pyseistr import gensyn
data=gensyn();
data=data[:,0::10];#or data[:,0:-1:10];
data=data/np.max(np.max(data));
np.random.seed(202122);
scnoi=(np.random.rand(data.shape[0],data.shape[1])*2-1)*0.2;

#add erratic noise
nerr=scnoi*0;inds=[10,20,50];
for i in range(len(inds)):
	nerr[:,inds[i]]=scnoi[:,inds[i]]*10;

dn=data+scnoi+nerr;


print('size of data is (%d,%d)'%data.shape)
print(data.flatten().max(),data.flatten().min())

def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ    
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))

## Slope estimation
dtemp=dn*0;#dtemp is the preprocessed data
for i in range(1,dn.shape[0]+1):
    dtemp[i-1,:]=smooth(dn[i-1,:],5);

dip=ps.dip2dc(dtemp,rect=[10,20,1]);
print(dn.shape)
print(dip.flatten().max(),dip.flatten().min())

## Structural smoothing
r=2;
eps=0.01;
order=2;
d1=ps.somean2dc(dn,dip,r,order,eps);
d2=ps.somf2dc(dn,dip,r,order,eps,1);

## plot results
fig = plt.figure(figsize=(10, 8))
ax=plt.subplot(2,4,1)
plt.imshow(data,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Clean data');
ax=plt.subplot(2,4,2)
plt.imshow(dn,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Noisy data');
ax=plt.subplot(2,4,3)
plt.imshow(d1,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Filtered (SOMEAN)');
ax=plt.subplot(2,4,4)
plt.imshow(dn-d1,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Noise (SOMEAN)');
ax=plt.subplot(2,4,6)
plt.imshow(dip,cmap='jet',clim=(-2, 2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Slope');
ax=plt.subplot(2,4,7)
plt.imshow(d2,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Filtered (SOMF)');
ax=plt.subplot(2,4,8)
plt.imshow(dn-d2,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Noise (SOMF)');
plt.savefig('test_pyseistr_somf2d.png',format='png',dpi=300)
plt.show()


