## This is a DEMO script for 2D structure-oriented interpolation 

import numpy as np
import matplotlib.pyplot as plt
import pyseistr as ps

## Generate synthetic data
[w,tw]=ps.ricker(30,0.001,0.1);
t=np.zeros([300,1000]);sigma=300;A=100;B=200;
data=np.zeros([400,1000]);

[m,n]=t.shape;
for i in range(1,n+1):
	k=np.floor(-A*np.exp(-np.power(i-n/2,2)/np.power(sigma,2))+B);k=int(k);
	if k>1 and k<=m:
		t[k-1,i-1]=1;
for i in range(1,n+1):
	data[:,i-1]=np.convolve(t[:,i-1],w);

data=data[:,0::10];#or data[:,0:-1:10];
data=data/np.max(np.max(data));
np.random.seed(202122);
scnoi=(np.random.rand(data.shape[0],data.shape[1])*2-1)*0.15;
dn=data+scnoi;

## Creating gaps
[n1,n2]=data.shape
raw=dn.flatten(order='F');
t=np.linspace(1,n1*n2,n1*n2);
x1=50;x2=60;inds=[ii for ii in range(0,n1*n2) if t[ii]>(x1-1)*n2 and t[ii]<=x2*n2]
raw[inds]=0;
x1=100;x2=120;inds=[ii for ii in range(0,n1*n2) if t[ii]>(x1-1)*n2 and t[ii]<=x2*n2]
raw[inds]=0;
x1=250;x2=280;inds=[ii for ii in range(0,n1*n2) if t[ii]>(x1-1)*n2 and t[ii]<=x2*n2]
raw[inds]=0;
raw=raw.reshape(n1,n2,order='F');

## Calculate local slope
dip=ps.dip2dc(raw,mask=raw,order=2,rect=[10,20,1],verb=0);

## SOINT2D
mask=0*dip;
recon=ps.soint2dc(raw,mask,dip,order=2,niter=100,njs=[1,1],drift=0,hasmask=0,twoplane=0,prec=0,verb=0);

## Structural smoothing
recon2=ps.somean2dc(recon,dip,2,2,0.01);

## plot results
fig = plt.figure(figsize=(8, 8))
ax=plt.subplot(2,3,1)
plt.imshow(data,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Clean data');
ax=plt.subplot(2,3,2)
plt.imshow(dn,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Noisy data');
ax=plt.subplot(2,3,3)
plt.imshow(raw,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Incomplete data');
ax=plt.subplot(2,3,4)
plt.imshow(dip,cmap='jet',clim=(-2, 2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Slope');
ax=plt.subplot(2,3,5)
plt.imshow(recon,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Reconstructed');
ax=plt.subplot(2,3,6)
plt.imshow(recon2,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Smoothed');

plt.savefig('test_pyseistr_soint2d.png',format='png',dpi=300)
plt.show()


