## DEMO for sparse data interpolation
# 
# 
# References
# [1] Chen, Y., Chen, X., Wang, Y. and Zu, S., 2019. The interpolation of sparse geophysical data. Surveys in Geophysics, 40(1), pp.73-105.
# [2] Chen et al., 2023, Pyseistr: a python package for structural denoising and interpolation of multi-channel seismic data, Seismological Research Letters, 94(3), 1703-1714.
# [3] Wang, H., Chen, Y., Saad, O.M., Chen, W., Oboué, Y.A.S.I., Yang, L., Fomel, S. and Chen, Y., 2022. A Matlab code package for 2D/3D local slope estimation and structural filtering. Geophysics, 87(3), pp.F1–F14.
# [4] Huang, G., Chen, X., Li, J., Saad, O.M., Fomel, S., Luo, C., Wang, H. and Chen, Y., 2021. The slope-attribute-regularized high-resolution prestack seismic inversion. Surveys in Geophysics, 42(3), pp.625-671.
# [5] Huang, G., Chen, X., Luo, C. and Chen, Y., 2020. Geological structure-guided initial model building for prestack AVO/AVA inversion. IEEE Transactions on Geoscience and Remote Sensing, 59(2), pp.1784-1793.


from pylib.io import binread
import pyseistr as ps
import matplotlib.pyplot as plt
import numpy as np

## Generate synthetic data
from pyseistr import gensyn
dc,dn=gensyn(noise=True);
dc=dc[:,::10];dn=dn[:,::10];

dc=dc[150:250,0:50];
dn=dn[150:250,0:50];

## Creating gaps
[n1,n2]=dc.shape
raw=dn.flatten(order='F');
t=np.linspace(1,n1*n2,n1*n2);
x1=30;x2=60;inds=[ii for ii in range(0,n1*n2) if t[ii]>(x1-1)*n2 and t[ii]<=x2*n2]
raw[inds]=0;
data=raw.reshape(n1,n2,order='F');

## Create mask
datam=np.ones(data.shape);
datam[np.where(data==0)]=0;

## Calculate local slope
# dip=ps.dip2dc(ps.bandpassc(data,0.004,flo=0,fhi=10),mask=datam,order=2,rect=[10,10,1],verb=0);
dip=ps.dip2dc(ps.bandpassc(data,0.004,flo=0,fhi=10),mask=datam,order=2,rect=[10,10,1],verb=0);

# recon=ps.sint2d(data,datam,dip,niter=8,ns=5,order=1,eps=0.01);
recon=ps.sint2dc(data,datam,dip,niter=8,ns=5,order=1,eps=0.01);

## plot results
fig = plt.figure(figsize=(8, 6))
ax=plt.subplot(2,3,1)
plt.imshow(dc,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Clean data');
ax=plt.subplot(2,3,2)
plt.imshow(datam,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Mask');
ax=plt.subplot(2,3,3)
plt.imshow(data,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Incomplete data');
ax=plt.subplot(2,3,4)
plt.imshow(dip,cmap='jet',clim=(-2, 2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Slope');
ax=plt.subplot(2,3,5)
plt.imshow(recon,cmap='jet',clim=(-0.2, 0.2),aspect=0.5);ax.set_xticks([]);ax.set_yticks([]);
plt.title('Reconstructed');

plt.savefig('test_pyseistr_sint2d.png',format='png',dpi=300)
plt.show()





