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
from pyseistr import plot3d
from pyseistr import sint3dc

import numpy as np

s3d_zero2=binread('/Users/chenyk/data/datapath/mada_codes/sfsint3/test/s3d-zero2.rsf@',n1=180,n2=61,n3=51)
# plot3d(s3d_zero2);

mask2=binread('/Users/chenyk/data/datapath/mada_codes/sfsint3/test/mask2.rsf@',n1=180,n2=61,n3=51)
# plot3d(mask2,vmin=0,vmax=1,cmap='jet');

s3d_slope=binread('/Users/chenyk/data/datapath/mada_codes/sfsint3/test/s3d-slope.rsf@',n1=180,n2=61,n3=51*2)
s3d_slope=s3d_slope.reshape(180,61,51,2,order='F')
# plot3d(s3d_slope[:,:,:,0],cmap='jet');

s3d_pws=binread('/Users/chenyk/data/datapath/mada_codes/sfsint3/test/s3d-pws2.rsf@',n1=180,n2=61,n3=51)
plot3d(s3d_pws,frames=[80,14,11],vmin=1500,vmax=3500,levels=np.linspace(1500,3500,100),cmap='jet',figname='sint3d');


dipi=s3d_slope[:,:,:,0]
dipx=s3d_slope[:,:,:,1]
dout=sint3dc(s3d_zero2,mask2,dipi,dipx,niter=30,eps=0.01,ns1=7,ns2=13,verb=1);
plot3d(dout,frames=[80,14,11],vmin=1500,vmax=3500,levels=np.linspace(1500,3500,100),cmap='jet',figname='sint3dc');

print(s3d_pws.max(),s3d_pws.min())
print(dout.max(),dout.min())








