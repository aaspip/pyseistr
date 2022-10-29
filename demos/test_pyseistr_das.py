## This is a DEMO script for structure-oriented distributed acoustic sensing (DAS) data processing

import os
import numpy as np

## You can either download the data using the following commands or using the binary file from https://github.com/aaspip/data
# https://github.com/aaspip/data/blob/main/forge0723.bin

#pip install segysak
# from segysak.segy import segy_loader, well_known_byte_locs
## test data 1
#P arrival: 1484
#window f1=1484 | window f2=200 n2=960 n1=2000
# os.system("wget -q https://pando-rgw01.chpc.utah.edu/silixa_das_apr_23_2019/FORGE_78-32_iDASv3-P11_UTC190423213209.sgy")
# segydata = segy_loader("FORGE_78-32_iDASv3-P11_UTC190423213209.sgy")
# data=np.zeros([1280,30000])
# data[:,:]=segydata.data
# dn=data[200:1160,1484-20:3484-20].transpose();

## test data 2
#P arrival: 24811
#window f1=24811 | window f2=200 n2=960 n1=2000
# os.system("wget -q https://pando-rgw01.chpc.utah.edu/silixa_das_apr_26_2019/FORGE_78-32_iDASv3-P11_UTC190426070723.sgy")
# segydata = segy_loader("FORGE_78-32_iDASv3-P11_UTC190426070723.sgy")
# data=np.zeros([1280,30000])
# data[:,:]=segydata.data
# pindex=24811
# dn=data[200:1160,pindex-20:pindex+500-20].transpose();

fid=open("forge0723.bin","rb");
dn = np.fromfile(fid, dtype = np.float32, count = 500*960).reshape([500,960],order='F')

import matplotlib.pyplot as plt
import pyseistr as ps
from pyseistr import cseis as seis

## BP
print(dn.max())
d1=ps.bandpassc(dn,0.0005,0,200,6,6,0,0);
d1_bp=d1.copy();
print(d1.max())

## SOMF
pp=ps.dip2dc(d1,2,10,2,0.01, 1, 0.000001,[50,50,1],1);
print('finished')
d1=ps.somf2dc(d1,pp,8,2,0.01,1);#SOMF
d1_bpsomf=d1.copy()

## FK
d1=d1_bpsomf-ps.fkdip(d1,0.02);
d1_bpsomffk=d1.copy()

## compare with matlab
# import scipy
# from scipy import io
# datas = {"dn":dn,"d1": d1_bp, "d2": d1_bpsomf, "d3": d1_bpsomffk}
# scipy.io.savemat("das2d.mat", datas)

## plot
clip=20;
fig = plt.figure(figsize=(6, 8))
ax=plt.subplot(3,2,1)
plt.imshow(dn,cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
plt.title('Raw DAS data');
ax=plt.subplot(3,2,3)
plt.imshow(d1_bp,cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
plt.title('BP');
ax=plt.subplot(3,2,4)
plt.imshow(d1_bpsomf,cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
plt.title('BPSOMF');
ax=plt.subplot(3,2,5)
plt.imshow(d1_bpsomffk,cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
plt.title('BPSOMFFK');
ax=plt.subplot(3,2,6)
plt.imshow(dn-d1_bpsomffk,cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
plt.title('Removed Noise');
plt.savefig('test_pyseistr_das.png',format='png',dpi=300)
plt.show()




