## This is a DEMO script for receiver function data enhancement

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 08:01:13 2022

@author: Yunfeng modificated by Quan Zhang and Yangkang Chen
"""

##DATAPATH
#https://github.com/aaspip/data/blob/main/rf2d_fromquan.txt

import os
#from obspy import read
#from glob import glob
import numpy as np
import matplotlib.pyplot as plt


#from obspy.taup import TauPyModel
#import somf2d.py

#import seistr
import pyseistr as ps
def smooth(a,WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a,np.ones(WSZ,dtype=int),'valid')/WSZ
    r = np.arange(1,WSZ-1,2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((  start , out0, stop  ))


#%% structural orientated filter
## Slope estimation
# Please download the data from https://github.com/aaspip/data/blob/main/rf2d_fromquan.txt
d2d = np.loadtxt('rf2d_fromquan.txt')
    
dtemp=d2d*0;#dtemp is the preprocessed data
for i in range(d2d.shape[0]):
    dtemp[i,:]=smooth(d2d[i,:],5);

dip=ps.dip2dc(dtemp);
print(d2d.shape)
print(dip.flatten().max(),dip.flatten().min())

## Structural smoothing
r=2
eps=0.01
order=2
d1=ps.somean2dc(d2d,dip,r,order,eps)
d2=ps.somf2dc(d2d,dip,r,order,eps,1)

dmax=np.max(d2d[:,30])

plt.figure(figsize=(10,16))
plt.subplot(4,2,1)
plt.imshow(d2d/dmax,cmap='jet',clim=(-0.2, 0.2),aspect=0.06)
#plt.xlabel('Trace#')
#plt.ylabel('Time (s)')
plt.title('Raw data')
plt.axis('off')

plt.subplot(4,2,2)
plt.imshow(dip[:,:]/np.max(dip[:,:]),cmap='jet',clim=(-0.2, 0.2),aspect=0.06)
#plt.xlabel('Trace#')
#plt.ylabel('Time (s)')
plt.title('Slope')
plt.axis('off')

plt.subplot(4,2,3)
plt.imshow(dtemp/dmax,cmap='jet',clim=(-0.2, 0.2),aspect=0.06)
#plt.xlabel('Trace#')
#plt.ylabel('Time (s)')
plt.title('Filtered (MEAN)')
plt.axis('off')

plt.subplot(4,2,4)
plt.imshow((dtemp-d2d)/dmax,cmap='jet',clim=(-0.2, 0.2),aspect=0.06)
#plt.xlabel('Trace#')
#plt.ylabel('Time (s)')
plt.title('Noise (MEAN)')
plt.axis('off')

plt.subplot(4,2,5)
plt.imshow(d1/dmax,cmap='jet',clim=(-0.2, 0.2),aspect=0.06)
#plt.xlabel('Trace#')
#plt.ylabel('Time (s)')
plt.title('Filtered (SOMEAN)')
plt.axis('off')

plt.subplot(4,2,6)
plt.imshow((d1-d2d)/dmax,cmap='jet',clim=(-0.2, 0.2),aspect=0.06)
#plt.xlabel('Trace#')
#plt.ylabel('Time (s)')
plt.title('Noise (SOMEAN)')
plt.axis('off')

plt.subplot(4,2,7)
plt.imshow(d2/dmax,cmap='jet',clim=(-0.2, 0.2),aspect=0.06)
#plt.xlabel('Trace#')
#plt.ylabel('Time (s)')
plt.title('Filtered (SOMF)')
plt.axis('off')

plt.subplot(4,2,8)
plt.imshow((d2-d2d)/dmax,cmap='jet',clim=(-0.2, 0.2),aspect=0.06)
#plt.xlabel('Trace#')
#plt.ylabel('Time (s)')
plt.title('Noise (SOMF)')
plt.axis('off')

plt.savefig('test_pyseistr_rf.png',bbox_inches='tight',dpi=300)
plt.show()


