#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## This is a DEMO script for SS precursor data enhancement using 3D structure-oriented median filter

## Download data from https://github.com/aaspip/data
# https://github.com/aaspip/data/blob/main/ssprecursor3D.mat

import numpy as np
import matplotlib.pyplot as plt
import pyseistr as ps

from scipy.io import loadmat
data= loadmat('ssprecursor3D.mat')
d0=data["d0"]
# print(d0)
# print(type(d0))

## 3D slope calculation (inline and xline)
import pyseistr as ps
import matplotlib.pyplot as plt
[dipi,dipx] = ps.dip3dc(d0);
#
## 3D slope calculation (inline and xline)
## Structural smoothing
r1=2;
r2=2;
eps=0.01;
order=2;
cmpn_d1=ps.somean3dc(d0,dipi,dipx,r1,r2,eps,order);
#cmpn_d2=ps.somf3d(d0,dipi,dipx,r1,r2,eps,order);

## plot results (with x & y labels)
# ig = plt.figure(figsize=(5, 8))
# ax=plt.subplot(5,1,1)
# plt.imshow(d0.reshape(201,20*16,order='F'),cmap='jet',clim=(-0.02, 0.02),aspect=0.25)
# #or ax.set_xticks([]);ax.set_yticks([]);
# ax.xaxis.set_tick_params(labelsize=6);ax.yaxis.set_tick_params(labelsize=6);
# ax.imshow(d0.reshape(201,20*16,order='F'),cmap='jet',clim=(-0.02, 0.02),aspect=0.25, extent=[1,310,-300,-100])
# plt.title('Raw ss-precursor data',fontsize=8);
# ax.set_xlabel('Trace', fontsize=8)
# ax.set_ylabel('Time to SS (sec)', fontsize=7)
# 
# ax=plt.subplot(5,1,2)
# plt.imshow(dipi.reshape(201,20*16,order='F'),cmap='jet',clim=(-2, 2),aspect=0.25)
# #or ax.set_xticks([]);ax.set_yticks([]);
# ax.xaxis.set_tick_params(labelsize=6);ax.yaxis.set_tick_params(labelsize=6);
# ax.imshow(dipi.reshape(201,20*16,order='F'),cmap='jet',clim=(-2, 2),aspect=0.25, extent=[1,310,-300,-100])
# plt.title('Iline slope',fontsize=8);
# ax.set_xlabel('Trace', fontsize=8)
# ax.set_ylabel('Time to SS (sec)', fontsize=7)
# 
# ax=plt.subplot(5,1,3)
# plt.imshow(dipx.reshape(201,20*16,order='F'),cmap='jet',clim=(-2, 2),aspect=0.25)
# #or ax.set_xticks([]);ax.set_yticks([]);
# ax.xaxis.set_tick_params(labelsize=6);ax.yaxis.set_tick_params(labelsize=6);
# ax.imshow(dipx.reshape(201,20*16,order='F'),cmap='jet',clim=(-2, 2),aspect=0.25, extent=[1,310,-300,-100])
# plt.title('Xline slope',fontsize=8);
# ax.set_xlabel('Trace', fontsize=8)
# ax.set_ylabel('Time to SS (sec)', fontsize=7)
# 
# ax=plt.subplot(5,1,4)
# plt.imshow(cmpn_d1.reshape(201,20*16,order='F'),cmap='jet',clim=(-0.02, 0.02),aspect=0.25)
# #or ax.set_xticks([]);ax.set_yticks([]);
# ax.xaxis.set_tick_params(labelsize=6);ax.yaxis.set_tick_params(labelsize=6);
# ax.imshow(cmpn_d1.reshape(201,20*16,order='F'),cmap='jet',clim=(-0.02, 0.02),aspect=0.25, extent=[1,310,-300,-100])
# plt.title('Denoised SS-precursor data',fontsize=8);
# ax.set_xlabel('Trace', fontsize=8)
# ax.set_ylabel('Time to SS (sec)', fontsize=7)
# 
# ax=plt.subplot(5,1,5)
# plt.imshow((d0-cmpn_d1).reshape(201,320,order='F'),cmap='jet',clim=(-0.02, 0.02),aspect=0.25)
# #or ax.set_xticks([]);ax.set_yticks([]);
# ax.xaxis.set_tick_params(labelsize=6);ax.yaxis.set_tick_params(labelsize=6);
# ax.imshow((d0-cmpn_d1).reshape(201,20*16,order='F'),cmap='jet',clim=(-0.02, 0.02),aspect=0.25, extent=[1,310,-300,-100])
# plt.title('Removed noise',fontsize=8);
# ax.set_xlabel('Trace', fontsize=8)
# ax.set_ylabel('Time to SS (sec)', fontsize=7)
# plt.savefig('test_pyseistr_ssprecursor_somean3d.png',format='png',dpi=300)
# plt.show()


## plot results (without x & y labels)
fig = plt.figure(figsize=(6, 8))
ax=plt.subplot(5,1,1)
plt.imshow(d0.reshape(201,20*16,order='F'),cmap='jet',clim=(-0.02, 0.02),aspect=0.3)
plt.title('Raw SS-precursor data',color='k');ax.set_xticks([]);ax.set_yticks([]);
ax=plt.subplot(5,1,2)
plt.imshow(dipi.reshape(201,20*16,order='F'),cmap='jet',clim=(-2,2),aspect=0.3)
plt.title('Iline slope',color='k');ax.set_xticks([]);ax.set_yticks([]);
ax=plt.subplot(5,1,3)
plt.imshow(dipx.reshape(201,20*16,order='F'),cmap='jet',clim=(-2,2),aspect=0.3)
plt.title('Xline slope',color='k');ax.set_xticks([]);ax.set_yticks([]);
ax=plt.subplot(5,1,4)
plt.imshow(cmpn_d1.reshape(201,20*16,order='F'),cmap='jet',clim=(-0.02, 0.02),aspect=0.3)
plt.title('Denoised SS-precursor data',color='k');ax.set_xticks([]);ax.set_yticks([]);
ax=plt.subplot(5,1,5)
plt.imshow((d0-cmpn_d1).reshape(201,20*16,order='F'),cmap='jet',clim=(-0.02, 0.02),aspect=0.3)
plt.title('Removed noise',color='k');ax.set_xticks([]);ax.set_yticks([]);
plt.savefig('test_pyseistr_ssprecursor.png',format='png',dpi=300)
plt.show()
