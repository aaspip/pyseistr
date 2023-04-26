import obspy
import numpy as np
import matplotlib.pyplot as plt
from pyseistr import cseis as seis
import pyseistr as ps
import os

if os.path.isdir('./FORGE') == False:  
	os.makedirs('./FORGE',exist_ok=True)

if os.path.isdir('./FORGEFIG') == False:  
	os.makedirs('./FORGEFIG',exist_ok=True)
	
fname="FORGE/FORGE_78-32_iDASv3-P11_UTC190423150554.sgy"
fname="FORGE/FORGE_78-32_iDASv3-P11_UTC190423213209.sgy"

# d=obspy.read(fname,format='SEGY')
# 
# data=[]
# 
# for ii in range(len(d)):
# 	data.append(np.expand_dims(d[ii],1))
# 
# dn=np.concatenate(data,axis=1);
# 
# dt=d[0].stats.delta;
# nt=d[0].stats.npts;
# 
# # dn=dn[1484-20:3484-20,200:1160]
# dn=dn[:,200:1160]
# 
# ## BP
# print(dn.max())
# d1=ps.bandpassc(dn,0.0005,0,200,6,6,0,0);
# d1_bp=d1.copy();
# print(d1.max())
# 
# ## SOMF
# pp=ps.dip2dc(d1,2,10,2,0.01, 1, 0.000001,[50,50,1],1);
# print('finished')
# d1=ps.somf2dc(d1,pp,8,2,0.01,1);#SOMF
# d1_bpsomf=d1.copy()
# 
# ## FK
# d1=d1_bpsomf-ps.fkdip(d1,0.02);
# d1_bpsomffk=d1.copy()
# 
# ## plot
# clip=20;
# fig = plt.figure(figsize=(6, 8))
# ax=plt.subplot(3,2,1)
# plt.imshow(dn,cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
# plt.title('Raw DAS data');
# ax=plt.subplot(3,2,3)
# plt.imshow(d1_bp,cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
# plt.title('BP');
# ax=plt.subplot(3,2,4)
# plt.imshow(d1_bpsomf,cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
# plt.title('BPSOMF');
# ax=plt.subplot(3,2,5)
# plt.imshow(d1_bpsomffk,cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
# plt.title('BPSOMFFK');
# ax=plt.subplot(3,2,6)
# plt.imshow(dn-d1_bpsomffk,cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
# plt.title('Removed Noise');
# plt.savefig('test_pyseistr_das_massive.png',format='png',dpi=300)
# plt.show()
# 
# 
# 
# 
# 
# clip=20;
# fig = plt.figure(figsize=(6, 8))
# ax=plt.subplot(3,2,1)
# plt.imshow(dn[0:3000,:],cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
# plt.title('Raw DAS data');
# 
# 
# ax=plt.subplot(3,2,2)
# plt.imshow(dn[0:6000,:],cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
# plt.title('Raw DAS data');
# 
# 
# ax=plt.subplot(3,2,3)
# plt.imshow(dn[0:9000,:],cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
# plt.title('Raw DAS data');
# 
# 
# ax=plt.subplot(3,2,4)
# plt.imshow(dn[0:12000,:],cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
# plt.title('Raw DAS data');
# 
# 
# ax=plt.subplot(3,2,5)
# plt.imshow(dn[0:15000,:],cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
# plt.title('Raw DAS data');
# 
# 
# ax=plt.subplot(3,2,6)
# plt.imshow(dn[0:-1,:],cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
# plt.title('Raw DAS data');
# plt.savefig('test_pyseistr_das_massive2.png',format='png',dpi=300)
# plt.show()
# 
# 
# 
# ## coherency measure
# d0=dn[0:3100,:]
# d0=dn[0+15000:3100+15000,:]
# [nt,nx]=d0.shape;
# v=np.linspace(-0.00013,0.00013,100);
# # v=np.linspace(-0.013,0.013,100);
# dt=d[0].stats.delta;
# h=np.linspace(0,nx-1,nx)
# par={'v':v,'nt':nt,'h':h,'dt':dt,'typ':1,'oper':1}
# 
# 
# c0=ps.cohc(d0,par);
# cmax=c0.max()
# print('Cmax=',cmax)
# 
# ## BP
# d1=ps.bandpassc(d0,0.0005,0,200,6,6,0,0);
# d1_bp=d1.copy();
# 
# c1=ps.cohc(d1,par);
# cmax=c1.max()
# print('Cmax=',cmax)
# 
# ## SOMF
# pp=ps.dip2dc(d1,2,10,2,0.01, 1, 0.000001,[50,50,1],1);
# print('finished')
# d1=ps.somf2dc(d1,pp,8,2,0.01,1);#SOMF
# d1_bpsomf=d1.copy()
# c2=ps.cohc(d1,par);
# cmax=c2.max()
# print('Cmax=',cmax)
# 
# ## FK
# d1=d1_bpsomf-ps.fkdip(d1,0.02);
# d1_bpsomffk=d1.copy()
# c3=ps.cohc(d1,par);
# cmax=c3.max()
# print('Cmax=',cmax)
# 
# 
# c0=ps.cohc(dtest,par);
# cmax=c0.max()
# print('Cmax=',cmax)

clip=20;
nwin=3000
thr=0.5
v=np.linspace(-0.00023,0.00023,100);
nt=nwin;nx=960;dt=0.0005;
h=np.linspace(0,nx-1,nx)
par={'v':v,'nt':nt,'h':h,'dt':dt,'typ':1,'oper':1}

files=["FORGE_78-32_iDASv3-P11_UTC190423213209.sgy"]

import glob
lines=glob.glob('FORGE/*.sgy')
files=[ii.split('/')[-1] for ii in lines]
files=files[:]
for ii in range(len(files)):
	ifile=files[ii]
	print(ii,'/',len(files),ifile)
	
	d=obspy.read('FORGE/'+ifile,format='SEGY')
	data=[]
	for ii in range(len(d)):
		data.append(np.expand_dims(d[ii],1))
	dn=np.concatenate(data,axis=1);

	if dn.shape[1]>1200:
		dn=dn[:,200:1160]
	else:
		dn=dn[:,100:1000]
	
	[nt,nx]=dn.shape;
	
	## Cmax parameter
	h=np.linspace(0,nx-1,nx);
	par={'v':v,'nt':nwin,'h':h,'dt':dt,'typ':1,'oper':1};
	
	ic=0
	t0=-nwin;
	while t0<nt-nwin:
		ic=ic+1;
		t0=t0+nwin
		d0=dn[t0:t0+nwin,:]
		if nwin != d0.shape[0]:
			print('nwin=',nwin,'n1=',d0.shape[0])
		d1=d0;
# 		d1=ps.bandpassc(d0,0.0005,0,200,6,6,0,0);d1_bp=d1.copy(); 	##BP
# 		pp=ps.dip2dc(d1,2,10,2,0.01, 1, 0.000001,[40,40,1],verb=0); ##SOMF
# 		d1=ps.somf2dc(d1,pp,8,2,0.01,verb=0);d1_bpsomf=d1.copy();	#SOMF
# 		d1=d1-ps.fkdip(d1,0.02);d1_bpsomffk=d1.copy()				#FK
		c=ps.cohc(d1,par);
		cmax=c.max()
		print('Window',ic,' Cmax=',cmax)
		if cmax>thr:
			print('In %s EQ detected (window %d) tbeg=%d: Cmax=%f'%(ifile,ic,t0,cmax))
			
			fig = plt.figure(figsize=(6, 8))
			ax=plt.subplot(3,2,1)
			plt.imshow(d0,cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
# 			plt.title('Raw DAS data');
			plt.title(ifile+' '+str(t0),fontsize=20)
			plt.xlabel('Cmax=%g'%cmax)
# 			ax=plt.subplot(3,2,3)
# 			plt.imshow(d1_bp,cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
# 			plt.title('BP');
# 			ax=plt.subplot(3,2,4)
# 			plt.imshow(d1_bpsomf,cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
# 			plt.title('BPSOMF');
# 			ax=plt.subplot(3,2,5)
# 			plt.imshow(d1_bpsomffk,cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
# 			plt.title('BPSOMFFK (Cmax=%g)'%cmax);
# 			ax=plt.subplot(3,2,6)
# 			plt.imshow(d0-d1_bpsomffk,cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
# 			plt.title('Removed Noise');
			plt.savefig('FORGEFIG/%s_%d_%d.png'%(ifile,ic,t0),format='png',dpi=200)
			plt.close;
		del d0
		del d1
# 		del d1_bp
# 		del d1_bpsomf
# 		del d1_bpsomffk
		del c
	del dn
	del d
	del data
