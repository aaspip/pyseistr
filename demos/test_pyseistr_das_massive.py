import obspy
import numpy as np
import matplotlib.pyplot as plt
from pyseistr import cseis as seis
import pyseistr as ps

fname="FORGE/FORGE_78-32_iDASv3-P11_UTC190423150554.sgy"
fname="FORGE/FORGE_78-32_iDASv3-P11_UTC190423213209.sgy"
    
d=obspy.read(fname,format='SEGY')

data=[]

for ii in range(len(d)):
	data.append(np.expand_dims(d[ii],1))

dn=np.concatenate(data,axis=1);

dt=d[0].stats.delta;
nt=d[0].stats.npts;

# dn=dn[1484-20:3484-20,200:1160]
dn=dn[:,200:1160]

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
plt.savefig('test_pyseistr_das_massive.png',format='png',dpi=300)
plt.show()





clip=20;
fig = plt.figure(figsize=(6, 8))
ax=plt.subplot(3,2,1)
plt.imshow(dn[0:3000,:],cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
plt.title('Raw DAS data');


ax=plt.subplot(3,2,2)
plt.imshow(dn[0:6000,:],cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
plt.title('Raw DAS data');


ax=plt.subplot(3,2,3)
plt.imshow(dn[0:9000,:],cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
plt.title('Raw DAS data');


ax=plt.subplot(3,2,4)
plt.imshow(dn[0:12000,:],cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
plt.title('Raw DAS data');


ax=plt.subplot(3,2,5)
plt.imshow(dn[0:15000,:],cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
plt.title('Raw DAS data');


ax=plt.subplot(3,2,6)
plt.imshow(dn[0:-1,:],cmap=seis(),clim=(-clip,clip),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);
plt.title('Raw DAS data');
plt.savefig('test_pyseistr_das_massive2.png',format='png',dpi=300)
plt.show()




