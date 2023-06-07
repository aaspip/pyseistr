#This is example showing how to calculate RGT using pyseistr
from pyseistr import smooth
from pyseistr import smoothc
from pyseistr import dip2dc
from pyseistr import rgt
import matplotlib.pyplot as plt;

from pyseistr import sigmoid
sig=sigmoid(n1=200,n2=210);
sig=smoothc(sig,rect=[3,1,1],diff=[1,0,0],adj=0);
sig=smoothc(sig,rect=[3,1,1])

sdip=dip2dc(sig,order=2,niter=10,rect=[4,4,1],verb=0)
time=rgt(sdip,o1=0,d1=0.004,order=2,i0=50,eps=0.1);

fig = plt.figure(figsize=(12, 6))
ax=plt.subplot(1,3,1)
plt.imshow(sig,cmap='gray',clim=(-0.02, 0.02),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);plt.title('Seismic');
plt.colorbar(orientation='horizontal');
ax=plt.subplot(1,3,2)
plt.imshow(sdip,cmap='jet',clim=(-2, 2),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);plt.title('Slope');
plt.colorbar(orientation='horizontal');
ax=plt.subplot(1,3,3)
plt.imshow(time,cmap='jet',clim=(0, 0.7),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);plt.title('RGT');
plt.colorbar(orientation='horizontal');

plt.savefig('test_pyseistr_rgt2d.png',format='png',dpi=300)
plt.show()






