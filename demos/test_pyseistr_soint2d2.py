#This is another SOINT example by spraying the trace along the seismic slope#
from pyseistr import smooth
from pyseistr import dip2dc
import matplotlib.pyplot as plt;
from pyseistr import sigmoid
sig=sigmoid(n1=200,n2=210);
sig=smooth(sig,rect=[3,1,1],diff=[1,0,0],adj=0);
sig=smooth(sig,rect=[3,1,1])

sdip=dip2dc(sig,order=2,niter=10,rect=[4,4,1],verb=0)
import numpy as np
from pyseistr import pwpaintc
data=np.zeros([sdip.shape[0],sdip.shape[1]])
data[:,50]=sig[:,50];
flat=pwpaintc(sdip,sig[:,50],order=2,i0=50,eps=0.1);
# plt.imshow(flat);plt.show();

fig = plt.figure(figsize=(12, 6))
ax=plt.subplot(1,3,1)
plt.imshow(data,cmap='gray',clim=(-0.01, 0.01),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);plt.title('Sparse Seismic');
plt.colorbar(orientation='horizontal');
ax=plt.subplot(1,3,2)
plt.imshow(sdip,cmap='jet',clim=(-2, 2),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);plt.title('Slope');
plt.colorbar(orientation='horizontal');
ax=plt.subplot(1,3,3)
plt.imshow(flat,cmap='gray',clim=(-0.02, 0.02),aspect='auto');ax.set_xticks([]);ax.set_yticks([]);plt.title('Reconstructed');
plt.colorbar(orientation='horizontal');

plt.savefig('test_pyseistr_soint2d2.png',format='png',dpi=300)
plt.show()

	
	
	





