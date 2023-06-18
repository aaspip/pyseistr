
## This DEMO is a 2D example [x,z] with velocity gradient and with one shot
# 
#  COPYRIGHT: Yangkang Chen, 2022, The University of Texas at Austin

import pyekfmm as fmm
import numpy as np
import matplotlib.pyplot as plt

import pyekfmm as fmm
import numpy as np


v1=1;
v2=3;
nz=101;
nx=101;
ny=101;
dx=0.01;
dz=0.01;
dy=0.01;
# vel=3.0*np.ones([101*101,1],dtype='float32'); #velocity axis must be x,y,z respectively
v=np.linspace(v1,v2,nz);
v=np.expand_dims(v,1);
h=np.ones([1,nx])
vel=np.multiply(v,h,dtype='float32'); #z,x

vel3d=np.zeros([nz,nx,ny],dtype='float32');
for ii in range(ny):
	vel3d[:,:,ii]=vel
# plt.figure();
# plt.imshow(vel3d[:,:,0]);
# plt.jet();plt.show()

vxyz=np.swapaxes(np.swapaxes(vel3d,0,1),1,2);
t=fmm.eikonal(vxyz.flatten(order='F'),xyz=np.array([0,0,0]),ax=[0,dx,nx],ay=[0,dy,ny],az=[0,dz,nz],order=2);
time=t.reshape(nx,ny,nz,order='F');#first axis (vertical) is x, second is z
# time=np.swapaxes(np.swapaxes(time,1,2),0,1);

# plt.figure();
# plt.imshow(time[:,:,0]);
# plt.jet();plt.show()

# tz=np.gradient(time,axis=1);
# tx=np.gradient(time,axis=0);
# # or
# tz,tx,ty = np.gradient(time)

tx,ty,tz = np.gradient(time)

receiverx=101.0
receivery=101.0
receiverz=101.0
paths,nrays=fmm.stream3d(-tx,-ty, -tz, receiverx, receivery, receiverz, step=0.1, maxvert=10000)
print('Before trim',paths.shape)
## trim the rays and add the source point
paths=fmm.trimrays(paths,start_points=np.array([1,1,1]),T=0.5)
print('After trim',paths.shape)


# 
# 
# vel=3.0*np.ones([101*101*101,1],dtype='float32');
# t=fmm.eikonal(vel,xyz=np.array([0.5,0,0]),ax=[0,0.01,101],ay=[0,0.01,101],az=[0,0.01,101],order=2);
# time=t.reshape(101,101,101,order='F'); #[x,y,z]
# 
# 
# ## Verify
# print(['Testing result:',time.max(),time.min(),time.std(),time.var()])
# print(['Correct result:',0.49965078, 0.0, 0.08905013, 0.007929926])
# 

import matplotlib.pyplot as plt
import numpy as np

# Define dimensions
Nx, Ny, Nz = 101, 101, 101
X, Y, Z = np.meshgrid(np.arange(Nx)*0.01, np.arange(Ny)*0.01, np.arange(Nz)*0.01)

# Specify the 3D data
data=np.transpose(time,(1,0,2)); ## data requires [y,x,z] so tranpose the first and second axis
# data=np.transpose(time,(2,1,0)); #[z,x,y] -> [y,x,z]

kw = {
    'vmin': data.min(),
    'vmax': data.max(),
    'levels': np.linspace(data.min(), data.max(), 10),
}

# Create a figure with 3D ax
fig = plt.figure(figsize=(8, 8))
# 
ax = fig.add_subplot(111, projection='3d')
# ax=plt.gca()
[n1,n2]=Z[:, -1, :].shape;

plt.jet()
# Plot contour surfaces
_ = ax.contourf(
    X[:, :, -1], Y[:, :, -1], np.random.rand(n1,n2),
    zdir='z', offset=0, alpha=0.7, **kw
)

_ = ax.contourf(
    X[0, :, :], np.random.rand(n1,n2), Z[0, :, :],
    zdir='y', offset=0, alpha=0.7, **kw
)

C = ax.contourf(
    np.random.rand(n1,n2), Y[:, -1, :], Z[:, -1, :],
    zdir='x', offset=X.max(), alpha=0.7, **kw
)


print(Z[0,:,:].shape)

plt.gca().set_xlabel("X",fontsize='large', fontweight='normal')
plt.gca().set_ylabel("Y",fontsize='large', fontweight='normal')
plt.gca().set_zlabel("Z",fontsize='large', fontweight='normal')
# --


# Set limits of the plot from coord limits
# xmin, xmax = X.min(), X.max()
# ymin, ymax = Y.min(), Y.max()
# zmin, zmax = Z.min(), Z.max()
# ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
# 
# # Plot edges
# # edges_kw = dict(color='0.4', linewidth=1, zorder=1e3)
# # ax.plot([xmax, xmax], [ymin, ymax], 0, **edges_kw)
# # ax.plot([xmin, xmax], [ymin, ymin], 0, **edges_kw)
# # ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], **edges_kw)
# 
# # Set labels and zticks
# ax.set(
#     xlabel='X (km)',
#     ylabel='Y (km)',
#     zlabel='Z (km)',
# #     zticks=[0, -150, -300, -450],
# )
# 
# # Set zoom and angle view
# # ax.view_init(40, -30, 0)
# # ax.set_box_aspect(None, zoom=0.9)
# 
# # Colorbar
# cbar=fig.colorbar(C, ax=ax, orientation='horizontal', fraction=0.02, pad=0.1, format= "%.2f", label='Traveltime (s)')
# cbar.ax.locator_params(nbins=5)
# 
# plt.gca().scatter(0.0,0,0,s=200,marker='*',color='r')
# plt.gca().set_xlim(0,1);
# plt.gca().set_ylim(0,1);
# plt.gca().set_zlim(0,1);
# 
# # plt.savefig('test_1_vgrad_ray3d.png',format='png',dpi=300,bbox_inches='tight', pad_inches=0)
# 
# plt.plot((receivery-1)*dy,(receiverx-1)*dx,(receiverz-1)*dz,'vb',markersize=15);
# ## plot rays
# plt.plot((paths[1,:]-1)*dy,(paths[0,:]-1)*dx,(paths[2,:]-1)*dz,'g--',markersize=20);
# 
# 
# for ii in range(1,102,10):
# 	paths,nrays=fmm.stream3d(-tx,-ty, -tz, 101, 101, ii, step=0.1, maxvert=10000)
# 	plt.plot((101-1)*dy,(101-1)*dx,(ii-1)*dz,'vb',markersize=10);
# 	## plot rays
# 	plt.plot((paths[1,:]-1)*dy,(paths[0,:]-1)*dx,(paths[2,:]-1)*dz,'g--',markersize=20);
# 	
# 	
plt.gca().invert_zaxis()
# plt.gca().text(-0.124, 0, -0.66, "b)", fontsize=28, color='k')
# 
# plt.savefig('test_pyekfmm_fig4.png',format='png',dpi=300,bbox_inches='tight', pad_inches=0)
# plt.savefig('test_pyekfmm_fig4.pdf',format='pdf',dpi=300,bbox_inches='tight', pad_inches=0)


# Show Figure
plt.show()

