def mutter(din,dt=1,dx=1,ot=0,ox=0,apex=(20,20),v=1,val=0):
	'''
	mutter: apply 2D muting
	
	INPUT
	din: input data
	ot/ox: obviously
	dt/dx: obviously
	apex(it,ix): apex 
	v:	  velocity for muting
	val: value to be replace, it can be 0 or 999999
	
	OUTPUT
	dout: output data
	
	
	EXAMPLE 1
	from pyseistr import mutter,gensyn,dither
	import matplotlib.pyplot as plt
	data=gensyn()
	plt.imshow(data);plt.show()
	n1,n2=data.shape
	data=dither(data,100)
	plt.imshow(data);plt.show()
	data2=mutter(data,apex=(int(n1/2),int(n2/2)))
	plt.imshow(data2);plt.show()

	EXAMPLE 2
	from pyseistr import mutter,gensyn,dither
	import matplotlib.pyplot as plt
	data=gensyn()
	plt.imshow(data);plt.show()
	n1,n2=data.shape
	data=dither(data,100)
	plt.imshow(data);plt.show()
	data2=mutter(data,apex=(int(n1/2),int(n2/2)), v=0.5, val=1.0)
	plt.imshow(data2);plt.show()

	EXAMPLE 3
	from pyseistr import mutter,gensyn,dither
	import matplotlib.pyplot as plt
	data=gensyn()
	plt.imshow(data);plt.show()
	n1,n2=data.shape
	data=dither(data,100)
	plt.imshow(data);plt.show()
	data2=mutter(data,apex=(0,0), v=0.5, val=1.0)
	plt.imshow(data2);plt.show()
	
	'''
	nt,nx=din.shape
	dout=din.copy()
	
	idt,idx=apex
	idt=int((idt-ot)/dt)
	idx=int((idx-ox)/dx)

	nw=1/v*dt/dx*(nt-idt); #half size of the bottom side of the cone, or reverse
# 	print('nw',nw)
	
	for it in range(1,nt+1):
		for ix in range(1,nx+1):
			if it> v*(ix-idx)*dx/dt+idt and it> v*(idx-ix)*dx/dt+idt:
				dout[it-1,ix-1]=val;
				
	return dout