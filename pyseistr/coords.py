import numpy as np
def line2cart(dim, nn, i):
	'''
	line2cart: Convert line (1D) to Cartesian (2D)
	
	INPUT
	dim:	number of dimensions
	nn:		box size [INT numpy array of size dim]
	i:		line coordinates (INT) in samples
	
	OUTPUT
	ii:		cartesian coordinates in samples [INT numpy array of size dim]
	
	EXAMPLE 1
	from pyseistr import line2cart
	coords = line2cart(2,[20,5], 25)
	print("The output coordinates of [1D coord 25] in 20*5 grid is ", coords)

	EXAMPLE 2
	from pyseistr import line2cart
	coords = line2cart(2,[20,5], 0)
	print("The output coordinates of [1D coord 0] in 20*5 grid is ", coords)
	'''
	
	ii=np.zeros(dim, dtype=np.int_)
	for axis in range(dim):
		ii[axis] = np.mod(i,nn[axis])
		i=int(i/nn[axis])

	return ii
	

def cart2line(dim, nn, ii):
	'''
	line2cart: Convert line (1D) to Cartesian (2D)
	
	INPUT
	dim:	number of dimensions
	nn:		box size [INT numpy array of size dim]
	ii:		cartesian coordinates in samples [INT numpy array of size dim]
	
	OUTPUT
	i:		line coordinates in samples (INT)
	
	EXAMPLE 1
	from pyseistr import cart2line
	coord = cart2line(2,[20,5], [5,1])
	print("The output coordinate [in 20*5 grid] of 2D coord [5,1] is ", coord)

	EXAMPLE 2
	from pyseistr import cart2line
	coord = cart2line(2,[20,5], [0,0])
	print("The output coordinate [in 20*5 grid] of 2D coord [0,0] is ", coord)
	'''
	if dim<1:
		return 0
		
	i=ii[dim-1];
	for axis in range(dim-2,-1,-1):
		i=i*nn[axis]+ii[axis];


	return int(i)