import numpy as np
def helix(nh=1):
	'''
	helix: create a Helix filter (a dictionary with a few keywords)
	
	INPUT
	nh:	size of the filter
	
	OUTPUT
	filter
	
	EXAMPLE
	from pyseistr import helix
	a=helix()
	print(a.keys())
	
	'''
	
	if nh>0:
		filt={'flt':np.zeros(nh,dtype=np.float_),'lag':np.zeros(nh,dtype=np.int_),'nh':nh, 'mis': None, 'h0': 0}
	else:
		filt={'flt':None,'lag':None,'nh':nh, 'mis': None, 'h0': 0}
	return filt
	

def nhelix(nh=1):
	'''
	nhelix: create a non-stationary Helix filter (a dictionary with a few keywords)
	
	INPUT
	nh:	size of the filter
	
	OUTPUT
	filter
	
	EXAMPLE
	from pyseistr import nhelix
	a=nhelix()
	print(a.keys())
	
	'''
	

	filt={'hlx':None, 'mis':None, 'pch':None, 'np': 1}
	
	#hlx is a list of helix filter 
		
	return filt