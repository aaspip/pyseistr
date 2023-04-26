import numpy as np
from cohcfun import *

def coh(inpt,par,operator=-1):
    '''
    Coherency measure based on the 
    Forward and Adjoint operators for Radon RT in the time domain.
    IN   	in:   		intput data 
      	Param:  		parameters combination
    par.h:		offset
    par.v:		velocity
    par.nt:		number of samples
    par.dt:		time interval
    par.typ:    1: linear 2: parablic 3: hyperbolic

   	operator: 
# 			operator =  1 means impute is m(tau,v) and output is d(t,x) FORWARD  OP
#			operator = -1 means input is d(t,x) and output is m(tau,v)  ADJOINT  OP 
     
 	OUT   out:  		output data

	 Copyright (C) 2014 The University of Texas at Austin
 	Copyright (C) 2014 Yangkang Chen

	REFERENCE
	Chen, 2018, GEO, Automatic velocity analysis using high-resolution hyperbolic Radon transform
	Chen et al., 2023, GEO, Enhancing earthquake detection from distributed acoustic sensing data by coherency measure and moving-rank-reduction filtering
    '''
    
    h = par['h'];
    v = par['v'];
    nt = par['nt'];
    dt = par['dt'];
    typ = par['typ'];
    
    nh = len(h)
    nv = len(v)
    
    if operator == -1:
        m = np.zeros((nt,nv),dtype='float32')
        num=np.zeros((nt,nv),dtype='float32')
        den=np.zeros((nt,nv),dtype='float32')
    if operator ==  1:
        d = np.zeros((nt,nh),dtype='float32')
    
    if operator == -1:
        d = inpt
    if operator ==  1:
        m = inpt
    
    hmax=np.max(abs(h))
    
    for itau in range(nt):
        #print(itau)
        for ih in range(nh):
            for iv in range(nv):
## This can also be replaced by Parabolic or linear integration 
                if typ == 1:
                    t = itau*dt + h[ih]*v[iv]	
                    it = int(np.floor(t/dt)+1)     
                elif typ == 2:
                    t = itau*dt + h[ih]*h[ih]*v[iv]/hmax/hmax  #curvature
                    it = int(np.floor(t/dt)+1)
            #if(it<=0) it=1;end
                elif typ == 3:
                    t = np.sqrt ((itau*dt)^2 + (h[ih]/v[iv])^2 )
                    it = int(np.floor(t/dt)+1)
                else:
                    t = np.sqrt ((itau*dt)^2 + (h[ih]/v[iv])^2 )
                    it = int(np.floor(t/dt)+1)
                #print(it),print(ih),print(itau),print(iv,'\n')
                if (it+1<=nt) & (it+1>0):  
                    if (operator==-1):
#                         m[itau,iv]=m[itau,iv]+d[it,ih]
                        num[itau,iv]=num[itau,iv]+d[it,ih]
                        den[itau,iv]=den[itau,iv]+d[it,ih]*d[it,ih]
                        
                    if (operator==1):
                        d[it,ih]=d[it,ih]+m[itau,iv]
    
    m=(num*num)/(nh*den);
    
    
    if operator == 1:
        out = d
    if operator ==-1:
        out = m

    return out
    
    
def cohc(din,par,operator=-1,verb=0):
	'''
	Forward and Adjoint operators for Radon RT in the time domain.
	radon implemented in C
	'''
	
	h = par['h'];
	v = par['v'];
	nt = par['nt'];
	dt = par['dt'];
	typ = par['typ'];

	nh = len(h);
	nv = len(v);
	
	din=np.float32(din.flatten(order='F'));
	v=np.float32(v);
	h=np.float32(h);
	
	dout=cohc_fb(din, v, h, typ, nt, nv, nh, dt, operator, verb);
	
	if operator==1: #forward
		dout=dout.reshape(nt,nh,order='F');
	if operator==-1: #backward/adjoint
		dout=dout.reshape(nt,nv,order='F');
	
	
	return dout
	
	
