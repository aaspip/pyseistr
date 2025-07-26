EPS=0.000000000001;

def srmp(din, ax1=[0,0.1,10], ax2=[0,0.1,10], ax3=[0,0.1,10], dif=None, verb=False, stack=False, both=False, jumpo=1, jumps=1):
	'''
	SRMP: 2D shot gather multiple prediction (Verified when stack=True)
	
	INPUT
	din: shot gather in the frequency domain 
	ax1: offset axis
	ax2: shot axis
	ax3: frequency axis
	dif: optional input data
	jumpo: jump in offset dimension, only for stack=n
	jumps: jump in shot dimension, only for stack=n
	stack: flag for stacking
	both:  flag for receiver; if True, receiver with both sides
	
	
	The axes in the input are {offset,shot,frequency}, (nh, ns, nf)
	The axes in the output are {prediction(if stack=n),offset,shot,frequency} (npre,nh,ns,nf)
	Requirement: offset interval = shot interval

	OUTPUT
	dout (if stack=True: dout(npre,nh,ns,nf); else: dout(nh,ns,nf))
	
	
	EXAMPLE
	from pyseistr import srmp
	import numpy as np
	d=np.zeros([300,20,20])
	d2=srmp(d)
	
	EXAMPLE2
	test_18_srme.py
	
	'''
	import numpy as np
	
	[n1,n2,nw]=din.shape
	newn1,newn2=n1,n2
	
	d1=ax1[1];d2=ax2[1];
	
	if np.mod(n1,jumpo)==0:
		newn1=int(n1/jumpo);
	else:
		newn1=int(n1/jumpo+1);
		
	if np.mod(n2,jumps)==0:
		newn2=int(n2/jumps);
	else:
		newn2=int(n2/jumps+1);
	
	newd1 = d1 * jumpo;
	newd2 = d2 * jumps;
	
	
	fold=0

	if stack:
		mm=np.zeros(n1*n2, dtype=np.complex_)
		dout=np.zeros([n1,n2,nw], dtype=np.complex_);
	else:
		if(~both):
			mm=np.zeros((2*n1-1)*n1*n2, dtype=np.complex_);
			mtemp=np.zeros((2*n1-1)*newn1*newn2, dtype=np.complex_);
		else:
			mm=np.zeros(n1*n1*n2, dtype=np.complex_);
			mtemp=np.zeros(n1*newn1*newn2, dtype=np.complex_);
		dout=np.zeros([2*n1-1,newn1,newn2,nw], dtype=np.complex_);
		
	for iw in range(nw):
		if verb:
			print("Frequency slice %d/%d is done"%(iw,nw))
		
		dd=din[:,:,iw].reshape(n1*n2,1, order='F')
		if dif is not None:
			ref=dif[:,:,iw].reshape(n1*n2,1, order='F')
		if(~both):
			for i2 in range(n2):
				for i1 in range(n1):
# 					print('i2,i1=',i2,i1)
					mm[i2*n1+i1] = 0;
					fold=0;
					for m in range(-1*n1+1,n1,1):
						if m<0:
							x1=-1*m;
							s1=i2+m;
							x2=i1-m;
							s2=m+i2;
						elif (i1-m)<0:
							x1=m;
							s1=i2;
							x2=m-i1;
							s2=i2+i1;
						else:
							x1=m;
							s1=i2;
							x2=i1-m;
							s2=m+i2;
						if stack:
							if (s1 >= 0 and s1 < n2 and x1 >= 0 and x1 < n1 and s2 >= 0 and s2 < n2 and x2 >= 0 and x2 < n1 ):
								if dif is not None:
										mm[i2*n1+i1] = mm[i2*n1+i1] + dd[s1*n1+x1]*ref[s2*n1+x2];
								else:
										mm[i2*n1+i1] = mm[i2*n1+i1] + dd[s1*n1+x1]*dd[s2*n1+x2];
							else:
								mm[i2*n1+i1] = 0;
						
							if 0!=np.abs(mm[i2*n1+i1]):
								fold=fold+1;
						else:
							if (s1 >= 0 and s1 < n2 and x1 >= 0 and x1 < n1 and s2 >= 0 and s2 < n2 and x2 >= 0 and x2 < n1):
								if dif is not None:
										mm[i2*(2*n1-1)*n1+i1*(2*n1-1)+m+n1-1] = dd[s1*n1+x1]*ref[s2*n1+x2];
								else:
										mm[i2*(2*n1-1)*n1+i1*(2*n1-1)+m+n1-1] = dd[s1*n1+x1]*dd[s2*n1+x2];
							else:
								mm[i2*(2*n1-1)*n1+i1*(2*n1-1)+m+n1-1]=0;
					
					if stack:
						mm[i2*n1+i1] = mm[i2*n1+i1]/(fold+EPS);
			
			if stack==False:
				tn=0;
				for i2 in range(n2):
					for i1 in range(n1):
						if (np.mod(i2,jumps)==0 and np.mod(i1,jumpo)==0):
							for m in range(-1*n1+1,n1,1):
								mtemp[tn] = mm[i2*(2*n1-1)*n1+i1*(2*n1-1)+m+n1-1];
								tn = tn+1;
				if (tn!=(2*n1-1)*newn1*newn2):
					Exception("Sorry, t2 needs to be 2*n1-1)*newn1*newn2");
				dout[:,:,:,iw]=mtemp.reshape(2*n1-1,newn1,newn2,order='F');
			else:
				dout[:,:,iw]=mm.reshape(n1,n2,order='F');
		else:
			for i2 in range(n2):
				for i1 in range(n1):
					mm[i2*n1+i1] = 0;
					fold=0;
					for m in range(0,n1,1):
						x1=m;
						s1=i2;
						x2=i1-m+n1/2;
						s2=m+i2-n1/2;
						if stack:
							if (s1 >= 0 and s1 < n2 and x1 >= 0 and x1 < n1 and s2 >= 0 and s2 < n2 and x2 >= 0 and x2 < n1 ):
								if dif is not None:
										mm[i2*n1+i1] = mm[i2*n1+i1] + dd[s1*n1+x1]*ref[s2*n1+x2];
								else:
										mm[i2*n1+i1] = mm[i2*n1+i1] + dd[s1*n1+x1]*dd[s2*n1+x2];
							else:
								mm[i2*n1+i1] = 0;
						
							if 0!=np.abs(mm[i2*n1+i1]):
								fold=fold+1;
						else:
							if (s1 >= 0 and s1 < n2 and x1 >= 0 and x1 < n1 and s2 >= 0 and s2 < n2 and x2 >= 0 and x2 < n1):
								if dif is not None:
										mm[i2*n1*n1+i1*n1+m] = dd[s1*n1+x1]*ref[s2*n1+x2];
								else:
										mm[i2*n1*n1+i1*n1+m] = dd[s1*n1+x1]*dd[s2*n1+x2];
							else:
								mm[i2*n1*n1+i1*n1+m]=0;
					
					if stack:
						mm[i2*n1+i1] = mm[i2*n1+i1]/(fold+EPS);
			
			if stack==False:
				tn=0;
				for i2 in range(n2):
					for i1 in range(n1):
						if (np.mod(i2,jumps)==0 and np.mod(i1,jumpo)==0):
							for m in range(0,n1,1):
								mtemp[tn] = mm[i2*n1*n1+i1*n1+m];
								tn = tn+1;
				if (tn!=n1*newn1*newn2):
					Exception("Sorry, t2 needs to be n1*newn1*newn2");
				dout[:,:,:,iw]=mtemp.reshape(n1,newn1,newn2,order='F');
			else:
				dout[:,:,iw]=mm.reshape(n1,n2,order='F');

					
	
	
# 	dout=din
	
	return dout