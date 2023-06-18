import numpy as np
def pwsmooth_set(dip,n1,n2,ns,order,eps):
	'''
	pwsmooth_set: set local slope for pwsmooth_lop
	
	INPUT
	dip:
	n1/n2:
	ns:
	order:
	eps:
	
	OUTPUT
	w1: 2D matrix [n1,n2]
	'''
	n12=n1*n2;
	w1=np.ones(n12);
	
	par={'nm':n12,'nd':n12,'dip':dip,'w1':w1.reshape(n1,n2,order='F'),'ns':ns,'order':order,'eps':eps}
	t=pwsmooth_lop(w1,par,False,False);
	for i1 in range(n12):
		if t[i1] != 0.0:
			w1[i1]=1.0/t[i1];
		else:
			w1[i1]=0.0;
	w1=w1.reshape(n1,n2,order='F');
	
	return w1
	
	
def pwsmooth_lop(din,par,adj,add):
	'''
	pwsmooth_lop: plane-wave smoothing operator
	BY Yangkang Chen, Aug, 10, 2021
	
	INPUT
	d: model/data
	par: parameter (dip,w1,ns,order,eps,nm,nd)
	adj: adj flag
	add: add flag
	
	OUTPUT
	m: data/model
	'''
	from .operators import adjnull
	from .pwspray2d import pwspray_lop
	
	dip=par['dip'];
	w1=par['w1'];
	[n1,n2]=dip.shape;
	ns=par['ns'];
	order=par['order'];
	eps=par['eps'];
	nm=par['nm']
	nd=par['nd']
	ndn=n1*n2;
	nds=n1*n2;

	if adj==1:
		d=din;
		if 'm' in par and add==1:
			m=par['m'];
		else:
			m=np.zeros(par['nm']);
	else:
		m=din;
		if 'd' in par and add==1:
			d=par['d'];
		else:
			d=np.zeros(par['nd']);

	dn=m;
	ds=d;
	
	if ndn!=nds:
		print('Wrong size %d = %d'%(ndn,nds))

	[ dn,ds ] = adjnull( adj,add,ndn,nds,dn,ds );

	ns2=2*ns+1;#spray diameter
	n12=n1*n2;

	u=np.zeros([n1,ns2,n2]);
	utmp=np.zeros(n1*n2*ns2);
	w=np.zeros(ns2);

	for iss in range(ns2):
		w[iss]=ns+1-abs(iss-ns);

	#for Normalization
	#t=zeros(n12,1);
	
	par.update({'nt':n1,'nx':n2,'nm':n1*n2,'nd':n1*n2*ns2})

	if adj:
		for i2 in range(n2):
			for i1 in range(n1):
				ws=w1[i1,i2];
				for iss in range(ns2):
					u[i1,iss,i2]=ds[i2*n1+i1]*w[iss]*ws;
	
		utmp=u.flatten(order='F');
		dn=pwspray_lop(utmp,par,True,True);
	else:
		utmp=pwspray_lop(dn,par,False,False);
		u=utmp.reshape(n1,ns2,n2,order='F');
		for i2 in range(n2):
			for i1 in range(n1):
				ws=w1[i1,i2]
				for iss in range(ns2):
					ds[i2*n1+i1]=ds[i2*n1+i1]+u[i1,iss,i2]*w[iss]*ws;
	
	if adj==1:
		dout=dn;
	else:
		dout=ds;
		
	return dout
	
	