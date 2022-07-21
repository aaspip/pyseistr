import numpy as np
def solver(opL,solv,nx,ny,x,dat,niter,par_L,par):
	#Generic linear solver. (same as Madagascar function: sf_solver)
	#
	#Solves
	#L{x}    =~ dat
	#
	#Yangkang Chen
	#Aug, 05, 2020
	#Ported to Python in Apr, 25, 2022
	#
	#INPUT
	#opL: forward/linear operator
	#solv: stepping function
	#nx:   size of x   (1D vector)
	#ny:   size of dat (1D vector)
	#x:    estimated model
	#dat:  data
	#niter:number of iterations
	#
	#par. (parameter struct)
	#  "wt":     float*:         weight      
	#  "wght":   sf_weight wght: weighting function
	#  "x0":     float*:         initial model
	#  "nloper": sf_operator:    nonlinear operator
	#  "mwt":    float*:         model weight
	#  "verb":   bool:           verbosity flag
	#  "known":  bool*:          known model mask
	#  "nmem":   int:            iteration memory
	#  "nfreq":  int:            periodic restart
	#  "xmov":   float**:        model iteration
	#  "rmov":   float**:        residual iteration
	#  "err":    float*:         final error
	#  "res":    float*:         final residual
	#
	#OUPUT
	#x: estimated model
	#
	#
	
	TOLERANCE=1.e-12;
	forget=0;
	x=x.flatten(order='F');
	dat=dat.flatten(order='F');
	xmov=None;
	rmov=None;
	wht=None;
	err=None;
	res=None;

	if 'wt' in par:
		wt=par['wt'];
	else:
		wt=None;
	
	if 'wght' in par:
		wght=par['wght'];
	else:
		wght=None;
	
	if 'x0' in par:
		x0=par['x0'];
	else:
		x0=None;
	
	if 'nloper' in par:
		nloper=par['nloper'];
		par_nloper=par['par_nloper'];
	else:
		nloper=None;
		
	if 'mwt' in par:
		mwt=par['mwt'];
	else:
		mwt=None;
		
	if 'verb' in par:
		verb=par['verb'];
	else:
		verb=0;
		
	if 'known' in par:
		known=par['known'];
	else:
		known=None;

	if 'nmem' in par:
		nmem=par['nmem'];
	else:
		nmem=-1;
	
	if 'nfreq' in par:
		nfreq=par['nfreq'];
	else:
		nfreq=0;
		
	if 'xmov' in par:
		xmov=par['xmov'];
	else:
		xmov=None;

	if 'rmov' in par:
		rmov=par['rmov'];
	else:
		rmov=None;
		
	if 'err' in par:
		err=par['err'];
	else:
		err=None;

	if 'res' in par:
		res=par['res'];
	else:
		res=None;
		
	g=np.zeros(nx);
	rr=np.zeros(ny);
	gg=np.zeros(ny);
	
	if wt  is not None or wght  is not None:
		td=zeros(ny);
		if wt  is not None:
			wht=wt;
		else:
			wht=np.ones(ny);
	
	if mwt  is not None:
		g2=np.zeros(nx);
	
	rr=-dat;
	if x0  is not None:
		x=x0;
		if mwt  is not None:
			x=x*mwt;
		if nloper  is not None:
			par_nloper['d']=rr;
			rr=nloper(x,par_nloper,0,1);
		else:
			par_L['d']=rr;
			rr=opL(x,par_L,0,1);
	else:
		x=np.zeros(nx);
	
	dpr0=np.sum(rr*rr);
	dpg0=1.0;
	
	for n in range(0,niter):
		if nmem>=0:
			forget= (n >= nmem);
			
		if wght  is not None and forget:
			wht=wght(ny,rr); #wght is a function
		
		if wht  is not None:
			rr=rr*wht;
			td=rr*wht;
			g=opL(td,par_L,1,0);
		else:
			g=opL(rr,par_L,1,0);
		
		if mwt  is not None:
			g=g*mwt;
			
		if known  is not None:
			for ii in range(0,nx):
				if known[ii]:
					g[ii]=0.0;
		
		if mwt  is not None:
			g2=g*mwt;
			gg=opL(g2,par_L,0,0);
		else:
			gg=opL(g,par_L,0,0);
		
		if wht  is not None:
			gg=gg*wht;
		
		if forget and (nfreq !=0): #periodic restart
			forget = (0 == np.mod(n+1,nfreq));
		
		if n==0:
			dpg0=np.sum(g*g);
			dpr=1.0;
			dpg=1.0;
		else:
			dpr=np.sum(rr*rr)/dpr0;
			dpg=np.sum(g*g)/dpg0;
		
		if verb:
			print('iteration %d res %f mod %f grad %f !'%(n+1, dpr,np.sum(x*x), dpg));
			
		if dpr < TOLERANCE or dpg < TOLERANCE:
			if verb:
				print('convergence in %d iterations\n',n+1);
			if mwt  is not None:
				x=x*mwt;
			break;
		
		x,rr = solv(forget,nx,ny,x,g,rr,gg);
		
		forget=0;
		if nloper  is not None:
			rr=-dat;
			if mwt  is not None:
				x=x*mwt;
			par_nloper['d']=rr;
			rr=nloper(x,par_nloper,0,1); 
		else:
			if wht is not None:
				rr=-dat;
				if mwt  is not None:
					x=x*mwt;
				par_L['d']=rr;
				rr=opL(x,par_L,0,1);
			else:
				if mwt  is not None and ( xmov  is not None or n==niter-1 ):
					x=x*mwt;
	
		if xmov  is not None:
			xmov[:,n]=x;
			par['xmov']=xmov;
		
		if rmov  is not None:
			rmov[:,n]=rr;
			par['rmov']=rmov;
		
		if err  is not None:
			err[n]=np.sum(r*r);
			par['err']=err;

	if res  is not None:
		res=rr;
		par['res']=res;
		
	return x,par

def conjgrad(opP,opL,opS, p, x, dat, eps_cg, tol_cg, N,ifhasp0,par_P,par_L,par_S,verb):
	#conjgrad: conjugate gradient with shaping
	#
	#by Yangkang Chen, 2022
	#
	#Modified by Yangkang Chen, Nov, 09, 2019 (fix the "adding" for each oper)
	#
	#INPUT
	#opP: preconditioning operator
	#opL: forward/linear operator
	#opS: shaping operator
	#d:  data
	#N:  number of iterations
	#eps_cg:  scaling
	#tol_cg:  tolerance
	#ifhasp0: flag indicating if has initial model
	#par_P: parameters for P
	#par_L: parameters for L
	#par_S: parameters for S
	#verb: verbosity flag
	#
	#OUPUT
	#x: estimated model
	#

	nnp=p.size;
	nx=par_L['nm'];	#model size
	nd=par_L['nd'];	#data size

	if opP  is not None:
		d=-dat; #nd*1
		r=opP(d,par_P,0,0);
	else:
		r=-dat;  

	if ifhasp0:
		x=op_S(p,par_S,0,0);
		if opP  is not None:
			d=opL(x,par_L,0,0);
			par_P['d']=r;#initialize data
			r=opP(d,par_P,0,1);
		else:
			par_P['d']=r;#initialize data
			r=opL(x,par_L,0,1);

	else:
		p=np.zeros(nnp);#define np!
		x=np.zeros(nx);#define nx!

	dg=0;
	g0=0;
	gnp=0;
	r0=np.sum(r*r);   #nr*1

	for n in range(1,N+1):
		gp=eps_cg*p; #np*1
		gx=-eps_cg*x; #nx*1
		
		if opP is not None:
			d=opP(r,par_P,1,0);#adjoint
			par_L['m']=gx;#initialize model
			gx=opL(d,par_L,1,1);#adjoint,adding
		else:
			par_L['m']=gx;#initialize model
			gx=opL(r,par_L,1,1);#adjoint,adding

	
		par_S['m']=gp;#initialize model
		gp=opS(gx,par_S,1,1);#adjoint,adding
		gx=opS(gp.copy(),par_S,0,0);#forward,adding
		#The above gp.copy() instead of gp is the most striking bug that has been found because otherwise gp was modified during the shaping operation (opS) (Mar, 28, 2022)
		
		if opP is not None:
			d=opL(gx,par_P,0,0);#forward
			gr=opP(d,par_L,0,0);#forward
		else:
			gr=opL(gx,par_L,0,0);#forward

		gn = np.sum(gp*gp); #np*1

		if n==1:
			g0=gn;
			sp=gp; #np*1
			sx=gx; #nx*1
			sr=gr; #nr*1
		else:
			alpha=gn/gnp;
			dg=gn/g0;
		
			if alpha < tol_cg or dg < tol_cg:
				return x;
				break;
		
			gp=alpha*sp+gp;
			t=sp;sp=gp;gp=t;
		
			gx=alpha*sx+gx;
			t=sx;sx=gx;gx=t;
		
			gr=alpha*sr+gr;
			t=sr;sr=gr;gr=t;

	 
		beta=np.sum(sr*sr)+eps_cg*(np.sum(sp*sp)-np.sum(sx*sx));
		
		if verb:
			print('iteration: %d, res: %g !'%(n,np.sum(r* r) / r0));  

		alpha=-gn/beta;
	
		p=alpha*sp+p;
		x=alpha*sx+x;
		r=alpha*sr+r;
	
		gnp=gn;

	return x


def cgstep(forget,nx,ny,x,g,rr,gg):
	#Step of conjugate-gradient iteration.
	# Yangkang Chen
	# Aug, 05, 2020
	# 
	# INPUT
	# forget:restart flag
	# nx:   model size
	# ny:   data size
	# x:    current model [nx]
	# g:    gradient [nx]
	# rr:   data residual [ny]
	# gg:   conjugate gradient [ny]
	# 
	# OUTPUT
	# x:    current model [nx]
	# g:    gradient [nx]
	# rr:   data residual [ny]
	# gg:   conjugate gradient [ny]
	
	EPSILON=1.e-12;
	Allocated=0;
	if not Allocated:
		Allocated =1;
		forget=1;
		S=np.zeros(nx);
		Ss=np.zeros(ny);
	if forget:
		S=np.zeros(nx);
		Ss=np.zeros(ny);
		beta=0;
		alfa=np.sum(gg*gg);
		#Solve G . ( R + G*alfa) = 0
		if alfa<=0:
			return x,rr
		alfa=-np.sum(gg*rr)/alfa;
	else:
		#search plane by solving 2-by-2
		#G . (R + G*alfa + S*beta) = 0
		#S . (R + G*alfa + S*beta) = 0
		gdg=np.sum(gg*gg);
		sds=np.sum(Ss*Ss);
		gds=np.sum(gg*Ss);
		if gdg == 0 or sds==0:
			return x,rr
			
		determ=1.0-(gds/gdg)*(gds/sds);
		if determ > EPSILON:
			determ = determ * gdg * sds;
		else:
			determ = gdg * sds * EPSILON;
		gdr=-np.sum(gg*rr);
		sdr=-np.sum(Ss*rr);
		
		alfa = ( sds * gdr - gds * sdr ) / determ;
		beta = (-gds * gdr + gdg * sdr ) / determ;
	S=S*beta;
	S=alfa*g+S;
	Ss=Ss*beta;
	Ss=alfa*gg+Ss;
	x=x+S;
	rr=rr+Ss;
	return x,rr