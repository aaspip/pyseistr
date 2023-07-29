#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <numpy/arrayobject.h>


/*from user/chenyk/vecoper.c */
#define RT_HUGE 9999999999999999
#define RT_EPS 0.00000000000000000000001 

void adjnull (bool adj /* adjoint flag */, 
		 bool add /* addition flag */, 
		 int nx   /* size of x */, 
		 int ny   /* size of y */, 
		 float* x, 
		 float* y) 
/*< Zeros out the output (unless add is true). 
  Useful first step for any linear operator. >*/
{
    int i;
    
    if(add) return;
    
    if(adj) {
	for (i = 0; i < nx; i++) {
	    x[i] = 0.;
	}
    } else {
	for (i = 0; i < ny; i++) {
	    y[i] = 0.;
	}
    }
}


float dotmultsum( float *xx, float *yy, int n)
/*< dot multiplication of vectors and sum up >*/
{	
	float zz=0;
	int i;
	for(i=0;i<n;i++)
		zz+=xx[i]*yy[i];
	return zz;
}

void scale( float *xx, float *yy, int n, float s)
/*< implement a scaled indentity operator >*/
{
	int i;
	for(i=0;i<n;i++)
		yy[i]=s*xx[i];
}

void vecabs( float *xx, float *yy, int n)
/*< Absolute value for vector >*/
{
	int i;
	for(i=0;i<n;i++)
		yy[i]=fabsf(xx[i]);
}

void scalesum(float *xx, float *yy, float *zz, int n, float sx, float sy)
/*< summation between two scaled vector >*/
{
	int i;
	for(i=0;i<n;i++)
		zz[i]=sx*xx[i]+sy*yy[i];
}

void scalesumreal(float *xx, float yy, float *zz, int n, float sx)
/*< summation between one scaled vector and one scaler  >*/
{
	int i;
	for(i=0;i<n;i++)
		zz[i]=sx*xx[i]+yy;
}

void vecdiv(float *xx, float *yy, float *zz, int n)
/*< division between two vector >*/
{
	int i;
	for(i=0;i<n;i++)
		zz[i]=xx[i]/(yy[i]+RT_EPS);
}

void vecmul(float *xx, float *yy, float *zz, int n)
/*< multiplication between two vector >*/
{
	int i;
	for(i=0;i<n;i++)
		zz[i]=xx[i]*yy[i];
}

float vecmax(float *xx, int n)
/*< maximum value in a vector >*/
{	
	float t=-RT_HUGE;
	int i;
	for(i=0;i<n;i++)
		if(t<xx[i]) t=xx[i];
	return t;
}

void consvec(float v, int n, float *x)
/*< Create constant-value vector >*/
{	
	int i;
	for(i=0;i<n;i++)
		x[i]=v;
}

float cblas_sdot(int n, const float *x, int sx, const float *y, int sy)
/*< x'y float -> complex >*/
{
    int i, ix, iy;
    float dot;

    dot = 0.;

    for (i=0; i < n; i++) {
	ix = i*sx;
	iy = i*sy;
	dot += x[ix] * y[iy];
    }

    return dot;
}

typedef void (*operator)(bool,bool,int,int,float*,float*);

/*same as user/chenyk/cohutil.c*/
static int nt,nh,nv,type;
static float dt,*v,*h,hmax=0,*den,*num;

void coh_init(int Nt, int Nh, int Nv, float Dt, float *V, float *H, int typ)
/*< Initialization for linear coh operator >*/
{
	nt=Nt;
	nh=Nh;
	nv=Nv;
	dt=Dt;
	v=V;
	h=H;
	type=typ;
	
	int i;
	if(type!=1 && type!=2 && type!=3) type=1;
	
	if(type==2)
	{
		for(i=0;i<nh;i++)
			if(hmax<fabs(h[i])) hmax=h[i];
	}
}

void coh_init2(int Nt, int Nh, int Nv, float Dt, float *V, float *H, int typ, float *Num, float *Den)
/*< Initialization for linear coh operator >*/
{
	nt=Nt;
	nh=Nh;
	nv=Nv;
	dt=Dt;
	v=V;
	h=H;
	type=typ;
	den=Den;
	num=Num;
	
	int i;
	if(type!=1 && type!=2 && type!=3) type=1;
	
	if(type==2)
	{
		for(i=0;i<nh;i++)
			if(hmax<fabs(h[i])) hmax=h[i];
	}
}

void coh(bool adj, bool add, int nx, int ny, float *x, float *y)
/*< General coh operator >*/
{   
  int itau,ih,iv,it,i;
	float tau,h_v,t;
	
	adjnull(adj,add,nx,ny,x,y);

	for(itau=0;itau<nt;itau++)
		for(ih=0;ih<nh;ih++)
			for (iv=0;iv<nv;iv++)
			{

				if(type==1) /*Linear*/
				{	
				t = itau*dt + h[ih]*v[iv];
				}
				if(type==2) /*Parabolic*/
				{
				t = itau*dt + h[ih]*h[ih]*v[iv]/hmax/hmax;
				}
				if(type==3) /*Hyperbolic*/
				{
				tau=itau*dt;
				h_v=h[ih]/v[iv];
				t=sqrtf(tau*tau+h_v*h_v);
				}	
				it=floorf(t/dt)+1;
				
				if(it<=nt && it+1>0)
				{
					if (adj)
// 					x[iv*nt+itau]+=y[ih*nt+it];
					{
					num[iv*nt+itau]=num[iv*nt+itau]+y[ih*nt+it];
                    den[iv*nt+itau]=den[iv*nt+itau]+y[ih*nt+it]*y[ih*nt+it];
                    }
					else
					y[ih*nt+it]+=x[iv*nt+itau]; 
				}	
			}	
	if(adj)
		for(itau=0;itau<nt;itau++)
		for (iv=0;iv<nv;iv++)
			x[iv*nt+itau]=(num[iv*nt+itau]*num[iv*nt+itau])/(nh*den[iv*nt+itau]+0.0000000000000001);
}


void coh_pcg(operator oper, 
			float *d, 
			float *x, 
			int itmax_internal,
			int itmax_external, 
			int verb,
			float *misfit)
/*< Preconditioned CG for high-resolution coh transform >*/
{
	int l,k,kc,nx,ny;
	float *z,*P,*s,*g,*xtemp,*di,*ss,*y,*r;
	float gammam,gamma,beta,den,alpha;
	nx=nt*nv;
	ny=nt*nh;
	
	/* Allocate memory */
	xtemp = (float*)malloc(nx * sizeof(float));
	y = (float*)malloc(nx * sizeof(float));
	z = (float*)malloc(nx * sizeof(float));
	P = (float*)malloc(nx * sizeof(float));
	di = (float*)malloc(ny * sizeof(float));
	r = (float*)malloc(ny * sizeof(float));
	g = (float*)malloc(nx * sizeof(float));
	s = (float*)malloc(nx * sizeof(float));
	ss = (float*)malloc(ny * sizeof(float));
	
	consvec(0,nx,z);				// z=zeros(size(m0));				
	consvec(1,nx,P);				// P=ones(size(z));	
			
	kc=1;						// kc=1;
	scale(z,x,nx,1);						// x=z;
	
	for(l=1;l<=itmax_external;l++)
	{
	vecmul(P,z,xtemp,nx);
	oper(false,false,nx,ny,xtemp,di);	// feval(operator,P.*z,Param,1);
	scalesum(d, di, r, ny, 1, -1);	// r=d-di;
	oper(true,false,nx,ny,g,r);		// g=feval(operator,r,Param,-1);
	vecmul(g,P,g,nx);				// g=g.*P;
	scale(g,s,nx,1);				// s=g;
	gammam=cblas_sdot(nx,g,1,g,1);	// gammam=cgdot(g);
	k=1;							// k=1;
	while (k<=itmax_internal)
	{
		vecmul(P,s,xtemp,nx);
		oper(false,false,nx,ny,xtemp,ss);	// ss=feval(operator,P.*s,Param,1);
		den = cblas_sdot(ny,ss,1,ss,1);	// den = cgdot(ss);	
		alpha = gammam/(den+0.00000001);		// alpha=gammam/(den+1.e-8);
		scalesum(z,s,z,nx,1,alpha);			// z=z+alpha*s;
		scalesum(r,ss,r,ny,1,-alpha);			// r=r-alpha*ss;
		misfit[kc-1]=cblas_sdot(ny,r,1,r,1);	// misfit(kc)=cgdot(r);
		oper(true,false,nx,ny,g,r);			// g=feval(operator,r,Param,-1);
		vecmul(g,P,g,nx);					// g=g.*P;
		gamma = cblas_sdot(nx,g,1,g,1);		// gamma=cgdot(g);
		beta=gamma/(gammam+0.0000001);			// beta=gamma/(gammam+1.e-7);	
		gammam=gamma;						// gammam=gamma;
		
		scalesum(g,s,s,nx,1,beta);			// s=g+beta*s;
		if(verb == 1)
			printf("Iteration = %d Misfit=%0.5g \n ",k,misfit[kc-1]);
		k=k+1;							// k=k+1;
		kc=kc+1;}							// kc=kc+1;
	vecmul(P,z,x,nx);						// x=P.*z;
	vecabs(x,xtemp,nx);						// y=x/max(abs(x(:)));
	scale(x,y,nx,1.0/vecmax(xtemp,nx));		// y=x/max(abs(x(:)));
	vecabs(y,P,nx);						// P=abs(y)+0.001;
	scalesumreal(P,0.001,P,nx,1);				// P=abs(y)+0.001;
	}	
	
	free(xtemp);
	free(y);
	free(z);
	free(P);
	free(di);
	free(r);
	free(g);
	free(s);
	free(ss);
}


static PyObject *cohc_inv(PyObject *self, PyObject *args){
	
    /*Below is the input part*/
    int f5,f6,f7,f8,f9,f10,f12;
    float f11;
    
	/**initialize data input**/
    int nd, nd2;
    
    PyObject *f1=NULL;
    PyObject *f2=NULL;
    PyObject *f3=NULL;
    PyObject *f4=NULL;
    PyObject *arrf1=NULL;
    PyObject *arrf2=NULL;
    PyObject *arrf3=NULL;
    PyObject *arrf4=NULL;
    
	PyArg_ParseTuple(args, "OOOOiiiiiifi", &f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8, &f9, &f10, &f11, &f12);

    int ndim, i, *p;
    
    int typ, niter_in, niter_out, nt0, nv0, nh0, verb, ndata, nmod;
    float *v0, *h0, *misfit, *data, *model; 
	float dt0;
	
	typ=f5;
	niter_in=f6;
	niter_out=f7;
	nt0=f8;
	nv0=f9;
	nh0=f10;
	dt0=f11;
	verb=f12;
	
    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    arrf2 = PyArray_FROM_OTF(f2, NPY_FLOAT, NPY_IN_ARRAY);
    arrf3 = PyArray_FROM_OTF(f3, NPY_FLOAT, NPY_IN_ARRAY);
    arrf4 = PyArray_FROM_OTF(f4, NPY_FLOAT, NPY_IN_ARRAY);
    
    nd2=PyArray_NDIM(arrf1);
    
    npy_intp *sp=PyArray_SHAPE(arrf1);
    npy_intp *spmod=PyArray_SHAPE(arrf2);

    ndata=nt0*nh0;
    nmod=nt0*nv0;

	data  = (float*)malloc(ndata * sizeof(float));
	model = (float*)malloc(nmod * sizeof(float));
	v0 = (float*)malloc(nv0 * sizeof(float));
	h0 = (float*)malloc(nh0 * sizeof(float));
	
    if (*sp != ndata)
    {
    	printf("Dimension mismatch, N_input = %d, N_data = %d\n", *sp, ndata);
    	return NULL;
    }
    
    /*reading data*/
    for (i=0; i<ndata; i++)
    {
        data[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    }

    for (i=0; i<nmod; i++)
    {
        model[i]=*((float*)PyArray_GETPTR1(arrf2,i));
        
    }

    for (i=0; i<nv0; i++)
    {
        v0[i]=*((float*)PyArray_GETPTR1(arrf3,i));
    }

    for (i=0; i<nh0; i++)
    {
        h0[i]=*((float*)PyArray_GETPTR1(arrf4,i));
    }
    
	misfit = (float*)malloc(niter_in*niter_out * sizeof(float));/*why in python-C env this needs to be before the pcg?*/
	
	coh_init(nt0, nh0, nv0, dt0, v0, h0, typ);
	
	coh_pcg(coh, data, model, niter_in, niter_out, verb, misfit);	
	
    /*Below is the output part*/
    PyArrayObject *vecout;
	npy_intp dims[2];
	dims[0]=nmod+niter_in*niter_out;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<dims[0];i++)
	if(i<nmod)
		(*((float*)PyArray_GETPTR1(vecout,i))) = model[i];
	else
		(*((float*)PyArray_GETPTR1(vecout,i))) = misfit[i-nmod];

	
	return PyArray_Return(vecout);
	
}

static PyObject *cohc_fb(PyObject *self, PyObject *args){
	/*forward and backward Radon*/
	
    /*Below is the input part*/
    int f4,f5,f6,f7,f9,f10;
    float f8;
    
	/**initialize data input**/
    int nd, nd2;
    
    PyObject *f1=NULL;
    PyObject *f2=NULL;
    PyObject *f3=NULL;

    PyObject *arrf1=NULL;
    PyObject *arrf2=NULL;
    PyObject *arrf3=NULL;

    
	PyArg_ParseTuple(args, "OOOiiiifii", &f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8, &f9, &f10);

    int ndim, i, *p;
    
    int typ, niter_in, niter_out, nt0, nv0, nh0, verb, ndata, nmod;
    float *v0, *h0, *misfit, *data, *model, *num0, *den0; 
	float dt0;
	int forw,adj;
	
	typ=f4;
	nt0=f5;
	nv0=f6;
	nh0=f7;
	dt0=f8;
	forw=f9; /*forw=1: forward, else: backward/adjoint*/
	verb=f10;
	
	if(verb)
		printf("typ=%d,nt0=%d,nv0=%d,nh0=%d,dt0=%g,forw=%d,verb=%d\n",typ,nt0,nv0,nh0,dt0,forw,verb);
	
    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    arrf2 = PyArray_FROM_OTF(f2, NPY_FLOAT, NPY_IN_ARRAY);
    arrf3 = PyArray_FROM_OTF(f3, NPY_FLOAT, NPY_IN_ARRAY);

    nd2=PyArray_NDIM(arrf1);
    
    npy_intp *sp=PyArray_SHAPE(arrf1);

    ndata=nt0*nh0;
    nmod=nt0*nv0;
	
	data  = (float*)malloc(ndata * sizeof(float));
	model = (float*)malloc(nmod * sizeof(float));
	v0 = (float*)malloc(nv0 * sizeof(float));
	h0 = (float*)malloc(nh0 * sizeof(float));
	
	den0=(float*)malloc(nmod * sizeof(float));
	num0=(float*)malloc(nmod * sizeof(float));
    
    for(i=0;i<nmod;i++)
    	{den0[i]=0;num0[i]=0;}
	
    /*reading data*/
    if(forw==-1)
    {
    	for (i=0; i<ndata; i++)
    	{
        	data[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    	}
    	for (i=0; i<nmod; i++)
    	{
        	model[i]=0;
    	}
    	adj=1;
	}
	
	if(forw==1)
	{
    	for (i=0; i<nmod; i++)
    	{
        	model[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    	}
    	for (i=0; i<ndata; i++)
    	{
        	data[i]=0;
    	}
    	adj=0;
	}

    for (i=0; i<nv0; i++)
    {
        v0[i]=*((float*)PyArray_GETPTR1(arrf2,i));
    }

    for (i=0; i<nh0; i++)
    {
        h0[i]=*((float*)PyArray_GETPTR1(arrf3,i));
    }

	coh_init2(nt0, nh0, nv0, dt0, v0, h0, typ, num0, den0);
	coh(adj, 0, nmod, ndata, model, data);

    /*Below is the output part*/
    PyArrayObject *vecout;
	npy_intp dims[2];
	
	if(forw==1)
	{dims[0]=ndata;}
	
	if(forw==-1)
	{dims[0]=nmod;
	}
	
	dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	if(forw==1)
	{
	for(i=0;i<dims[0];i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = data[i];
	}
	
	if(forw==-1)
	{
	for(i=0;i<dims[0];i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = model[i];
	}

	free(data);
	free(model);
	free(v0);
	free(h0);
	free(den0);
	free(num0);
	
	return PyArray_Return(vecout);
	
	
}

// documentation for each functions.
static char cohcfun_document[] = "Document stuff for coh...";

// defining our functions like below:
// function_name, function, METH_VARARGS flag, function documents
static PyMethodDef functions[] = {
  {"cohc_inv", cohc_inv, METH_VARARGS, cohcfun_document},
  {"cohc_fb", cohc_fb, METH_VARARGS, cohcfun_document},
  {NULL, NULL, 0, NULL}
};

// initializing our module informations and settings in this structure
// for more informations, check head part of this file. there are some important links out there.
static struct PyModuleDef cohcfunModule = {
  PyModuleDef_HEAD_INIT, // head informations for Python C API. It is needed to be first member in this struct !!
  "cohcfun",  // module name
  NULL, // means that the module does not support sub-interpreters, because it has global state.
  -1,
  functions  // our functions list
};

// runs while initializing and calls module creation function.
PyMODINIT_FUNC PyInit_cohcfun(void){
  
    PyObject *module = PyModule_Create(&cohcfunModule);
    import_array();
    return module;
}
