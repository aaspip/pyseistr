#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <numpy/arrayobject.h>

/*NOTE: PS indicates PySeistr*/
#define PS_NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))
#define PS_MAX_DIM 9
#define PS_PI (3.14159265358979323846264338328)

#define SF_MAX(a,b) ((a) < (b) ? (b) : (a))
#define SF_MIN(a,b) ((a) < (b) ? (a) : (b))
#define SF_ABS(a)   ((a) >= 0  ? (a) : (-(a)))

/*sf functions*/
typedef void (*operator)(bool,bool,int,int,float*,float*);

void ps_adjnull (bool adj /* adjoint flag */, 
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

void *ps_alloc (size_t n    /* number of elements */, 
			  size_t size /* size of one element */)
	  /*< output-checking allocation >*/
{
    void *ptr; 
    
    size *= n;
    
    ptr = malloc (size);

    if (NULL == ptr)
	{
	printf("cannot allocate %lu bytes:", size);
	return NULL;
	}

    return ptr;
}

float *ps_floatalloc (size_t n /* number of elements */)
	  /*< float allocation >*/ 
{
    float *ptr;
    ptr = (float*) ps_alloc (n,sizeof(float));
    return ptr;
}

float **ps_floatalloc2 (size_t n1 /* fast dimension */, 
				  size_t n2 /* slow dimension */)
/*< float 2-D allocation, out[0] points to a contiguous array >*/ 
{
    size_t i2;
    float **ptr;
    
    ptr = (float**) ps_alloc (n2,sizeof(float*));
    ptr[0] = ps_floatalloc (n1*n2);
    for (i2=1; i2 < n2; i2++) {
	ptr[i2] = ptr[0]+i2*n1;
    }
    return ptr;
}

float ***ps_floatalloc3 (size_t n1 /* fast dimension */, 
				   size_t n2 /* slower dimension */, 
				   size_t n3 /* slowest dimension */)
/*< float 3-D allocation, out[0][0] points to a contiguous array >*/ 
{
    size_t i3;
    float ***ptr;
    
    ptr = (float***) ps_alloc (n3,sizeof(float**));
    ptr[0] = ps_floatalloc2 (n1,n2*n3);
    for (i3=1; i3 < n3; i3++) {
	ptr[i3] = ptr[0]+i3*n2;
    }
    return ptr;
}

int *ps_intalloc (size_t n /* number of elements */)
	  /*< int allocation >*/  
{
    int *ptr;
    ptr = (int*) ps_alloc (n,sizeof(int));
    return ptr;
}

bool *ps_boolalloc (size_t n /* number of elements */)
/*< bool allocation >*/
{
    bool *ptr;
    ptr = (bool*) ps_alloc (n,sizeof(bool));
    return ptr;
}

bool **ps_boolalloc2 (size_t n1 /* fast dimension */, 
				size_t n2 /* slow dimension */)
/*< bool 2-D allocation, out[0] points to a contiguous array >*/
{
    size_t i2;
    bool **ptr;
    
    ptr = (bool**) ps_alloc (n2,sizeof(bool*));
    ptr[0] = ps_boolalloc (n1*n2);
    for (i2=1; i2 < n2; i2++) {
	ptr[i2] = ptr[0]+i2*n1;
    }
    return ptr;
}

/*banded.c*/
typedef struct sf_Bands *sf_bands;
/* abstract data type */
/*^*/

struct sf_Bands {
    int n, band;
    float *d, **o;
};

sf_bands sf_banded_init (int n    /* matrix size */, 
			 int band /* band size */)
/*< initialize >*/
{
    sf_bands slv;
    int i;
    
    slv = (sf_bands) ps_alloc (1,sizeof(*slv));
    slv->o = (float**) ps_alloc (band,sizeof(float*));
    for (i = 0; i < band; i++) {
	slv->o[i] = ps_floatalloc (n-1-i);
    }
    slv->d = ps_floatalloc (n);
    slv->n = n;
    slv->band = band;

    return slv;
}

void sf_banded_define (sf_bands slv, 
		       float* diag  /* diagonal [n] */, 
		       float** offd /* off-diagonal [band][n] */)
/*< define the matrix >*/
{
    int k, m, m1, n, n1;
    float t;
    
    for (k = 0; k < slv->n; k++) {
	t = diag[k];
	m1 = SF_MIN(k,slv->band);
	for (m = 0; m < m1; m++)
	    t -= (slv->o[m][k-m-1])*(slv->o[m][k-m-1])*(slv->d[k-m-1]);
	slv->d[k] = t;
	n1 = SF_MIN(slv->n-k-1,slv->band);
	for (n = 0; n < n1; n++) {
	    t = offd[n][k];
	    m1 = SF_MIN(k,slv->band-n-1);
	    for (m = 0; m < m1; m++) {
		t -= (slv->o[m][k-m-1])*(slv->o[n+m+1][k-m-1])*(slv->d[k-m-1]);
	    }
	    slv->o[n][k] = t/slv->d[k];
	}
    }
}


void sf_banded_const_define (sf_bands slv, 
			     float diag        /* diagonal */, 
			     const float* offd /* off-diagonal [band] */)
/*< define matrix with constant diagonal coefficients >*/
{
    int k, m, m1, n, n1;
    float t;
    
    for (k = 0; k < slv->n; k++) {   
	t = diag;
	m1 = SF_MIN(k,slv->band);
	for (m = 0; m < m1; m++)
	    t -= (slv->o[m][k-m-1])*(slv->o[m][k-m-1])*(slv->d[k-m-1]);
	slv->d[k] = t;
	n1 = SF_MIN(slv->n-k-1,slv->band);
	for (n = 0; n < n1; n++) {
	    t = offd[n];
	    m1 = SF_MIN(k,slv->band-n-1);
	    for (m = 0; m < m1; m++) {
		t -= (slv->o[m][k-m-1])*(slv->o[n+m+1][k-m-1])*(slv->d[k-m-1]);
	    }
	    slv->o[n][k] = t/slv->d[k];
	}
    }
}

void sf_banded_const_define_eps (sf_bands slv, 
				 float diag        /* diagonal */, 
				 const float* offd /* off-diagonal [band] */, 
				 int nb            /* size of the boundary */,
				 float eps         /* regularization parameter */)
/*< define matrix with constant diagonal coefficients 
  and regularized b.c. >*/
{
    int k, m, m1, n, n1;
    float t;
    
    for (k = 0; k < slv->n; k++) {   
	t = diag;
	if (k < nb || slv->n-k-1 < nb) t += eps;
	m1 = SF_MIN(k,slv->band);
	for (m = 0; m < m1; m++)
	    t -= (slv->o[m][k-m-1])*(slv->o[m][k-m-1])*(slv->d[k-m-1]);
	slv->d[k] = t;
	n1 = SF_MIN(slv->n-k-1,slv->band);
	for (n = 0; n < n1; n++) {
	    t = offd[n];
	    m1 = SF_MIN(k,slv->band-n-1);
	    for (m = 0; m < m1; m++) {
		t -= (slv->o[m][k-m-1])*(slv->o[n+m+1][k-m-1])*(slv->d[k-m-1]);
	    }
	    slv->o[n][k] = t/slv->d[k];
	}
    }
}


void sf_banded_const_define_reflect (sf_bands slv, 
				     float diag        /* diagonal */, 
				     const float* offd /* off-diagonal [band] */)
/*< define matrix with constant diagonal coefficients  
  and reflecting b.c. >*/
{
    int k, m, n, b, i;
    float t;
    
    b = slv->band;
    
    slv->d[0] = diag+offd[0];
    for (k = 0; k < b-1; k++) {
	for (m = k; m >= 0; m--) {
	    i = 2*k-m+1;
	    t = (i < b)? offd[m]+offd[i]: offd[m];
	    for (n = m+1; n < k-1; n++) 
		t -= (slv->o[n][k-n])*(slv->o[n-m-1][k-n])*(slv->d[k-n]);
	    slv->o[m][k-m] = t/slv->d[k-m];
	}
	i = 2*(k+1);
	t = (i < b)? diag + offd[i]: diag;
	for (m = 0; m <= k; m++)
	    t -= (slv->o[m][k-m])*(slv->o[m][k-m])*(slv->d[k-m]);
	slv->d[k+1] = t;
    }
    for (k = b-1; k < slv->n-1; k++) {
	for (m = b-1; m >= 0; m--) {
	    i = 2*k-m+1;
	    t = (i < b)? offd[m]+offd[i]: offd[m];
	    for (n = m+1; n < b; n++) 
		t -= (slv->o[n][k-n])*(slv->o[n-m-1][k-n])*(slv->d[k-n]);
	    slv->o[m][k-m] = t/(slv->d[k-m]);
	}
	i = 2*(k+1);
	t = (i < b)? diag + offd[i]: diag;
	for (m = 0; m < b; m++) 
	    t -= (slv->o[m][k-m])*(slv->o[m][k-m])*slv->d[k-m];
	slv->d[k+1] = t;
    }
}

void sf_banded_solve (const sf_bands slv, float* b)
/*< invert (in place) >*/
{
    int k, m, m1;
    float t;

    for (k = 1; k < slv->n; k++) {
	t = b[k];
	m1 = SF_MIN(k,slv->band);
	for (m = 0; m < m1; m++)
	    t -= (slv->o[m][k-m-1]) * b[k-m-1];
	b[k] = t;
    }
    for (k = slv->n-1; k >= 0; k--) {
	t = b[k]/slv->d[k];
	m1 = SF_MIN(slv->n -k-1,slv->band);
	for (m = 0; m < m1; m++)
	    t -= slv->o[m][k] * b[k+m+1];
	b[k] = t;
    }
}

void sf_banded_close (sf_bands slv)
/*< free allocated storage >*/
{
    int i;

    for (i = 0; i < slv->band; i++) {
	free(slv->o[i]);
    }
    free (slv->o);
    free (slv->d);
    free (slv);
}

/*apfilt.c*/
static int nf; /*size of filter, nf=nw*2*/
static double *b;

void apfilt_init(int nw /* filter order */)
/*< initialize >*/
{
    int j, k;
    double bk;

    nf = nw*2;
    b = (double*) ps_alloc(nf+1,sizeof(double));

    for (k=0; k <= nf; k++) {
	bk = 1.0;
	for (j=0; j < nf; j++) {
	    if (j < nf-k) {
		bk *= (k+j+1.0)/(2*(2*j+1)*(j+1));
	    } else {
		bk *= 1.0/(2*(2*j+1));
	    }
	}
	b[k] = bk;
    }
}

void apfilt_close(void)
/*< free allocated storage >*/
{
    free(b);
}

void passfilter (float p  /* slope */, 
		 float* a /* output filter [n+1] */)
/*< find filter coefficients >*/
{
    int j, k;
    double ak;
    
    for (k=0; k <= nf; k++) {
	ak = b[k];
	for (j=0; j < nf; j++) {
	    if (j < nf-k) {
		ak *= (nf-j-p);
	    } else {
		ak *= (p+j+1);
	    }
	}
	a[k] = ak;
    }
}

void aderfilter (float p  /* slope */, 
		 float* a /* output filter [n+1] */)
/*< find coefficients for filter derivative >*/
{

    int i, j, k;
    double ak, ai;
    
    for (k=0; k <= nf; k++) {
	ak = 0.;
	for (i=0; i < nf; i++) {
	    ai = -1.0;
	    for (j=0; j < nf; j++) {
		if (j != i) {			
		    if (j < nf-k) {
			ai *= (nf-j-p);
		    } else {
			ai *= (p+j+1);
		    }
		} else if (j < nf-k) {
		    ai *= (-1);
		}
	    }
	    ak += ai;
	}
	a[k] = ak*b[k];
    }
}




#ifndef _sf_butter_h

typedef struct Sf_Butter *sf_butter;
/* abstract data type */
/*^*/

#endif

struct Sf_Butter {
    bool low;
    int nn;
    float **den, mid;
};


sf_butter sf_butter_init(bool low     /* low-pass (or high-pass) */, 
		   float cutoff /* cut off frequency */, 
		   int nn       /* number of poles */)
/*< initialize >*/
{
    int j;
    float arg, ss, sinw, cosw, fact;
    sf_butter bw;

    arg = 2.*PS_PI*cutoff;
    sinw = sinf(arg);
    cosw = cosf(arg);

    bw = (sf_butter) ps_alloc (1,sizeof(*bw));
    bw->nn = nn;
    bw->low = low;
    bw->den = ps_floatalloc2(2,(nn+1)/2);

    if (nn%2) {
	if (low) {
	    fact = (1.+cosw)/sinw;
	    bw->den[nn/2][0] = 1./(1.+fact);
	    bw->den[nn/2][1] = 1.-fact;
	} else {
	    fact = sinw/(1.+cosw);
	    bw->den[nn/2][0] = 1./(fact+1.);
	    bw->den[nn/2][1] = fact-1.;
	}
    }

    fact = low? sinf(0.5*arg): cosf(0.5*arg);
    fact *= fact;
    
    for (j=0; j < nn/2; j++) {
	ss = sinf(PS_PI*(2*j+1)/(2*nn))*sinw;
	bw->den[j][0] = fact/(1.+ss);
	bw->den[j][1] = (1.-ss)/fact;
    }
    bw->mid = -2.*cosw/fact;

    return bw;
}

void sf_butter_close(sf_butter bw)
/*< Free allocated storage >*/
{
    free(bw->den[0]);
    free(bw->den);
    free(bw);
}

void sf_butter_apply (const sf_butter bw, int nx, float *x /* data [nx] */)
/*< filter the data (in place) >*/
{
    int ix, j, nn;
    float d0, d1, d2, x0, x1, x2, y0, y1, y2;

    d1 = bw->mid;
    nn = bw->nn;

    if (nn%2) {
	d0 = bw->den[nn/2][0];
	d2 = bw->den[nn/2][1];
	x0 = y1 = 0.;
	for (ix=0; ix < nx; ix++) { 
	    x1 = x0; x0 = x[ix];
	    y0 = (bw->low)? 
		(x0 + x1 - d2 * y1)*d0:
		(x0 - x1 - d2 * y1)*d0;
	    x[ix] = y1 = y0;
	}
    }

    for (j=0; j < nn/2; j++) {
	d0 = bw->den[j][0];
	d2 = bw->den[j][1];
	x1 = x0 = y1 = y2 = 0.;
	for (ix=0; ix < nx; ix++) { 
	    x2 = x1; x1 = x0; x0 = x[ix];
	    y0 = (bw->low)? 
		(x0 + 2*x1 + x2 - d1 * y1 - d2 * y2)*d0:
		(x0 - 2*x1 + x2 - d1 * y1 - d2 * y2)*d0;
	    y2 = y1; x[ix] = y1 = y0;
	}
    }
}

void sf_reverse (int n1, float* trace)
/*< reverse a trace >*/
{
    int i1;
    float t;

    for (i1=0; i1 < n1/2; i1++) { 
        t=trace[i1];
        trace[i1]=trace[n1-1-i1];
        trace[n1-1-i1]=t;
    }
}



static PyObject *cbp(PyObject *self, PyObject *args){
	
	int phase, verb;
    int i, i2, n1, n2, n3, n123, nplo, nphi;
    float d1, flo, fhi, *trace;
    const float eps=0.0001;
    sf_butter blo=NULL, bhi=NULL;

    
	/**initialize data input**/
    int nd, nd2;
    
    PyObject *f1=NULL;
    PyObject *arrf1=NULL;

    
	PyArg_ParseTuple(args, "Oiiifffiiii", &f1, &n1, &n2, &n3, &d1, &flo, &fhi, &nplo, &nphi, &phase, &verb);

	
	verb=1;
	n123=n1*n2*n3;
	printf("nplo=%d,n123=%d\n",nplo,n123);
	
    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);

    nd2=PyArray_NDIM(arrf1);
    npy_intp *sp=PyArray_SHAPE(arrf1);

	if(flo<0)
		printf('Negative flo');
	else
		flo=flo*d1;


	if(fhi<0)
		printf('Negative flo');
	else
	{	fhi=fhi*d1;
		if(flo>fhi)
			printf('Need flo < fhi\n');

		if(0.5<fhi)
			printf('Need fhi < Nyquist\n');
	}		

    if (nplo < 1)            nplo = 1;
    if (nplo > 1 && !phase)  nplo /= 2; 

    /* number of poles for high cutoff */
    if (nphi < 1)            nphi = 1;
    if (nphi > 1 && !phase)  nphi /= 2; 

    if (verb) printf("flo=%g fhi=%g nplo=%d nphi=%d\n",
			 flo,fhi,nplo,nphi);

	float *trace1;int i1;
    trace = ps_floatalloc(n123);
    trace1 = ps_floatalloc(n1);

    /*reading data*/
    for (i=0; i<n123; i++)
    {
        trace[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    }
    
    if (flo > eps)     blo = sf_butter_init(false, flo, nplo);
    if (fhi < 0.5-eps) bhi = sf_butter_init(true,  fhi, nphi);

    for (i2=0; i2 < n2*n3; i2++) {
	for(i1=0;i1<n1;i1++)
		trace1[i1]=trace[i1+i2*n1];
	
	if (NULL != blo) {
	    sf_butter_apply (blo, n1, trace1); 

	    if (!phase) {
		sf_reverse (n1, trace1);
		sf_butter_apply (blo, n1, trace1); 
		sf_reverse (n1, trace1);	
	    }
	}

	if (NULL != bhi) {
	    sf_butter_apply (bhi, n1, trace1); 

	    if (!phase) {
		sf_reverse (n1, trace1);
		sf_butter_apply (bhi, n1, trace1); 
		sf_reverse (n1, trace1);		
	    }
	}
 
	for(i1=0;i1<n1;i1++)
	{	trace[i1+i2*n1]=trace1[i1];
		}
    }

    if (NULL != blo) sf_butter_close(blo);
    if (NULL != bhi) sf_butter_close(bhi);


    
    /*Below is the output part*/
    PyArrayObject *vecout;
	npy_intp dims[2];
	dims[0]=n123;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<dims[0];i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = trace[i];

    free(trace);
	return PyArray_Return(vecout);
	
}




// documentation for each functions.
static char bpcfun_document[] = "Document stuff for dip...";

// defining our functions like below:
// function_name, function, METH_VARARGS flag, function documents
static PyMethodDef functions[] = {
  {"cbp", cbp, METH_VARARGS, bpcfun_document},
  {NULL, NULL, 0, NULL}
};

// initializing our module informations and settings in this structure
// for more informations, check head part of this file. there are some important links out there.
static struct PyModuleDef bpcfunModule = {
  PyModuleDef_HEAD_INIT, // head informations for Python C API. It is needed to be first member in this struct !!
  "bpcfun",  // module name
  NULL, // means that the module does not support sub-interpreters, because it has global state.
  -1,
  functions  // our functions list
};

// runs while initializing and calls module creation function.
PyMODINIT_FUNC PyInit_bpcfun(void){
  
    PyObject *module = PyModule_Create(&bpcfunModule);
    import_array();
    return module;
}
