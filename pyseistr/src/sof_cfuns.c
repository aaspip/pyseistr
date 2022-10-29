#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <numpy/arrayobject.h>

/*NOTE: PS indicates PySeistr*/
#define PS_NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))
#define PS_MAX_DIM 9

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



/*apfilt.c*/

static int n1, n2, ns, ns2, n12;
static float ***u, *w, **w1, *t;
static float *trace, **pspr;

/*pwd.c*/

#ifndef _pwd_h

typedef struct Pwd *pwd; /* abstract data type */
/*^*/

#endif

struct Pwd {
    int n, na;
    float **a, *b;
};

pwd pwd_init(int n1 /* trace length */, 
	     int nw /* filter order */)
/*< initialize >*/
{
    pwd w;

    w = (pwd) ps_alloc(1,sizeof(*w));
    w->n = n1;
    w->na = 2*nw+1;
 
    w->a = ps_floatalloc2 (n1,w->na);
    w->b = ps_floatalloc (w->na);

    apfilt_init(nw);

    return w;
}

void pwd_close (pwd w)
/*< free allocated storage >*/
{
    free (w->a[0]);
    free (w->a);
    free (w->b);
    free (w);
    apfilt_close();
}

void pwd_define (bool adj        /* adjoint flag */, 
		 pwd w           /* pwd object */, 
		 const float* pp /* slope */, 
		 float* diag     /* defined diagonal */, 
		 float** offd    /* defined off-diagonal */)
/*< fill the matrix >*/
{
    int i, j, k, m, n, nw;
    float am, aj;
    
    nw = (w->na-1)/2;
    n = w->n;

    for (i=0; i < n; i++) {
	passfilter (pp[i], w->b);
	
	for (j=0; j < w->na; j++) {
	    if (adj) {
		w->a[j][i] = w->b[w->na-1-j];
	    } else {
		w->a[j][i] = w->b[j];
	    }
	}
    }
    
    for (i=0; i < n; i++) {
	for (j=0; j < w->na; j++) {
	    k = i+j-nw;
	    if (k >= nw && k < n-nw) {
		aj = w->a[j][k];
		diag[i] += aj*aj;
	    }
	} 
	for (m=0; m < 2*nw; m++) {
	    for (j=m+1; j < w->na; j++) {
		k = i+j-nw;
		if (k >= nw && k < n-nw) {
		    aj = w->a[j][k];
		    am = w->a[j-m-1][k];
		    offd[m][i] += am*aj;
		}
	    }
	}
    }
}

void pwd_set (bool adj   /* adjoint flag */,
	      pwd w      /* pwd object */, 
	      float* inp /* input */, 
	      float* out /* output */, 
	      float* tmp /* temporary storage */)
/*< matrix multiplication >*/
{
    int i, j, k, n, nw;

    nw = (w->na-1)/2;
    n = w->n;

    if (adj) {
	for (i=0; i < n; i++) {
	    tmp[i]=0.;
	}
	for (i=0; i < n; i++) {
	    for (j=0; j < w->na; j++) {
		k = i+j-nw;
		if (k >= nw && k < n-nw) 
		    tmp[k] += w->a[j][k]*out[i];
	    }
	}
	for (i=0; i < n; i++) {
	    inp[i]=0.;
	}
	for (i=nw; i < n-nw; i++) {
	    for (j=0; j < w->na; j++) {
		k = i+j-nw;
		inp[k] += w->a[j][i]*tmp[i];
	    }
	}
    } else {
	for (i=0; i < n; i++) {
	    tmp[i] = 0.;
	}
	for (i=nw; i < n-nw; i++) {
	    for (j=0; j < w->na; j++) {
		k = i+j-nw;
		tmp[i] += w->a[j][i]*inp[k];
	    }
	}
	for (i=0; i < n; i++) {
	    out[i] = 0.;
	    for (j=0; j < w->na; j++) {
		k = i+j-nw;
		if (k >= nw && k < n-nw) 
		    out[i] += w->a[j][k]*tmp[k];
	    }
	}
    }
}




/*predict */
/* Trace prediction with plane-wave destruction */

static int nb, k2;
static sf_bands slv;
static float *diag, **offd, eps, eps2, **dip, *tt;
static pwd W1, W2;

static void stepper(bool adj /* adjoint flag */,
		    int i2   /* trace number */);

void predict_init (float e        /* regularization parameter */,
		   int nw         /* accuracy order */,
		   int k          /* radius */,
		   bool two       /* if two predictions */)
/*< initialize >*/
{

    nb = 2*nw;

    eps = e;
    eps2 = e;

    slv = sf_banded_init (n1, nb);
    diag = ps_floatalloc (n1);
    offd = ps_floatalloc2 (n1,nb);

    W1 = pwd_init (n1, nw);
    W2 = two? pwd_init(n1,nw): NULL;

    tt = NULL;

    k2 = k;
//     if (k2 > n2-1) sf_error("%s: k2=%d > n2-1=%d",__FILE__,k2,n2-1);
}

void predict_close (void)
/*< free allocated storage >*/
{
    sf_banded_close (slv);
    free (diag);
    free (*offd);
    free (offd);
    pwd_close (W1);
    if (NULL != W2) pwd_close (W2);
    if (NULL != tt) free(tt);
}

static void regularization(void)
/* fill diag and offd using regularization */ 
{
    int i1, ib;

    for (i1=0; i1 < n1; i1++) {
	diag[i1] = 6.*eps;
    	offd[0][i1] = -4.*eps;
    	offd[1][i1] = eps;
	for (ib=2; ib < nb; ib++) {
	    offd[ib][i1] = 0.0;
	}
    }

    diag[0] = diag[n1-1] = eps2+eps;
    diag[1] = diag[n1-2] = eps2+5.*eps;
    offd[0][0] = offd[0][n1-2] = -2.*eps;
}

void predict_step(bool adj            /* adjoint flag */,
		  bool forw           /* forward or backward */, 
		  float* trace        /* input/output trace */,
		  const float* pp    /* slope */)
/*< prediction step >*/
{
    float t0, t1, t2, t3;
    
    regularization();
    pwd_define (forw, W1, pp, diag, offd);
    sf_banded_define (slv, diag, offd);

    if (adj) sf_banded_solve (slv, trace);

    t0 = trace[0];
    t1 = trace[1];
    t2 = trace[n1-2];
    t3 = trace[n1-1];

    pwd_set (adj, W1, trace, trace, diag);

    trace[0] += eps2*t0;
    trace[1] += eps2*t1;
    trace[n1-2] += eps2*t2;
    trace[n1-1] += eps2*t3;

    if (!adj) sf_banded_solve (slv, trace);
}

void predict1_step(bool forw      /* forward or backward */, 
		   float* trace1  /* input trace */,
		   const float* pp /* slope */,
		   float* trace /* output trace */)
/*< prediction step from one trace >*/
{
    float t0, t1, t2, t3;
    
    regularization();

    pwd_define (forw, W1, pp, diag, offd);
    sf_banded_define (slv, diag, offd);

    t0 = trace1[0];
    t1 = trace1[1];
    t2 = trace1[n1-2];
    t3 = trace1[n1-1];

    pwd_set (false, W1, trace1, trace, diag);

    trace[0] += eps2*t0;
    trace[1] += eps2*t1;
    trace[n1-2] += eps2*t2;
    trace[n1-1] += eps2*t3;

    sf_banded_solve (slv, trace);
}

void predict2_step(bool forw1        /* forward or backward */, 
		   bool forw2,
		   float* trace1     /* input trace */,
		   float* trace2,
		   const float* pp1  /* slope */,
		   const float* pp2,
		   float *trace      /* output trace */)
/*< prediction step from two traces>*/
{
    int i1;
    float t0, t1, t2, t3;
    
    regularization();

    pwd_define (forw1, W1, pp1, diag, offd);
    pwd_define (forw2, W2, pp2, diag, offd);

    sf_banded_define (slv, diag, offd);

    t0 = 0.5*(trace1[0]+trace2[0]);
    t1 = 0.5*(trace1[1]+trace2[1]);
    t2 = 0.5*(trace1[n1-2]+trace2[n1-2]);
    t3 = 0.5*(trace1[n1-1]+trace2[n1-1]);

    pwd_set (false, W1, trace1, offd[0], trace);
    pwd_set (false, W2, trace2, offd[1], trace);

    for (i1=0; i1 < n1; i1++) {
	trace[i1] = offd[0][i1]+offd[1][i1];
    }

    trace[0] += eps2*t0;
    trace[1] += eps2*t1;
    trace[n1-2] += eps2*t2;
    trace[n1-1] += eps2*t3;

    sf_banded_solve (slv, trace);
}

void predict_set(float **dip1 /* dip field [n2][n1] */)
/*< set the local slopes for applying the linear operator >*/
{
    dip=dip1;
    if (NULL == tt) tt = ps_floatalloc(n1);
}

static void stepper(bool adj /* adjoint flag */,
		    int i2   /* trace number */)
{
    if (i2 < k2) {
	predict_step(adj,false,tt,dip[k2-1-i2]);
    } else if (i2 < n2+k2-1) {
	predict_step(adj,true,tt,dip[i2-k2]);
    } else {
	predict_step(adj,false,tt,dip[2*n2+k2-3-i2]);
    }
}

void predict_lop(bool adj, bool add, int nx, int ny, float *xx, float *yy)
/*< linear operator >*/
{
    int i1, i2;

//     if (nx != ny || nx != n1*n2) sf_error("%s: Wrong dimensions",__FILE__);

    ps_adjnull(adj,add,nx,ny,xx,yy);

    for (i1=0; i1 < n1; i1++) {
	tt[i1] = 0.;
    }

    if (adj) {
	for (i2=n2-1; i2 >= 0; i2--) {
	    predict_step(true,true,tt,dip[i2]);
	    for (i1=0; i1 < n1; i1++) {
		tt[i1] += yy[i1+i2*n1];
	    }
	    for (i1=0; i1 < n1; i1++) {
		xx[i1+i2*n1] += tt[i1];
	    }
	}
    } else {
	for (i2=0; i2 < n2; i2++) {
	    for (i1=0; i1 < n1; i1++) {
		tt[i1] += xx[i1+i2*n1];
	    }
	    for (i1=0; i1 < n1; i1++) {
		yy[i1+i2*n1] += tt[i1];
	    }
	    predict_step(false,true,tt,dip[i2]);
	}
    }
}

void predicter_lop(bool adj, bool add, int nx, int ny, float *xx, float *yy)
/*< linear operator >*/
{
    int i1, i2;

//     if (nx != ny || nx != n1*(n2+2*k2)) 
// 	sf_error("%s: Wrong dimensions",__FILE__);

    ps_adjnull(adj,add,nx,ny,xx,yy);

    for (i1=0; i1 < n1; i1++) {
	tt[i1] = 0.;
    }

    if (adj) {
	for (i2=n2+2*k2-1; i2 >= 0; i2--) {
	    stepper(true,i2);
	    for (i1=0; i1 < n1; i1++) {
		tt[i1] += yy[i1+i2*n1];
	    }
	    for (i1=0; i1 < n1; i1++) {
		xx[i1+i2*n1] += tt[i1];
	    }
	}
    } else {
	for (i2=0; i2 < n2+2*k2; i2++) {
	    for (i1=0; i1 < n1; i1++) {
		tt[i1] += xx[i1+i2*n1];
	    }
	    for (i1=0; i1 < n1; i1++) {
		yy[i1+i2*n1] += tt[i1];
	    }
	    stepper(false,i2);
	}
    }
}

void subtracter_lop(bool adj, bool add, int nx, int ny, float *xx, float *yy)
/*< linear operator >*/
{
    int i1, i2, j2, m2;

//     if (nx != ny || nx != n1*(n2+2*k2)) 
// 	sf_error("%s: Wrong dimensions",__FILE__);

    ps_adjnull(adj,add,nx,ny,xx,yy);

    if (adj) {
	for (j2=0; j2 < n2+2*k2; j2++) {
	    i2=j2+k2;
	    if (i2 < n2+2*k2) {
		for (i1=0; i1 < n1; i1++) {
		    tt[i1] = yy[i1+i2*n1];
		}
		for (m2=i2-1; m2 >= j2; m2--) {
		    stepper(true,m2);
		}
		for (i1=0; i1 < n1; i1++) {
		    xx[i1+j2*n1] += yy[i1+j2*n1]-tt[i1];
		}
	    } else {
		for (i1=0; i1 < n1; i1++) {
		    xx[i1+j2*n1] += yy[i1+j2*n1];
		}
	    }
	}
    } else {
	for (i2=0; i2 < n2+2*k2; i2++) { 
	    j2=i2-k2;
	    if (j2 >=0) {
		for (i1=0; i1 < n1; i1++) {
		    tt[i1] = xx[i1+j2*n1];
		}
		for (m2=j2; m2 < i2; m2++) {
		    stepper(false,m2);
		}
		for (i1=0; i1 < n1; i1++) {
		    yy[i1+i2*n1] += xx[i1+i2*n1]-tt[i1];
		}
	    } else {
		for (i1=0; i1 < n1; i1++) {
		    yy[i1+i2*n1] += xx[i1+i2*n1];
		}
	    }
	}
    }
}

void subtract_lop(bool adj, bool add, int nx, int ny, float *xx, float *yy)
/*< linear operator >*/
{
    int i1, i2, j2, m2;

//     if (nx != ny || nx != n1*n2) sf_error("%s: Wrong dimensions",__FILE__);

    ps_adjnull(adj,add,nx,ny,xx,yy);

    if (adj) {
	for (j2=0; j2 < n2; j2++) {
	    i2=j2+k2;
	    if (i2 < n2) {
		for (i1=0; i1 < n1; i1++) {
		    tt[i1] = yy[i1+i2*n1];
		}
		for (m2=i2-1; m2 >= j2; m2--) {
		    predict_step(true,true,tt,dip[m2]);
		}
		for (i1=0; i1 < n1; i1++) {
		    xx[i1+j2*n1] += yy[i1+j2*n1]-tt[i1];
		}
	    } else {
		for (i1=0; i1 < n1; i1++) {
		    xx[i1+j2*n1] += yy[i1+j2*n1];
		}
	    }
	}
    } else {
	for (i2=0; i2 < n2; i2++) { 
	    j2=i2-k2;
	    if (j2 >=0) {
		for (i1=0; i1 < n1; i1++) {
		    tt[i1] = xx[i1+j2*n1];
		}
		for (m2=j2; m2 < i2; m2++) {
		    predict_step(false,true,tt,dip[m2]);
		}
		for (i1=0; i1 < n1; i1++) {
		    yy[i1+i2*n1] += xx[i1+i2*n1]-tt[i1];
		}
	    } else {
		for (i1=0; i1 < n1; i1++) {
		    yy[i1+i2*n1] += xx[i1+i2*n1];
		}
	    }
	}
    }
}



/*pwspray.c*/
    
int pwspray_init(int nr      /* spray radius */, 
		 int nt      /* trace length */, 
		 int n       /* number of traces */,
		 int order   /* PWD order */,
		 float eps   /* regularization */)		 
/*< initialize >*/
{
    n1=nt;
    n2=n;

    ns=nr;
    ns2=2*ns+1;

    predict_init (eps*eps, order, 1, false);
    trace = ps_floatalloc(n1);

    return ns2;
}

void pwspray_set(float **dip /* local slope */)
/*< set local slope >*/
{
    pspr = dip;
}


void pwspray_close(void)
/*< free allocated storage >*/
{
    predict_close();
    free(trace);
}

void pwspray_lop(bool adj, bool add, int n, int nu, float* u1, float *u)
/*< linear operator >*/
{
    int i, is, ip, j, i1;

//     if (n  != n1*n2) sf_error("%s: wrong size %d != %d*%d",__FILE__,n, n1,n2);
//     if (nu != n*ns2) sf_error("%s: wrong size %d != %d*%d",__FILE__,nu,n,ns2);

    ps_adjnull(adj,add,n,nu,u1,u);

    for (i=0; i < n2; i++) { 	
	if (adj) {
	    for (i1=0; i1 < n1; i1++) {
		trace[i1] = 0.0f;
	    }

	    /* predict forward */
	    for (is=ns-1; is >= 0; is--) {
		ip = i+is+1;
		if (ip >= n2) continue;
		j = ip*ns2+ns+is+1;
		for (i1=0; i1 < n1; i1++) {
		    trace[i1] += u[j*n1+i1];
		}
		predict_step(true,true,trace,pspr[ip-1]);
	    }

	    for (i1=0; i1 < n1; i1++) {
		u1[i*n1+i1] += trace[i1];
		trace[i1] = 0.0f;
	    }

	    /* predict backward */
	    for (is=ns-1; is >= 0; is--) {
		ip = i-is-1;
		if (ip < 0) continue;
		j = ip*ns2+ns-is-1;
		for (i1=0; i1 < n1; i1++) {
		    trace[i1] += u[j*n1+i1];
		}
		predict_step(true,false,trace,pspr[ip]);
	    }
	    
	    for (i1=0; i1 < n1; i1++) {
		u1[i*n1+i1] += trace[i1];
		trace[i1] = u[(i*ns2+ns)*n1+i1];
		u1[i*n1+i1] += trace[i1];
	    }
	    
	} else {

	    for (i1=0; i1 < n1; i1++) {
		trace[i1] = u1[i*n1+i1];
		u[(i*ns2+ns)*n1+i1] += trace[i1];
	    }

            /* predict forward */
	    for (is=0; is < ns; is++) {
		ip = i-is-1;
		if (ip < 0) break;
		j = ip*ns2+ns-is-1;
		predict_step(false,false,trace,pspr[ip]);
		for (i1=0; i1 < n1; i1++) {
		    u[j*n1+i1] += trace[i1];
		}
	    }

	    for (i1=0; i1 < n1; i1++) {
		trace[i1] = u1[i*n1+i1];
	    }
	    
	    /* predict backward */
	    for (is=0; is < ns; is++) {
		ip = i+is+1;
		if (ip >= n2) break;
		j = ip*ns2+ns+is+1;
		predict_step(false,true,trace,pspr[ip-1]);
		for (i1=0; i1 < n1; i1++) {
		    u[j*n1+i1] += trace[i1];
		}
	    }
	}
    }
}




void pwsmooth_init(int nr      /* spray radius */,
		   int m1      /* trace length */,
		   int m2      /* number of traces */,
		   int order   /* PWD order */,
		   float eps   /* regularization */)
/*< initialize >*/
{
    int is;

	ns=nr;
    n1 = m1;
    n2 = m2;
    n12 = n1*n2;

    ns2 = pwspray_init(nr,n1,n2,order,eps);

    u = ps_floatalloc3(n1,ns2,n2);
    w = ps_floatalloc(ns2);
    w1 = ps_floatalloc2(n1,n2);

    for (is=0; is < ns2; is++) {
	w[is]=ns+1-SF_ABS(is-ns);
    }

    /* Normalization */
    t = ps_floatalloc(n12);
}

void pwsmooth_lop(bool adj, bool add, 
		  int nin, int nout, float* trace, float *smooth)
/*< linear operator >*/
{
    int i1, i2, is;
    float ws;

//     if (nin != nout || nin != n1*n2) 
// 	sf_error("%s: wrong size %d != %d",__FILE__,nin,nout);
    
    ps_adjnull(adj,add,nin,nout,trace,smooth);

    if (adj) {
	for (i2=0; i2 < n2; i2++) {
	    for (i1=0; i1 < n1; i1++) {
		ws=w1[i2][i1]; 
		for (is=0; is < ns2; is++) {
		    u[i2][is][i1] = smooth[i2*n1+i1]*w[is]*ws;
		}
	    }
	}

	pwspray_lop(true,  true,  nin, nin*ns2, trace, u[0][0]);
    } else {
	pwspray_lop(false, false, nin, nin*ns2, trace, u[0][0]);

	for (i2=0; i2 < n2; i2++) {
	    for (i1=0; i1 < n1; i1++) {
		ws=w1[i2][i1]; 
		for (is=0; is < ns2; is++) {
		    smooth[i2*n1+i1] += u[i2][is][i1]*w[is]*ws;
		}
	    }
	}
    }
}


void pwsmooth_set(float **dip /* local slope */)
/*< set local slope >*/
{
    int i1;
    
    pwspray_set(dip);

    for (i1=0; i1 < n12; i1++) {
	w1[0][i1]=1.0f;
    }

    pwsmooth_lop(false,false,n12,n12,w1[0],t);

    for (i1=0; i1 < n12; i1++) {
	if (0.0f != t[i1]) {
	    w1[0][i1]=1.0/t[i1];
	} else {
	    w1[0][i1]=0.0f;
	}
    }
}

void pwsmooth_close(void)
/*< free allocated storage >*/
{
    free(**u);
    free(*u);
    free(u);
    free(w);
    free(*w1);
    free(w1);
    free(t);
    pwspray_close();
}

float sf_quantile(int q    /* quantile */, 
		  int n    /* array length */, 
		  float* a /* array [n] */) 
/*< find quantile (caution: a is changed) >*/ 
{
    float *i, *j, ak, *low, *hi, buf, *k;

    low=a;
    hi=a+n-1;
    k=a+q; 
    while (low<hi) {
	ak = *k;
	i = low; j = hi;
	do {
	    while (*i < ak) i++;     
	    while (*j > ak) j--;     
	    if (i<=j) {
		buf = *i;
		*i++ = *j;
		*j-- = buf;
	    }
	} while (i<=j);
	if (j<k) low = i; 
	if (k<i) hi = j;
    }
    return (*k);
}

static float *extendt, *temp1, *trace2, *win_len;
static int mfn1, mfn2;
static int nfilter2, nfilter, nfilter0, axis, ifbound;
static int l1,l2,l3,l4;

void mf_init(int n1, int n2, int nfw, int axis0, bool ifbound0)
{
	mfn1=n1;
	mfn2=n2;
	
	nfilter=nfw;
	nfilter2=(nfw-1)/2;
	
	axis=axis0;
	ifbound=ifbound0;
	
	if(axis==1)
	{
	extendt=ps_floatalloc((n1+2*nfilter2)*n2 );
	}else{
	extendt=ps_floatalloc(n1*(n2+2*nfilter2) );
	}
	temp1=ps_floatalloc(nfilter);
}

void svmf_init(int n1, int n2, int nfw, int axis0, bool ifbound0)
{

	l1=2,l2=0,l3=2,l4=4;
	
	mfn1=n1;
	mfn2=n2;
	
	nfilter0=nfw;
	nfilter=nfw+l1;
	nfilter2=(nfilter-1)/2;
	
	axis=axis0;
	ifbound=ifbound0;
	
	if(axis==1)
	{
	extendt=ps_floatalloc((n1+2*nfilter2)*n2 );
	}else{
	extendt=ps_floatalloc(n1*(n2+2*nfilter2) );
	}
	temp1=ps_floatalloc(nfilter);
	
	trace2=ps_floatalloc(n1*n2);
	
	win_len=ps_floatalloc(n1*n2);
}

void boundary(float* tempt,float* extendt)
/*<extend seismic data>*/
{
	int m=nfilter2;
    int i,j;
    int n1,n2;
    n1=mfn1;
    n2=mfn2;
    
    if(axis==1)
    {
    for(i=0;i<(n1+2*m)*(n2);i++){
	extendt[i]=0.0;
    }
    /*extend the number of samples*/
    for(i=0;i<n2;i++){
	for(j=0;j<m;j++){
	    if (ifbound){
		extendt[(n1+2*m)*i+j]=tempt[n1*i+0];
	    }
	    else{
		extendt[(n1+2*m)*i+j]=0.0;
	    }
	}
    }
    for(i=0;i<n2;i++){
	for(j=0;j<n1;j++){
	    extendt[(n1+2*m)*i+j+m]=tempt[n1*i+j];
	}
    }
    for(i=0;i<n2;i++){
	for(j=0;j<m;j++){
	    if (ifbound){
		extendt[(n1+2*m)*i+j+n1+m]=tempt[n1*i+n1-1];
	    }
	    else{
		extendt[(n1+2*m)*i+j+n1+m]=0.0;
	    }
	}
    }
    }else /*suppose axis is either 1 or 2*/
    {
    for(i=0;i<n1*(n2+2*m);i++){
	extendt[i]=0.0;
    }
    
    /*extend the number of samples*/
    for(i=0;i<m;i++){
	for(j=0;j<n1;j++){
	    if (ifbound){
		extendt[n1*i+j]=tempt[n1*0+j];
	    }
	    else{
		extendt[n1*i+j]=0.0;
	    }
	}
    }
    for(i=0;i<n2;i++){
	for(j=0;j<n1;j++){
	    extendt[n1*(i+m)+j]=tempt[n1*i+j];
	}
    }
    for(i=0;i<m;i++){
	for(j=0;j<n1;j++){
	    if (ifbound){
		extendt[n1*(i+n2+m)+j]=tempt[n1*(n2-1)+j];
	    }
	    else{
		extendt[n1*(i+n2+m)+j]=0.0;
	    }
	}
    }
    
    }
}

void mf(float *trace)
{

	int n1,n2,k,i,j;
	n1=mfn1;
	n2=mfn2;
	int m=nfilter2;
	
	boundary(trace,extendt);

	/************1D median filter****************/
	if(axis==1)
	{
		for(i=0;i<n2;i++){
	    for(j=0;j<n1;j++){
		for(k=0;k<nfilter;k++){
		    temp1[k]=extendt[(n1+2*m)*i+j+k];
		}
		trace[n1*i+j]=sf_quantile(m,nfilter,temp1); 
	    }
	    }
	}else
	{
	if(axis==2)
	{
		for(i=0;i<n2;i++){
	    for(j=0;j<n1;j++){
		for(k=0;k<nfilter;k++){
		    temp1[k]=extendt[n1*(i+k)+j];
		}
		trace[n1*i+j]=sf_quantile(m,nfilter,temp1); 
	    }
	    }
		
	}else{
	printf("Wrong axis number for 2D dataset");
	}
	}
}

void svmf(float *trace)
{

	int n1,n2,k,i,j;
	n1=mfn1;
	n2=mfn2;
	int m=nfilter2, m2=(nfilter0-1)/2;
	
	float sum=0, avg=0;
	
	for(i=0;i<n1*n2;i++)
	trace2[i]=trace[i];
	
	boundary(trace,extendt);
	
	/************1D median filter****************/
	if(axis==1)
	{
		for(i=0;i<n2;i++){
	    for(j=0;j<n1;j++){
		for(k=0;k<nfilter;k++){
		    temp1[k]=extendt[(n1+2*m)*i+j+k+m-m2];
		}
		trace2[n1*i+j]=sf_quantile(m,nfilter,temp1); 
	    }
	    }
	}else
	{
	if(axis==2)
	{
		for(i=0;i<n2;i++){
	    for(j=0;j<n1;j++){
		for(k=0;k<nfilter;k++){
		    temp1[k]=extendt[n1*(m+i+k-m2)+j];
		}
		trace2[n1*i+j]=sf_quantile(m,nfilter,temp1); 
	    }
	    }
		
	}else{
	printf("Wrong axis number for 2D dataset");
	}
	}
	

	
	for(i=0;i<n1*n2;i++)
		sum=sum+fabs(trace2[i]);
	avg=sum/(n1*n2);
	
	for(i=0;i<n1*n2;i++)
	{
		if(fabs(trace[i])<avg)
		{
			if(fabs(trace[i])<avg/2)
				win_len[i] = nfilter0+l1;
			else
				win_len[i] = nfilter0+l2;
		}else{
			if(fabs(trace[i])>avg*2)
				win_len[i] = nfilter0-l4;
			else
				win_len[i] = nfilter0-l3;
		}
	}


	/************1D median filter****************/
	if(axis==1)
	{
		for(i=0;i<n2;i++){
	    for(j=0;j<n1;j++){
	    
	    m2=(win_len[i*n1+j]-1)/2;
		for(k=0;k<win_len[i*n1+j];k++){
		    temp1[k]=extendt[(n1+2*m)*i+j+k+m-m2];
		}
		trace[n1*i+j]=sf_quantile(m2,win_len[i*n1+j],temp1); 
	    }
	    }
	}else
	{
	if(axis==2)
	{
		for(i=0;i<n2;i++){
	    for(j=0;j<n1;j++){
	    m2=(win_len[i*n1+j]-1)/2;
		for(k=0;k<win_len[i*n1+j];k++){
		    temp1[k]=extendt[n1*(i+m+k-m2)+j];
		}
		trace[n1*i+j]=sf_quantile(m2,win_len[i*n1+j],temp1); 
	    }
	    }
	}else{
	printf("Wrong axis number for 2D dataset");
	}
	}
}

static PyObject *csomean2d(PyObject *self, PyObject *args){
		
    /*Below is the input part*/
    int f3,f4,f5,f6,f7,f8;
    float f9;
    int f10;
    
	/**initialize data input**/
    int nd, nd2;
    
    PyObject *f1=NULL;
    PyObject *f2=NULL;
    PyObject *arrf1=NULL;
    PyObject *arrf2=NULL;
    
	PyArg_ParseTuple(args, "OOiiiiiifi", &f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8, &f9, &f10);

    int i1,i2,i3;
    int n123, niter, order, nj1,nj2, i,j, liter, dim;
    int n[PS_MAX_DIM], rect[3], n4, nr, ir; 
    float p0, q0, *u, *p, *pi=NULL, *qi=NULL;
    float pmin, pmax, qmin, qmax, eps;
    char key[4];
    bool verb, both, adj;

    int n1, n2, n3, ns;
    float *input, *smooth, ***slope;
    
	n1=f3;
	n2=f4;
	n3=f5;
	n123=n1*n2*n3;
	
	ns=f6;
	order=f7;/*default order=1*/
	adj=f8; /*adjoint flag*/
	eps=f9; /*regularization*/
	verb=f10;
	
    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
	arrf2 = PyArray_FROM_OTF(f2, NPY_FLOAT, NPY_IN_ARRAY);

    nd2=PyArray_NDIM(arrf1);
    npy_intp *sp=PyArray_SHAPE(arrf1);

	
    if (*sp != n123)
    {
    	printf("Dimension mismatch, N_input = %d, N_data = %d\n", *sp, n123);
    	return NULL;
    }

    input = ps_floatalloc(n123);
    smooth = ps_floatalloc(n123);
    slope = ps_floatalloc3(n1,n2,n3);
    
    /*reading data*/
    for (i=0; i<n123; i++)
    {
        input[i]=*((float*)PyArray_GETPTR1(arrf1,i));
        smooth[i]=0.0;
    }

	for (i1=0;i1<n1;i1++)
	for (i2=0;i2<n2;i2++)
	for (i3=0;i3<n3;i3++)
	{
		i=i3*n2*n1+i2*n1+i1;
		slope[i3][i2][i1] = *((float*)PyArray_GETPTR1(arrf2,i));
	}

    pwsmooth_init(ns, n1, n2, order, eps);

    for (i3=0; i3 < n3; i3++) {
	if (verb) printf("slice %d of %d;\n",i3+1,n3);

	pwsmooth_set(slope[i3]);

	if (adj) {
	    pwsmooth_lop(true,false,n1*n2,n1*n2,smooth+i3*n1*n2,input+i3*n1*n2);
	} else {
	    pwsmooth_lop(false,false,n1*n2,n1*n2,input+i3*n1*n2,smooth+i3*n1*n2);
	}

    }
    
    /*Below is the output part*/
    PyArrayObject *vecout;
	npy_intp dims[2];
	dims[0]=n123;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<dims[0];i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = smooth[i];


	return PyArray_Return(vecout);
	
}

static PyObject *csomf2d(PyObject *self, PyObject *args){
		
    /*Below is the input part*/
    int f3,f4,f5,f6,f7,f8;
    float f9;
    int f10;
    int nmf, option=0;
    
	/**initialize data input**/
    int nd, nd2;
    
    PyObject *f1=NULL;
    PyObject *f2=NULL;
    PyObject *arrf1=NULL;
    PyObject *arrf2=NULL;
    
	PyArg_ParseTuple(args, "OOiiiiiiifi", &f1, &f2, &f3, &f4, &f5, &f6, &nmf, &option, &f7, &f9, &f10);

    int i1,i2,i3;
    int n123, niter, order, nj1,nj2, i,j, liter, dim;
    int n[PS_MAX_DIM], rect[3], n4, nr, ir; 
    float p0, q0, *p, *pi=NULL, *qi=NULL;
    float pmin, pmax, qmin, qmax, eps;
    char key[4];
    bool verb, both, adj;

    int n1, n2, n3, ns;
    float *input, *smooth, ***slope;
    
	n1=f3;
	n2=f4;
	n3=f5;
	n123=n1*n2*n3;
	
	ns=f6;
	order=f7;/*default order=1*/
	eps=f9; /*regularization*/
	verb=f10;
	
    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
	arrf2 = PyArray_FROM_OTF(f2, NPY_FLOAT, NPY_IN_ARRAY);

    nd2=PyArray_NDIM(arrf1);
    npy_intp *sp=PyArray_SHAPE(arrf1);
	
    if (*sp != n123)
    {
    	printf("Dimension mismatch, N_input = %d, N_data = %d\n", *sp, n123);
    	return NULL;
    }

    input = ps_floatalloc(n123);
    smooth = ps_floatalloc(n123);
    slope = ps_floatalloc3(n1,n2,n3);
    
    /*reading data*/
    for (i=0; i<n123; i++)
    {
        input[i]=*((float*)PyArray_GETPTR1(arrf1,i));
        smooth[i]=0.0;
    }

	for (i1=0;i1<n1;i1++)
	for (i2=0;i2<n2;i2++)
	for (i3=0;i3<n3;i3++)
	{
		i=i3*n2*n1+i2*n1+i1;
		slope[i3][i2][i1] = *((float*)PyArray_GETPTR1(arrf2,i));
	}

    pwsmooth_init(ns, n1, n2, order, eps);
	int np=2*ns+1;
	int k;
    float *tt;
    float sum;
    tt=ps_floatalloc(n1*np);
    
    for (i3=0; i3 < n3; i3++) {
	if (verb) printf("slice %d of %d;\n",i3+1,n3);

	pwsmooth_set(slope[i3]);
	
	pwspray_lop(false, false, n1*n2, n1*n2*np, input+i3*n1*n2, u[0][0]);
	

	    
	/*below is MF/SVMF*/
    if(option==1)
    {
    mf_init(n1, np, nmf, 2, 1);
    	printf("running MF\n");
    }
    else
    {
    svmf_init(n1, np, nmf, 2, 1);
    	printf("running SVMF\n");
    }
    



    for(i=0;i<n2;i++)
    {
    	sum=0;
    	for(j=0;j<n1*np;j++)
    	{tt[j]=u[i][0][j];sum=sum+tt[j];}
    	
    	if(option==1)
    	mf(tt);
    	else
    	svmf(tt);
    	
    	sum=0;
    	for(j=0;j<n1*np;j++)
    	{u[i][0][j]=tt[j];sum=sum+tt[j];}

	for(k=0;k<n1;k++)
	smooth[i*n1+k+i3*n1*n2] = u[i][(np-1)/2][k];
    }
    

    }

    
    

    /*Below is the output part*/
    PyArrayObject *vecout;
	npy_intp dims[2];
	dims[0]=n123;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<dims[0];i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = smooth[i];


	return PyArray_Return(vecout);
	
}







// documentation for each functions.
static char sofcfun_document[] = "Document stuff for dip...";

// defining our functions like below:
// function_name, function, METH_VARARGS flag, function documents
static PyMethodDef functions[] = {
  {"csomean2d", csomean2d, METH_VARARGS, sofcfun_document},
  {"csomf2d", csomf2d, METH_VARARGS, sofcfun_document},
  {NULL, NULL, 0, NULL}
};

// initializing our module informations and settings in this structure
// for more informations, check head part of this file. there are some important links out there.
static struct PyModuleDef sofcfunModule = {
  PyModuleDef_HEAD_INIT, // head informations for Python C API. It is needed to be first member in this struct !!
  "sofcfun",  // module name
  NULL, // means that the module does not support sub-interpreters, because it has global state.
  -1,
  functions  // our functions list
};

// runs while initializing and calls module creation function.
PyMODINIT_FUNC PyInit_sofcfun(void){
  
    PyObject *module = PyModule_Create(&sofcfunModule);
    import_array();
    return module;
}
