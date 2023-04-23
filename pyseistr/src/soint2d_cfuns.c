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


#define ps_MAX(a,b) ((a) < (b) ? (b) : (a))
#define ps_MIN(a,b) ((a) < (b) ? (a) : (b))
#define ps_ABS(a)   ((a) >= 0  ? (a) : (-(a)))

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


static const bool *mmask;

void ps_mask_init(const bool *m_in)
/*< initialize with mask >*/
{
    mmask = m_in;
}

void ps_mask_lop(bool adj, bool add, int nx, int ny, float *x, float *y)
/*< linear operator >*/
{
    int ix;

//     if (nx != ny) ps_error("%s: wrong size: %d != %d",nx,ny);

    ps_adjnull (adj,add,nx,ny,x,y);

    for (ix=0; ix < nx; ix++) {
	if (mmask[ix]) {
	    if (adj) x[ix] += y[ix];
	    else     y[ix] += x[ix];
	}
    }
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

#ifndef _allp2_h

typedef struct Allpass2 *allpas2;
/* abstract data type */
/*^*/

#endif

struct Allpass2 {
    int nx, ny, nw, nj;
    bool drift;
    float *flt, **pp;
};

static allpas2 ap2;

allpas2 allpass2_init(int nw         /* filter order */, 
		      int nj         /* filter step */, 
		      int nx, int ny /* data size */, 
		      bool drift     /* shit filter */,
		      float **pp     /* dip [ny][nx] */) 
/*< Initialize >*/
{
    allpas2 ap;
    
    ap = (allpas2) ps_alloc(1,sizeof(*ap));
    
    ap->nw = nw;
    ap->nj = nj;
    ap->nx = nx;
    ap->ny = ny;
    ap->drift = drift;
    ap->pp = pp;

    ap->flt = ps_floatalloc(2*nw+1);
    
    apfilt_init(nw);
    return ap;
}

void allpass2_close(allpas2 ap)
/*< free allocated storage >*/
{
    apfilt_close();
    free(ap->flt);
    free(ap);
}

void allpass22_init (allpas2 ap1)
/*< Initialize linear operator >*/
{
    ap2 = ap1;
}

void allpass21_lop (bool adj, bool add, int n1, int n2, float* xx, float* yy)
/*< PWD as linear operator >*/
{

    int i, ix, iy, iw, is, nx, ny, id;

    ps_adjnull(adj,add,n1,n2,xx,yy);
  
    nx = ap2->nx;
    ny = ap2->ny;

    for (iy=0; iy < ny-1; iy++) {
	for (ix = ap2->nw*ap2->nj; ix < nx-ap2->nw*ap2->nj; ix++) {
	    i = ix + iy*nx;

	    if (ap2->drift) {
		id = PS_NINT(ap2->pp[iy][ix]);
		if (ix-ap2->nw*ap2->nj-id < 0 || 
		    ix+ap2->nw*ap2->nj-id >= nx) continue;

		passfilter(ap2->pp[iy][ix]-id, ap2->flt);		
		
		for (iw = 0; iw <= 2*ap2->nw; iw++) {
		    is = (iw-ap2->nw)*ap2->nj;
		    
		    if (adj) {
			xx[i+is+nx] += yy[i]*ap2->flt[iw];
			xx[i-is-id] -= yy[i]*ap2->flt[iw];
		    } else {
			yy[i] += (xx[i+is+nx] - xx[i-is-id]) * ap2->flt[iw];
		    }
		}
	    } else {
		passfilter(ap2->pp[iy][ix], ap2->flt);
		
		for (iw = 0; iw <= 2*ap2->nw; iw++) {
		    is = (iw-ap2->nw)*ap2->nj;
		    
		    if (adj) {
			xx[i+is+nx] += yy[i]*ap2->flt[iw];
			xx[i-is]    -= yy[i]*ap2->flt[iw];
		    } else {
			yy[i] += (xx[i+is+nx] - xx[i-is]) * ap2->flt[iw];
		    }
		}
	    }
	}
    }
}

void allpass21 (bool der          /* derivative flag */, 
		const allpas2 ap /* PWD object */, 
		float** xx        /* input */, 
		float** yy        /* output */)
/*< plane-wave destruction >*/
{
    int ix, iy, iw, is, id;

    for (iy=0; iy < ap->ny; iy++) {
	for (ix=0; ix < ap->nx; ix++) {
	    yy[iy][ix] = 0.;
	}
    }
  
    for (iy=0; iy < ap->ny-1; iy++) {
	for (ix = ap->nw*ap->nj; ix < ap->nx-ap->nw*ap->nj; ix++) {
	    if (ap->drift) {
		id = PS_NINT(ap->pp[iy][ix]);
		if (ix-ap->nw*ap->nj-id < 0 || 
		    ix+ap->nw*ap->nj-id >= ap->nx) continue;

		if (der) {
		    aderfilter(ap->pp[iy][ix]-id, ap->flt);
		} else {
		    passfilter(ap->pp[iy][ix]-id, ap->flt);
		}
		
		for (iw = 0; iw <= 2*ap->nw; iw++) {
		    is = (iw-ap->nw)*ap->nj;
		    
		    yy[iy][ix] += (xx[iy+1][ix+is] - 
				   xx[iy  ][ix-is-id]) * ap->flt[iw];
		}
	    } else {
		if (der) {
		    aderfilter(ap->pp[iy][ix], ap->flt);
		} else {
		    passfilter(ap->pp[iy][ix], ap->flt);
		}
		
		for (iw = 0; iw <= 2*ap->nw; iw++) {
		    is = (iw-ap->nw)*ap->nj;
		    
		    yy[iy][ix] += (xx[iy+1][ix+is] - 
				   xx[iy  ][ix-is]) * ap->flt[iw];
		}
	    }
	}
    }
}


/*from cblas */
double ps_cblas_dsdot(int n, const float *x, int sx, const float *y, int sy)
/*< x'y float -> double >*/
{
    int i, ix, iy;
    double dot;

    dot = 0.;

    for (i=0; i < n; i++) {
	ix = i*sx;
	iy = i*sy;
        dot += (double) x[ix] * y[iy];
    }

    return dot;
}

void ps_cblas_sscal(int n, float alpha, float *x, int sx)
/*< x = alpha*x >*/
{
    int i, ix;

    for (i=0; i < n; i++) {
        ix = i*sx;
	x[ix] *= alpha;
    }
}


void ps_cblas_saxpy(int n, float a, const float *x, int sx, float *y, int sy)
/*< y += a*x >*/
{
    int i, ix, iy;

    for (i=0; i < n; i++) {
	ix = i*sx;
	iy = i*sy;
	y[iy] += a * x[ix];
    }
}

void ps_cblas_sswap(int n, float *x, int sx, float* y, int sy) 
/*< swap x and y >*/
{
    int i, ix, iy;
    float t;

    for (i=0; i < n; i++) {
	ix = i*sx;
	iy = i*sy;
	t = x[ix];
	x[ix] = y[iy];
	y[iy] = t;
    }
}

float ps_cblas_sdot(int n, const float *x, int sx, const float *y, int sy)
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

float ps_cblas_snrm2 (int n, const float* x, int sx) 
/*< sum x_i^2 >*/
{
    int i, ix;
    float xn;

    xn = 0.0;

    for (i=0; i < n; i++) {
	ix = i*sx;
	xn += x[ix]*x[ix];
    }
    return xn;
}

static const float EPSILON=1.e-12;

static float* S;  /* model step */
static float* Ss; /* residual step */
static bool Allocated = false; /* if S and Ss are allocated */

void ps_cgstep( bool forget     /* restart flag */, 
		int nx, int ny  /* model size, data size */, 
		float* x        /* current model [nx] */, 
		const float* g  /* gradient [nx] */, 
		float* rr       /* data residual [ny] */, 
		const float* gg /* conjugate gradient [ny] */) 
/*< Step of conjugate-gradient iteration. >*/
{
    double sds, gdg, gds, determ, gdr, sdr, alfa, beta;
    int i;
    if (!Allocated) {
	Allocated = forget = true;
	S  = ps_floatalloc (nx);
	Ss = ps_floatalloc (ny);
    }
    if (forget) {
	for (i = 0; i < nx; i++) S[i] = 0.;
	for (i = 0; i < ny; i++) Ss[i] = 0.;
	beta = 0.0;
	alfa = ps_cblas_dsdot( ny, gg, 1, gg, 1);
	/* Solve G . ( R + G*alfa) = 0 */
	if (alfa <= 0.) return;
	alfa = - ps_cblas_dsdot( ny, gg, 1, rr, 1) / alfa;
    } else {
	/* search plane by solving 2-by-2
	   G . (R + G*alfa + S*beta) = 0
	   S . (R + G*alfa + S*beta) = 0 */
	gdg = ps_cblas_dsdot( ny, gg, 1, gg, 1);       
	sds = ps_cblas_dsdot( ny, Ss, 1, Ss, 1);       
	gds = ps_cblas_dsdot( ny, gg, 1, Ss, 1);       
	if (gdg == 0. || sds == 0.) return;
	determ = 1.0 - (gds/gdg)*(gds/sds);
	if (determ > EPSILON) determ *= gdg * sds;
	else determ = gdg * sds * EPSILON;
	gdr = - ps_cblas_dsdot( ny, gg, 1, rr, 1);
	sdr = - ps_cblas_dsdot( ny, Ss, 1, rr, 1);
	alfa = ( sds * gdr - gds * sdr ) / determ;
	beta = (-gds * gdr + gdg * sdr ) / determ;
    }
    ps_cblas_sscal(nx,beta,S,1);
    ps_cblas_saxpy(nx,alfa,g,1,S,1);

    ps_cblas_sscal(ny,beta,Ss,1);
    ps_cblas_saxpy(ny,alfa,gg,1,Ss,1);

    for (i = 0; i < nx; i++) {
	x[i] +=  S[i];
    }
    for (i = 0; i < ny; i++) {
	rr[i] += Ss[i];
    }
}

void ps_cgstep_close (void) 
/*< Free allocated space. >*/ 
{
    if (Allocated) {
	free (S);
	free (Ss);
	Allocated = false;
    }
}

static const float TOLERANCE=1.e-12;
typedef void (*ps_operator)(bool,bool,int,int,float*,float*);
typedef void (*ps_solverstep)(bool,int,int,float*,
			   const float*,float*,const float*);
typedef void (*ps_weight)(int,const float*,float*);
void ps_solver (ps_operator oper   /* linear operator */, 
		ps_solverstep solv /* stepping function */, 
		int nx             /* size of x */, 
		int ny             /* size of dat */, 
		float* x           /* estimated model */, 
		const float* dat   /* data */, 
		int niter          /* number of iterations */, 
		...                /* variable number of arguments */)
/*< Generic linear solver.
  ---
  Solves
  oper{x}    =~ dat
  ---
  The last parameter in the call to this function should be "end".
  Example: 
  ---
  ps_solver (oper_lop,ps_cgstep,nx,ny,x,y,100,"x0",x0,"end");
  ---
  Parameters in ...:
  ---
  "wt":     float*:         weight      
  "wght":   ps_weight wght: weighting function
  "x0":     float*:         initial model
  "nloper": ps_operator:    nonlinear operator
  "mwt":    float*:         model weight
  "verb":   bool:           verbosity flag
  "known":  bool*:          known model mask
  "nmem":   int:            iteration memory
  "nfreq":  int:            periodic restart
  "xmov":   float**:        model iteration
  "rmov":   float**:        residual iteration
  "err":    float*:         final error
  "res":    float*:         final residual
  >*/ 
{
	
    va_list args;
    char* par;
    float* wt = NULL;
    ps_weight wght = NULL;
    float* x0 = NULL;
    ps_operator nloper = NULL;
    float* mwt = NULL;
    bool verb = false;
    bool* known = NULL;
    int nmem = -1;
    int nfreq = 0;
    float** xmov = NULL;
    float** rmov = NULL;
    float* err = NULL;
    float* res = NULL;
    float* wht = NULL;
    float *g, *rr, *gg, *td = NULL, *g2 = NULL;
    float dpr, dpg, dpr0, dpg0;
    int i, iter; 
    bool forget = false;
	
    va_start (args, niter);
    for (;;) {
	par = va_arg (args, char *);
	if      (0 == strcmp (par,"end")) {break;}
	else if (0 == strcmp (par,"wt"))      
	{                    wt = va_arg (args, float*);}
	else if (0 == strcmp (par,"wght"))      
	{                    wght = va_arg (args, ps_weight);}
	else if (0 == strcmp (par,"x0"))      
	{                    x0 = va_arg (args, float*);}
	else if (0 == strcmp (par,"nloper"))      
	{                    nloper = va_arg (args, ps_operator);}
	else if (0 == strcmp (par,"mwt"))      
	{                    mwt = va_arg (args, float*);}
	else if (0 == strcmp (par,"verb"))      
	{                    verb = (bool) va_arg (args, int);}    
	else if (0 == strcmp (par,"known"))      
	{                    known = va_arg (args, bool*);}  
	else if (0 == strcmp (par,"nmem"))      
	{                    nmem = va_arg (args, int);}
	else if (0 == strcmp (par,"nfreq"))      
	{                    nfreq = va_arg (args, int);}
	else if (0 == strcmp (par,"xmov"))      
	{                    xmov = va_arg (args, float**);}
	else if (0 == strcmp (par,"rmov"))      
	{                    rmov = va_arg (args, float**);}
	else if (0 == strcmp (par,"err"))      
	{                    err = va_arg (args, float*);}
	else if (0 == strcmp (par,"res"))      
	{                    res = va_arg (args, float*);}
	else 
	{ printf("solver: unknown argument %s\n",par);}
    }
    va_end (args);
	
    g =  ps_floatalloc (nx);
    rr = ps_floatalloc (ny);
    gg = ps_floatalloc (ny);

    if (wt != NULL || wght != NULL) {
	td = ps_floatalloc (ny);
	if (wt != NULL) {
	    wht = wt;
	} else {
	    wht = ps_floatalloc (ny);
	    for (i=0; i < ny; i++) {
		wht[i] = 1.0;
	    }
	} 
    }
	
    if (mwt != NULL) {
	g2 = ps_floatalloc (nx);
    }

    for (i=0; i < ny; i++) {
	rr[i] = - dat[i];
    }
    if (x0 != NULL) {
	for (i=0; i < nx; i++) {
	    x[i] = x0[i];
	} 	
	if (mwt != NULL) {
	    for (i=0; i < nx; i++) {
		x[i] *= mwt[i];
	    }
	} 
	if (nloper != NULL) {
	    nloper (false, true, nx, ny, x, rr);
	} else {
	    oper (false, true, nx, ny, x, rr);
            
	}
    } else {
	for (i=0; i < nx; i++) {
	    x[i] = 0.0;
	} 
    }
	
    dpr0 = ps_cblas_snrm2(ny, rr, 1);
    dpg0 = 1.;

    for (iter=0; iter < niter; iter++) {
	if ( nmem >= 0) {  /* restart */
	    forget = (bool) (iter >= nmem);
	}
	if (wght != NULL && forget) {
	    wght (ny, rr, wht);
	}
	if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] *= wht[i];
		td[i] = rr[i]*wht[i];
	    }
      
	    oper (true, false, nx, ny, g, td);
	} else {
	    oper (true, false, nx, ny, g, rr);
	} 

	if (mwt != NULL) {
	    for (i=0; i < nx; i++) {
		g[i] *= mwt[i];
	    }
	}
	if (known != NULL) {
	    for (i=0; i < nx; i++) {
		if (known[i]) g[i] = 0.0;
	    }
	} 

	if (mwt != NULL) {
	    for (i=0; i < nx; i++) {
		g2[i] = g[i]*mwt[i];
	    }
	    oper (false, false, nx, ny, g2, gg);
	} else {
	    oper (false, false, nx, ny, g, gg);
	}

	if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		gg[i] *= wht[i];
	    }
	}
 
	if (forget && nfreq != 0) { /* periodic restart */
	    forget = (bool) (0 == (iter+1)%nfreq); 
	}
	
	if (iter == 0) {
	    dpg0  = ps_cblas_snrm2 (nx, g, 1);
	    dpr = 1.;
	    dpg = 1.;
	} else {
	    dpr = ps_cblas_snrm2 (ny, rr, 1)/dpr0;
	    dpg = ps_cblas_snrm2 (nx, g , 1)/dpg0;
	}    

	if (verb) 
	    printf ("iteration %d res %f mod %f grad %f\n",
			iter+1, dpr, ps_cblas_snrm2 (nx, x, 1), dpg);
	

	if (dpr < TOLERANCE || dpg < TOLERANCE) {
	    if (verb) 
		printf("convergence in %d iterations\n",iter+1);

	    if (mwt != NULL) {
		for (i=0; i < nx; i++) {
		    x[i] *= mwt[i];
		}
	    }
	    break;
	}

	solv (forget, nx, ny, x, g, rr, gg);
	forget = false;

	if (nloper != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = -dat[i]; 
	    }

	    if (mwt != NULL) {
		for (i=0; i < nx; i++) {
		    x[i] *= mwt[i];
		}
	    }
	    nloper (false, true, nx, ny, x, rr);
	} else if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = -dat[i]; 
	    }
	    if (mwt != NULL) {
		for (i=0; i < nx; i++) {
		    x[i] *= mwt[i];
		}
	    }
	    oper (false, true, nx, ny, x, rr);
	}  else if (mwt != NULL && (xmov != NULL || iter == niter-1)) {
	    for (i=0; i < nx; i++) {
		x[i] *= mwt[i];
	    }
	}

	if (xmov != NULL) {
	    for (i=0; i < nx; i++) {
		xmov[iter][i] =  x[i];
	    }
	}
	if (rmov != NULL) {
	    for (i=0; i < ny; i++) {
		rmov[iter][i] =  rr[i];
	    }
	}
    
	if (err != NULL) {
	    err[iter] = ps_cblas_snrm2(ny, rr, 1);
	}
    }

    if (res != NULL) {
	for (i=0; i < ny; i++) {
	    res[i] = rr[i];
	}
    }
  
    free (g);
    free (rr);
    free (gg);

    if (wht != NULL) {
	free (td);
	if (wt == NULL) {
	    free (wht);
	}
    }

    if (mwt != NULL) {
	free (g2);
    }
}

/*from pwd.c*/
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

/*from api/c/banded.c*/
#ifndef _ps_banded_h

typedef struct ps_Bands *ps_bands;
/* abstract data type */
/*^*/

#endif

struct ps_Bands {
    int n, band;
    float *d, **o;
};

ps_bands ps_banded_init (int n    /* matrix size */, 
			 int band /* band size */)
/*< initialize >*/
{
    ps_bands slv;
    int i;
    
    slv = (ps_bands) ps_alloc (1,sizeof(*slv));
    slv->o = (float**) ps_alloc (band,sizeof(float*));
    for (i = 0; i < band; i++) {
	slv->o[i] = ps_floatalloc (n-1-i);
    }
    slv->d = ps_floatalloc (n);
    slv->n = n;
    slv->band = band;

    return slv;
}

void ps_banded_define (ps_bands slv, 
		       float* diag  /* diagonal [n] */, 
		       float** offd /* off-diagonal [band][n] */)
/*< define the matrix >*/
{
    int k, m, m1, n, n1;
    float t;
    
    for (k = 0; k < slv->n; k++) {
	t = diag[k];
	m1 = ps_MIN(k,slv->band);
	for (m = 0; m < m1; m++)
	    t -= (slv->o[m][k-m-1])*(slv->o[m][k-m-1])*(slv->d[k-m-1]);
	slv->d[k] = t;
	n1 = ps_MIN(slv->n-k-1,slv->band);
	for (n = 0; n < n1; n++) {
	    t = offd[n][k];
	    m1 = ps_MIN(k,slv->band-n-1);
	    for (m = 0; m < m1; m++) {
		t -= (slv->o[m][k-m-1])*(slv->o[n+m+1][k-m-1])*(slv->d[k-m-1]);
	    }
	    slv->o[n][k] = t/slv->d[k];
	}
    }
}


void ps_banded_const_define (ps_bands slv, 
			     float diag        /* diagonal */, 
			     const float* offd /* off-diagonal [band] */)
/*< define matrix with constant diagonal coefficients >*/
{
    int k, m, m1, n, n1;
    float t;
    
    for (k = 0; k < slv->n; k++) {   
	t = diag;
	m1 = ps_MIN(k,slv->band);
	for (m = 0; m < m1; m++)
	    t -= (slv->o[m][k-m-1])*(slv->o[m][k-m-1])*(slv->d[k-m-1]);
	slv->d[k] = t;
	n1 = ps_MIN(slv->n-k-1,slv->band);
	for (n = 0; n < n1; n++) {
	    t = offd[n];
	    m1 = ps_MIN(k,slv->band-n-1);
	    for (m = 0; m < m1; m++) {
		t -= (slv->o[m][k-m-1])*(slv->o[n+m+1][k-m-1])*(slv->d[k-m-1]);
	    }
	    slv->o[n][k] = t/slv->d[k];
	}
    }
}

void ps_banded_const_define_eps (ps_bands slv, 
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
	m1 = ps_MIN(k,slv->band);
	for (m = 0; m < m1; m++)
	    t -= (slv->o[m][k-m-1])*(slv->o[m][k-m-1])*(slv->d[k-m-1]);
	slv->d[k] = t;
	n1 = ps_MIN(slv->n-k-1,slv->band);
	for (n = 0; n < n1; n++) {
	    t = offd[n];
	    m1 = ps_MIN(k,slv->band-n-1);
	    for (m = 0; m < m1; m++) {
		t -= (slv->o[m][k-m-1])*(slv->o[n+m+1][k-m-1])*(slv->d[k-m-1]);
	    }
	    slv->o[n][k] = t/slv->d[k];
	}
    }
}


void ps_banded_const_define_reflect (ps_bands slv, 
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

void ps_banded_solve (const ps_bands slv, float* b)
/*< invert (in place) >*/
{
    int k, m, m1;
    float t;

    for (k = 1; k < slv->n; k++) {
	t = b[k];
	m1 = ps_MIN(k,slv->band);
	for (m = 0; m < m1; m++)
	    t -= (slv->o[m][k-m-1]) * b[k-m-1];
	b[k] = t;
    }
    for (k = slv->n-1; k >= 0; k--) {
	t = b[k]/slv->d[k];
	m1 = ps_MIN(slv->n -k-1,slv->band);
	for (m = 0; m < m1; m++)
	    t -= slv->o[m][k] * b[k+m+1];
	b[k] = t;
    }
}

void ps_banded_close (ps_bands slv)
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


/*from predict.c*/
static int n1, n2, nb, k2;
static ps_bands slv;
static float *diag, **offd, eps, eps2, **dip, *tt;
static pwd w1, w2;

static void stepper(bool adj /* adjoint flag */,
		    int i2   /* trace number */);

void predict_init (int nx, int ny /* data size */, 
		   float e        /* regularization parameter */,
		   int nw         /* accuracy order */,
		   int k          /* radius */,
		   bool two       /* if two predictions */)
/*< initialize >*/
{
    n1 = nx;
    n2 = ny;
    nb = 2*nw;

    eps = e;
    eps2 = e;

    slv = ps_banded_init (n1, nb);
    diag = ps_floatalloc (n1);
    offd = ps_floatalloc2 (n1,nb);

    w1 = pwd_init (n1, nw);
    w2 = two? pwd_init(n1,nw): NULL;

    tt = NULL;

    k2 = k;
//     if (k2 > n2-1) ps_error("%s: k2=%d > n2-1=%d",__FILE__,k2,n2-1);
}



void predict_close (void)
/*< free allocated storage >*/
{
    ps_banded_close (slv);
    free (diag);
    free (*offd);
    free (offd);
    pwd_close (w1);
    if (NULL != w2) pwd_close (w2);
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
    pwd_define (forw, w1, pp, diag, offd);
    ps_banded_define (slv, diag, offd);

    if (adj) ps_banded_solve (slv, trace);

    t0 = trace[0];
    t1 = trace[1];
    t2 = trace[n1-2];
    t3 = trace[n1-1];

    pwd_set (adj, w1, trace, trace, diag);

    trace[0] += eps2*t0;
    trace[1] += eps2*t1;
    trace[n1-2] += eps2*t2;
    trace[n1-1] += eps2*t3;

    if (!adj) ps_banded_solve (slv, trace);
}

void predict1_step(bool forw      /* forward or backward */, 
		   float* trace1  /* input trace */,
		   const float* pp /* slope */,
		   float* trace /* output trace */)
/*< prediction step from one trace >*/
{
    float t0, t1, t2, t3;
    
    regularization();

    pwd_define (forw, w1, pp, diag, offd);
    ps_banded_define (slv, diag, offd);

    t0 = trace1[0];
    t1 = trace1[1];
    t2 = trace1[n1-2];
    t3 = trace1[n1-1];

    pwd_set (false, w1, trace1, trace, diag);

    trace[0] += eps2*t0;
    trace[1] += eps2*t1;
    trace[n1-2] += eps2*t2;
    trace[n1-1] += eps2*t3;

    ps_banded_solve (slv, trace);
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

    pwd_define (forw1, w1, pp1, diag, offd);
    pwd_define (forw2, w2, pp2, diag, offd);

    ps_banded_define (slv, diag, offd);

    t0 = 0.5*(trace1[0]+trace2[0]);
    t1 = 0.5*(trace1[1]+trace2[1]);
    t2 = 0.5*(trace1[n1-2]+trace2[n1-2]);
    t3 = 0.5*(trace1[n1-1]+trace2[n1-1]);

    pwd_set (false, w1, trace1, offd[0], trace);
    pwd_set (false, w2, trace2, offd[1], trace);

    for (i1=0; i1 < n1; i1++) {
	trace[i1] = offd[0][i1]+offd[1][i1];
    }

    trace[0] += eps2*t0;
    trace[1] += eps2*t1;
    trace[n1-2] += eps2*t2;
    trace[n1-1] += eps2*t3;

    ps_banded_solve (slv, trace);
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

//     if (nx != ny || nx != n1*n2) ps_error("%s: Wrong dimensions",__FILE__);

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

static int n12;
static float **p, **q, *tmp;

void predict2_init(int m1, int m2           /* data dimensions */, 
		   float eps                /* regularization parameter */,
		   int order                /* accuracy order */,
		   float** pp, float **qq   /* slopes [m1][m2] */)
/*< initialize >*/
{
    p=pp;
    q=qq;
    n12=m1*m2;

    predict_init(m1,m2,eps,order,1,false);
    tmp = ps_floatalloc(n12);
}

void predict2_close(void)
/*< free allocated storage >*/
{
    predict_close();
    free(tmp);
}

void predict2_lop(bool adj, bool add, int nx, int ny, float* x, float* y)
/*< linear operator >*/
{
//     if (nx != n12 || ny != nx) ps_error("%s: wrong dimensions",__FILE__);

    ps_adjnull(adj,add,nx,ny,x,y);

    if (adj) {
	predict_set(q);
	predict_lop (true, false, nx, nx, tmp, y);
	predict_set(p);
	predict_lop (true, add, nx, nx, x, tmp);
    } else {
	predict_set(p);
	predict_lop (false, false, nx, nx, x, tmp);
	predict_set(q);
	predict_lop (false, add, nx, nx, tmp, y);
    }
}

/*from chain.c*/
void ps_chain( ps_operator oper1     /* outer operator */, 
	       ps_operator oper2     /* inner operator */, 
	       bool adj              /* adjoint flag */, 
	       bool add              /* addition flag */, 
	       int nm                /* model size */, 
	       int nd                /* data size */, 
	       int nt                /* intermediate size */, 
	       /*@out@*/ float* mod  /* [nm] model */, 
	       /*@out@*/ float* dat  /* [nd] data */, 
	       float* tmp            /* [nt] intermediate */) 
/*< Chains two operators, computing oper1{oper2{mod}} 
  or its adjoint. The tmp array is used for temporary storage. >*/
{
    if (adj) {
	oper1 (true, false, nt, nd, tmp, dat);
	oper2 (true, add, nm, nt, mod, tmp);
    } else {
	oper2 (false, false, nm, nt, mod, tmp);
	oper1 (false, add, nt, nd, tmp, dat);
    }
}

void ps_array( ps_operator oper1     /* top operator */, 
	       ps_operator oper2     /* bottom operator */, 
	       bool adj              /* adjoint flag */, 
	       bool add              /* addition flag */, 
	       int nm                /* model size */, 
	       int nd1               /* top data size */, 
	       int nd2               /* bottom data size */, 
	       /*@out@*/ float* mod  /* [nm] model */, 
	       /*@out@*/ float* dat1 /* [nd1] top data */, 
	       /*@out@*/ float* dat2 /* [nd2] bottom data */) 
/*< Constructs an array of two operators, 
  computing {oper1{mod},oper2{mod}} or its adjoint. >*/
{
    if (adj) {
	oper1 (true, add,  nm, nd1, mod, dat1);
	oper2 (true, true, nm, nd2, mod, dat2);
    } else {
	oper1 (false, add, nm, nd1, mod, dat1);
	oper2 (false, add, nm, nd2, mod, dat2);
    }
}


void ps_normal (ps_operator oper /* operator */, 
		bool add         /* addition flag */, 
		int nm           /* model size */, 
		int nd           /* data size */, 
		float *mod       /* [nd] model */, 
		float *dat       /* [nd] data */, 
		float *tmp       /* [nm] intermediate */)
/*< Applies a normal operator (self-adjoint) >*/
{
    oper (true, false, nm, nd, tmp, mod);
    oper (false, add,  nm, nd, tmp, dat);
}

void ps_chain3 (ps_operator oper1 /* outer operator */, 
		ps_operator oper2 /* middle operator */, 
		ps_operator oper3 /* inner operator */, 
		bool adj          /* adjoint flag */, 
		bool add          /* addition flag */, 
		int nm            /* model size */, 
		int nt1           /* inner intermediate size */, 
		int nt2           /* outer intermediate size */, 
		int nd            /* data size */, 
		float* mod        /* [nm] model */, 
		float* dat        /* [nd] data */, 
		float* tmp1       /* [nt1] inner intermediate */, 
		float* tmp2       /* [nt2] outer intermediate */)
/*< Chains three operators, computing oper1{oper2{poer3{{mod}}} or its adjoint.
  The tmp1 and tmp2 arrays are used for temporary storage. >*/
{
    if (adj) {
	oper1 (true, false, nt2, nd, tmp2, dat);
	oper2 (true, false, nt1, nt2, tmp1, tmp2);
	oper3 (true, add,   nm, nt1, mod, tmp1);
    } else {
	oper3 (false, false, nm, nt1, mod, tmp1);
	oper2 (false, false, nt1, nt2, tmp1, tmp2);
	oper1 (false, add, nt2, nd, tmp2, dat);
    }
}

/*from big solver*/
void ps_solver_prec (ps_operator oper   /* linear operator */, 
		     ps_solverstep solv /* stepping function */, 
		     ps_operator prec   /* preconditioning operator */, 
		     int nprec          /* size of p */, 
		     int nx             /* size of x */, 
		     int ny             /* size of dat */, 
		     float* x           /* estimated model */, 
		     const float* dat   /* data */, 
		     int niter          /* number of iterations */, 
		     double eps          /* regularization parameter */, 
		     ...                /* variable number of arguments */) 
/*< Generic preconditioned linear solver.
 ---
 Solves
 oper{x} =~ dat
 eps p   =~ 0
 where x = prec{p}
 ---
 The last parameter in the call to this function should be "end".
 Example: 
 ---
 ps_solver_prec (oper_lop,ps_cgstep,prec_lop,
 np,nx,ny,x,y,100,1.0,"x0",x0,"end");
 ---
 Parameters in ...:
 ... 
 "wt":     float*:         weight      
 "wght":   ps_weight wght: weighting function
 "x0":     float*:         initial model
 "nloper": ps_operator:    nonlinear operator  
 "mwt":    float*:         model weight
 "verb":   bool:           verbosity flag
 "known":  bool*:          known model mask
 "nmem":   int:            iteration memory
 "nfreq":  int:            periodic restart
 "xmov":   float**:        model iteration
 "rmov":   float**:        residual iteration
 "err":    float*:         final error
 "res":    float*:         final residual
 "xp":     float*:         preconditioned model
 >*/
{
    va_list args;
    char* par;
    float* wt = NULL;
    ps_weight wght = NULL;
    float* x0 = NULL;
    ps_operator nloper = NULL;
    float* mwt = NULL;
    bool verb = false;
    bool* known = NULL;
    int nmem = -1;
    int nfreq = 0;
    float** xmov = NULL;
    float** rmov = NULL;
    float* err = NULL;
    float* res = NULL;
    float* xp = NULL;
    float* wht = NULL;
    float *p, *g, *rr, *gg, *tp = NULL, *td = NULL;
    int i, iter;
    double dprr, dppd, dppm, dpgm, dprr0=1., dpgm0=1.;
    bool forget = false;

    va_start (args, eps);
    for (;;) {
	par = va_arg (args, char *);
	if      (0 == strcmp (par,"end")) {break;}
	else if (0 == strcmp (par,"wt"))      
	{                    wt = va_arg (args, float*);}
	else if (0 == strcmp (par,"wght"))      
	{                    wght = va_arg (args, ps_weight);}
	else if (0 == strcmp (par,"x0"))      
	{                    x0 = va_arg (args, float*);}
	else if (0 == strcmp (par,"nloper"))      
	{                    nloper = va_arg (args, ps_operator);}
	else if (0 == strcmp (par,"mwt"))      
	{                    mwt = va_arg (args, float*);}
	else if (0 == strcmp (par,"verb"))      
	{                    verb = (bool) va_arg (args, int);}    
	else if (0 == strcmp (par,"known"))      
	{                    known = va_arg (args, bool*);}  
	else if (0 == strcmp (par,"nmem"))      
	{                    nmem = va_arg (args, int);}
	else if (0 == strcmp (par,"nfreq"))      
	{                    nfreq = va_arg (args, int);}
	else if (0 == strcmp (par,"xmov"))      
	{                    xmov = va_arg (args, float**);}
	else if (0 == strcmp (par,"rmov"))      
	{                    rmov = va_arg (args, float**);}
	else if (0 == strcmp (par,"err"))      
	{                    err = va_arg (args, float*);}
	else if (0 == strcmp (par,"res"))      
	{                    res = va_arg (args, float*);}
	else if (0 == strcmp (par,"xp"))      
	{                    xp = va_arg (args, float*);}
	else 
	{ 
		//ps_error("%s: unknown parameter %s",__FILE__,par);
	}
    }
    va_end (args);
  
    p = ps_floatalloc (ny+nprec);
    g = ps_floatalloc (ny+nprec);
    rr = ps_floatalloc (ny);
    gg = ps_floatalloc (ny);
    for (i=0; i < ny; i++) {
	rr[i] = -dat[i];
	p[i+nprec] = 0.0;
    }

    if (wt != NULL || wght != NULL) {
	td = ps_floatalloc (ny);
	if (wt != NULL) {
	    wht = wt;
	} else {
	    wht = ps_floatalloc (ny);
	    for (i=0; i < ny; i++) {
		wht[i] = 1.0;
	    }
	} 
    }

    if (mwt != NULL) tp = ps_floatalloc (nprec);

    if (x0 != NULL) {
	for (i=0; i < nprec; i++) {
	    p[i] = x0[i]; 
	}
	if (nloper != NULL) {
	    if (mwt != NULL) {
		for (i=0; i < nprec; i++) {
		    tp[i] = p[i]*mwt[i];
		}
		ps_chain (nloper, prec, false, true, nprec, ny, nx, tp, rr, x);
	    } else { 
		ps_chain (nloper, prec, false, true, nprec, ny, nx,  p, rr, x);
	    }
	} else {
	    if (mwt != NULL) {
		for (i=0; i < nprec; i++) {
		    tp[i] = p[i]*mwt[i];
		}
		ps_chain (  oper, prec, false, true, nprec, ny, nx, tp, rr, x);
	    } else { 
		ps_chain (  oper, prec, false, true, nprec, ny, nx,  p, rr, x);
	    }
	}
    } else {
	for (i=0; i < nprec; i++) {
	    p[i] = 0.0; 
	}
    }

    for (iter = 0; iter < niter; iter++) {
	if (nmem >= 0) {
	    forget = (bool) (iter >= nmem);
	}
	if (wght != NULL && forget) {
	    wght (ny, rr, wht);
	}
	if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = eps*p[i+nprec] + wht[i]*rr[i];
		td[i] = rr[i]*wht[i];
	    } 
	    ps_chain (oper, prec, true, false, nprec, ny, nx, g, td, x); 
	} else {
	    ps_chain (oper, prec, true, false, nprec, ny, nx, g, rr, x);
	}
	if (mwt != NULL) {
	    for (i=0; i < nprec; i++) {
		g[i] *= mwt[i];
	    }
	}
	for (i=0; i < ny; i++) {
	    g[i+nprec] = eps*rr[i];
	}
	if (known != NULL) {
	    for (i=0; i < nprec; i++) {
		if (known[i]) {
		    g[i] = 0.0;
		} 
	    }
	}

	if (mwt != NULL) {
	    for (i=0; i < nprec; i++) {
		tp[i] = g[i]*mwt[i];
	    }
	    ps_chain (oper, prec, false, false, nprec, ny, nx, tp, gg, x);
	} else {
	    ps_chain (oper, prec, false, false, nprec, ny, nx,  g, gg, x);
	}
	if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		gg[i] *= wht[i];
	    }
	}
	ps_cblas_saxpy(ny,eps,g+nprec,1,gg,1);

	if (forget && nfreq != 0) {  /* periodic restart */
	    forget = (bool) (0 == (iter+1)%nfreq);
	} 
	
	if (iter == 0) {
	    dprr0 = ps_cblas_snrm2 (ny, rr, 1);
	    dpgm0 = ps_cblas_snrm2 (nprec, g, 1);
	    dprr = 1.;
	    dpgm = 1.;
	} else {
	    dprr = ps_cblas_snrm2 (ny, rr, 1)/dprr0;
	    dpgm = ps_cblas_snrm2 (nprec, g, 1)/dpgm0;
	}
	dppd = ps_cblas_snrm2 (ny, p+nprec, 1);
	dppm = ps_cblas_snrm2 (nprec, p, 1);

	if (verb) 
	    printf("iteration %d res %g prec dat %g prec mod %g grad %g\n", 
		       iter, dprr, dppd, dppm, dpgm);
	
	if (dprr < TOLERANCE || dpgm < TOLERANCE) {
	    if (verb) 
		printf("convergence in %d iterations\n",iter+1);

	    if (mwt != NULL) {
		for (i=0; i < nprec; i++) {
		    tp[i] = p[i]*mwt[i];
		}
		prec (false, false, nprec, nx, tp, x);
	    } else {
		prec (false, false, nprec, nx,  p, x);
	    }

	    break;
	}

	solv (forget, nprec+ny, ny, p, g, rr, gg);
	forget = false;

	if (nloper != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = eps*p[i+nprec] - dat[i];
	    }
	    if (mwt != NULL) {
		for (i=0; i < nprec; i++) {
		    tp[i] = p[i]*mwt[i];
		}
		ps_chain (nloper, prec, false, true, nprec, ny, nx, tp, rr, x);
	    } else { 
		ps_chain (nloper, prec, false, true, nprec, ny, nx,  p, rr, x);
	    }
	} else if (wht != NULL) {
	    for (i=0; i < ny; i++) {
		rr[i] = -dat[i];
	    }
	    if (mwt != NULL) {
		for (i=0; i < nprec; i++) {
		    tp[i] = p[i]*mwt[i];
		}
		ps_chain (  oper, prec, false, true, nprec, ny, nx, tp, rr, x);
	    } else { 
		ps_chain (  oper, prec, false, true, nprec, ny, nx,  p, rr, x);
	    }	
	} else if (xmov != NULL || iter == niter-1) {
	    if (mwt != NULL) {
		for (i=0; i < nprec; i++) {
		    tp[i] = p[i]*mwt[i];
		}
		prec (false, false, nprec, nx, tp, x);
	    } else {
		prec (false, false, nprec, nx,  p, x);
	    }
	}
	if (xmov != NULL) {
	    for (i=0; i < nx; i++) {
		xmov[iter][i] =  x[i];
	    }
	}
	if (rmov != NULL) {
	    for (i=0; i < ny; i++) {
		rmov[iter][i] =  p[i+nprec] * eps;
	    }
	}
	if (err != NULL) err[iter] = ps_cblas_snrm2(ny, rr, 1);
    } /* iter */

    if (xp != NULL) {
	for (i=0; i < nprec; i++) {
	    xp[i] = p[i];
	}
    }
    if (res != NULL) {
	for (i=0; i < ny; i++) {
	    res[i] = rr[i];
	}
    }

    for (; iter < niter; iter++) {
	if (xmov != NULL) {
	    for (i=0; i < nx; i++) {
		xmov[iter][i] =  x[i];
	    }
	}
	if (rmov != NULL) {
	    for (i=0; i < ny; i++) {
		rmov[iter][i] =  p[i+nprec] * eps;
	    }
	}    
	if (err != NULL) err[iter] = ps_cblas_snrm2(ny, rr, 1);
    }  

    free (p);
    free (g);
    free (rr);
    free (gg);

    if (wht != NULL) {
	free (td);
	if (wt == NULL) {
	    free (wht);
	}
    }

    if (mwt != NULL) {
	free (tp);
    }

}


static PyObject *csoint2d(PyObject *self, PyObject *args){
    
	/**initialize data input**/
    int nd, nd2;
    int i, niter, nw, n1, n2, n12, nj1, nj2, i4, n4;
    float *mm, *dd, **pp, **qq, a;
    bool *known;
    int verb, drift, hasmask, twoplane, prec;
    
    PyObject *f1=NULL;
    PyObject *f2=NULL;
    PyObject *f3=NULL;
    PyObject *f4=NULL;
    PyObject *arrf1=NULL;
    PyObject *arrf2=NULL;
    PyObject *arrf3=NULL;
    PyObject *arrf4=NULL;
    
	PyArg_ParseTuple(args, "OOOOiiiiiiiiiii", &f1, &f2, &f3, &f4, &n1, &n2, &nw, &nj1, &nj2, &niter, &drift, &hasmask, &twoplane, &prec, &verb);

	/*f1: mm*/
	/*f2: dd*/
	/*pp: slope*/
	/*qq: second slope*/
	/*n1/n2: first/second dimension*/
	/*nw: accuracy order*/
	/*nj1: antialiasing for first dip*/
	/*nj2: antialiasing for second dip*/
	/*niter: number of iterations*/
	/*drift: shift filter*/
	/*hasmaks: if has mask*/
	/*twoplane: if has two plane-wave components*/
	/*prec: if apply preconditioning*/
	/*verb: verbosity flag*/
	
	/*var is noise variance*/
//     a = sqrtf(var);
    
    printf("n1=%d,n2=%d,nw=%d,nj1=%d,nj2=%d,niter=%d,drift=%d\n",n1,n2,nw,nj1,nj2,niter,drift);
    printf("hasmask=%d,twoplane=%d,prec=%d,verb=%d\n",hasmask,twoplane,prec,verb);
    
	n12=n1*n2;
	
    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
	arrf2 = PyArray_FROM_OTF(f2, NPY_FLOAT, NPY_IN_ARRAY);
	arrf3 = PyArray_FROM_OTF(f3, NPY_FLOAT, NPY_IN_ARRAY);
	arrf4 = PyArray_FROM_OTF(f4, NPY_FLOAT, NPY_IN_ARRAY);
	
    nd2=PyArray_NDIM(arrf1);
    npy_intp *sp=PyArray_SHAPE(arrf1);
	
    if (*sp != n12)
    {
    	printf("Dimension mismatch, N_input = %d, N_data = %d\n", *sp, n12);
    	return NULL;
    }

    mm = ps_floatalloc(n12);
    dd = ps_floatalloc(n12);
    known = ps_boolalloc(n12);

    pp = ps_floatalloc2(n1,n2);
    
    /*reading data*/
    for (i=0; i<n12; i++)
    {
        mm[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    }
    
    for (i=0; i<n12; i++)
    {
        dd[i]=*((float*)PyArray_GETPTR1(arrf2,i));
    }
    
    for (i=0; i<n12; i++)
    {
        pp[0][i]=*((float*)PyArray_GETPTR1(arrf3,i));
    }
    
    
    if(twoplane)
    {
    	qq = ps_floatalloc2(n1,n2);
    	for (i=0; i<n12; i++)
    	{
        qq[0][i]=*((float*)PyArray_GETPTR1(arrf4,i));
    	}
    }
    else
    {qq = NULL;}
    
    /*NOTE: if twoplane==0, pp = qq eactly*/

//     allpass3_init(allpass_init(nw, nj1, n1,n2,n3, drift, pp),
// 		  allpass_init(nw, nj2, n1,n2,n3, drift, qq));
	
//     if (!ps_getint("niter",&niter)) niter=100;
//     /* number of iterations */
// 
//     if (!ps_getint("order",&nw)) nw=1;
//     /* accuracy order */
//     if (!ps_getint("nj1",&nj1)) nj1=1;
//     /* antialiasing for first dip */
//     if (!ps_getint("nj2",&nj2)) nj2=1;
//     /* antialiasing for second dip */
// 
//     if (!ps_getbool("drift",&drift)) drift=false;
//     /* if shift filter */
// 
//     if (!ps_getbool("prec",&prec)) prec = false;
//     /* if y, apply preconditioning */
// 
//     if (!ps_getbool("verb",&verb)) verb = false;
//     /* verbosity flag */

//     np = ps_leftsize(dip,2);

//     pp = ps_floatalloc2(n1,n2);

//     if (np > n3) {
// 	qq = ps_floatalloc2(n1,n2);
//     } else {
// 	qq = NULL;
//     }

//     mm = ps_floatalloc(n12);
//     dd = ps_floatalloc(n12);
//     known = ps_boolalloc(n12);
    
//     if (NULL != ps_getstring ("mask")) {
// 	mask = ps_input("mask");
//     } else {
// 	mask = NULL;
//     }

    if (twoplane) {
	if (prec) {
	    predict2_init(n1,n2,0.0001,nw,pp,qq);
	    ps_mask_init(known);
	} else {
// 	    twoplane2_init(nw, nj1,nj2, n1,n2, drift, pp, qq);
	}
    } else {
	if (prec) {
	    predict_init(n1,n2,0.0001,nw,1,false);
	    predict_set(pp);
	    ps_mask_init(known);
	} else {
	    allpass22_init(allpass2_init(nw, nj1, n1,n2, drift, pp));
	}
    }
    
	
	if (hasmask==1) {    
	    for (i=0; i < n12; i++) {
		known[i] = (bool) (dd[i] != 0.);
		dd[i] = 0.;
	    }
	} else {
	    for (i=0; i < n12; i++) {
		known[i] = (bool) (mm[i] != 0.);
		dd[i] = 0.;
	    }
	}
	
// 	for (i=0; i < 2*n123; i++) {
// 	    dd[i] = a*ps_randn_one_bm();
// 	}
	
// 	if (NULL != mask) {
// 	    ps_floatread(dd,n12,mask);
// 
// 	    for (i=0; i < n12; i++) {
// 		known[i] = (bool) (dd[i] != 0.);
// 		dd[i] = 0.;
// 	    }
// 	} else {
// 	    for (i=0; i < n12; i++) {
// 		known[i] = (bool) (mm[i] != 0.);
// 		dd[i] = 0.;
// 	    }
// 	}	
	
// 	ps_solver(allpass3_lop, ps_cgstep, n123, 2*n123, mm, dd, niter,
// 		  "known", known, "x0", mm, "verb", verb, "end");
// 	ps_cgstep_close();

// 	n12=n123;
// 	ps_floatread(pp[0],n12,dip);

	
	if (NULL != qq) {
// 	    ps_floatread(qq[0],n12,dip);

	    if (prec) {
		ps_solver_prec(ps_mask_lop, ps_cgstep, predict2_lop, 
			       n12, n12, n12, 
			       mm, mm, niter, 0.,"verb", verb,"end");
	    } else {
// 		ps_solver(twoplane2_lop, ps_cgstep, n12, n12, mm, dd, niter,
// 			  "known", known, "x0", mm, "verb", verb, "end");
	    }
	} else {
	    if (prec) {
		ps_solver_prec(ps_mask_lop, ps_cgstep, predict_lop, 
			       n12, n12, n12, 
			       mm, mm, niter, 0.,"verb", verb,"end");
	    } else {

		ps_solver(allpass21_lop, ps_cgstep, n12, n12, mm, dd, niter,
			  "known", known, "x0", mm, "verb", verb, "end");
	    }
	}
	ps_cgstep_close();
    
    /*Below is the output part*/
    PyArrayObject *vecout;
	npy_intp dims[2];
	dims[0]=n12;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<dims[0];i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = mm[i];

	return PyArray_Return(vecout);
	
}

// documentation for each functions.
static char soint2dcfun_document[] = "Document stuff for dip...";

// defining our functions like below:
// function_name, function, METH_VARARGS flag, function documents
static PyMethodDef functions[] = {
  {"csoint2d", csoint2d, METH_VARARGS, soint2dcfun_document},
  {NULL, NULL, 0, NULL}
};

// initializing our module informations and settings in this structure
// for more informations, check head part of this file. there are some important links out there.
static struct PyModuleDef soint2dcfunModule = {
  PyModuleDef_HEAD_INIT, // head informations for Python C API. It is needed to be first member in this struct !!
  "soint2dcfun",  // module name
  NULL, // means that the module does not support sub-interpreters, because it has global state.
  -1,
  functions  // our functions list
};

// runs while initializing and calls module creation function.
PyMODINIT_FUNC PyInit_soint2dcfun(void){
  
    PyObject *module = PyModule_Create(&soint2dcfunModule);
    import_array();
    return module;
}
