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


#define PS_MAX(a,b) ((a) < (b) ? (b) : (a))
#define PS_MIN(a,b) ((a) < (b) ? (a) : (b))
#define PS_ABS(a)   ((a) >= 0  ? (a) : (-(a)))

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


#ifndef _allp3_h

typedef struct Allpass *allpass;
/* abstract data type */
/*^*/

#endif

struct Allpass {
    int nx, ny, nz, nw, nj;
    bool drift;
    float *flt, *pp;
};

static allpass ap1, ap2;

allpass allpass_init(int nw                 /* filter size */, 
		     int nj                 /* filter step */, 
		     int nx, int ny, int nz /* data size */, 
		     bool drift             /* if shift filter */,
		     float *pp              /* dip [nz*ny*nx] */)
/*< Initialize >*/
{
    allpass ap;

    ap = (allpass) ps_alloc(1,sizeof(*ap));

    ap->nw = nw;
    ap->nj = nj;
    ap->nx = nx;
    ap->ny = ny;
    ap->nz = nz;
    ap->drift = drift;
    ap->pp = pp;

    ap->flt = ps_floatalloc(2*nw+1);
    apfilt_init(nw);

    return ap;
}

void allpass_close(allpass ap)
/*< free allocated storage >*/
{
    apfilt_close();
    free(ap->flt);
    free(ap);
}

void allpass1 (bool left        /* left or right prediction */,
	       bool der         /* derivative flag */, 
	       const allpass ap /* PWD object */, 
	       float* xx        /* input */, 
	       float* yy        /* output */)
/*< in-line plane-wave destruction >*/
{
    int ix, iy, iz, iw, is, i, nx, ny, nz, i1, i2, ip, id;

    nx = ap->nx;
    ny = ap->ny;
    nz = ap->nz;

    if (left) {
	i1=1; i2=ny;   ip=-nx;
    } else {
	i1=0; i2=ny-1; ip=nx;
    }

    for (iz=0; iz < nz; iz++) {
	for (iy=0; iy < ny; iy++) {
	    for (ix=0; ix < nx; ix++) {
		i = ix + nx * (iy + ny * iz);
		yy[i] = 0.;
	    }
	}
    }
  
    for (iz=0; iz < nz; iz++) {
	for (iy=i1; iy < i2; iy++) {
	    for (ix = ap->nw*ap->nj; ix < nx-ap->nw*ap->nj; ix++) {
		i = ix + nx * (iy + ny * iz);

		if (ap->drift) {
		    id = PS_NINT(ap->pp[i]);
		    if (ix-ap->nw*ap->nj-id < 0 || 
			ix+ap->nw*ap->nj-id >= nx) continue;

		    if (der) {
			aderfilter(ap->pp[i]-id, ap->flt);
		    } else {
			passfilter(ap->pp[i]-id, ap->flt);
		    }
		    
		    for (iw = 0; iw <= 2*ap->nw; iw++) {
			is = (iw-ap->nw)*ap->nj;
			
			yy[i] += (xx[i+is+ip] - xx[i-is-id]) * ap->flt[iw];
		    }		    
		} else {
		    if (der) {
			aderfilter(ap->pp[i], ap->flt);
		    } else {
			passfilter(ap->pp[i], ap->flt);
		    }
		    
		    for (iw = 0; iw <= 2*ap->nw; iw++) {
			is = (iw-ap->nw)*ap->nj;
			
			yy[i] += (xx[i+is+ip] - xx[i-is]) * ap->flt[iw];
		    }
		}
	    }
	}
    }
}

void allpass1t (bool left        /* left or right prediction */,
	       bool der         /* derivative flag */, 
	       const allpass ap /* PWD object */, 
	       float* xx        /* input */, 
	       float* yy        /* output */)
/*< adjoint of in-line plane-wave destruction >*/
{
    int ix, iy, iz, iw, is, i, nx, ny, nz, i1, i2, ip, id;

    nx = ap->nx;
    ny = ap->ny;
    nz = ap->nz;

    if (left) {
	i1=1; i2=ny;   ip=-nx;
    } else {
	i1=0; i2=ny-1; ip=nx;
    }

    for (iz=0; iz < nz; iz++) {
	for (iy=0; iy < ny; iy++) {
	    for (ix=0; ix < nx; ix++) {
		i = ix + nx * (iy + ny * iz);
		xx[i] = 0.;
	    }
	}
    }
  
    for (iz=0; iz < nz; iz++) {
	for (iy=i1; iy < i2; iy++) {
	    for (ix = ap->nw*ap->nj; ix < nx-ap->nw*ap->nj; ix++) {
		i = ix + nx * (iy + ny * iz);

		if (ap->drift) {
		    id = PS_NINT(ap->pp[i]);
		    if (ix-ap->nw*ap->nj-id < 0 || 
			ix+ap->nw*ap->nj-id >= nx) continue;

		    if (der) {
			aderfilter(ap->pp[i]-id, ap->flt);
		    } else {
			passfilter(ap->pp[i]-id, ap->flt);
		    }
		    
		    for (iw = 0; iw <= 2*ap->nw; iw++) {
			is = (iw-ap->nw)*ap->nj;
			
			xx[i+is+ip] += yy[i] * ap->flt[iw];
			xx[i-is-id] -= yy[i] * ap->flt[iw];
		    }
		} else {
		    if (der) {
			aderfilter(ap->pp[i], ap->flt);
		    } else {
			passfilter(ap->pp[i], ap->flt);
		    }
		    
		    for (iw = 0; iw <= 2*ap->nw; iw++) {
			is = (iw-ap->nw)*ap->nj;
			
			xx[i+is+ip] += yy[i] * ap->flt[iw];
			xx[i-is]    -= yy[i] * ap->flt[iw];
		    }
		}
	    }
	}
    }
}

void left1 (bool left        /* left or right prediction */,
	       bool der         /* derivative flag */, 
	       const allpass ap /* PWD object */, 
	       float* xx        /* input */, 
	       float* yy        /* output */)
/*< left part of in-line plane-wave destruction >*/
{
    int ix, iy, iz, iw, is, i, nx, ny, nz, i1, i2, ip, id;

    nx = ap->nx;
    ny = ap->ny;
    nz = ap->nz;

    if (left) {
	i1=1; i2=ny;   ip=-nx;
    } else {
	i1=0; i2=ny-1; ip=nx;
    }

    for (iz=0; iz < nz; iz++) {
	for (iy=0; iy < ny; iy++) {
	    for (ix=0; ix < nx; ix++) {
		i = ix + nx * (iy + ny * iz);
		yy[i] = 0.;
	    }
	}
    }
  
    for (iz=0; iz < nz; iz++) {
	for (iy=i1; iy < i2; iy++) {
	    for (ix = ap->nw*ap->nj; ix < nx-ap->nw*ap->nj; ix++) {
		i = ix + nx * (iy + ny * iz);

		if (ap->drift) {
		    id = PS_NINT(ap->pp[i]);
		    if (ix-ap->nw*ap->nj-id < 0 || 
			ix+ap->nw*ap->nj-id >= nx) continue;

		    if (der) {
			aderfilter(ap->pp[i]-id, ap->flt);
		    } else {
			passfilter(ap->pp[i]-id, ap->flt);
		    }
		} else {
		    if (der) {
			aderfilter(ap->pp[i], ap->flt);
		    } else {
			passfilter(ap->pp[i], ap->flt);
		    }
		}

		for (iw = 0; iw <= 2*ap->nw; iw++) {
		    is = (iw-ap->nw)*ap->nj;
		    
		    yy[i] += xx[i+is+ip] * ap->flt[iw];
		}
	    }
	}
    }
}

void right1 (bool left        /* left or right prediction */,
	       bool der         /* derivative flag */, 
	       const allpass ap /* PWD object */, 
	       float* xx        /* input */, 
	       float* yy        /* output */)
/*< right part of in-line plane-wave destruction >*/
{
    int ix, iy, iz, iw, is, i, nx, ny, nz, i1, i2, id;

    nx = ap->nx;
    ny = ap->ny;
    nz = ap->nz;

    if (left) {
	i1=1; i2=ny;   
    } else {
	i1=0; i2=ny-1;
    }

    for (iz=0; iz < nz; iz++) {
	for (iy=0; iy < ny; iy++) {
	    for (ix=0; ix < nx; ix++) {
		i = ix + nx * (iy + ny * iz);
		yy[i] = 0.;
	    }
	}
    }
  
    for (iz=0; iz < nz; iz++) {
	for (iy=i1; iy < i2; iy++) {
	    for (ix = ap->nw*ap->nj; ix < nx-ap->nw*ap->nj; ix++) {
		i = ix + nx * (iy + ny * iz);

		if (ap->drift) {
		    id = PS_NINT(ap->pp[i]);
		    if (ix-ap->nw*ap->nj-id < 0 || 
			ix+ap->nw*ap->nj-id >= nx) continue;
		    
		    if (der) {
			aderfilter(ap->pp[i]-id, ap->flt);
		    } else {
			passfilter(ap->pp[i]-id, ap->flt);
		    }
		    
		    for (iw = 0; iw <= 2*ap->nw; iw++) {
			is = (iw-ap->nw)*ap->nj;
			
			yy[i] += xx[i-is-id] * ap->flt[iw];
		    }
		} else {
		    if (der) {
			aderfilter(ap->pp[i], ap->flt);
		    } else {
			passfilter(ap->pp[i], ap->flt);
		    }
		    
		    for (iw = 0; iw <= 2*ap->nw; iw++) {
			is = (iw-ap->nw)*ap->nj;
			
			yy[i] += xx[i-is] * ap->flt[iw];
		    }
		}
	    }
	}
    }
}

void allpass2 (bool left        /* left or right prediction */,
	       bool der         /* derivative flag */, 
	       const allpass ap /* PWD object */, 
	       float* xx        /* input */, 
	       float* yy        /* output */)
/*< cross-line plane-wave destruction >*/
{
    int ix, iy, iz, iw, is, i, nx, ny, nz, i1, i2, ip, id;

    nx = ap->nx;
    ny = ap->ny;
    nz = ap->nz;

    if (left) {
	i1=1; i2=nz;   ip=-nx*ny;
    } else {
	i1=0; i2=nz-1; ip=nx*ny;
    }
    
    for (iz=0; iz < nz; iz++) {
	for (iy=0; iy < ny; iy++) {
	    for (ix=0; ix < nx; ix++) {
		i = ix + nx * (iy + ny * iz);
		yy[i] = 0.;
	    }
	}
    }
    
    for (iz=i1; iz < i2; iz++) {
	for (iy=0; iy < ny; iy++) {
	    for (ix = ap->nw*ap->nj; ix < nx-ap->nw*ap->nj; ix++) {
		i = ix + nx * (iy + ny * iz);
		
		if (ap->drift) {
		    id = PS_NINT(ap->pp[i]);
		    if (ix-ap->nw*ap->nj-id < 0 || 
			ix+ap->nw*ap->nj-id >= nx) continue;

		    if (der) {
			aderfilter(ap->pp[i]-id, ap->flt);
		    } else {
			passfilter(ap->pp[i]-id, ap->flt);
		    }
		    
		    for (iw = 0; iw <= 2*ap->nw; iw++) {
			is = (iw-ap->nw)*ap->nj;
			
			yy[i] += (xx[i+is+ip] - xx[i-is-id]) * ap->flt[iw];
		    }

		} else {
		    if (der) {
			aderfilter(ap->pp[i], ap->flt);
		    } else {
			passfilter(ap->pp[i], ap->flt);
		    }
		    
		    for (iw = 0; iw <= 2*ap->nw; iw++) {
			is = (iw-ap->nw)*ap->nj;
			
			yy[i] += (xx[i+is+ip] - xx[i-is]) * ap->flt[iw];
		    }
		}
	    }
	}
    }
}

void allpass3_init (allpass ap, allpass aq)
/*< Initialize linear operator >*/
{
    ap1 = ap;
    ap2 = aq;
}

void allpass3_lop (bool adj, bool add, int n1, int n2, float* xx, float* yy)
/*< PWD as linear operator >*/
{
    int i, ix, iy, iz, iw, is, nx, ny, nz, nw, nj, id;

//     if (n2 != 2*n1) sf_error("%s: size mismatch: %d != 2*%d",__FILE__,n2,n1);

    ps_adjnull(adj, add, n1, n2, xx, yy);

    nx = ap1->nx;
    ny = ap1->ny;
    nz = ap1->nz;
    nw = ap1->nw;
    nj = ap1->nj;

//     if (nx*ny*nz != n1) sf_error("%s: size mismatch",__FILE__);
    
    for (iz=0; iz < nz; iz++) {
	for (iy=0; iy < ny-1; iy++) {
	    for (ix = nw*nj; ix < nx-nw*nj; ix++) {
		i = ix + nx*(iy + ny*iz);

		if (ap1->drift) {
		    id = PS_NINT(ap1->pp[i]);
		    if (ix-nw*nj-id < 0 || 
			ix+nw*nj-id >= nx) continue;

		    passfilter(ap1->pp[i]-id, ap1->flt);
		    
		    for (iw = 0; iw <= 2*nw; iw++) {
			is = (iw-nw)*nj;
			
			if (adj) {
			    xx[i+nx+is] += yy[i] * ap1->flt[iw];
			    xx[i-is-id] -= yy[i] * ap1->flt[iw];
			} else {
			    yy[i] += (xx[i+nx+is] - xx[i-is-id]) * ap1->flt[iw];
			}
		    }
		} else {
		    passfilter(ap1->pp[i], ap1->flt);
		    
		    for (iw = 0; iw <= 2*nw; iw++) {
			is = (iw-nw)*nj;
			
			if (adj) {
			    xx[i+nx+is] += yy[i] * ap1->flt[iw];
			    xx[i-is]    -= yy[i] * ap1->flt[iw];
			} else {
			    yy[i] += (xx[i+nx+is] - xx[i-is]) * ap1->flt[iw];
			}
		    }
		}
	    }
	}
    }

    nx = ap2->nx;
    ny = ap2->ny;
    nz = ap2->nz;
    nw = ap2->nw;
    nj = ap2->nj;

//     if (nx*ny*nz != n1) sf_error("%s: size mismatch",__FILE__);
    
    for (iz=0; iz < nz-1; iz++) {
	for (iy=0; iy < ny; iy++) {
	    for (ix = nw*nj; ix < nx-nw*nj; ix++) {
		i = ix + nx*(iy + ny*iz);

		if (ap2->drift) {
		    id = PS_NINT(ap2->pp[i]);
		    if (ix-nw*nj-id < 0 || 
			ix+nw*nj-id >= nx) continue;

		    passfilter(ap2->pp[i]-id, ap2->flt);
		    
		    for (iw = 0; iw <= 2*nw; iw++) {
			is = (iw-nw)*nj;
			
			if (adj) {
			    xx[i+nx*ny+is] += yy[i+n1] * ap2->flt[iw];
			    xx[i-is-id]    -= yy[i+n1] * ap2->flt[iw];
			} else {
			    yy[i+n1] += (xx[i+nx*ny+is] - xx[i-is-id]) * ap2->flt[iw];
			}
		    }
		} else {
		    passfilter(ap2->pp[i], ap2->flt);
		    
		    for (iw = 0; iw <= 2*nw; iw++) {
			is = (iw-nw)*nj;
			
			if (adj) {
			    xx[i+nx*ny+is] += yy[i+n1] * ap2->flt[iw];
			    xx[i-is]       -= yy[i+n1] * ap2->flt[iw];
			} else {
			    yy[i+n1] += (xx[i+nx*ny+is] - xx[i-is]) * ap2->flt[iw];
			}
		    }
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


/* Period parameters */  
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */

static unsigned long mt[N]; /* the array for the state vector  */
static int mti=N+1; /* mti==N+1 means mt[N] is not initialized */

/* initializes mt[N] with a seed */
void init_genrand(unsigned long s)
{
    mt[0]= s & 0xffffffffUL;
    for (mti=1; mti<N; mti++) {
        mt[mti] = 
	    (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti); 
        /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
        /* In the previous versions, MSBs of the seed affect   */
        /* only MSBs of the array mt[].                        */
        /* 2002/01/09 modified by Makoto Matsumoto             */
        mt[mti] &= 0xffffffffUL;
        /* for >32 bit machines */
    }
}


/* generates a random number on [0,0xffffffff]-interval */
unsigned long genrand_int32(void)
{
    unsigned long y;
    static unsigned long mag01[2]={0x0UL, MATRIX_A};
    /* mag01[x] = x * MATRIX_A  for x=0,1 */

    if (mti >= N) { /* generate N words at one time */
        int kk;

        if (mti == N+1)   /* if init_genrand() has not been called, */
            init_genrand(5489UL); /* a default initial seed is used */

        for (kk=0;kk<N-M;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+M] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        for (;kk<N-1;kk++) {
            y = (mt[kk]&UPPER_MASK)|(mt[kk+1]&LOWER_MASK);
            mt[kk] = mt[kk+(M-N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
        }
        y = (mt[N-1]&UPPER_MASK)|(mt[0]&LOWER_MASK);
        mt[N-1] = mt[M-1] ^ (y >> 1) ^ mag01[y & 0x1UL];

        mti = 0;
    }
  
    y = mt[mti++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}

/* generates a random number on [0,1]-real-interval */
double genrand_real1(void)
{
    return genrand_int32()*(1.0/4294967295.0); 
    /* divided by 2^32-1 */ 
}

static bool   iset = true;
static float  vset = 0.0;

float ps_randn_one_bm (void)
/*< return a random number (normally distributed, Box-Muller method) >*/
{
    double x1, x2, y1, y2, z1, z2;
    
    if (iset) {
	do {
	    x1 = genrand_real1 ();
	} while (x1 == 0.0);
        x2 = genrand_real1 ();
        
        z1 = sqrt(-2.0*log(x1));
        z2 = 2.0*PS_PI*x2;
        
        y1 = z1*cos(z2);
        y2 = z1*sin(z2);

        iset = false;
        vset = y1;
        return ((float) y2);
    } else {
        iset = true;
        return vset;
    }
}


static PyObject *csoint3d(PyObject *self, PyObject *args){
    
	/**initialize data input**/
    int nd, nd2;
    int i, niter, nw, n1, n2, n3, n123, nj1, nj2, seed, i4, n4;
    float *mm, *dd, *pp, *qq, a, var;
    bool *known;
    int verb, drift, hasmask;
    
    PyObject *f1=NULL;
    PyObject *f2=NULL;
    PyObject *f3=NULL;
    PyObject *f4=NULL;
    PyObject *arrf1=NULL;
    PyObject *arrf2=NULL;
    PyObject *arrf3=NULL;
    PyObject *arrf4=NULL;
    
	PyArg_ParseTuple(args, "OOOOiiiiiiiiiifi", &f1, &f2, &f3, &f4, &n1, &n2, &n3, &nw, &nj1, &nj2, &niter, &drift, &seed, &hasmask, &var, &verb);


	/*var is noise variance*/
    a = sqrtf(var);
    
	n123=n1*n2*n3;
    init_genrand((unsigned long) seed);
	
    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
	arrf2 = PyArray_FROM_OTF(f2, NPY_FLOAT, NPY_IN_ARRAY);
	arrf3 = PyArray_FROM_OTF(f3, NPY_FLOAT, NPY_IN_ARRAY);
	arrf4 = PyArray_FROM_OTF(f4, NPY_FLOAT, NPY_IN_ARRAY);
	
    nd2=PyArray_NDIM(arrf1);
    npy_intp *sp=PyArray_SHAPE(arrf1);
	
    if (*sp != n123)
    {
    	printf("Dimension mismatch, N_input = %d, N_data = %d\n", *sp, n123);
    	return NULL;
    }

    pp = ps_floatalloc(n123);
    qq = ps_floatalloc(n123);

    mm = ps_floatalloc(n123);
    dd = ps_floatalloc(2*n123);
    known = ps_boolalloc(n123);

    /*reading data*/
    for (i=0; i<n123; i++)
    {
        mm[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    }
    
    for (i=0; i<n123; i++)
    {
        dd[i]=*((float*)PyArray_GETPTR1(arrf2,i));
    }
    
    for (i=0; i<n123; i++)
    {
        pp[i]=*((float*)PyArray_GETPTR1(arrf3,i));
    }
    for (i=0; i<n123; i++)
    {
        qq[i]=*((float*)PyArray_GETPTR1(arrf4,i));
    }
    

    allpass3_init(allpass_init(nw, nj1, n1,n2,n3, drift, pp),
		  allpass_init(nw, nj2, n1,n2,n3, drift, qq));
	
	if (hasmask==1) {    
	    for (i=0; i < n123; i++) {
		known[i] = (bool) (dd[i] != 0.);
	    }
	} else {
	    for (i=0; i < n123; i++) {
		known[i] = (bool) (mm[i] != 0.);
	    }
	}
	
	for (i=0; i < 2*n123; i++) {
	    dd[i] = a*ps_randn_one_bm();
	}
	
	ps_solver(allpass3_lop, ps_cgstep, n123, 2*n123, mm, dd, niter,
		  "known", known, "x0", mm, "verb", verb, "end");
	ps_cgstep_close();

    
    /*Below is the output part*/
    PyArrayObject *vecout;
	npy_intp dims[2];
	dims[0]=n123;dims[1]=1;
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<dims[0];i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = mm[i];

	return PyArray_Return(vecout);
	
}

// documentation for each functions.
static char soint3dcfun_document[] = "Document stuff for dip...";

// defining our functions like below:
// function_name, function, METH_VARARGS flag, function documents
static PyMethodDef functions[] = {
  {"csoint3d", csoint3d, METH_VARARGS, soint3dcfun_document},
  {NULL, NULL, 0, NULL}
};

// initializing our module informations and settings in this structure
// for more informations, check head part of this file. there are some important links out there.
static struct PyModuleDef soint3dcfunModule = {
  PyModuleDef_HEAD_INIT, // head informations for Python C API. It is needed to be first member in this struct !!
  "soint3dcfun",  // module name
  NULL, // means that the module does not support sub-interpreters, because it has global state.
  -1,
  functions  // our functions list
};

// runs while initializing and calls module creation function.
PyMODINIT_FUNC PyInit_soint3dcfun(void){
  
    PyObject *module = PyModule_Create(&soint3dcfunModule);
    import_array();
    return module;
}
