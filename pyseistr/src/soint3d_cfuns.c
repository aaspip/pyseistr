#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <numpy/arrayobject.h>


/*NOTE: PS indicates PySeistr*/
#define FLT_EPSILON 1.19209290E-07F /*delete #include <float.h> on July 28, 2023*/
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

float ****ps_floatalloc4 (size_t n1 /* fast dimension */, 
				    size_t n2 /* slower dimension */, 
				    size_t n3 /* slower dimension */, 
				    size_t n4 /* slowest dimension */)
/*< float 4-D allocation, out[0][0][0] points to a contiguous array >*/ 
{
    size_t i4;
    float ****ptr;
    
    ptr = (float****) ps_alloc (n4,sizeof(float***));
    ptr[0] = ps_floatalloc3 (n1,n2,n3*n4);
    for (i4=1; i4 < n4; i4++) {
	ptr[i4] = ptr[0]+i4*n3;
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

//     if (n2 != 2*n1) ps_error("%s: size mismatch: %d != 2*%d",__FILE__,n2,n1);

    ps_adjnull(adj, add, n1, n2, xx, yy);

    nx = ap1->nx;
    ny = ap1->ny;
    nz = ap1->nz;
    nw = ap1->nw;
    nj = ap1->nj;

//     if (nx*ny*nz != n1) ps_error("%s: size mismatch",__FILE__);
    
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

//     if (nx*ny*nz != n1) ps_error("%s: size mismatch",__FILE__);
    
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

/*from conjugate*/
static int np, nx, nr, nd;
static float *r, *sp, *sx, *sr, *gp, *gx, *gr;
static float eps, tol;
static bool verb, hasp0;

void ps_conjgrad_init(int np1     /* preconditioned size */, 
		      int nx1     /* model size */, 
		      int nd1     /* data size */, 
		      int nr1     /* residual size */, 
		      float eps1  /* scaling */,
		      float tol1  /* tolerance */, 
		      bool verb1  /* verbosity flag */, 
		      bool hasp01 /* if has initial model */) 
/*< solver constructor >*/
{
    np = np1; 
    nx = nx1;
    nr = nr1;
    nd = nd1;
    eps = eps1*eps1;
    tol = tol1;
    verb = verb1;
    hasp0 = hasp01;

    r = ps_floatalloc(nr);  
    sp = ps_floatalloc(np);
    gp = ps_floatalloc(np);
    sx = ps_floatalloc(nx);
    gx = ps_floatalloc(nx);
    sr = ps_floatalloc(nr);
    gr = ps_floatalloc(nr);
}

void ps_conjgrad_close(void) 
/*< Free allocated space >*/
{
    free (r);
    free (sp);
    free (gp);
    free (sx);
    free (gx);
    free (sr);
    free (gr);
}

void ps_conjgrad(operator prec  /* data preconditioning */, 
		 operator oper  /* linear operator */, 
		 operator shape /* shaping operator */, 
		 float* p          /* preconditioned model */, 
		 float* x          /* estimated model */, 
		 float* dat        /* data */, 
		 int niter         /* number of iterations */) 
/*< Conjugate gradient solver with shaping >*/
{
    double gn, gnp, alpha, beta, g0, dg, r0;
    float *d=NULL;
    int i, iter;
    
    if (NULL != prec) {
	d = ps_floatalloc(nd); 
	for (i=0; i < nd; i++) {
	    d[i] = - dat[i];
	}
	prec(false,false,nd,nr,d,r);
    } else {
	for (i=0; i < nr; i++) {
	    r[i] = - dat[i];
	}
    }
    
    if (hasp0) { /* initial p */
	shape(false,false,np,nx,p,x);
	if (NULL != prec) {
	    oper(false,false,nx,nd,x,d);
	    prec(false,true,nd,nr,d,r);
	} else {
	    oper(false,true,nx,nr,x,r);
	}
    } else {
	for (i=0; i < np; i++) {
	    p[i] = 0.;
	}
	for (i=0; i < nx; i++) {
	    x[i] = 0.;
	}
    } 
    
    dg = g0 = gnp = 0.;
    r0 = ps_cblas_dsdot(nr,r,1,r,1);
    if (r0 == 0.) {
	if (verb) printf("zero residual: r0=%g \n",r0);
	return;
    }

    for (iter=0; iter < niter; iter++) {
	for (i=0; i < np; i++) {
	    gp[i] = eps*p[i];
	}
	for (i=0; i < nx; i++) {
	    gx[i] = -eps*x[i];
	}

	if (NULL != prec) {
	    prec(true,false,nd,nr,d,r);
	    oper(true,true,nx,nd,gx,d);
	} else {
	    oper(true,true,nx,nr,gx,r);
	}

	shape(true,true,np,nx,gp,gx);
	shape(false,false,np,nx,gp,gx);

	if (NULL != prec) {
	    oper(false,false,nx,nd,gx,d);
	    prec(false,false,nd,nr,d,gr);
	} else {
	    oper(false,false,nx,nr,gx,gr);
	}

	gn = ps_cblas_dsdot(np,gp,1,gp,1);

	if (iter==0) {
	    g0 = gn;

	    for (i=0; i < np; i++) {
		sp[i] = gp[i];
	    }
	    for (i=0; i < nx; i++) {
		sx[i] = gx[i];
	    }
	    for (i=0; i < nr; i++) {
		sr[i] = gr[i];
	    }
	} else {
	    alpha = gn / gnp;
	    dg = gn / g0;

	    if (alpha < tol || dg < tol) {
		if (verb) 
		    printf(
			"convergence in %d iterations, alpha=%g, gd=%g \n",
			iter,alpha,dg);
		break;
	    }

	    ps_cblas_saxpy(np,alpha,sp,1,gp,1);
	    ps_cblas_sswap(np,sp,1,gp,1);

	    ps_cblas_saxpy(nx,alpha,sx,1,gx,1);
	    ps_cblas_sswap(nx,sx,1,gx,1);

	    ps_cblas_saxpy(nr,alpha,sr,1,gr,1);
	    ps_cblas_sswap(nr,sr,1,gr,1);
	}

	beta = ps_cblas_dsdot(nr,sr,1,sr,1) + eps*(ps_cblas_dsdot(np,sp,1,sp,1) - ps_cblas_dsdot(nx,sx,1,sx,1));
	
	if (verb) printf("iteration %d res: %f grad: %f\n",
			     iter,ps_cblas_snrm2(nr,r,1)/r0,dg);

	alpha = - gn / beta;

	ps_cblas_saxpy(np,alpha,sp,1,p,1);
	ps_cblas_saxpy(nx,alpha,sx,1,x,1);
	ps_cblas_saxpy(nr,alpha,sr,1,r,1);

	gnp = gn;
    }

    if (NULL != prec) free (d);

}

/*mask.c*/
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

    ps_adjnull (adj,add,nx,ny,x,y);

    for (ix=0; ix < nx; ix++) {
	if (mmask[ix]) {
	    if (adj) x[ix] += y[ix];
	    else     y[ix] += x[ix];
	}
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
	m1 = PS_MIN(k,slv->band);
	for (m = 0; m < m1; m++)
	    t -= (slv->o[m][k-m-1])*(slv->o[m][k-m-1])*(slv->d[k-m-1]);
	slv->d[k] = t;
	n1 = PS_MIN(slv->n-k-1,slv->band);
	for (n = 0; n < n1; n++) {
	    t = offd[n][k];
	    m1 = PS_MIN(k,slv->band-n-1);
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
	m1 = PS_MIN(k,slv->band);
	for (m = 0; m < m1; m++)
	    t -= (slv->o[m][k-m-1])*(slv->o[m][k-m-1])*(slv->d[k-m-1]);
	slv->d[k] = t;
	n1 = PS_MIN(slv->n-k-1,slv->band);
	for (n = 0; n < n1; n++) {
	    t = offd[n];
	    m1 = PS_MIN(k,slv->band-n-1);
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
	m1 = PS_MIN(k,slv->band);
	for (m = 0; m < m1; m++)
	    t -= (slv->o[m][k-m-1])*(slv->o[m][k-m-1])*(slv->d[k-m-1]);
	slv->d[k] = t;
	n1 = PS_MIN(slv->n-k-1,slv->band);
	for (n = 0; n < n1; n++) {
	    t = offd[n];
	    m1 = PS_MIN(k,slv->band-n-1);
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
	m1 = PS_MIN(k,slv->band);
	for (m = 0; m < m1; m++)
	    t -= (slv->o[m][k-m-1]) * b[k-m-1];
	b[k] = t;
    }
    for (k = slv->n-1; k >= 0; k--) {
	t = b[k]/slv->d[k];
	m1 = PS_MIN(slv->n -k-1,slv->band);
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
static float *diag, **offd, eps0, eps2, **dip, *tt;
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

    eps0 = e;
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
	diag[i1] = 6.*eps0;
    	offd[0][i1] = -4.*eps0;
    	offd[1][i1] = eps0;
	for (ib=2; ib < nb; ib++) {
	    offd[ib][i1] = 0.0;
	}
    }

    diag[0] = diag[n1-1] = eps2+eps0;
    diag[1] = diag[n1-2] = eps2+5.*eps0;
    offd[0][0] = offd[0][n1-2] = -2.*eps0;
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

/*pwspray.c*/
static int n1, n2, ns, ns2sp;
static float *trace, **pspr;
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
    ns2sp=2*ns+1;

    predict_init (n1, n2, eps*eps, order, 1, false);
    trace = ps_floatalloc(n1);
    return ns2sp;
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
		j = ip*ns2sp+ns+is+1;
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
		j = ip*ns2sp+ns-is-1;
		for (i1=0; i1 < n1; i1++) {
		    trace[i1] += u[j*n1+i1];
		}
		predict_step(true,false,trace,pspr[ip]);
	    }
	    
	    for (i1=0; i1 < n1; i1++) {
		u1[i*n1+i1] += trace[i1];
		trace[i1] = u[(i*ns2sp+ns)*n1+i1];
		u1[i*n1+i1] += trace[i1];
	    }
	    
	} else {

	    for (i1=0; i1 < n1; i1++) {
		trace[i1] = u1[i*n1+i1];
		u[(i*ns2sp+ns)*n1+i1] += trace[i1];
	    }

            /* predict forward */
	    for (is=0; is < ns; is++) {
		ip = i-is-1;
		if (ip < 0) break;
		j = ip*ns2sp+ns-is-1;
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
		j = ip*ns2sp+ns+is+1;
		predict_step(false,true,trace,pspr[ip-1]);
		for (i1=0; i1 < n1; i1++) {
		    u[j*n1+i1] += trace[i1];
		}
	    }
	}
    }
}

// static int n1, n2, ns2, n12;
static int n12;
static float ***u, *w, **ww1, *t;
static float **p1, **p2, *smooth1, *smooth2, *smooth3;

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

    ns2sp = pwspray_init(nr,n1,n2,order,eps);

    u = ps_floatalloc3(n1,ns2sp,n2);
    w = ps_floatalloc(ns2sp);
    ww1 = ps_floatalloc2(n1,n2);

    for (is=0; is < ns2sp; is++) {
	w[is]=ns+1-PS_ABS(is-ns);
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
    
    ps_adjnull(adj,add,nin,nout,trace,smooth);

    if (adj) {
	for (i2=0; i2 < n2; i2++) {
	    for (i1=0; i1 < n1; i1++) {
		ws=ww1[i2][i1]; 
		for (is=0; is < ns2sp; is++) {
		    u[i2][is][i1] = smooth[i2*n1+i1]*w[is]*ws;
		}
	    }
	}

	pwspray_lop(true,  true,  nin, nin*ns2sp, trace, u[0][0]);
    } else {
	pwspray_lop(false, false, nin, nin*ns2sp, trace, u[0][0]);

	for (i2=0; i2 < n2; i2++) {
	    for (i1=0; i1 < n1; i1++) {
		ws=ww1[i2][i1]; 
		for (is=0; is < ns2sp; is++) {
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
	ww1[0][i1]=1.0f;
    }

    pwsmooth_lop(false,false,n12,n12,ww1[0],t);

    for (i1=0; i1 < n12; i1++) {
	if (0.0f != t[i1]) {
	    ww1[0][i1]=1.0/t[i1];
	} else {
	    ww1[0][i1]=0.0f;
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
    free(*ww1);
    free(ww1);
    free(t);
    pwspray_close();
}

/*pwsmooth3.c*/
static int ps3n1,ps3n2,ps3n3,ps3n12,ps3n13, ns1,ns2, order1,order2;
static float ***idip, ***xdip, ***itmp, ***itmp2, ***xtmp;
static float ps3eps;

void pwsmooth3_init(int ns1_in      /* spray radius */,
		    int ns2_in      /* spray radius */,
		    int m1          /* trace length */,
		    int m2          /* number of traces */,
		    int m3          /* number of traces */,
		    int order1_in   /* PWD order */,
		    int order2_in   /* PWD order */,
		    float eps_in    /* regularization */,
		    float ****dip   /* local slope */)
/*< initialize >*/
{
    int i2, i3;

    ps3n1 = m1;
    ps3n2 = m2;
    ps3n3 = m3;
    ps3n12 = ps3n1*ps3n2;
    ps3n13 = ps3n1*ps3n3;


    ns1 = ns1_in;
    ns2 = ns2_in;
	
    order1 = order1_in;
    order2 = order2_in;
    
    ps3eps = eps_in;

    idip = dip[0];
    xdip = (float***) ps_alloc(ps3n2,sizeof(float**));
    for (i2=0; i2 < ps3n2; i2++) {
	xdip[i2] = (float**) ps_alloc(ps3n3,sizeof(float*));
	for (i3=0; i3 < ps3n3; i3++) {
	    xdip[i2][i3] = dip[1][i3][i2];
	}
    }
    
    itmp = ps_floatalloc3(ps3n1,ps3n2,ps3n3);
    xtmp = ps_floatalloc3(ps3n1,ps3n3,ps3n2);
    itmp2 = ps_floatalloc3(ps3n1,ps3n3,ps3n2);

//     xtmp2 = (float***) ps_alloc(n3,sizeof(float**));
//     for (i3=0; i3 < n3; i3++) {
// 	xtmp2[i3] = (float**) ps_alloc(n2,sizeof(float*));
// 	for (i2=0; i2 < n2; i2++) {
// 	    xtmp2[i3][i2] = xtmp[i2][i3];
// 	}
//     }
}

void pwsmooth3_close(void)
/*< free allocated storage >*/
{
    int i2, i3;

    for (i2=0; i2 < ps3n2; i2++) {
	free(xdip[i2]);
    }
    free(xdip);
    free(**itmp);
    free(*itmp);
    free(itmp);
    free(**xtmp);
    free(*xtmp);
    free(xtmp);
    free(**itmp2);
    free(*itmp2);
    free(itmp2);
//     for (i3=0; i3 < n3; i3++) {
// 	free(xtmp2[i3]);
//     }
//     free(xtmp2);
}

void pwsmooth3_lop(bool adj, bool add, 
		  int nin, int nout, float* trace, float *smooth)
/*< linear operator >*/
{
    int i1, i2, i3;

    ps_adjnull(adj,add,nin,nout,trace,smooth);

    if (adj) {
	for (i3=0; i3 < ps3n3; i3++) {
	    for (i2=0; i2 < ps3n2; i2++) {
		for (i1=0; i1 < ps3n1; i1++) {
		    xtmp[i2][i3][i1] = smooth[i1+ps3n1*(i2+ps3n2*i3)];
		}
	    }
	}
	/* crossline */
	pwsmooth_init(ns2,ps3n1,ps3n3,order2,ps3eps);
	for (i2=0; i2 < ps3n2; i2++) {
	    pwsmooth_set(xdip[i2]);
	    pwsmooth_lop(true,false,ps3n13,ps3n13,itmp2[i2][0],xtmp[i2][0]);
	}
	pwsmooth_close();
	/* transpose */
	for (i3=0; i3 < ps3n3; i3++) {
	    for (i2=0; i2 < ps3n2; i2++) {
		for (i1=0; i1 < ps3n1; i1++) {
		    itmp[i3][i2][i1] = itmp2[i2][i3][i1];
		}
	    }
	}
	/* inline */
	pwsmooth_init(ns1,ps3n1,ps3n2,order1,ps3eps);
	for (i3=0; i3 < ps3n3; i3++) {
	    pwsmooth_set(idip[i3]);
	    pwsmooth_lop(true,true,ps3n12,ps3n12,trace+i3*ps3n12,itmp[i3][0]);
	}
	pwsmooth_close();
    } else {
	/* inline */
	pwsmooth_init(ns1,ps3n1,ps3n2,order1,ps3eps);
	for (i3=0; i3 < ps3n3; i3++) {
	    pwsmooth_set(idip[i3]);
	    pwsmooth_lop(false,false,ps3n12,ps3n12,trace+i3*ps3n12,itmp[i3][0]);
	}
	pwsmooth_close();
	/* transpose */
	for (i3=0; i3 < ps3n3; i3++) {
	    for (i2=0; i2 < ps3n2; i2++) {
		for (i1=0; i1 < ps3n1; i1++) {
		    itmp2[i2][i3][i1] = itmp[i3][i2][i1];
		}
	    }
	}
	/* crossline */
	pwsmooth_init(ns2,ps3n1,ps3n3,order2,ps3eps);
	for (i2=0; i2 < ps3n2; i2++) {
	    pwsmooth_set(xdip[i2]);
	    pwsmooth_lop(false,false,ps3n13,ps3n13,itmp2[i2][0],xtmp[i2][0]);
	}
	pwsmooth_close();
	for (i3=0; i3 < ps3n3; i3++) {
	    for (i2=0; i2 < ps3n2; i2++) {
		for (i1=0; i1 < ps3n1; i1++) {
		    smooth[i1+ps3n1*(i2+ps3n2*i3)] += xtmp[i2][i3][i1];
		}
	    }
	}
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

static PyObject *csint3d(PyObject *self, PyObject *args){
    
	/**initialize data input**/
    int nd, nd2;
    int i, niter, nw1, nw2, ns1, ns2, n1, n2, n3, n123, nj1, nj2, seed, i4, n4;
    float *mm, *dd, ****pp, *qq, *xx, eps, var, lam;
    bool *known;
    int verb, drift, hasmask;
    
    PyObject *f1=NULL;/*input data*/
    PyObject *f2=NULL;/*inline dip*/
    PyObject *f3=NULL;/*xline dip*/
    PyObject *f4=NULL;/*mask*/
    PyObject *arrf1=NULL;
    PyObject *arrf2=NULL;
    PyObject *arrf3=NULL;
    PyObject *arrf4=NULL;
    
	PyArg_ParseTuple(args, "OOOOiiiiiiiiif", &f1, &f2, &f3, &f4, &n1, &n2, &n3, &niter, &ns1, &ns2, &nw1, &nw2, &verb, &eps);
	/*n1/n2/n3: first/second/third dimension*/
	/*nw1/nw2: accuracy order*/
	/*niter: number of iterations*/
	/*verb: verbosity flag*/
	/*eps: stability factor*/
	
    printf("n1=%d,n2=%d,n3=%d,ns1=%d,ns2=%d,nw1=%d,nw2=%d,niter=%d,verb=%d,eps=%g\n",n1,n2,n3,ns1,ns2,nw1,nw2,niter,verb,eps);
    
	n123=n1*n2*n3;

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

    pp = ps_floatalloc4(n1,n2,n3,2);
    mm = ps_floatalloc(n123);
    xx = ps_floatalloc(n123);
    known = ps_boolalloc(n123);
    dd = ps_floatalloc(n123);

    /*reading data*/
    for (i=0; i<n123; i++)
    {
        mm[i]=*((float*)PyArray_GETPTR1(arrf1,i));
    }

	for (i=0; i < n123; i++) {
	    xx[i] = mm[i];
	}
	
    for (i=0; i<n123; i++)
    {
        dd[i]=*((float*)PyArray_GETPTR1(arrf4,i));
    }
    
    for (i=0; i<n123; i++)
    {
        pp[0][0][0][i]=*((float*)PyArray_GETPTR1(arrf2,i));
    }
    for (i=0; i<n123; i++)
    {
        pp[0][0][0][i+n123]=*((float*)PyArray_GETPTR1(arrf3,i));
    }
    
    ps_mask_init(known);
    
	/* figure out scaling and make known data mask */
	lam = 0.;
	for (i=0; i < n123; i++) {
	    if (dd[i] != 0.) {
		known[i] = true;
		lam += 1.;
	    } else {
		known[i] = false;
	    }
	}
	lam = sqrtf(lam/n123);
	
	pwsmooth3_init(ns1,ns2, n1, n2, n3, nw1, nw2, eps, pp);
	ps_conjgrad_init(n123, n123, n123, n123, lam, 10*FLT_EPSILON, true, true); 
	ps_conjgrad(NULL,ps_mask_lop,pwsmooth3_lop,xx,mm,mm,niter);
	ps_conjgrad_close();

	pwsmooth3_close();
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
  {"csint3d", csint3d, METH_VARARGS, soint3dcfun_document},
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
