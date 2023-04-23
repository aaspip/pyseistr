#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdio.h>
#include <numpy/arrayobject.h>

/*NOTE: PS indicates PySeistr*/
#define PS_NINT(x) ((int)((x)>0.0?(x)+0.5:(x)-0.5))
#define PS_MAX_DIM 9

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

/*from decart.c*/
int ps_first_index (int i          /* dimension [0...dim-1] */, 
		    int j        /* line coordinate */, 
		    int dim        /* number of dimensions */, 
		    const int *n /* box size [dim] */, 
		    const int *s /* step [dim] */)
/*< Find first index for multidimensional transforms >*/
{
    int i0, n123, ii;
    int k;

    n123 = 1;
    i0 = 0;
    for (k=0; k < dim; k++) {
	if (k == i) continue;
	ii = (j/n123)%n[k]; /* to cartesian */
	n123 *= n[k];	
	i0 += ii*s[k];      /* back to line */
    }

    return i0;
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

/*from triangle.c*/
typedef struct ps_Triangle *ps_triangle;
/* abstract data type */

struct ps_Triangle {
    float *tmp, wt;
    int np, nb, nx;
    bool box;
};

static void fold (int o, int d, int nx, int nb, int np, 
		  const float *x, float* tmp);
static void fold2 (int o, int d, int nx, int nb, int np, 
		   float *x, const float* tmp);
static void doubint (int nx, float *x, bool der);
static void triple (int o, int d, int nx, int nb, 
		    float* x, const float* tmp, bool box, float wt);
static void triple2 (int o, int d, int nx, int nb, 
		     const float* x, float* tmp, bool box, float wt);

ps_triangle ps_triangle_init (int  nbox /* triangle length */, 
			      int  ndat /* data length */,
                              bool box  /* if box instead of triangle */)
/*< initialize >*/
{
    ps_triangle tr;

    tr = (ps_triangle) ps_alloc(1,sizeof(*tr));

    tr->nx = ndat;
    tr->nb = nbox;
    tr->box = box;
    tr->np = ndat + 2*nbox;
    
    if (box) {
	tr->wt = 1.0/(2*nbox-1);
    } else {
	tr->wt = 1.0/(nbox*nbox);
    }
    
    tr->tmp = ps_floatalloc(tr->np);

    return tr;
}

static void fold (int o, int d, int nx, int nb, int np, 
		  const float *x, float* tmp)
{
    int i, j;

    /* copy middle */
    for (i=0; i < nx; i++) 
	tmp[i+nb] = x[o+i*d];
    
    /* reflections from the right side */
    for (j=nb+nx; j < np; j += nx) {
	for (i=0; i < nx && i < np-j; i++)
	    tmp[j+i] = x[o+(nx-1-i)*d];
	j += nx;
	for (i=0; i < nx && i < np-j; i++)
	    tmp[j+i] = x[o+i*d];
    }
    
    /* reflections from the left side */
    for (j=nb; j >= 0; j -= nx) {
	for (i=0; i < nx && i < j; i++)
	    tmp[j-1-i] = x[o+i*d];
	j -= nx;
	for (i=0; i < nx && i < j; i++)
	    tmp[j-1-i] = x[o+(nx-1-i)*d];
    }
}

static void fold2 (int o, int d, int nx, int nb, int np, 
		   float *x, const float* tmp)
{
    int i, j;

    /* copy middle */
    for (i=0; i < nx; i++) 
	x[o+i*d] = tmp[i+nb];

    /* reflections from the right side */
    for (j=nb+nx; j < np; j += nx) {
	for (i=0; i < nx && i < np-j; i++)
	    x[o+(nx-1-i)*d] += tmp[j+i];
	j += nx;
	for (i=0; i < nx && i < np-j; i++)
	    x[o+i*d] += tmp[j+i];
    }
    
    /* reflections from the left side */
    for (j=nb; j >= 0; j -= nx) {
	for (i=0; i < nx && i < j; i++)
	    x[o+i*d] += tmp[j-1-i];
	j -= nx;
	for (i=0; i < nx && i < j; i++)
	    x[o+(nx-1-i)*d] += tmp[j-1-i];
    }
}
    
static void doubint (int nx, float *xx, bool der)
{
    int i;
    float t;

    /* integrate backward */
    t = 0.;
    for (i=nx-1; i >= 0; i--) {
	t += xx[i];
	xx[i] = t;
    }

    if (der) return;

    /* integrate forward */
    t=0.;
    for (i=0; i < nx; i++) {
	t += xx[i];
	xx[i] = t;
    }
}

static void doubint2 (int nx, float *xx, bool der)
{
    int i;
    float t;


    /* integrate forward */
    t=0.;
    for (i=0; i < nx; i++) {
	t += xx[i];
	xx[i] = t;
    }

    if (der) return;

    /* integrate backward */
    t = 0.;
    for (i=nx-1; i >= 0; i--) {
	t += xx[i];
	xx[i] = t;
    }
}

static void triple (int o, int d, int nx, int nb, float* x, const float* tmp, bool box, float wt)
{
    int i;
    const float *tmp1, *tmp2;
    
    if (box) {
	tmp2 = tmp + 2*nb;

	for (i=0; i < nx; i++) {
	    x[o+i*d] = (tmp[i+1] - tmp2[i])*wt;
	}
    } else {
	tmp1 = tmp + nb;
	tmp2 = tmp + 2*nb;

	for (i=0; i < nx; i++) {
	    x[o+i*d] = (2.*tmp1[i] - tmp[i] - tmp2[i])*wt;
	}
    }
}

static void dtriple (int o, int d, int nx, int nb, float* x, const float* tmp, float wt)
{
    int i;
    const float *tmp2;

    tmp2 = tmp + 2*nb;
    
    for (i=0; i < nx; i++) {
	x[o+i*d] = (tmp[i] - tmp2[i])*wt;
    }
}

static void triple2 (int o, int d, int nx, int nb, const float* x, float* tmp, bool box, float wt)
{
    int i;

    for (i=0; i < nx + 2*nb; i++) {
	tmp[i] = 0;
    }

    if (box) {
	ps_cblas_saxpy(nx,  +wt,x+o,d,tmp+1   ,1);
	ps_cblas_saxpy(nx,  -wt,x+o,d,tmp+2*nb,1);
    } else {
	ps_cblas_saxpy(nx,  -wt,x+o,d,tmp     ,1);
	ps_cblas_saxpy(nx,2.*wt,x+o,d,tmp+nb  ,1);
	ps_cblas_saxpy(nx,  -wt,x+o,d,tmp+2*nb,1);
    }
}

static void dtriple2 (int o, int d, int nx, int nb, const float* x, float* tmp, float wt)
{
    int i;

    for (i=0; i < nx + 2*nb; i++) {
	tmp[i] = 0;
    }

    ps_cblas_saxpy(nx,  wt,x+o,d,tmp     ,1);
    ps_cblas_saxpy(nx, -wt,x+o,d,tmp+2*nb,1);
}

void ps_smooth (ps_triangle tr  /* smoothing object */, 
		int o, int d    /* trace sampling */, 
		bool der        /* if derivative */, 
		float *x        /* data (smoothed in place) */)
/*< apply triangle smoothing >*/
{
    fold (o,d,tr->nx,tr->nb,tr->np,x,tr->tmp);
    doubint (tr->np,tr->tmp,(bool) (tr->box || der));
    triple (o,d,tr->nx,tr->nb,x,tr->tmp, tr->box, tr->wt);
}

void ps_dsmooth (ps_triangle tr  /* smoothing object */, 
		int o, int d    /* trace sampling */, 
		bool der        /* if derivative */, 
		float *x        /* data (smoothed in place) */)
/*< apply triangle smoothing >*/
{
    fold (o,d,tr->nx,tr->nb,tr->np,x,tr->tmp);
    doubint (tr->np,tr->tmp,(bool) (tr->box || der));
    dtriple (o,d,tr->nx,tr->nb,x,tr->tmp, tr->wt);
}

void ps_smooth2 (ps_triangle tr  /* smoothing object */, 
		 int o, int d    /* trace sampling */, 
		 bool der        /* if derivative */,
		 float *x        /* data (smoothed in place) */)
/*< apply adjoint triangle smoothing >*/
{
    triple2 (o,d,tr->nx,tr->nb,x,tr->tmp, tr->box, tr->wt);
    doubint2 (tr->np,tr->tmp,(bool) (tr->box || der));
    fold2 (o,d,tr->nx,tr->nb,tr->np,x,tr->tmp);
}

void ps_dsmooth2 (ps_triangle tr  /* smoothing object */, 
		 int o, int d    /* trace sampling */, 
		 bool der        /* if derivative */,
		 float *x        /* data (smoothed in place) */)
/*< apply adjoint triangle smoothing >*/
{
    dtriple2 (o,d,tr->nx,tr->nb,x,tr->tmp, tr->wt);
    doubint2 (tr->np,tr->tmp,(bool) (tr->box || der));
    fold2 (o,d,tr->nx,tr->nb,tr->np,x,tr->tmp);
}

void  ps_triangle_close(ps_triangle tr)
/*< free allocated storage >*/
{
    free (tr->tmp);
    free (tr);
}

/*from trianglen.c*/
static int *ntri, s[PS_MAX_DIM], nd, dim;
static ps_triangle *tr;
static float *tmp;

void ps_trianglen_init (int ndim  /* number of dimensions */, 
			int *nbox /* triangle radius [ndim] */, 
			int *ndat /* data dimensions [ndim] */)
/*< initialize >*/
{
    int i;

    dim = ndim;
    ntri = ps_intalloc(dim);

    tr = (ps_triangle*) ps_alloc(dim,sizeof(ps_triangle));

    nd = 1;
    for (i=0; i < dim; i++) {
	tr[i] = (nbox[i] > 1)? ps_triangle_init (nbox[i],ndat[i],false): NULL;
	s[i] = nd;
	ntri[i] = ndat[i];
	nd *= ndat[i];
    }
    tmp = ps_floatalloc (nd);
}

void ps_trianglen_lop (bool adj, bool add, int nx, int ny, float* x, float* y)
/*< linear operator >*/
{
    int i, j, i0;

//     if (nx != ny || nx != nd) 
// 	sf_error("%s: Wrong data dimensions: nx=%d, ny=%d, nd=%d",
// 		 __FILE__,nx,ny,nd);

    ps_adjnull (adj,add,nx,ny,x,y);
  
    if (adj) {
	for (i=0; i < nd; i++) {
	    tmp[i] = y[i];
	}
    } else {
	for (i=0; i < nd; i++) {
	    tmp[i] = x[i];
	}
    }

  
    for (i=0; i < dim; i++) {
	if (NULL != tr[i]) {
	    for (j=0; j < nd/ntri[i]; j++) {
		i0 = ps_first_index (i,j,dim,ntri,s);
		ps_smooth2 (tr[i], i0, s[i], false, tmp);
	    }
	}
    }
	
    if (adj) {
	for (i=0; i < nd; i++) {
	    x[i] += tmp[i];
	}
    } else {
	for (i=0; i < nd; i++) {
	    y[i] += tmp[i];
	}
    }    
}

void ps_trianglen_close(void)
/*< free allocated storage >*/
{
    int i;

    free (tmp);

    for (i=0; i < dim; i++) {
	if (NULL != tr[i]) ps_triangle_close (tr[i]);
    }

    free(tr);
    free(ntri);
}


/*from weight.c*/
static float* weig;

void ps_weight_init(float *w1)
/*< initialize >*/
{
    weig = w1;
}

void ps_weight_lop (bool adj, bool add, int nx, int ny, float* xx, float* yy)
/*< linear operator >*/
{
    int i;

//     if (ny!=nx) sf_error("%s: size mismatch: %d != %d",__FILE__,ny,nx);

    ps_adjnull (adj, add, nx, ny, xx, yy);
  
    if (adj) {
        for (i=0; i < nx; i++) {
	    xx[i] += yy[i] * weig[i];
	}
    } else {
        for (i=0; i < nx; i++) {
            yy[i] += xx[i] * weig[i];
	}
    }

}


/*from divn.c*/
static int niter, ndivn;
static float *p;

void ps_divn_init(int ndim   /* number of dimensions */, 
		  int nd     /* data size */, 
		  int *ndat  /* data dimensions [ndim] */, 
		  int *nbox  /* smoothing radius [ndim] */, 
		  int niter1 /* number of iterations */,
		  bool verb  /* verbosity */) 
/*< initialize >*/
{
    niter = niter1;
    ndivn = nd;

    ps_trianglen_init(ndim, nbox, ndat);
    ps_conjgrad_init(nd, nd, nd, nd, 1., 1.e-6, verb, false);
    p = ps_floatalloc (nd);
}

void ps_divn_close (void)
/*< free allocated storage >*/
{
    ps_trianglen_close();
    ps_conjgrad_close();
    free (p);
}

void ps_divn (float* num, float* den,  float* rat)
/*< smoothly divide rat=num/den >*/
{
    ps_weight_init(den);
    ps_conjgrad(NULL, ps_weight_lop,ps_trianglen_lop,p,rat,num,niter); 
}

void ps_divne (float* num, float* den,  float* rat, float eps)
/*< smoothly divide rat=num/den with preconditioning >*/
{
    int i;
    double norm;

    if (eps > 0.0f) {
	for (i=0; i < ndivn; i++) {
	    norm = 1.0/hypot(den[i],eps);

	    num[i] *= norm;
	    den[i] *= norm;
	}
    } 

    norm = ps_cblas_dsdot(ndivn,den,1,den,1);
    if (norm == 0.0) {
	for (i=0; i < ndivn; i++) {
	    rat[i] = 0.0;
	}
	return;
    }
    norm = sqrt(ndivn/norm);

    for (i=0; i < ndivn; i++) {
	num[i] *= norm;
	den[i] *= norm;
    }   

    ps_weight_init(den);
    ps_conjgrad(NULL, ps_weight_lop,ps_trianglen_lop,p,rat,num,niter); 
}



/*from apfilt.c*/
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


/*from mask6.c*/
void mask32 (bool both              /* left and right predictions */,
	     int nw                 /* filter size */, 
	     int nj1, int nj2       /* dealiasing stretch */, 
	     int nx, int ny, int nz /* data size */, 
	     float *yy              /* data [nz*ny*nx] */, 
	     bool **m               /* dip mask [both? 4:2][nz*ny*nx] */) 
/*< two-dip masks in 3-D >*/
{
    int ix, iy, iz, iw, is, i, n;
    bool *xx;

    n = nx*ny*nz;

    xx = ps_boolalloc(n);

    for (i=0; i < n; i++) {
	xx[i] = (bool) (yy[i] == 0.);
	m[0][i] = false;
	m[1][i] = false;
	if (both) {
	    m[2][i] = false;
	    m[3][i] = false;
	}
    }

    for (iz=0; iz < nz; iz++) {
	for (iy=0; iy < ny-1; iy++) {
	    for (ix = nw*nj1; ix < nx-nw*nj1; ix++) {
		i = ix + nx * (iy + ny * iz);

		for (iw = 0; iw <= 2*nw; iw++) {
		    is = (iw-nw)*nj1;		  
		    m[0][i] = (bool) (m[0][i] || xx[i-is] || xx[i+nx+is]);
		}
	    }
	}
    }

    for (iz=0; iz < nz-1; iz++) {
	for (iy=0; iy < ny; iy++) {
	    for (ix = nw*nj2; ix < nx-nw*nj2; ix++) {
		i = ix + nx * (iy + ny * iz);

		for (iw = 0; iw <= 2*nw; iw++) {
		    is = (iw-nw)*nj2;		  
		    m[1][i] = (bool) (m[1][i] || xx[i-is] || xx[i+ny*nx+is]);
		}
	    }
	}
    }

    if (!both) {
	free(xx);
	return;
    }

    for (iz=0; iz < nz; iz++) {
	for (iy=1; iy < ny; iy++) {
	    for (ix = nw*nj1; ix < nx-nw*nj1; ix++) {
		i = ix + nx * (iy + ny * iz);

		for (iw = 0; iw <= 2*nw; iw++) {
		    is = (iw-nw)*nj1;		  
		    m[2][i] = (bool) (m[2][i] || xx[i-is] || xx[i-nx+is]);
		}
	    }
	}
    }

    for (iz=1; iz < nz; iz++) {
	for (iy=0; iy < ny; iy++) {
	    for (ix = nw*nj2; ix < nx-nw*nj2; ix++) {
		i = ix + nx * (iy + ny * iz);

		for (iw = 0; iw <= 2*nw; iw++) {
		    is = (iw-nw)*nj2;		  
		    m[3][i] = (bool) (m[3][i] || xx[i-is] || xx[i-ny*nx+is]);
		}
	    }
	}
    }
	
    free(xx); 
}

void mask3 (int nw         /* filter size */, 
	    int nj         /* dealiasing stretch */, 
	    int nx, int ny /* data size */, 
	    float **yy     /* data */, 
	    bool **mm      /* mask */) 
/*< one-dip mask in 2-D >*/
{
    int ix, iy, iw, is;
    bool **xx;

    xx = ps_boolalloc2(nx,ny);
    
    for (iy=0; iy < ny; iy++) {
	for (ix=0; ix < nx; ix++) {
	    xx[iy][ix] = (bool) (yy[iy][ix] == 0.);
	    mm[iy][ix] = false;
	}
    }

    for (iy=0; iy < ny-1; iy++) {
	for (ix = nw*nj; ix < nx-nw*nj; ix++) {
	    for (iw = 0; iw <= 2*nw; iw++) {
		is = (iw-nw)*nj;
		mm[iy][ix] = (bool) (mm[iy][ix] || xx[iy+1][ix+is] || xx[iy][ix-is]);
	    }
	}
    }
    
    free(xx[0]);
    free(xx);
}

void mask6 (int nw           /* filter size */, 
	    int nj1, int nj2 /* dealiasing stretch */, 
	    int nx, int ny   /* data size */, 
	    float *yy       /* data [ny][nx] */, 
	    bool *mm        /* mask [ny][nx] */) 
/*< two-dip mask in 2-D >*/
{
    int ix, iy, iw, is, n, i;
    bool *xx;

    n = nx*ny;

    xx = ps_boolalloc(n);
    
    for (i=0; i < n; i++) {
	mm[i] = (bool) (yy[i] == 0.);
	xx[i] = false;
    }

    for (iy=0; iy < ny-1; iy++) {
	for (ix = nw*nj1; ix < nx-nw*nj1; ix++) {
	    i = ix + nx*iy;

	    for (iw = 0; iw <= 2*nw; iw++) {
		is = (iw-nw)*nj1;
		xx[i] = (bool) (xx[i] || mm[i+nx+is] || mm[i-is]);
	    }
	}
    }
    
    for (i=0; i < n; i++) {
	mm[i] = false;
    }
    
    for (iy=0; iy < ny-1; iy++) {
	for (ix = nw*nj2; ix < nx-nw*nj2; ix++) {
	    i = ix + nx*iy;
	    
	    for (iw = 0; iw <= 2*nw; iw++) {
		is = (iw-nw)*nj2;
		mm[i] = (bool) (mm[i] || xx[i+nx+is] || xx[i-is]);
	    }
	}
    }
    
    free(xx);
}






/*from allp3.c*/
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


/*from dip3.c*/
static float *u1, *u2, *dp, *p0, eps;
static int nd, n1, n2, n3, nn[3];

void dip3_init(int m1, int m2, int m3 /* dimensions */, 
	       int* rect              /* smoothing radius [3] */, 
	       int niter              /* number of iterations */,
	       float eps1             /* regularization */,      
	       bool verb              /* verbosity flag */)
/*< initialize >*/
{
    n1=m1;
    n2=m2;
    n3=m3;
    nd = n1*n2*n3; /*number of data samples*/
    eps = eps1;

    u1 = ps_floatalloc(nd);
    u2 = ps_floatalloc(nd);
    dp = ps_floatalloc(nd);
    p0 = ps_floatalloc(nd);

    nn[0]=n1;
    nn[1]=n2;
    nn[2]=n3;

    ps_divn_init (3, nd, nn, rect, niter, verb);
}

void dip3_close(void)
/*< free allocated storage >*/
{
    free (u1);
    free (u2);
    free (dp);
    ps_divn_close();
}

void dip3(bool left               /* left or right prediction */,
	  int dip                 /* 1 - inline, 2 - crossline */, 
	  int niter               /* number of nonlinear iterations */, 
	  int nw                  /* filter size */, 
	  int nj                  /* filter stretch for aliasing */, 
	  bool drift              /* if shift filter */,
	  float *u                /* input data */, 
	  float* p                /* output dip */, 
	  bool* mask              /* input mask for known data */,
	  float pmin, float pmax  /* minimum and maximum dip */)
/*< estimate local dip >*/
{
    int i, iter, k;
    float usum, usum2, pi, lam;
    allpass ap;
 
    ap = allpass_init (nw,nj,n1,n2,n3,drift,p);

    if (dip == 1) {
	allpass1 (left, false, ap, u,u2);
    } else {
	allpass2 (left, false, ap, u,u2);
    }

    for (iter =0; iter < niter; iter++) {
	if (dip == 1) {
	    allpass1 (left, true,  ap, u,u1);
	} else {
	    allpass2 (left, true,  ap, u,u1);
	}

	usum = 0.0;
	for(i=0; i < nd; i++) {
	    p0[i] = p[i];
	    usum += u2[i]*u2[i];
	}
	
	if (NULL != mask) {
	    for(i=0; i < nd; i++) {
		if (mask[i]) {
		    u1[i] = 0.;
		    u2[i] = 0.;
		}
	    }
	}

	ps_divne (u2, u1, dp, eps);

	lam = 1.;
	for (k=0; k < 8; k++) {
	    for(i=0; i < nd; i++) {
		pi = p0[i]+lam*dp[i];
		if (pi < pmin) pi=pmin;
		if (pi > pmax) pi=pmax;
		p[i] = pi;
	    }
	    if (dip == 1) {
		allpass1 (left, false, ap, u,u2);
	    } else {
		allpass2 (left, false, ap, u,u2);
	    }

	    usum2 = 0.;
	    for(i=0; i < nd; i++) {
		usum2 += u2[i]*u2[i];
	    }
	    if (usum2 < usum) break;
	    lam *= 0.5;
	}
    } /* iter */

    allpass_close(ap);
}


static PyObject *dipc(PyObject *self, PyObject *args){
	
    /*Below is the input part*/
    int f2,f3,f4,f5,f6,f7;
    float f8,f9,f10;
    int f11,f12,f13,f14,f15;
    
	/**initialize data input**/
    int nd, nd2;
    
    PyObject *f1=NULL;
    PyObject *arrf1=NULL;

    
	PyArg_ParseTuple(args, "Oiiiiiifffiiiii", &f1, &f2, &f3, &f4, &f5, &f6, &f7, &f8, &f9, &f10, &f11, &f12, &f13, &f14, &f15); 	
    
    int n123, niter, order, nj1,nj2, i,j, liter, dim;
    int n[PS_MAX_DIM], rect[3], n4, nr, ir; 
    float p0, q0, *u, *um, *p, *pi=NULL, *qi=NULL;
    float pmin, pmax, qmin, qmax, eps;
    char key[4];
    bool verb, both, **mm, drift, hasmask;

    dim=4;
    if (dim < 2) n[1]=1;
    if (dim < 3) n[2]=1;
    n123 = n[0]*n[1]*n[2];
    nr = 1;
    for (j=3; j < dim; j++) {
	nr *= n[j];
    }

//     if (!sf_getbool("both",&both)) 
    both=false;
    /* if y, compute both left and right predictions */

//     if (1 == n[2]) {
// 	n4=0;
// 	if (both) sf_putint(out,"n3",2);
//     } else {
// 	if (!sf_getint("n4",&n4)) n4=2;
// 	/* what to compute in 3-D. 0: in-line, 1: cross-line, 2: both */ 
// 	if (n4 > 2) n4=2;
// 	if (2==n4) {
// 	    sf_putint(out,"n4",both? 4:2);
// 	    for (j=3; j < dim; j++) {
// 		snprintf(key,4,"n%d",both? j+4:j+2);
// 		sf_putint(out,key,n[j]);
// 	    }
// 	} else if (both) {
// 	    sf_putint(out,"n4",2);
// 	    for (j=3; j < dim; j++) {
// 		snprintf(key,4,"n%d",j+2);
// 		sf_putint(out,key,n[j]);
// 	    }
// 	}
//     }

// 	n4=0;
	
    niter=5;
    /* number of iterations */
    liter=20;
    /* number of linear iterations */
    rect[0]=1;
    /* dip smoothness on 1st axis */
    rect[1]=1;
    /* dip smoothness on 2nd axis */
    rect[2]=1;
    /* dip smoothness on 3rd axis */
    p0=0.;
    /* initial in-line dip */
    q0=0.;
    /* initial cross-line dip */
    order=1;
    /* accuracy order */
    nj1=1;
    /* in-line antialiasing */
    nj2=1;
    /* cross-line antialiasing */
    drift=false;
    /* if shift filter */
    verb = false;
    /* verbosity flag */
    
    /* minimum inline dip */
    /* maximum inline dip */
    /* minimum cross-line dip */
    /* maximum cross-line dip */
	pmin=-340282346638528859811704183484516925440.000000;
	pmax=340282346638528859811704183484516925440.000000;
	qmin=-340282346638528859811704183484516925440.000000;
	qmax=340282346638528859811704183484516925440.000000;
	/*printf("pmin pmax qmin qmax=%f,%f,%f,%f\n",pmin,pmax,qmin,qmax);*/

	n[0]=f2;
	n[1]=f3;
	n[2]=f4;
	n123=n[0]*n[1]*n[2];
	niter=f5;
	liter=f6;
	order=f7;
	eps=f8;
	
    rect[0]=f11;
    rect[1]=f12;
    rect[2]=f13;
	hasmask=f14;
	verb=f15;
// 	verb=true;
	
    arrf1 = PyArray_FROM_OTF(f1, NPY_FLOAT, NPY_IN_ARRAY);
    
    nd2=PyArray_NDIM(arrf1);
    npy_intp *sp=PyArray_SHAPE(arrf1);

	
	if(hasmask==0)
    	if (*sp != n123)
    	{
    		printf("Dimension mismatch, N_input = %d, N_data = %d\n", *sp, n123);
    		return NULL;
    	}
    printf("rect1=%d,rect2=%d,rect3=%d\n",rect[0],rect[1],rect[2]);
	printf("both=%d,hasmask=%d,verb=%d\n",both,hasmask,verb);
	printf("n123=%d\n",n123);
	printf("eps_dv=%f\n",eps);


	if(n[2]==1)
	{
	n4=0;
	p = ps_floatalloc(n123);
	}
	else
	{
	n4=2; /*calculate both inline and crossline dips*/
	p = ps_floatalloc(n123*2);
	}
    u = ps_floatalloc(n123);
    um = ps_floatalloc(n123); /*temporary array for mask*/
    
    
    /*reading data*/
    for (i=0; i<n123; i++)
    {
        u[i]=*((float*)PyArray_GETPTR1(arrf1,i));
        p[i]=0;
        if(n4==2)
        	p[i+n123]=0;
    }

    /* initialize dip estimation */
    dip3_init(n[0], n[1], n[2], rect, liter, eps, verb);



    if (hasmask) {
	mm = ps_boolalloc2(n123,both? 4:2);
// 	mask = sf_input("mask");
    } else {
	mm = (bool**) ps_alloc(4,sizeof(bool*));
	mm[0] = mm[1] = mm[2] = mm[3] = NULL;
// 	mask = NULL;
    }

//     if (NULL != sf_getstring("idip")) {
// 	/* initial in-line dip */
// 	idip0 = sf_input("idip");
// 	if (both) pi = ps_floatalloc(n123);
//     } else {
// 	idip0 = NULL;
//     }

//     if (NULL != sf_getstring("xdip")) {
// 	/* initial cross-line dip */
// 	xdip0 = sf_input("xdip");
// 	if (both) qi = ps_floatalloc(n123);
//     } else {
// 	xdip0 = NULL;
//     }

	nr=1;
    for (ir=0; ir < nr; ir++) {
    	if (hasmask) {
// 	    sf_floatread(u,n123,mask);
    	for (i=0; i<n123; i++)
        	um[i]=*((float*)PyArray_GETPTR1(arrf1,i+n123));
	    mask32 (both, order, nj1, nj2, n[0], n[1], n[2], um, mm);
	}
	
	if (1 != n4) {
	    /* initialize t-x dip */
// 	    if (NULL != idip0) {
// 		if (both) {
// // 		    sf_floatread(pi,n123,idip0);
// 		    for(i=0; i < n123; i++) {
// 			p[i] = pi[i];
// 		    }
// 		} else {
// // 		    sf_floatread(p,n123,idip0);
// 		}
// 	    } else {
		for(i=0; i < n123; i++) {
		    p[i] = p0;
// 		}
	    }
	    
	    /* estimate t-x dip */
	    dip3(false, 1, niter, order, nj1, drift, u, p, mm[0], pmin, pmax);
	    
	}

	if (0 != n4) {
	    /* initialize t-y dip */
// 	    if (NULL != xdip0) {
// 		if (both) {
// 		    sf_floatread(qi,n123,xdip0);
// 		    for(i=0; i < n123; i++) {
// 			p[i] = qi[i];
// 		    }
// 		} else {
// 		    sf_floatread(p,n123,xdip0);
// 		}
// 	    } else {
		for(i=0; i < n123; i++) {
		    p[i+n123] = q0;
		}
// 	    }	
	    
	    /* estimate t-y dip */
	    dip3(false, 2, niter, order, nj2, drift, u, p+n123, mm[1], qmin, qmax);

	}

	if (!both) continue;

	if (1 != n4) {
	    /* initialize t-x dip */
// 	    if (NULL != idip0) {
// 		for(i=0; i < n123; i++) {
// 		    p[i] = -pi[i];
// 		}
// 	    } else {
		for(i=0; i < n123; i++) {
		    p[i] = -p0;
		}
// 	    }
	    
	    /* estimate t-x dip */
	    dip3(true, 1, niter, order, nj1, drift, u, p+n123*2, mm[2], -pmax, -pmin);

	}

	if (0 != n4) {
	    /* initialize t-y dip */
// 	    if (NULL != xdip0) {
// 		for(i=0; i < n123; i++) {
// 		    p[i] = -qi[i];
// 		}
// 	    } else {
		for(i=0; i < n123; i++) {
		    p[i] = -q0;
		}
// 	    }	
	    
	    /* estimate t-y dip */
	    dip3(true, 2, niter, order, nj2, drift, u, p+n123*3, mm[3], -qmax, -qmin);
	}	
    }

    /*Below is the output part*/
    PyArrayObject *vecout;
	npy_intp dims[2];
	if(n4==0)
	{dims[0]=n123;dims[1]=1;}
	else
	{
	dims[0]=n123*2;dims[1]=1;
	}
	/* Parse tuples separately since args will differ between C fcns */
	/* Make a new double vector of same dimension */
	vecout=(PyArrayObject *) PyArray_SimpleNew(1,dims,NPY_FLOAT);
	for(i=0;i<dims[0];i++)
		(*((float*)PyArray_GETPTR1(vecout,i))) = p[i];

	
	return PyArray_Return(vecout);
	
}


























// documentation for each functions.
static char dipcfun_document[] = "Document stuff for dip...";

// defining our functions like below:
// function_name, function, METH_VARARGS flag, function documents
static PyMethodDef functions[] = {
  {"dipc", dipc, METH_VARARGS, dipcfun_document},
  {NULL, NULL, 0, NULL}
};

// initializing our module informations and settings in this structure
// for more informations, check head part of this file. there are some important links out there.
static struct PyModuleDef dipcfunModule = {
  PyModuleDef_HEAD_INIT, // head informations for Python C API. It is needed to be first member in this struct !!
  "dipcfun",  // module name
  NULL, // means that the module does not support sub-interpreters, because it has global state.
  -1,
  functions  // our functions list
};

// runs while initializing and calls module creation function.
PyMODINIT_FUNC PyInit_dipcfun(void){
  
    PyObject *module = PyModule_Create(&dipcfunModule);
    import_array();
    return module;
}
