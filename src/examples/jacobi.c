/*


   12/05/2010   Jacobi obtenido de 

	http://www.openmp.org/samples/jacobi.f


    Traducido por Kiko y Ruyman


*/


/*  program to solve a finite difference  */
/*  discretization of Helmholtz equation :   */
/*  (d2/dx2)u + (d2/dy2)u - alpha u = f  */
/*  using Jacobi iterative method.  */
/*   */
/*  Directives are used in this code to achieve paralleism.  */
/*  All do loops are parallized with 'static even' scheduling to  */
/*  maximize performance.  */
/*   */
/*  Input :  n - grid dimension in x direction  */
/*           m - grid dimension in y direction */
/*           alpha - Helmholtz constant (always greater than 0.0) */
/*           tol   - error tolerance for iterative solver */
/*           relax - Successice over relaxation parameter */
/*           mits  - Maximum iterations for iterative solver */
/*  */
/*  On output  */
/*        : u(n,m) - Dependent variable (solutions) */
/*        : f(n,m) - Right hand side function  */
/* ************************************************************ */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/* Constants */
#define N 500
#define M 500
#define ALPHA 0.00000005L
#define RELAX 0.01L
#define TOL 0.00001L
#define MITS 100

double u[N][M];
double f[N][M];


/* ***************************************************** */
/*  Initializes data  */
/*  Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2) */
/*  */
/* ***************************************************** */
void initialize(int n, int m, double alpha, double *_dx, double *_dy,
	   double u[N][M], double f[N][M])
{
    int i, j, xx, yy;
    double dx = 2.0 / (n - 1);
    double dy = 2.0 / (m - 1);
/*  Initilize initial condition and RHS */

    printf("*** Initialize  n: %d m: %d alpha : %g\n", n, m, alpha);
/* $omp parallel do  schedule(static)  */
/* $omp& shared(n,m,dx,dy,u,f,alpha) */
/* $omp& private(i,j,xx,yy) */
    for (i = 0; i < m; i++) {
	for (j = 0; j < n; j++) {
	    xx = -1.0 + dx * (double) (i);	/* -1 < x < 1 */
	    yy = -1.0 + dy * (double) (j);	/* -1 < y < 1 */
	    u[i][j] = 0.0;
	    f[i][j] = -alpha * (1.0 - xx * xx) * (1.0 - yy * yy) - 2.0 * (1.0 - xx * xx) - 2.0 * (1.0 - yy * yy);
	}
    }
/*  $omp end parallel do */

    printf("*** Initialize  n: %d m: %d alpha : %g\n", n, m, alpha);
    *_dx = dx;
    *_dy = dy;
}


/* ***************************************************************** */
/*  Subroutine HelmholtzJ */
/*  Solves poisson equation on rectangular grid assuming :  */
/*  (1) Uniform discretization in each direction, and  */
/*  (2) Dirichlect boundary conditions  */
/*   */
/*  Jacobi method is used in this routine  */
/*  */
/*  Input : n,m   Number of grid points in the X/Y directions  */
/*          dx,dy Grid spacing in the X/Y directions  */
/*          alpha Helmholtz eqn. coefficient  */
/*          omega Relaxation factor  */
/*          f(n,m) Right hand side function  */
/*          u(n,m) Dependent variable/Solution */
/*          tol    Tolerance for iterative solver  */
/*          maxit  Maximum number of iterations  */
/*  */
/*  Output : u(n,m) - Solution  */
/* **************************************************************** */

void jacobi(int n, int m, double *_dx, double *_dy, double alpha, double omega,
       double u[N][M], double f[N][M], double tol, double maxit)
{
    int i, j, k;
    double error, resid, ax, ay, b;
    double uold[N][M];

    double dx = *_dx;
    double dy = *_dy;


    /*  Initialize coefficients */
    ax = 1.0 / (dx * dx);	/*  X-direction coef */
    ay = 1.0 / (dy * dy);	/*   Y-direction coef */
      b = -2.0 / (dx * dx) - 2.0 / (dy * dy) - alpha;	/*  Central coeff   */

    error = 10.0 * tol;
    k = 1;

     while (k < maxit && error > tol) { 
	error = 0.0;
/*  Copy new solution into old */
/* $omp paralleldo schedule(static)  */
/* $omp& shared(n,m,uold,u) */
/* $omp&private(i,j) */
#pragma omp parallel shared(omega,error,tol,n,m,ax,ay,b,alpha,uold,u,f) private(i, j, resid)
{
   #pragma omp for
	for (i = 0; i < m; i++)
	    for (j = 0; j < n; j++)
		uold[i][j] = u[i][j];

/* $omp end paralleldo */

/*  Compute stencil, residual, & update */

/* $omp paralleldo schedule(static) */
/* $omp& shared(omega,error,tol,n,m,ax,ay,b,alpha,uold,u,f) */
/* $omp&private(i,j,resid) */
/* $omp&reduction(+:error)  */

   #pragma omp for reduction(+ : error)
	for (i = 1; i < m - 1; i++) {
	    for (j = 1; j < n - 1; j++) {
		   resid = (ax * (uold[i - 1][j] + uold[i + 1][j])  /*      Evaluate residual  */
		    + ay * (uold[i][j - 1] + uold[i][j + 1])
		    + b * uold[i][j] - f[i][j]) / b;
		   u[i][j] = uold[i][j] - omega * resid;  /*  Update solution  */
		   error += resid * resid;                /*  Accumulate residual error */
	    }
}

	}
/* $omp end paralleldo  */

	/*  Error check  */
	k++;
	error = sqrt(error) / (double) (n * m);
    } /*  End iteration loop  */

    printf("Total Number of Iterations %d \n", k);
    printf("Residual %g \n", error);
    *_dx = dx;
    *_dy = dy;

}


/* *********************************************************** */
/*  Checks error between numerical and exact solution          */
/* *********************************************************** */
void error_check(int n, int m, double alpha, double *_dx, double *_dy,
	    double u[N][M], double f[N][M])
{
    int i, j;
    double xx, yy, temp, error;

    double dx = 2.0 / (n - 1);
    double dy = 2.0 / (m - 1);
    error = 0.0;

/* $omp parallel do schedule(static) */
/* $omp& shared(n,m,dx,dy,u,error) */
/* $omp& private(i,j,xx,yy,temp) */
/* $omp& reduction(+:error) */
    for (i = 0; i < m; i++)
	for (j = 0; j < n; j++) {
	    xx = -1.0L + dx * (double) (i - 1);
	    yy = -1.0L + dy * (double) (j - 1);
	    temp = u[i][j] - (1.0L - xx * xx) * (1.0L - yy * yy);
	    error += temp * temp;
	}
    error = sqrt(error) / (double) (n * m);
    printf("Solution Error : %g \n", error);
    *_dx = dx;
    *_dy = dy;
}

/* ************************************************************ */
/*  Subroutine driver ()  */
/*  This is where the arrays are allocated and initialzed.  */
/*  */
/*  Working varaibles/arrays  */
/*      dx  - grid spacing in x direction  */
/*      dy  - grid spacing in y direction  */
/* ************************************************************ */
void driver()
{
    double dx, dy;

    initialize(N, M, ALPHA, &dx, &dy, u, f);
    jacobi(N, M,  &dx, &dy, ALPHA, RELAX, u, f, TOL, MITS);
    error_check(N, M, ALPHA, &dx, &dy, u, f);

}



int main(int argc, char *argv[])
{

    driver();

   return 0;
}


