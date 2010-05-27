#define N 2048

double * u; 

double * f; 

double * uold; 

void * mimalloc(unsigned int size) {
        void * tmp = malloc(size);

        if (!tmp) {
                printf("** Error malloc \n");
                exit(1);
        }

        return tmp;
}


double t(void) {
    struct timeval tv;
    assert (gettimeofday(&tv, 0) == 0);
    return (double)tv.tv_sec + (double)tv.tv_usec/1000000.0;
}
double time_start, time_end;



void initialize(int n, int m, double alpha, double *_dx, double *_dy,double * u, double * f)
{

        int i;
        int j;
        int xx;
        int yy;
        double dx = 2.0 / (n - 1);
        double dy = 2.0 / (m - 1);
        printf("*** Initialize  n: %d m: %d alpha : %g\n", n, m, alpha);
        for (i = 0; i < m; i++) {

                for (j = 0; j < n; j++) {

                        xx = -1.0 + (dx * (double) (i));
                        yy = -1.0 + (dy * (double) (j));
                        u[(j * N) + i] = 0.0;
                        f[(j * N) + i] =
                                (((-alpha * (1.0 - (xx * xx))) * (1.0 - (yy * yy))) -
                                 (2.0 * (1.0 - (xx * xx)))) - (2.0 * (1.0 - (yy * yy)));
                }

        }

        printf("*** Initialize  n: %d m: %d alpha : %g\n", n, m, alpha);
        *_dx = dx;
        *_dy = dy;
}


void jacobi(int n, int m, double *_dx, double *_dy, double alpha, double omega,
                double u[N*N], double f[N*N], double uold[N*N], double tol, double maxit)
{

        int i;
        int j;
        int k;
        double error;
        double resid;
        double ax;
        double ay;
        double b;
        double dx = *_dx;
        double dy = *_dy;

        ax = 1.0 / (dx * dx);
        ay = 1.0 / (dy * dy);
        b = ((-2.0 / (dx * dx)) - (2.0 / (dy * dy))) - alpha;
        error = 10.0 * tol;
        k = 1;
        while ((k < maxit) && (error > tol)) {

                error = 0.0;
                {
#pragma omp target device (cuda) copy_out(uold) 
#pragma omp parallel shared(omega,error,tol,n,m,ax,ay,b,alpha,uold,u,f)  private(i,j,resid) 
{
 #pragma omp for
                        for (i = 0; i < m; i++)
                                for (j = 0; j < n; j++)
                                        uold[(j * N) + i] = u[(j * N) + i];
}

#pragma omp target device (cuda) copy_out(u)
#pragma omp parallel shared(omega,error,tol,n,m,ax,ay,b,alpha,uold,u,f)  private(i,j,resid) 
{
#pragma omp for reduction(+:error )
                        for (i = 0; i < (m - 2); i++) {

                                for (j = 0; j < (n - 2); j++) {

                                        resid =
                                                ((((ax *
                                                    (uold[((j + 1) * N) + ((i + 1) - 1)] +
                                                     uold[((j + 1) * N) + ((i + 1) + 1)])) +
                                                   (ay *
                                                    (uold[(((j + 1) - 1) * N) + (i + 1)] +
                                                     uold[(((j + 1) + 1) * N) + (i + 1)]))) +
                                                  (b * uold[((j + 1) * N) + (i + 1)])) -
                                                 f[((j + 1) * N) + (i + 1)]) / b;
                                        u[((j + 1) * N) + (i + 1)] =
                                                uold[((j + 1) * N) + (i + 1)] - (omega * resid);
                                        error += resid * resid;
                                }

                        }
}

                }

                k++;
                error = sqrt(error) / (double) (n * m);
        }

        printf("Total Number of Iterations %d \n", k);
        printf("Residual %g \n", error);
        *_dx = dx;
        *_dy = dy;
}

void error_check(int n, int m, double alpha, double *_dx, double *_dy,
                double * u, double *f)
{

        int i;
        int j;
        double xx;
        double yy;
        double temp;
        double error;
        double dx = 2.0 / (n - 1);
        double dy = 2.0 / (m - 1);
        error = 0.0;
        for (i = 0; i < m; i++)
                for (j = 0; j < n; j++) {
                        xx = -1.0L + (dx * (double) (i - 1));
                        yy = -1.0L + (dy * (double) (j - 1));
                        temp = u[(j * N) + i] - ((1.0L - (xx * xx)) * (1.0L - (yy * yy)));
                        error += temp * temp;
                }

        error = sqrt(error) / (double) (n * m);
        printf("Solution Error : %g \n", error);
        *_dx = dx;
        *_dy = dy;
}



void driver()
{

   double dx;
   double dy;

   u = (double *) mimalloc(N*N*sizeof(double));
   f = (double *) mimalloc(N*N*sizeof(double));
   uold = (double *) mimalloc(N*N*sizeof(double));

   initialize(N, N, 0.00000005L, &dx, &dy, u, f);

   time_start = t();

   jacobi(N, N, &dx, &dy, 0.00000005L, 0.01L, u, f, uold, 0.00001L, 100);

   time_end = t();

   error_check(N, N, 0.00000005L, &dx, &dy, u, f);


   printf("** Jacobi time : %g \n", time_end - time_start);
}


int main(int argc, char *argv[])
{

        driver();
        return 0;
}

