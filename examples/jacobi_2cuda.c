/* #include <stdio.h> */
/* #include <stdlib.h> */
/* #include <math.h> */

/* #include "include/llcomp_cuda.h" */
/* #include <sys/time.h>  */
/* #include "examples/mytime.h" */

#define N_DEF 200
#define TOL_DEF 0.0000001
#define MAXIT_DEF 500


// Reserva dinamicamente espacio para un vector de tamanyo m
void ReservaVector (double **v, int m) {

  *v = (double *) malloc(m * sizeof(double) );

}

// Libera el espacio reservado dinamicamente para un vector
void LiberaVector (double *v) {
  free(v);
}

/*
// Reserva dinamicamente espacio para una matrix a de mxn
void ReservaMatriz (double ***a, int m, int n) {
  int i;

  *a = (double **) malloc(m * sizeof(double *) );
  for (i = 0; i < m; i++) 
    (*a)[i] = (double *) malloc(n * sizeof(double) );

}

// Libera el espacio reservado dinamicamente para una matrix a de m filas
void LiberaMatriz( double **a, int m) {
  int i;

  for (i = 0; i < m; i++)
    free(a[i]);
  free(a);


}
*/


/* Otra forma de reservar y liberar matrices */
/* En este caso el espacio ocupado por los elementos es un
bloque de memoria consecutiva */

void ReservaMatriz (double ***a, int m, int n) {
  int i;

//  *espacio = (double *) malloc(m * n * sizeof(double) );
  *a = (double **) malloc(m * sizeof(double *) );
  (*a)[0] = (double *) malloc(m * n * sizeof(double) );
  for (i = 1; i < m; i++) 
    (*a)[i] = (*a)[i-1] + n;
}

/*
void ReservaMatriz (double ***a, int m, int n, double **espacio) {
  int i;

  *espacio = (double *) malloc(m * n * sizeof(double) );
  *a = (double **) malloc(m * sizeof(double *) );
  for (i = 0; i < m; i++) 
    (*a)[i] = &(*espacio)[i*n];
}
*/

// Libera el espacio reservado dinamicamente para una matrix a de m filas
void LiberaMatriz( double **a) {
  free(a[0]);
  free(a);
}

/*
void LiberaMatriz( double **a, double *espacio) {
  free(espacio);
  free(a);
}
*/

// Genera aleatoriamente un vector v de tamanyo m
void GeneraVector (double *v, int m) {
  int i;
  unsigned short semilla[3];
  /*time_t t;*/

  semilla[0] = 1;
  semilla[1] = (unsigned) 25; /*time(&t);*/
  semilla[1] = 3; 

  for (i = 0; i < m; i++) 
    v[i] = 10.0 * erand48(semilla);

}

// Genera aleatoriamente una matriz a de tamanyo mxn
void GeneraMatriz (double **a, int m, int n) {
  int i, j;
  double *ptr;
  unsigned short semilla[3];
/*  time_t t;*/

  semilla[0] = 1;
  semilla[1] = 2;
  semilla[2] = (unsigned) 25; /*time(&t);*/

  for (i = 0; i < m; i++) {
    ptr = &a[i][0];
    for (j = 0; j < n; j++, ptr++)
      *ptr = 10.0 * erand48(semilla);
  }

}

// Copia el vector x en y
void CopiaVector (double *x, int n, double *y) {
  int i;
 
  for (i = 0; i < n; i++) 
    y[i] = x[i]; 

}

// Copia la matriz a en b
void CopiaMatriz (double **a, int m, int n, double **b) {
  int i, j;
  double *ptra, *ptrb;
 
  for (i = 0; i < m; i++) {
    ptra = &a[i][0];
    ptrb = &b[i][0];
    for (j = 0; j < n; j++, ptra++, ptrb++)
      *ptrb = *ptra; 
  }      

}

// Escribe por pantalla una matriz a de tamanyo mxn
void EscribeMatriz (double **a, int m, int n) {
  int i, j;
  double *ptr;
 
  for (i = 0; i < m; i++) {
    ptr = &a[i][0];
    for (j = 0; j < n; j++, ptr++)
      printf(" %9.4g ", *ptr);
    printf("\n");
  }      

}

// Escribe por pantalla un vector v de tamanyo m
void EscribeVector (double *v, int m){
  int i;

  for (i = 0; i < m; i++)
    printf(" %9.4g ", v[i]);
  printf("\n");

}



/****************/
int *despl;
int *count;
/****************/

void CompruebaSolucion(double **a, double *x, double *b, int  n) {
  int i, j;
  double sum, dif = 0.0;

  for (i = 0; i < n; i++) {
    sum = 0.0;
    for (j = 0; j < n; j++)
      sum += a[i][j]*x[j];
    dif += (b[i] - sum)*(b[i] - sum);
  }
  dif = sqrt(dif);
  printf("||b-x||_2 = %g\n", dif);
}


void print_matrix (double **a, int f, int c) {
   int i,j;
   for (i=0; i < f; i++) {
      for (j=0; j < c; j++)
         printf ("%f ", a[i][j]);
      printf ("\n");
   }
}
void print_vector (int *v, int *h,  int c) {
   int j;
   printf ("vector --\t");
   /*for (j=0; j < c; j++)
       printf ("%d ", v[j]);
   printf ("\t---- ");*/
   for (j=0; j < c; j++)
       printf ("%d ", h[j]);
   printf ("\n");
}

double jacobi_sec(double **a, int n, double *b, double *x, double tol, int maxit) {
  int i, j, it = 0;
  double sum, dif, resi = 10.0, *new_x;

  new_x = (double *) malloc(n*sizeof(double));

  for (i = 0; i < n; i++) x[i] = b[i];
  /* while ( (it < maxit) && (resi > tol) ) { */
  while (it < maxit) {
    resi = 0.0;
    for (i = 0; i < n; i++) {
      sum = 0.0;
      for (j = 0; j < n; j++) 
        sum += a[i][j]*x[j];
      resi += fabs(sum - b[i]);
      sum += -a[i][i]*x[i];
      new_x[i] = (b[i] - sum) / a[i][i];
    }
    for (i = 0; i < n; i++) x[i] = new_x[i];
    it++;
  } 

/*  printf("Secuencial Iteraciones: %d, resi: %g\n", it, resi); */
  free(new_x);
  return resi;  
}

double jacobi_llc(double a[N_DEF][N_DEF], int n, double b[N_DEF], double x[N_DEF], double tol, int maxit) {
  int i, j, it = 0;
  double sum, dif, resi = 10.0;/*, *new_x;*/
  double new_x[N_DEF];

/*  new_x = (double *) malloc(n*sizeof(double)); */
  

  for (i = 0; i < n; i++) x[i] = b[i];
  /* while ( (it < maxit) && (resi > tol) ) { */
  while (it < maxit) {
    resi = 0.0;
#pragma omp parallel  private (sum, j)  shared(a, b, new_x, x)
{
   #pragma omp for reduction (+ : resi)
    for (i = 0; i < n; i++) {
      sum = 0.0;
      for (j = 0; j < n; j++) 
        sum += a[i][j]*x[j];
      resi += fabs(sum - b[i]);
      sum += -a[i][i]*x[i];
      new_x[i] = (b[i] - sum) / a[i][i];
    }
    for (i = 0; i < n; i++) x[i] = new_x[i];
    it++;
  } 
}

/*  printf("llc Iteraciones: %d, resi: %g\n", it, resi); */
  free(new_x);

  return resi;
}


int main(int argc, char *argv[]) {
  double tiempo, tol;
  double **a, *b, *x;
  int i, j, n, nfilas, diag, maxit, it;

/*   CLOCK_TYPE chrono;*/
  double seq_time, llc_time, mpi_time;
  double seq_resi, llc_resi, mpi_resi;

  n = N_DEF;
  maxit = MAXIT_DEF;
  tol = TOL_DEF;

 /*  switch (argc) {
    case 4: tol = atof(argv[3]);
    case 3: maxit = atoi(argv[2]);
    case 2: n = atoi(argv[1]);
  } */

  printf ("\n\nEjecutando jacobi con n = %d, maxit = %d, tol = %g\n\n", n, maxit, tol);
  
  /* Por simplicidad supondremos que n es divisible por el numero de procesos: nprocs */
  nfilas = n;

  ReservaMatriz(&a, n, n);
  ReservaVector(&b, n);
  ReservaVector(&x, n);

  GeneraVector(b, n);

  /* matriz aleatoria */
  GeneraMatriz(a, n, n);
  /* Para garantizar la convergencia del metodo de Jacobi */
  /* Convierte en diagonal dominante la matriz distribuida */
  diag = 0;
  for (i = 0; i < n; i++) {
     for (j = 0; j < n; j++) 
        a[i][diag] += a[i][j];
     diag++;
  }
 
  /****************************************************************************
   algoritmo secuencial */
/*   CLOCK_Start(chrono);*/
  seq_resi = jacobi_sec(a, n, b, x, tol, maxit);
/*  CLOCK_End(chrono, seq_time);*/
 /* CompruebaSolucion (a, x, b, n); */

  /****************************************************************************
   jacobi llc */
/*  CLOCK_Start (chrono);*/
  llc_resi = jacobi_llc(a, n, b, x, tol, maxit);
/*  CLOCK_End(chrono, llc_time);*/
/*  CompruebaSolucion (a, x, b, n);*/

  /***************************************************************************
  ****************************************************************************/
  
 /* 
  printf ("%d\t%g\t#llc_plot0 RAP: N=%d, IT=%d, TOL=%g (t_seq=%g/t_llc=%g) SEQ_SOL=%g, LLC_SOL=%g\n",
                            LLC_NUMPROCESSORS, seq_time/llc_time, n, maxit, tol, seq_time, llc_time, seq_resi, llc_resi);
  printf ("%d\t%g\t#llc_plot1 RAP: N=%d, IT=%d, TOL=%g (t_seq=%g/t_mpi=%g) SEQ_SOL=%g, MPI_SOL=%g\n",
                            LLC_NUMPROCESSORS, seq_time/mpi_time, n, maxit, tol, seq_time, mpi_time, seq_resi, mpi_resi);
  printf ("%d\t%g\t#llc_plot2 RAP: N=%d, IT=%d, TOL=%g (t_mpi=%g/t_llc=%g) MPI_SOL=%g, LLC_SOL=%g\n",
                            LLC_NUMPROCESSORS, mpi_time/llc_time, n, maxit, tol, mpi_time, llc_time, mpi_resi, llc_resi);

*/
  
  /**************************************************************************
   Libeacion de memoria */
  LiberaMatriz(a);
  LiberaVector(b);
  LiberaVector(x);

  return 0;

}
