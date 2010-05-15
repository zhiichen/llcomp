#include <stdio.h>
#include <stdlib.h> 
#include <math.h>


/* ***********************************************************************
LIBRARY MYTIME

File: mytime.h
Version: 0.2
Date:    01.11.1999
Description: This library contains macros for handling time. It is based
  in the system routine gettimeofday() and associated types defined in
  <sys/time.h>.
History:
  21.01.1999 First version of GETTIME and DIFTIME
  01.11.1999 Modifications introduced during the TRACS visit to EPCC.
             CLOCK_Start, CLOCK_End
  19.11.1999 Include mpi.h to perform collective-reduction operations.
             It is necessary to introduce mytime.c to create functions.
************************************************************************ */
#ifndef MYTIME_H
#define MYTIME_H

 #include <time.h> 

#define MICROSEC_PER_SEC 1.0e6

typedef struct timeval CLOCK_TYPE;

/* ---------------------------------------------------------------------
Macro: GETTIME
Description: Gets system clock time in a data struct of type timeval.
------------------------------------------------------------------------ */
#define GETTIME(tv) gettimeofday(&(tv), NULL)

/* ---------------------------------------------------------------------
Macro: DIFTIME
Description: Compute the time in microsec between the time passing in
  the data structs tv2 and tv1. The result is giving in t (double).
------------------------------------------------------------------------ */
#define DIFTIME(tv2, tv1, t) t=(double)((tv2).tv_sec-(tv1).tv_sec)+    \
               (double)((tv2).tv_usec-(tv1).tv_usec) / MICROSEC_PER_SEC 

/* ---------------------------------------------------------------------
Macro: SEC2USEC
Description: Traslate ts seconds into tus microseconds.
------------------------------------------------------------------------ */
#define SEC2USEC(ts, tus) tus=(double)(ts)*MICROSEC_PER_SEC

/* ---------------------------------------------------------------------
Macro: CLOCK_Start
Description: Starts a chrono
------------------------------------------------------------------------ */
#define CLOCK_Start(ch) gettimeofday(&(ch), NULL)

/* ---------------------------------------------------------------------
Macro: CLOCK_End
Description: Ends a chrono
------------------------------------------------------------------------ */
#define CLOCK_End(ch, tm) {                                            \
  CLOCK_TYPE ch2;                                                      \
                                                                       \
  gettimeofday(&(ch2), NULL);                                          \
  tm=(double)((ch2).tv_sec - (ch).tv_sec) +                            \
     (double)((ch2).tv_usec - (ch).tv_usec) / MICROSEC_PER_SEC;        \
}

/* ---------------------------------------------------------------------
Function: CLOCK_Avg
Description: Computes the average value of t among processors
------------------------------------------------------------------------ */
double CLOCK_Avg(double t);

/* ---------------------------------------------------------------------
Function: CLOCK_Min
Description: Computes the minimum value of t among processors
------------------------------------------------------------------------ */
double CLOCK_Min(double t);

/* ---------------------------------------------------------------------
Function: CLOCK_Max
Description: Computes the maximum value of t among processors
------------------------------------------------------------------------ */
double CLOCK_Max(double t);

#endif  /* MYTIME_H */


#define LLC_printMaster printf
#define LLC_NUMPROCESSORS 1
#define LLC_NAME 

# define MAXITER 1000000
# define THRESOLD 2.0
/* # define npoints 4092 */
/* #define npoints 8184 */
#define npoints 16368

float uni (void) {
	return (float) random() / RAND_MAX;
}

void rinit (int seed) {
	srandom(seed);
}

  struct complex{
      double creal;
      double cimag;
  };

  int min(int a, int b) {
   return((a < b) ? a : b); 
  }

  main(int argc, char **argv){
    int i, j, numinside;
    int num_threads = 1;
    double area_seq, error_seq;
    double area_llc, error_llc;
    double area_mpi, error_mpi;
    struct complex z, c[npoints];

    double ztemp;
    int numoutside, gnumoutside;
    int nt;                 /* No. of threads */
		int MPI_NUMPROCESSORS, MPI_NAME;

		CLOCK_TYPE chrono;
		double t_llc, t_seq, t_omp;

		LLC_printMaster ("\n\n\n*************** NUMPROCESSORS = %d ***************************\n\n",
				LLC_NUMPROCESSORS);
    rinit (54321);
    for (i=0; i<npoints; i++) {
        c[i].creal = -2.0+2.5*uni();
        c[i].cimag = 1.125*uni();
    }



   printf ("** Serial Loop **\n");
	CLOCK_Start(chrono);

   numoutside = 0;
    for(i = 0; i<npoints; i++) {
      z.creal = c[i].creal;
      z.cimag = c[i].cimag;
     for (j = 0; j < MAXITER; j++) { 
        ztemp = (z.creal * z.creal) - (z.cimag * z.cimag) + c[i].creal;
        z.cimag = z.creal * z.cimag * 2 + c[i].cimag;
        z.creal = ztemp;
        if (z.creal * z.creal + z.cimag * z.cimag > 2.0) {
          numoutside++;
          break;
        }
     } /* for j */
    }  /* for i */

	numinside = npoints - numoutside;

/* *  5. PARALLEL llc: Calculate area and error */
  area_llc = 2.0 * 2.5 * 1.125 * numinside / npoints;
  error_llc = area_llc / sqrt(npoints);

	CLOCK_End(chrono, t_seq);
   printf("*** Area serial %g \n", area_llc);
   printf("*** Error serial %g \n", error_llc);

   printf ("** CUDA Loop **\n");
	CLOCK_Start(chrono);
/* #pragma omp parallel for reduction(+:numoutside) private(i,j,ztemp,z) shared(nt,c) */
  #pragma omp parallel private(z, ztemp, j) shared(nt, c)
  {
	 numoutside = 0;
    #pragma omp for reduction (+:numoutside)
    for(i = 0; i<npoints; i++) {
      z.creal = c[i].creal;
      z.cimag = c[i].cimag;
      for (j = 0; j < MAXITER; j++) {
        ztemp = (z.creal * z.creal) - (z.cimag * z.cimag) + c[i].creal;
        z.cimag = z.creal * z.cimag * 2 + c[i].cimag;
        z.creal = ztemp;
        if (z.creal * z.creal + z.cimag * z.cimag > THRESOLD) {
          numoutside++;
          break;
        } 
      } /* for j */
    } /* for i */
  }
	numinside = npoints - numoutside;

/* *  5. PARALLEL llc: Calculate area and error */
  area_llc = 2.0 * 2.5 * 1.125 * numinside / npoints;
  error_llc = area_llc / sqrt(npoints);
	
	CLOCK_End(chrono, t_llc);
   printf("*** Area llc %g \n", area_llc);
   printf("*** Error llc %g \n", error_llc);


	CLOCK_Start(chrono);
  #pragma omp parallel private(z, ztemp, j) shared(nt, c)
  {
	 numoutside = 0;
    #pragma omp for reduction (+:numoutside)
    for(i = 0; i<npoints; i++) {
      z.creal = c[i].creal;
      z.cimag = c[i].cimag;
      for (j = 0; j < MAXITER; j++) {
        ztemp = (z.creal * z.creal) - (z.cimag * z.cimag) + c[i].creal;
        z.cimag = z.creal * z.cimag * 2 + c[i].cimag;
        z.creal = ztemp;
        if (z.creal * z.creal + z.cimag * z.cimag > THRESOLD) {
          numoutside++;
          break;
        } 
      } /* for j */
    } /* for i */
  }
	numinside = npoints - numoutside;

/* *  5. PARALLEL llc: Calculate area and error */
  area_llc = 2.0 * 2.5 * 1.125 * numinside / npoints;
  error_llc = area_llc / sqrt(npoints);
	
	CLOCK_End(chrono, t_omp);
   printf("*** Area omp %g \n", area_llc);
   printf("*** Error omp %g \n", error_llc);


  LLC_printMaster ("%d\t%d\t%g\t#llc_plot0 MANDEL: N = %ld. [seq_time(%g)/llc_time(%g)]\n", num_threads, 
			              LLC_NUMPROCESSORS, (t_seq/t_llc), npoints, t_seq, t_llc);
  LLC_printMaster ("%d\t%d\t%g\t#llc_plot1 MANDEL: N = %ld. [seq_time(%g)/omp_time(%g)]\n", num_threads, 
			              LLC_NUMPROCESSORS, (t_seq/t_omp), npoints, t_seq, t_omp);

  }
