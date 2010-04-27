
/* #include <stdio.h> */

#define N_ELEM 5120000



int
main (int argc, char *argv[])
{
  int done = 0, i;
  int n = N_ELEM;
  double PI25DT = 3.141592653589793238462643;
  double pi, sum, x;
  double mysum = 0.0;
  double pi_time = 0.0;
  double h;
  double array[10];


  h = 1.0 / (double) n;
  sum = 0.0;
 
 /* reduction(+: sum) */
#pragma omp parallel for shared(h, array) private(x) reduction(+ : sum)
  for (i = 0; i <= n; i++)
    {
      x = h * ((double) i - 0.5);
      sum += 4.0 / (1.0 + x * x);
    }

  pi = h * sum;


  printf ("Pi %f , time: %f\n", pi, pi_time);


}
