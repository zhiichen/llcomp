
/*#include <stdio.h>*/

#define N_ELEM 5120000



int
main ()
{
  int i;
  int n = N_ELEM;
  double pi, sum, x;
  double mysum = 0.0;
  double h;
  int size_t;


  h = 1.0 / (double) n;
  sum = 0.0;
  size_t = 3;
 
 /* reduction(+: sum) */
#pragma omp parallel for shared(h) private(x) reduction(+ : sum)
  for (i = 0; i <= n; i++)
    {
      x = h * ((double) i - 0.5);
      sum += 4.0 / (1.0 + x * x);
    }

  pi = h * sum;


  /* printf ("Pi %f , time: %f\n", pi, pi_time); */


}
