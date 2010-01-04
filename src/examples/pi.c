
double
f (double a)
{
  return (4.0 / (1.0 + a * a));
}



int
main (int argc, char *argv[])
{
  int done = 0, n, i;
  double PI25DT = 3.141592653589793238462643;
  double pi, h, sum, x;
  double mysum = 0.0;
  double pi_time = 0.0;

  for (i = 1; i <= n; i++); 

  n = 1000000000.0;

  h = 1.0 / (double) n;
  sum = 0.0;
 
 
#pragma omp /* for reduction(+: sum) */
  for (i = 0; i <= n; i++)
    {
      x = h * ((double) i - 0.5);
      sum += f (x);
    }

  pi = h * sum;


  printf ("Pi %f , time: %f\n", pi, pi_time);


}
