
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


  n = 1000000000.0;
  h = 1.0 / (double) n;
  sum = 0.0;
 
 /* reduction(+: sum) private(x) shared(h)*/
#pragma omp parallel for shared(h, pi_time) 
  for (i = 0; i <= n; i++)
    {
      x = h * ((double) i - 0.5);
      sum += f (x);
    }

  pi = h * sum;


  printf ("Pi %f , time: %f\n", pi, pi_time);


}
