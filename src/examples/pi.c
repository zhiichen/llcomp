
double f (double);

double f (double a) {
	return (4.0 / (1.0 + a * a));
}

int main (int argc, char * argv[]) {
	int done = 0, n, i;
	double PI25DT = 3.141592653589793238462643;
	double pi, h, sum, x;
	double mysum = 0.0;
	double pi_time = 0.0;
	int num_threads = 0;
	char name [20];


	n = 1000000000.0;

	h = 1.0 / (double) n;
	sum = 0.0;

	for (i = 0; i <= n; i++) {
		x = h * ((double)i - 0.5);
		sum += f(x);
	}

	pi = h * sum;

	fprintf(stderr, "pi: %g, Time: %g \n", pi, pi_time); 


  printf("Seq Time: %g \n", pi_time);
}

