

int main()
{
    int i;
    int sum[10];

    for (i = 0; i <= 10; i++) {
	sum[i] = i;
    }

    #pragma omp parallel for reduction(+ : sum)
    for (i = 0; i <= 10; i++) {
	sum[i] = i;
    }

}