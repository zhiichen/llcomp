

int main()
{
    int i;
    int sum[10];

    for (i = 0; i <= 10; i++) {
	sum[i] = i;
    }

#pragma omp parallel for shared(sum)
    for (i = 0; i <= 10; i++) {
	sum[i] = i;
    }

   sum[i] = ' ';
}
