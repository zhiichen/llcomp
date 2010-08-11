

int main()
{
    int i;
    int sum[10];

    #pragma llc swap
    for (j = 0; j <= 10; j++)
        for (i = 0; i <= 10; i++) {
    	sum[i] = i;
        }

}
