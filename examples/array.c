
void testfunc (double vnd[10][10]) {
  int i, j;


  for (i = 0; i < 10; i++) {
    for (j = 0; i < 10; i++) 
     vnd[i][j] = -1.0;
    
  }

}

int main () {
  double vnd[10][20];
  int i, j;
  vnd[i][j] = 3.0;
  #pragma omp target device(cuda) copy_in(vnd) copy_out(vnd)
  #pragma omp parallel  shared(vnd) private(i,j)
  {
  #pragma omp for 
  for (i = 1; i < 10; i++) {
    for (j = i; j < 10; j++) 
     vnd[i][j] = -1.0;
    
  }
  }

  testfunc(vnd);

}
