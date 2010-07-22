
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
  for (i = 0; i < 10; i++) {
    for (j = 0; j < 10; j++) 
     vnd[i][j] = -1.0;
    
  }

  testfunc(vnd);

}
