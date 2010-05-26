
/***********************
    Reduction declaratations (header file to C parser)



    Ruymán Reyes Castro, 25 Mar 2010
*/

void checkCUDAError(const char *msg);


#define EMUSYNC
#define LLC_REDUCTION_FUNC(dest, fuente) dest = dest + fuente;

double kernelReduction_double(double *in_data, int numElems, double ini_value);

int kernelReduction_int(int *in_data, int numElems, int ini_value);

