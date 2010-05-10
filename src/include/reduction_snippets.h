
unsigned int nextPow2( unsigned int x ) {
    --x; 
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}



#define EMUSYNC
// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};


// specialize for double to avoid unaligned memory 
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double*()
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }

    __device__ inline operator const double*() const
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }
};


template <class T>
__global__ void reduce3(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;
    if (i + blockDim.x < n)
        sdata[tid] += g_idata[i+blockDim.x];

    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}



template <class T>
T kernelReduction_template(T *in_data, int numElems, T ini_value)
 {
    T result = ini_value;
    T check_result = ini_value; /* Variable to check results with host reduction */
    T *reduction_loc_sum;
    int maxThreads = 512;
    /* Num elems must be a power of two */
    double exponent = ceil(log2((double) numElems));
    /* reduction kernel asume potencia de dos */
    int n =  (1<<(int)exponent); 
    int threads;
    int blocks;
    int i, j, k;
    int __i__;
    int smemSize;
    dim3 dimGrid, dimBlock;
    /* Partial reduction var */
    T *reduction_block_cu_sum;

    /* Reduction array on host */
    reduction_loc_sum = (T *) (malloc(n * sizeof(T))); 
    for (i = 0; i < n; i++) 
	reduction_loc_sum[i] = 0;

    /* printf("*** Exponent : %f \n", exponent); */
    
    threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);

    /* Partial reduction array on device */
    cudaMalloc((void **) (&reduction_block_cu_sum), blocks * sizeof(T));
    cudaMemcpy(reduction_block_cu_sum, reduction_loc_sum, blocks*sizeof(T), cudaMemcpyHostToDevice);


    smemSize = threads * sizeof(T);

    dimBlock.x = threads;
    dimBlock.y = 1;
    dimBlock.z = 1;
    dimGrid.x = blocks;
    dimGrid.y = 1;
    dimGrid.z = 1;

/*    printf("*** N: %d  == %d\n", n, numElems);
    printf("*** Shared Mem Size: %d, Threads: %d , Blocks : %d \n", smemSize, threads, blocks);*/

    /* Call to the partial reduction kernel */
    reduce3<T> <<< dimGrid, dimBlock, smemSize >>> (in_data, reduction_block_cu_sum, n);

    /* Retrieve partial reduction array */
    cudaMemcpy(reduction_loc_sum, reduction_block_cu_sum,  blocks*sizeof(T), cudaMemcpyDeviceToHost);

    checkCUDAError("memcpy"); 
    /* Final reduction (one per block) */
/*    printf("*** Final reduction *** \n");*/
   
    for (__i__ = 0; __i__ < blocks; __i__++) {

	result += reduction_loc_sum[__i__];
/* DEBUG:	printf("r[%d] : %g\n", __i__, reduction_loc_sum[__i__]);*/
    }

#ifdef CHECK_WITH_HOST
    printf("Result on device %f \n", result);
    /* Retrieve partial reduction array */
    cudaMemcpy(reduction_loc_sum, in_data,  numElems*sizeof(T), cudaMemcpyDeviceToHost);
    checkCUDAError("memcpy");
    check_result = 0;  
    /* Final reduction (all elements) */
    for (k = 0; k < numElems; k++) {
		check_result += reduction_loc_sum[k];
    }
    printf("Result on host %f \n", check_result);
#endif
/* clean */

  cudaFree(reduction_block_cu_sum);
  free(reduction_loc_sum);
  return result;
} 



double kernelReduction_double(double *in_data, int numElems, double ini_value) {
   return kernelReduction_template<double>(in_data, numElems, ini_value);
}

int kernelReduction_int(int *in_data, int numElems, int ini_value) {
   return kernelReduction_template<int>(in_data, numElems, ini_value);
}


