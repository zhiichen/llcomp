// includes, system
#include <stdio.h>
#include <assert.h>

#define N_ELEM 5120000

// Simple utility function to check for CUDA runtime errors 
void checkCUDAError (const char *msg);

///////////////////////////////////////////////////////////////////// 
// Program main //
/////////////////////////////////////////////////////////////////// 

int
main (int argc, char **argv)
{
// pointer for host memory and size 
  int n = N_ELEM;
  int dimA = n;			// 256K elements (1MB total)
  double h = 1.0 / (double) n;

// pointer for device memory 
  double *d_a;
  double sum = 0.0;

// define grid and block size 
  int numThreadsPerBlock = 512;	// Blocks

// Part 1: compute number of blocks needed based on 
// array size and desired block size 
  int numBlocks = dimA / numThreadsPerBlock;

// allocate host and device memory 
  size_t memSize = numBlocks * numThreadsPerBlock * sizeof (double);
  double *h_a = (double *) malloc (memSize);
  cudaMalloc ((void **) &d_a, memSize);
  cudaMalloc ((void *) &h, sizeof (double));

  printf ("** Array Size %d", memSize);
  printf ("** Num elem of array %d", memSize / sizeof (double));
//
  printf ("** Launch kernel (Grid: %d, Block: %d) **\n", numBlocks,
	  numThreadsPerBlock);

// launch kernel 
  dim3 dimGrid (numBlocks);
  dim3 dimBlock (numThreadsPerBlock);
  piLoop <<< dimGrid, dimBlock >>> (d_a);

// block until the device has completed 
  cudaThreadSynchronize ();

// check if kernel execution generated an error 
// Check for any CUDA errors 
  checkCUDAError ("kernel invocation");

  printf ("*** Get data *** \n");
// device to host copy 
  cudaMemcpy (h_a, d_a, memSize, cudaMemcpyDeviceToHost);

// Check for any CUDA errors 
  checkCUDAError ("memcpy");

// verify the data returned to the host is correct 
  printf ("*** Reduce ***\n");
  for (int i = 0; i < dimA; i++)
    {
      sum += h_a[i];
    }


  printf ("*** Pi: %f\n", sum * (1.0 / n));

// free device memory 
  cudaFree (d_a);

// free host memory
  free (h_a);

// If the program makes it this far, then the results are 
// correct and there are no run-time errors. Good work! 
  printf ("Correct!\n");
  return 0;
}


// Part3: implement the kernel 
__global__ void
piLoop (double *d_in, double h)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  double x = h * ((double) idx - 0.5);
  // printf ("** Parameters: %f, %f, %d **\n", h, tmp, idx);
  d_in[idx] = 4.0 / (1.0 + x * x);

}



void
checkCUDAError (const char *msg)
{
  cudaError_t err = cudaGetLastError ();
  if (cudaSuccess != err)
    {
      fprintf (stderr, "Cuda error: %s: %s.\n", msg,
	       cudaGetErrorString (err));
      exit (EXIT_FAILURE);
    }
}


