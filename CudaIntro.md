For llCoMP info, visit the Project Home section.

# CUDA Introduction #


In order to prepare the compiler to generate CUDA, we'll make an example CUDA code.

## Setup ##

Easy setup, just an rpm or tgz, get it from [NVIDIA](http://www.nvidia.com/object/cuda_get.html)

## Tools ##

When installed, new programs  will be in your path:

  * nvcc : Nvidia cuda compiler, your main tool
  * cudafe: C Cuda frontend
  * cudafe++: C++ Cuda frontend

## Sources ##

Cuda source files have .cu extension, written in C plus CUDA addons. Nvcc, by default, builds a binary file.

## Glossary ##

Kernel: function callable from the host and executed on the CUDA device -- simultaneously by many threads in parallel.

## Vim syntax plugin for CUDA ##

  1. Download [this](http://vim.cybermirror.org/runtime/syntax/cuda.vim)
  1. Copy it on your vim syntax directory
  1. Edit your .vimrc
```
" CUDA
au BufNewFile,BufRead *.cu set ft=cuda
```

## Example ##

Example code, using memory transfer (from [here](http://llpanorama.wordpress.com/2008/05/21/my-first-cuda-program/):

```
// moveArrays.cu
//
// demonstrates CUDA interface to data allocation on device (GPU)
// and data movement between host (CPU) and device.
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
int main(void)
{  
   float *a_h, *b_h;     // pointers to host memory
   float *a_d, *b_d;     // pointers to device memory
   int N = 14;
   int i;
   // allocate arrays on host
   a_h = (float *)malloc(sizeof(float)*N);
   b_h = (float *)malloc(sizeof(float)*N);
   // allocate arrays on device
   cudaMalloc((void **) &a_d, sizeof(float)*N);
   cudaMalloc((void **) &b_d, sizeof(float)*N);
   // initialize host data
   for (i=0; i<N; i++) {
      a_h[i] = 10.f+i;
      b_h[i] = 0.f;
   }
   // send data from host to device: a_h to a_d 
   cudaMemcpy(a_d, a_h, sizeof(float)*N, cudaMemcpyHostToDevice);
   // copy data within device: a_d to b_d
   cudaMemcpy(b_d, a_d, sizeof(float)*N, cudaMemcpyDeviceToDevice);
   // retrieve data from device: b_d to b_h
   cudaMemcpy(b_h, b_d, sizeof(float)*N, cudaMemcpyDeviceToHost);
   // check result
   for (i=0; i<N; i++)
      assert(a_h[i] == b_h[i]);
   // cleanup
   free(a_h); free(b_h);
   cudaFree(a_d); cudaFree(b_d);
}
```


## Building ##

To compile, just use:

```
nvcc ej1.cu -o ej1
```

You'll get an ELF exectuable file, which contains calls to the GPU hardware.

## Reference ##

  * cudaMalloc : Allocate memory on device
  * cudaMemcpy: Copy memory to/from device. Waits for all threads on device to finish
  * cudaThreadSyncronize: Blocks until the device has completed all previous calls
  * Kernel Declaration:

```
__global__ void kernel_name (parameters)
{
code
}
```

  * Calling a kernel

```
kernel_name << config >> (parameters)
```

where:
  * config : numBlocks blockSize
  * paramaters : C-style parameters

  * Variables inside a thread:
    * blockIdx My block id (Three-dimensional)
    * threadIdx My id
    * blockDim My number of brothers (threads within the same block) (bidimensional)
    * gridDim : Number of blocks

  * Simple Thread id calc:
{{
int idx = blockIdx.x\*blockDim.x + threadIdx.x;
}}


## Threads ##

  * Threads within a block have the ability to communicate and syncronize
  * Think as it were a grid.
  * blockSize -> number of threads per block (from device)
  * nblocks -> number of blocks (from device)
  * Some devices may launch several blocks at the same time, but you cannot garantize that, so you cannot sync between blocks.
  * If blocks cannot be executed on parallel, the'll be executed sequencially.
  * Each kernel uses a different grid configuration.

[![](http://www.behardware.com/medias/photos_news/00/20/IMG0020712.jpg)](http://www.behardware.com/)


## Double Precision ##

If you need to use double vars, you'll need to activate them on the compiler:
```
nvcc -arch sm_13
```

Otherwise, your double vars inside kernel code will translate them to float
without warning.  But only on the kernel code, so, if you use double on the
host code, your program will break.

## Debugging ##

Use device emulation to run on the host
```
nvcc --device-emulation program
```

You can use gdb and printf inside kernel

## Tutorial ##

http://www.ddj.com/hpc-high-performance-computing/207402986