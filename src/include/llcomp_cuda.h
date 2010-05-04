#define  __attribute__(x)  /*NOTHING*/


#define __const
#define __addr
#define __THROW
#define __extension__

# define __inline
# define __THROW
# define __P(args)   args
# define __PMT(args) args
# define __restrict__
# define __restrict


#include "/usr/local/cuda/include/cuda.h"
#include "/usr/local/cuda/include/builtin_types.h" 


/* Keep kernel identifiers as original ... */

#define __global__ __global__
#define __host__ __host__
#define __device__ __device__

void
checkCUDAError (const char *msg);

