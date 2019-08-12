#include "spmm.h"
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
//#include <thrust/execution_policy.h>

/*-----------------------------------------------*/
void cuda_init(int argc, char **argv) {
  int deviceCount, dev;
  cudaGetDeviceCount(&deviceCount);
  printf("=========================================\n");
  if (deviceCount == 0)
    printf("There is no device supporting CUDA\n");

  for (dev = 0; dev < deviceCount; ++dev) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    if (dev == 0) {
      if (deviceProp.major == 9999 && deviceProp.minor == 9999)
        printf("There is no device supporting CUDA.\n");
      else if (deviceCount == 1)
        printf("There is 1 device supporting CUDA\n");
      else
        printf("There are %d devices supporting CUDA\n", deviceCount);
    }
    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
    printf("  Major revision number:          %d\n",
           deviceProp.major);
    printf("  Minor revision number:          %d\n",
           deviceProp.minor);
    printf("  Total amount of global memory:  %.2f GB\n",
           deviceProp.totalGlobalMem/1e9);
  }
  dev = 0;
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("\nRunning on Device %d: \"%s\"\n", dev, deviceProp.name);
  printf("=========================================\n");
}

/*---------------------------------------------------*/
void cuda_check_err() {
  cudaError_t cudaerr = cudaGetLastError() ;
  if (cudaerr != cudaSuccess)
    printf("error: %s\n",cudaGetErrorString(cudaerr));
}

/*
__global__
void csr_spmm_kernel(int m, int k, int n, int *d_ia, int *d_ja, REAL *d_a,
                                          int *d_ib, int *d_jb, REAL *d_b,
                                          int *d_ic, int *d_jc, REAL *d_c)
{
}
*/

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600 && DOUBLEPRECISION == 1
static __inline__ __device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

static __device__ __forceinline__ int get_warp_id()
{
    return threadIdx.x >> 5;
}

static __device__ __forceinline__ int get_lane_id()
{
   return threadIdx.x & (WARP-1);
}

static __device__ int HashFunc(int m, int key, int i)
{
   return ((key % m) + i) % m;
}

static __device__ int HashInsert(int   HashSize,      /* capacity of the hash table */
                                 int  *HashKeys,      /* assumed to be initialized as all -1's */
                                 REAL *HashVals,      /* assumed to be initialized as all 0's */
                                 int   key,           /* assumed to be nonnegative */
                                 REAL  val)
{
   int i, ret=-1;
   for (i = 0; i < HashSize; i++)
   {
      /* compute the hash value of key */
      int j = HashFunc(HashSize, key, i);
      /* try to insert key+1 into slot j */
      int old = atomicCAS(HashKeys+j, -1, key);
      if (old == -1 || old == key)
      {
         /* this slot was open or contained 'key', update value */
         atomicAdd(HashVals+j, val);
         ret = j;
         break;
      }
   }
   return ret;
}

__global__
void csr_merge_row(int rowi, int *ia, int *ja, REAL *a,
                             int *ib, int *jb, REAL *b,
                             int *nz, int *ic, REAL *c,
                   int HashSize, int *HashKeys, REAL *HashVals)
{
   int i, j, istart, iend;
   const int NUM_WARPS = BLOCKDIM / WARP;
   const int warp_id = get_warp_id();
   const int lane_id = get_lane_id();

   /*
   int rowA = blockIdx.x * NUM_WARPS + warp_id;
   volatile __shared__ int rownnz[NUM_WARPS];
   rownnz[warp_id] = 0;
   */

   istart = ia[rowi];
   iend = ia[rowi+1];

   for (i = istart + threadIdx.y; i < iend; i += WARP)
   {
      int rowB = ja[i];
      REAL va = a[i];
      for (j = ib[rowB]; j < ib[rowB+1]; j++)
      {
         int pos = HashInsert(HashSize, HashKeys, HashVals, jb[j], va*b[j]);
         assert(pos != -1);
      }
   }
}

void csr_spmm(struct csr_t *A, struct csr_t *B, struct csr_t *C)
{
   int m, k, n, nnzA, nnzB;
   int *d_ia, *d_ja, *d_ib, *d_jb, *d_ic, d_jc;
   REAL *d_a, *d_b, *d_c;
   csr_t C0;

   m = A->nrow;
   k = A->ncol;
   n = B->ncol;
   nnzA = A->nnz;
   nnzB = B->nnz;
   /*---------- Device Memory */
   cudaMalloc((void **)&d_ia, (m+1)*sizeof(int));
   cudaMalloc((void **)&d_ja,  nnzA*sizeof(int));
   cudaMalloc((void **)&d_a,   nnzA*sizeof(REAL));
   cudaMalloc((void **)&d_ib, (k+1)*sizeof(int));
   cudaMalloc((void **)&d_jb,  nnzB*sizeof(int));
   cudaMalloc((void **)&d_b,   nnzB*sizeof(REAL));
   /*---------- Memcpy */
   cudaMemcpy(d_ia, A->ia, (m+1)*sizeof(int),  cudaMemcpyHostToDevice);
   cudaMemcpy(d_ja, A->ja,  nnzA*sizeof(int),  cudaMemcpyHostToDevice);
   cudaMemcpy(d_a,  A->a,   nnzA*sizeof(REAL), cudaMemcpyHostToDevice);
   cudaMemcpy(d_ib, B->ia, (k+1)*sizeof(int),  cudaMemcpyHostToDevice);
   cudaMemcpy(d_jb, B->ja,  nnzB*sizeof(int),  cudaMemcpyHostToDevice);
   cudaMemcpy(d_b,  B->a,   nnzB*sizeof(REAL), cudaMemcpyHostToDevice);

   csr_spmm_cpu(A, B, &C0);
   int rowi = 70;
   int rowi_len = C0.ia[rowi+1] - C0.ia[rowi];

   printf("\n=====\n");
   for (int i = C0.ia[rowi]; i < C0.ia[rowi+1]; i++)
   {
      printf("(%d, %e)\n", C0.ja[i], C0.a[i]);
   }
   printf("\n=====\n");

   int  *d_hash_keys, *h_hash_keys;
   REAL *d_hash_vals, *h_hash_vals;
   cudaMalloc((void **)&d_hash_keys, rowi_len*sizeof(int));
   cudaMalloc((void **)&d_hash_vals, rowi_len*sizeof(REAL));

   thrust::device_ptr<int> d_hash_keys_ptr(d_hash_keys);
   thrust::fill(d_hash_keys_ptr, d_hash_keys_ptr+rowi_len, -1);
   cudaMemset(d_hash_vals, 0, rowi_len*sizeof(REAL));

   dim3 bDim(4,8,1);
   csr_merge_row<<<1,bDim>>>(rowi, d_ia, d_ja, d_a, d_ib, d_jb, d_b, NULL, NULL, NULL,
                             rowi_len, d_hash_keys, d_hash_vals);

   h_hash_keys = (int *)  malloc(rowi_len*sizeof(int));
   h_hash_vals = (REAL *) malloc(rowi_len*sizeof(REAL));
   cudaMemcpy(h_hash_keys, d_hash_keys, rowi_len*sizeof(int),  cudaMemcpyDeviceToHost);
   cudaMemcpy(h_hash_vals, d_hash_vals, rowi_len*sizeof(REAL), cudaMemcpyDeviceToHost);
   printf("\n");
   for (int i = 0; i < rowi_len; i++)
   {
      printf("(%d, %e)\n", h_hash_keys[i], h_hash_vals[i]);
   }
   printf("\n");

   cudaFree(d_ia);
   cudaFree(d_ja);
   cudaFree(d_a);
   cudaFree(d_ib);
   cudaFree(d_jb);
   cudaFree(d_b);
   cudaFree(d_hash_keys);
   cudaFree(d_hash_vals);
   free(h_hash_keys);
   free(h_hash_vals);
}
