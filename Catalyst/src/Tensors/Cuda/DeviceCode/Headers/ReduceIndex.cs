using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.CUDA.RuntimeCompiler;

namespace Catalyst.Tensors.CUDA.DeviceCode.Headers
{
    [CudaInclude("Code", "ReduceIndex")]
    public static class ReduceIndex
    {
        // kernel_transformReduceOuterDimIndex and kernel_transformReduceInnermostDimIndex are from cuTorch

        public static readonly string Code = @"

// Simple pair type, compatible with thrust::pair but only supporting
// the operations we need, and able to be compiled with NVRTC
template <typename T1, typename T2>
struct pair
{
  typedef T1 first_type;
  typedef T2 second_type;

  first_type first;
  second_type second;

  __device__ pair(void):first(),second() {}
  
  inline __device__ pair(const T1 &x, const T2 &y) :first(x),second(y) {}

  // copy constructor from a pair with types convertible to T1 and T2
  template <typename U1, typename U2>
  inline __device__ pair(const pair<U1,U2> &p) :first(p.first),second(p.second) {}
};

template <typename T1, typename T2>
inline __device__ pair<T1,T2> make_pair(T1 x, T2 y)
{
  return pair<T1,T2>(x,y);
}


#define REDUCE_INDEX_KERNELS(KERNEL_NAME, REDUCE_OP_CODE) \
struct ReduceIndexOp##KERNEL_NAME { __device__ __forceinline__ pair<float,float> operator()(const pair<float,float> &a, const pair<float,float> &b) { REDUCE_OP_CODE } };\
extern ""C"" {\
    __global__ void outer_index_##KERNEL_NAME(float *tgt1, float *tgt2, float *src_, unsigned __int64 num_orows, unsigned __int64 num_irows, unsigned __int64 row_size, float initVal, float initIdx) {\
        kernel_transformReduceOuterDimIndex<ReduceIndexOp##KERNEL_NAME>(tgt1, tgt2, src_, num_orows, num_irows, row_size, pair<float, float>(initVal, initIdx), ReduceIndexOp##KERNEL_NAME());\
    }\
    __global__ void inner_index_##KERNEL_NAME(float *tgt1, float* tgt2, float *src_, unsigned __int64 num_rows, unsigned __int64 row_size, float initVal, float initIdx) {\
        kernel_transformReduceInnermostDimIndex<ReduceIndexOp##KERNEL_NAME>(tgt1, tgt2, src_, num_rows, row_size, pair<float, float>(initVal, initIdx), ReduceIndexOp##KERNEL_NAME());\
    }\
}


template<class BinaryFunction>
__device__ void kernel_transformReduceOuterDimIndex(float *tgt1, float *tgt2,
                                                             float *src_,
                                                             unsigned num_orows,
                                                             unsigned num_irows,
                                                             unsigned row_size,
                                                             pair<float,float> init,
                                                             BinaryFunction binary_op)
{
  for (unsigned orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (unsigned irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows; irow += gridDim.y * blockDim.x) {
      float *src = src_ + orow * row_size * num_irows + irow;
      pair<float,float> acc = init;

      for (unsigned col = 0; col < row_size; ++col) {
        acc = binary_op(make_pair(*src, col), acc);
        src += num_irows;
      }
      tgt1[orow * num_irows + irow] = acc.first;
      tgt2[orow * num_irows + irow] = acc.second;
    }
  }
}


/* Reduce the innermost dimension of a tensor (on pair functors which are (value, index))
 *
 * For an n-d tensor (n <= 4) where the reduction is along the innermost dimension:
 *
 * - block.x is the innermost dimension, i.e. dimension 0;
 * - block.y and grid.y make up dimension 1; and
 * - grid.x and grid z are the remaining two outer dimensions (if any)
 *
 * Reduction along other dimensions is handled in a separate kernel.
 */
template<class BinaryFunction>
__device__ void kernel_transformReduceInnermostDimIndex(
  float *tgt1, float* tgt2, float *src_,
  unsigned num_rows, unsigned row_size,
  pair<float,float> init, BinaryFunction binary_op)
{
  __shared__ float sbuf[32][16];
  __shared__ float ibuf[32][16];

  for (unsigned block_row = blockIdx.x * blockDim.y; block_row < num_rows; block_row += blockDim.y * gridDim.x) {
    unsigned row = block_row + threadIdx.y;
    pair<float,float> acc = init;
    if (row < num_rows) {
      float *src = src_ + row * row_size;
      // Sequential reduction within a thread.
      for (unsigned col = threadIdx.x; col < row_size; col += blockDim.x) {
        acc = binary_op(make_pair(src[col], col), acc);
      }
    }

    sbuf[threadIdx.y][threadIdx.x] = acc.first;
    ibuf[threadIdx.y][threadIdx.x] = acc.second;

    // Reduce intermediate values to single value.
    float* sline = &sbuf[threadIdx.y][0];
    float* iline = &ibuf[threadIdx.y][0];
    for (unsigned s = 8; s > 0; s >>= 1) {
      if (row < num_rows && threadIdx.x < s) {
        pair<float,float> arg1 = make_pair<float,float>(sline[threadIdx.x], iline[threadIdx.x]);
        pair<float,float> arg2 = make_pair<float,float>(sline[threadIdx.x + s], iline[threadIdx.x + s]);
        pair<float,float> res = binary_op(arg1, arg2);
        sline[threadIdx.x] = res.first;
        iline[threadIdx.x] = res.second;
      }
      __syncthreads();
    }

    if (row < num_rows && threadIdx.x == 0) {
      tgt1[row] = sline[0];
      tgt2[row] = iline[0];
    }
    __syncthreads();
  }
}

";
    }
}
