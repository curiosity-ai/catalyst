using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.CUDA.RuntimeCompiler;

namespace Catalyst.Tensors.CUDA.DeviceCode.Headers
{
    [CudaInclude("Code", "ReduceAll")]
    public static class ReduceAll
    {

        // Reduce all functions from cuTorch

        public static readonly string Code = @"

//
// One-pass implementation for small tensors
//

template <typename ModifyOp, typename ReduceOp, typename IndexType, int ADims>
__global__ void
reduceAll(TensorInfo<IndexType> in,
                       IndexType totalElements,
                       float init,
                       ModifyOp modifyOp,
                       ReduceOp reduceOp,
                       float* out) {
  // With a block-wide stride, have each thread perform its own reduction.
  float r = init;
  for (IndexType i = threadIdx.x; i < totalElements; i += blockDim.x) {
    const IndexType inOffset = IndexToOffset<IndexType, ADims>::get(i, in);
    r = reduceOp(r, modifyOp(in.data[inOffset]));
  }

  // Reduce within the block
  extern __shared__ float smem[];
  r = reduceBlock<float, ReduceOp>(smem, blockDim.x, r, reduceOp, init);

  if (threadIdx.x == 0) {
    // Write out reduced value
    *out = r;
  }
}


//
// Two-pass implementation
//


//Computes ceil(a / b)
template <typename T>
__host__ __device__ __forceinline__ T CeilDiv(T a, T b) {
  return (a + b - 1) / b;
}

template <typename IndexType>
__device__ __forceinline__ IndexType getStartIndex(IndexType totalSize) {
  IndexType sizePerBlock = CeilDiv(totalSize, (IndexType) gridDim.x);
  return blockIdx.x * sizePerBlock;
}

template <typename IndexType>
__device__ __forceinline__ IndexType getEndIndex(IndexType totalSize) {
  IndexType sizePerBlock = CeilDiv(totalSize, (IndexType) gridDim.x);
  return min((IndexType) ((blockIdx.x + 1) * sizePerBlock), totalSize);
}

// Kernel that handles an entire reduction of a tensor in two passes
template <typename ModifyOp, typename ReduceOp, typename IndexType, int ADims>
__global__ void
reduceAllPass1(TensorInfo<IndexType> in,
                IndexType totalElements,
                float init,
                ModifyOp modifyOp,
                ReduceOp reduceOp,
                float* scratchSpace) {
  const IndexType startIndex = getStartIndex<IndexType>(totalElements);
  const IndexType endIndex = getEndIndex<IndexType>(totalElements);

  // With a block-wide stride, have each thread perform its own reduction.
  float r = init;
  for (IndexType i = startIndex + threadIdx.x; i < endIndex; i += blockDim.x) {
    const IndexType inOffset = IndexToOffset<IndexType, ADims>::get(i, in);
    r = reduceOp(r, modifyOp(in.data[inOffset]));
  }

  // Reduce within the block
  extern __shared__ float smem[];
  r = reduceBlock<float, ReduceOp>(smem, blockDim.x, r, reduceOp, init);

  if (threadIdx.x == 0) {
    // Write out block-wide reduced value
    scratchSpace[blockIdx.x] = r;
  }
}

template <typename ReduceOp, typename IndexType>
__global__ void
reduceAllPass2(int numPass1Blocks,
                float init,
                ReduceOp reduceOp,
                float* scratchSpace,
                float* out) {
  float r = init;
  if (threadIdx.x < numPass1Blocks) {
    r = scratchSpace[threadIdx.x];
  }

  // Reduce within the block
  extern __shared__ float smem[];
  r = reduceBlock<float, ReduceOp>(smem, numPass1Blocks, r, reduceOp, init);

  if (threadIdx.x == 0) {
    *out = r;
  }
}



";
    }
}
