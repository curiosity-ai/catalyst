using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.CUDA.RuntimeCompiler;

namespace Catalyst.Tensors.CUDA.DeviceCode.Headers
{
    [CudaInclude("Code", "Reduce")]
    public static class Reduce
    {
        public const int NonContigReduceBlockSize = 32 * 16;

        // Reduce functions from cuTorch

        public static readonly string Code = @"
template <typename IndexType>
__device__ __forceinline__ IndexType getReduceContigDimSliceIndex() {
  // Each block handles one slice
  return getLinearBlockId<IndexType>();
}


// Kernel that handles an entire reduction of a slice of a tensor per
// each block
template <typename ModifyOp, typename ReduceOp, typename IndexType, int ADims, int BDims>
__device__ void
reduceContigDim_apply(TensorInfo<IndexType> out,
                             TensorInfo<IndexType> in,
                             IndexType reductionSize,
                             IndexType totalSlices,
                             float init,
                             ModifyOp modifyOp,
                             ReduceOp reduceOp) {
  const IndexType sliceIndex = getReduceContigDimSliceIndex<IndexType>();

  if (sliceIndex >= totalSlices) {
    return;
  }

  // Get the offset in `out` for the reduction
  const IndexType outOffset =
    IndexToOffset<IndexType, ADims>::get(sliceIndex, out);

  // Get the base offset in `in` for this block's reduction
  const IndexType inBaseOffset =
    IndexToOffset<IndexType, BDims>::get(sliceIndex, in);

  // Each thread in the block will reduce some subset of elements in
  // the slice. The elements are guaranteed contiguous starting at
  // `inBaseOffset`.
  float r = init;
  for (IndexType i = threadIdx.x; i < reductionSize; i += blockDim.x) {
    r = reduceOp(r, modifyOp(in.data[inBaseOffset + i]));
  }

  // Reduce within the block
  extern __shared__ float smem[];
  r = reduceBlock<float, ReduceOp>(smem, blockDim.x, r, reduceOp, init);

  if (threadIdx.x == 0) {
    // Write out reduced value
    out.data[outOffset] = r;
  }
}

// Threads per thread block
#define THC_NONCONTIG_REDUCE_BLOCK_SIZE " + NonContigReduceBlockSize + "\n" + @"

template <typename IndexType>
__device__ __forceinline__ IndexType getReduceNoncontigDimSliceIndex() {
  // Each thread handles one slice
  return getLinearBlockId<IndexType>() * THC_NONCONTIG_REDUCE_BLOCK_SIZE + threadIdx.x;
}

// Kernel that handles an entire reduction of a slice of a tensor per each thread
template <typename ModifyOp, typename ReduceOp, typename IndexType, int ADims, int BDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
reduceNoncontigDim_apply(TensorInfo<IndexType> out,
                                TensorInfo<IndexType> in,
                                IndexType reductionStride,
                                IndexType reductionSize,
                                IndexType totalSlices,
                                float init,
                                ModifyOp modifyOp,
                                ReduceOp reduceOp) {
  const IndexType sliceIndex = getReduceNoncontigDimSliceIndex<IndexType>();

  if (sliceIndex >= totalSlices) {
    return;
  }

  // Each thread picks a point in `out` and `in` for which it is
  // producing the reduction
  const IndexType outOffset =
    IndexToOffset<IndexType, ADims>::get(sliceIndex, out);
  const IndexType inBaseOffset =
    IndexToOffset<IndexType, BDims>::get(sliceIndex, in);

  // For each point in reductionSize, reduce into `r`
  IndexType inOffset = inBaseOffset;
  float r = init;

  for (IndexType i = 0; i < reductionSize; ++i) {
    r = reduceOp(r, modifyOp(in.data[inOffset]));
    inOffset += reductionStride;
  }

  // Write out reduced value
  out.data[outOffset] = r;


}

";
    }
}
