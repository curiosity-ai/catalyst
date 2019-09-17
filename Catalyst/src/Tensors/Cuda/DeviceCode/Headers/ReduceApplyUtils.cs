using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.CUDA.RuntimeCompiler;

namespace Catalyst.Tensors.CUDA.DeviceCode.Headers
{
    [CudaInclude("Code", "ReduceApplyUtils")]
    public static class ReduceApplyUtils
    {
        // ReduceApplyUtils functions from cuTorch

        public static readonly string Code = @"

// Translate a linear index for the apply to a float* offset;
// specialized on `Dims` to reduce nvcc compilation time
template <typename IndexType, int Dims>
struct IndexToOffset {
  static __host__ __device__ IndexType get(
    IndexType linearId,
    const TensorInfo<IndexType>& info) {
    IndexType offset = 0;

    // Use static dims
    for (int i = Dims - 1; i >= 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;

      if (i > 0) {
        linearId /= info.sizes[i];
      }
    }

    return offset;
  }
};

template <typename IndexType>
struct IndexToOffset<IndexType, -2> {
  static __forceinline__ __host__ __device__ IndexType
    get(IndexType linearId, const TensorInfo<IndexType>& info) {
    return linearId;
  }
};

template <typename IndexType>
struct IndexToOffset<IndexType, -1> {
  static __forceinline__ __host__ __device__ IndexType
    get(IndexType linearId, const TensorInfo<IndexType>& info) {
    IndexType offset = 0;

    // Use dynamic dims
    for (int i = info.dims - 1; i >= 0; --i) {
      IndexType curDimIndex = linearId % info.sizes[i];
      IndexType curDimOffset = curDimIndex * info.strides[i];
      offset += curDimOffset;

      linearId /= info.sizes[i];
    }

    return offset;
  }
};

template <typename IndexType>
__device__ __forceinline__ IndexType getLinearBlockId() {
  return blockIdx.z * gridDim.y * gridDim.x +
    blockIdx.y * gridDim.x +
    blockIdx.x;
}

";
    }
}
