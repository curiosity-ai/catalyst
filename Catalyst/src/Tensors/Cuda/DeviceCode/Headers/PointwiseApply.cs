using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.CUDA.RuntimeCompiler;

namespace Catalyst.Tensors.CUDA.DeviceCode.Headers
{
    // Depends on: general, ReduceApplyUtils
    [CudaInclude("Code", "PointwiseApply")]
    public static class PointwiseApply
    {
        // Apply functions from cuTorch

        public static readonly string Code = @"

template <typename Op, typename IndexType, int ADims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
pointwiseApply1(TensorInfo<IndexType> a,
                             IndexType totalElements,
                             Op op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<IndexType, ADims>::get(linearIndex, a);

    op(&a.data[aOffset]);
  }
}

template <typename Op, typename IndexType, int ADims, int BDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
pointwiseApply2(TensorInfo<IndexType> a,
                             TensorInfo<IndexType> b,
                             IndexType totalElements,
                             Op op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<IndexType, ADims>::get(linearIndex, a);

    // Convert `linearIndex` into an offset of `b`
    const IndexType bOffset =
      IndexToOffset<IndexType, BDims>::get(linearIndex, b);

    op(&a.data[aOffset], &b.data[bOffset]);
  }
}

template <typename Op, typename IndexType, int ADims, int BDims, int CDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
pointwiseApply3(TensorInfo<IndexType> a,
                             TensorInfo<IndexType> b,
                             TensorInfo<IndexType> c,
                             IndexType totalElements,
                             Op op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<IndexType, ADims>::get(linearIndex, a);

    // Convert `linearIndex` into an offset of `b`
    const IndexType bOffset =
      IndexToOffset<IndexType, BDims>::get(linearIndex, b);

    // Convert `linearIndex` into an offset of `c`
    const IndexType cOffset =
      IndexToOffset<IndexType, CDims>::get(linearIndex, c);

    op(&a.data[aOffset], &b.data[bOffset], &c.data[cOffset]);
  }
}


template <typename Op, typename IndexType, int ADims, int BDims, int CDims, int DDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
pointwiseApply4(TensorInfo<IndexType> a,
                             TensorInfo<IndexType> b,
                             TensorInfo<IndexType> c,
                             TensorInfo<IndexType> d,
                             IndexType totalElements,
                             Op op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<IndexType, ADims>::get(linearIndex, a);

    // Convert `linearIndex` into an offset of `b`
    const IndexType bOffset =
      IndexToOffset<IndexType, BDims>::get(linearIndex, b);

    // Convert `linearIndex` into an offset of `c`
    const IndexType cOffset =
      IndexToOffset<IndexType, CDims>::get(linearIndex, c);

    // Convert `linearIndex` into an offset of `d`
    const IndexType dOffset =
      IndexToOffset<IndexType, DDims>::get(linearIndex, d);

    op(&a.data[aOffset], &b.data[bOffset], &c.data[cOffset], &d.data[dOffset]);
  }
}


template <typename Op, typename IndexType, int ADims, int BDims, int CDims, int DDims, int EDims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
pointwiseApply5(TensorInfo<IndexType> a,
                             TensorInfo<IndexType> b,
                             TensorInfo<IndexType> c,
                             TensorInfo<IndexType> d,
                             TensorInfo<IndexType> e,
                             IndexType totalElements,
                             Op op) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    // Convert `linearIndex` into an offset of `a`
    const IndexType aOffset =
      IndexToOffset<IndexType, ADims>::get(linearIndex, a);

    // Convert `linearIndex` into an offset of `b`
    const IndexType bOffset =
      IndexToOffset<IndexType, BDims>::get(linearIndex, b);

    // Convert `linearIndex` into an offset of `c`
    const IndexType cOffset =
      IndexToOffset<IndexType, CDims>::get(linearIndex, c);

    // Convert `linearIndex` into an offset of `d`
    const IndexType dOffset =
      IndexToOffset<IndexType, DDims>::get(linearIndex, d);

    // Convert `linearIndex` into an offset of `e`
    const IndexType eOffset =
      IndexToOffset<IndexType, EDims>::get(linearIndex, e);

    op(&a.data[aOffset], &b.data[bOffset], &c.data[cOffset], &d.data[dOffset], &e.data[eOffset]);
  }
}

";
    }
}
