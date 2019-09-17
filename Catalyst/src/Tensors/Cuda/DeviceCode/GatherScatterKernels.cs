using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.Core;
using Catalyst.Tensors.CUDA.RuntimeCompiler;

namespace Catalyst.Tensors.CUDA.DeviceCode
{
    [Precompile]
    public class GatherScatterKernels : CudaCode
    {
        public static readonly string Code = @"

// Compute the offsets into the given tensors for a linear index. For the 't2'
// tensor, dimension 'dim' is skipped. The tensors are assumed to have the same
// size (with the exception of 't2' in dimension 'dim').
// This version uses a static number of dimensions.
template <typename IndexType, int Dims>
struct IndexToScatterGatherOffsets {
  static __device__ void compute(
      IndexType linearId, const int dim,
      const TensorInfo<IndexType>& index, IndexType* indexOffset,
      const TensorInfo<IndexType>& t1, IndexType* t1Offset,
      const TensorInfo<IndexType>& t2, IndexType* t2Offset) {
    for (int d = Dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      *t1Offset += curDimIndex * t1.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }

  static __device__ void compute(
      IndexType linearId, const int dim,
      const TensorInfo<IndexType>& index, IndexType* indexOffset,
      const TensorInfo<IndexType>& t2, IndexType* t2Offset) {
    for (int d = Dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }
};

// Same as above but using a dynamic number of dimensions.
template <typename IndexType>
struct IndexToScatterGatherOffsets<IndexType, -1> {
  static __device__ void compute(
      IndexType linearId, const int dim,
      const TensorInfo<IndexType>& index, IndexType* indexOffset,
      const TensorInfo<IndexType>& t1, IndexType* t1Offset,
      const TensorInfo<IndexType>& t2, IndexType* t2Offset) {
    for (int d = index.dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      *t1Offset += curDimIndex * t1.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }

  static __device__ void compute(
      IndexType linearId, const int dim,
      const TensorInfo<IndexType>& index, IndexType* indexOffset,
      const TensorInfo<IndexType>& t2, IndexType* t2Offset) {
    for (int d = index.dims - 1; d >= 0; d--) {
      IndexType curDimIndex = linearId % index.sizes[d];
      *indexOffset += curDimIndex * index.strides[d];
      if (d != dim) {
        *t2Offset += curDimIndex * t2.strides[d];
      }
      linearId /= index.sizes[d];
    }
  }
};


template <typename IndexType, int Dims>
__global__ void gather_kernel(
    TensorInfo<IndexType> tensor,
    TensorInfo<IndexType> src,
    TensorInfo<IndexType> index,
    const int dim,
    const IndexType totalElements) {
  for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    IndexType tensorOffset = 0;
    IndexType srcOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Dims>::compute(linearId, dim,
                                                          index, &indexOffset,
                                                          tensor, &tensorOffset,
                                                          src, &srcOffset);

    IndexType indexValue = (IndexType)index.data[indexOffset];
    srcOffset += indexValue * src.strides[dim];

    tensor.data[tensorOffset] = src.data[srcOffset];
  }
}

template <typename IndexType, int Dims>
__global__ void scatter_kernel(
    TensorInfo<IndexType> tensor,
    TensorInfo<IndexType> src,
    TensorInfo<IndexType> index,
    const int dim,
    const IndexType totalElements) {
  for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    IndexType tensorOffset = 0;
    IndexType srcOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Dims>::compute(linearId, dim,
                                                          index, &indexOffset,
                                                          src, &srcOffset,
                                                          tensor, &tensorOffset);

    IndexType indexValue = (IndexType)index.data[indexOffset];
    tensorOffset += indexValue * tensor.strides[dim];

    tensor.data[tensorOffset] = src.data[srcOffset];
  }
}


template <typename IndexType, int Dims>
__global__ void scatterFill_kernel(
    TensorInfo<IndexType> tensor,
    TensorInfo<IndexType> index,
    float value,
    const int dim,
    const IndexType totalElements) {
  for (IndexType linearId = blockIdx.x * blockDim.x + threadIdx.x;
       linearId < totalElements;
       linearId += gridDim.x * blockDim.x) {
    IndexType tensorOffset = 0;
    IndexType indexOffset = 0;

    IndexToScatterGatherOffsets<IndexType, Dims>::compute(linearId, dim,
                                                          index, &indexOffset,
                                                          tensor, &tensorOffset);

    IndexType indexValue = (IndexType)index.data[indexOffset];
    tensorOffset += indexValue * tensor.strides[dim];

    tensor.data[tensorOffset] = value;
  }
}


#define DECLARE_GATHER(KERNEL_NAME, INDEX_TYPE, DIMS) \
    extern ""C"" {\
        __global__ void KERNEL_NAME(\
                                          TensorInfo<INDEX_TYPE> tensor,\
                                          TensorInfo<INDEX_TYPE> src,\
                                          TensorInfo<INDEX_TYPE> indices,\
                                          const int dim,\
                                          INDEX_TYPE totalElements)\
        {\
            gather_kernel<INDEX_TYPE, DIMS>(tensor, src, indices, dim, totalElements);\
        }\
    }

#define DECLARE_SCATTER(KERNEL_NAME, INDEX_TYPE, DIMS) \
    extern ""C"" {\
        __global__ void KERNEL_NAME(\
                                          TensorInfo<INDEX_TYPE> tensor,\
                                          TensorInfo<INDEX_TYPE> src,\
                                          TensorInfo<INDEX_TYPE> indices,\
                                          const int dim,\
                                          INDEX_TYPE totalElements)\
        {\
            scatter_kernel<INDEX_TYPE, DIMS>(tensor, src, indices, dim, totalElements);\
        }\
    }

#define DECLARE_SCATTERFILL(KERNEL_NAME, INDEX_TYPE, DIMS) \
    extern ""C"" {\
        __global__ void KERNEL_NAME(\
                                          TensorInfo<INDEX_TYPE> tensor,\
                                          TensorInfo<INDEX_TYPE> indices,\
                                          float value,\
                                          const int dim,\
                                          INDEX_TYPE totalElements)\
        {\
            scatterFill_kernel<INDEX_TYPE, DIMS>(tensor, indices, value, dim, totalElements);\
        }\
    }
";

        private const string GatherBaseName = "gather_";
        private const string ScatterBaseName = "scatter_";
        private const string ScatterFillBaseName = "scatterFill_";

        public GatherScatterKernels() : base(GetCode(), "General", "ReduceApplyUtils")
        {
        }


        private static string GetCode()
        {
            var sb = new StringBuilder(Code);
            sb.AppendLine(GetMacroInvocations(true, 1));
            sb.AppendLine(GetMacroInvocations(true, 2));
            sb.AppendLine(GetMacroInvocations(true, 3));
            sb.AppendLine(GetMacroInvocations(true, -1));
            sb.AppendLine(GetMacroInvocations(false, -1));
            return sb.ToString();
        }

        private static string GetMacroInvocations(bool is32, int dims)
        {
            var indexType = is32 ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;

            return
                string.Format("DECLARE_GATHER({0}, {1}, {2})\n", MakeKernelName(GatherBaseName, is32, dims), indexType, dims) +
                string.Format("DECLARE_SCATTER({0}, {1}, {2})\n", MakeKernelName(ScatterBaseName, is32, dims), indexType, dims) +
                string.Format("DECLARE_SCATTERFILL({0}, {1}, {2})\n", MakeKernelName(ScatterFillBaseName, is32, dims), indexType, dims);
        }
        

        private static string MakeKernelName(string baseName, bool is32, int dims)
        {
            return string.Format("{0}{1}_{2}",
                baseName,
                is32 ? "__int32" : "__int64",
                dims.ToString().Replace('-', 'M')
                );
        }



        public Tensor Gather(Tensor result, Tensor src, int dim, Tensor indices)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var cudaContext = context.CudaContextForTensor(src);

            if (result != null && result.DimensionCount != src.DimensionCount) throw new InvalidOperationException("result and src must have same number of dimensions");
            if (result != null && dim < 0 && dim >= result.DimensionCount) throw new ArgumentOutOfRangeException("dim");
            if (indices.DimensionCount != src.DimensionCount) throw new InvalidOperationException("src and indices must have same number of dimensions");
            if (result != null && !result.IsSameSizeAs(indices)) throw new InvalidOperationException("result and indices must be the same size");
            if (result != null && !TensorResultBuilder.ArrayEqualExcept(src.Sizes, result.Sizes, dim)) throw new InvalidOperationException("result and src must be the same size except in dimension dim");

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, indices.Allocator, src.ElementType, false, indices.Sizes);

            var nElement = indices.ElementCount();
            var block = ApplyUtils.GetApplyBlock();
            var grid = ApplyUtils.GetApplyGrid(context.DeviceInfoForContext(cudaContext), nElement);

            if (ApplyUtils.CanUse32BitIndexMath(writeTarget) &&
                ApplyUtils.CanUse32BitIndexMath(src) &&
                ApplyUtils.CanUse32BitIndexMath(indices))
            {
                var dims = indices.DimensionCount <= 3 ? indices.DimensionCount : -1;
                var kernelName = MakeKernelName(GatherBaseName, true, dims);
                Invoke(context, cudaContext, kernelName, grid, block, 0, CUstream.NullStream, true,
                    writeTarget, src, indices, dim, (int)nElement);
            }
            else
            {
                var kernelName = MakeKernelName(GatherBaseName, false, -1);
                Invoke(context, cudaContext, kernelName, grid, block, 0, CUstream.NullStream, false,
                   writeTarget, src, indices, dim, (long)nElement);
            }

            return writeTarget;
        }

        public Tensor Scatter(Tensor result, Tensor src, int dim, Tensor indices)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var cudaContext = context.CudaContextForTensor(src);

            if (result == null) throw new ArgumentNullException("result");

            if (result.DimensionCount != src.DimensionCount) throw new InvalidOperationException("result and src must have same number of dimensions");
            if (dim < 0 && dim >= result.DimensionCount) throw new ArgumentOutOfRangeException("dim");
            if (indices.DimensionCount != src.DimensionCount) throw new InvalidOperationException("src and indices must have same number of dimensions");
            if (!src.IsSameSizeAs(indices)) throw new InvalidOperationException("src and indices must be the same size");
            if (!TensorResultBuilder.ArrayEqualExcept(src.Sizes, result.Sizes, dim)) throw new InvalidOperationException("result and src must be the same size except in dimension dim");

            var writeTarget = result;

            var nElement = indices.ElementCount();
            var block = ApplyUtils.GetApplyBlock();
            var grid = ApplyUtils.GetApplyGrid(context.DeviceInfoForContext(cudaContext), nElement);

            if (ApplyUtils.CanUse32BitIndexMath(writeTarget) &&
                ApplyUtils.CanUse32BitIndexMath(src) &&
                ApplyUtils.CanUse32BitIndexMath(indices))
            {
                var dims = indices.DimensionCount <= 3 ? indices.DimensionCount : -1;
                var kernelName = MakeKernelName(ScatterBaseName, true, dims);
                Invoke(context, cudaContext, kernelName, grid, block, 0, CUstream.NullStream, true,
                    writeTarget, src, indices, dim, (int)nElement);
            }
            else
            {
                var kernelName = MakeKernelName(ScatterBaseName, false, -1);
                Invoke(context, cudaContext, kernelName, grid, block, 0, CUstream.NullStream, false,
                   writeTarget, src, indices, dim, (long)nElement);
            }

            return writeTarget;
        }

        public Tensor ScatterFill(Tensor result, float value, int dim, Tensor indices)
        {
            var context = CudaHelpers.TSContextForTensor(indices);
            var cudaContext = context.CudaContextForTensor(indices);

            if (result == null) throw new ArgumentNullException("result");

            if (dim < 0 && dim >= result.DimensionCount) throw new ArgumentOutOfRangeException("dim");
            if (indices.DimensionCount != result.DimensionCount) throw new InvalidOperationException("result and indices must have same number of dimensions");
            if (!TensorResultBuilder.ArrayEqualExcept(indices.Sizes, result.Sizes, dim)) throw new InvalidOperationException("result and indices must be the same size except in dimension dim");

            var writeTarget = result;

            var nElement = indices.ElementCount();
            var block = ApplyUtils.GetApplyBlock();
            var grid = ApplyUtils.GetApplyGrid(context.DeviceInfoForContext(cudaContext), nElement);

            if (ApplyUtils.CanUse32BitIndexMath(writeTarget) &&
                ApplyUtils.CanUse32BitIndexMath(indices))
            {
                var dims = indices.DimensionCount <= 3 ? indices.DimensionCount : -1;
                var kernelName = MakeKernelName(ScatterFillBaseName, true, dims);
                Invoke(context, cudaContext, kernelName, grid, block, 0, CUstream.NullStream, true,
                    writeTarget, indices, value, dim, (int)nElement);
            }
            else
            {
                var kernelName = MakeKernelName(ScatterFillBaseName, false, -1);
                Invoke(context, cudaContext, kernelName, grid, block, 0, CUstream.NullStream, false,
                   writeTarget, indices, value, dim, (long)nElement);
            }

            return writeTarget;
        }


        private void Invoke(TSCudaContext context, CudaContext cudaContext, string kernelName, dim3 grid, dim3 block, uint smemSize, CUstream stream, bool index32, params object[] args)
        {
            ConvertTensorArgs.Convert(cudaContext, index32, args);

            var ptx = GetPtx(context.Compiler);
            var kernel = context.KernelCache.Get(cudaContext, ptx, kernelName);
            kernel.GridDimensions = grid;
            kernel.BlockDimensions = block;
            kernel.DynamicSharedMemory = smemSize;
            kernel.RunAsync(stream, args);
        }
    }
}
