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
    public class IndexSelectKernels : CudaCode
    {
        public static readonly string Code = @"

// We prefer this kernel to avoid reloading index points if the number
// of indices is a small number.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is large, then the
// indexSelectLargeIndex kernel is a better choice to increase
// parallelism.
template <typename IndexType, int DstDim, int SrcDim, int IdxDim>
__device__ void indexSelectSmallIndex(TensorInfo<IndexType> dst,
                                      TensorInfo<IndexType> src,
                                      TensorInfo<IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType innerSize,
                                      __int64 srcSelectDimSize) {
  // In order to avoid reloading the index that we are copying, load
  // it once to handle all of the points that are being selected, so
  // it can be reused as much as possible. This kernel is chosen when
  // this is a good choice (small number of chosen indices), since
  // re-accessing indices in addition to src elements can be slow.
  for (IndexType dstIndex = 0; dstIndex < indices.sizes[0]; ++dstIndex) {

    IndexType srcIndex =
      indices.data[IndexToOffset<IndexType, IdxDim>::get(dstIndex, indices)];

    if (srcIndex < srcSelectDimSize) {
      // We stride over the output ignoring the indexed dimension
      // (innerSize), whose offset calculation is handled differently
      for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
           linearIndex < innerSize;
           linearIndex += gridDim.x * blockDim.x) {
        IndexType dstOffset =
          IndexToOffset<IndexType, DstDim>::get(linearIndex, dst);
        dstOffset += dstIndex * dst.strides[dstSelectDim];

        IndexType srcOffset =
          IndexToOffset<IndexType, SrcDim>::get(linearIndex, src);
        srcOffset += srcIndex * src.strides[srcSelectDim];

        dst.data[dstOffset] = src.data[srcOffset];
      }
    }
  }
}




// We prefer this kernel to balance parallelism across index points,
// if there are a large number of indices.
// This kernel in fact works for all choices of problem size, but if
// the number of indices chosen is small, then the
// indexSelectSmallIndex kernel is a better choice to reduce memory
// accesses.
template <typename IndexType, int DstDim, int SrcDim, int IdxDim>
__device__ void indexSelectLargeIndex(TensorInfo<IndexType> dst,
                                      TensorInfo<IndexType> src,
                                      TensorInfo<IndexType> indices,
                                      int dstSelectDim,
                                      int srcSelectDim,
                                      IndexType totalSize,
                                      IndexType innerSize,
                                      __int64 srcSelectDimSize) {
  // We stride over the output including the indexed dimension
  // (totalSize), and calculate the destination index point based on that
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalSize;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType dstIndex = linearIndex / innerSize;
    IndexType elementInSlice = linearIndex % innerSize;

    IndexType srcIndex =
      indices.data[IndexToOffset<IndexType, IdxDim>::get(dstIndex, indices)];

    if (srcIndex < srcSelectDimSize) {
      IndexType dstOffset =
        IndexToOffset<IndexType, DstDim>::get(elementInSlice, dst);
      dstOffset += dstIndex * dst.strides[dstSelectDim];

      IndexType srcOffset =
        IndexToOffset<IndexType, SrcDim>::get(elementInSlice, src);
      srcOffset += srcIndex * src.strides[srcSelectDim];

      dst.data[dstOffset] = src.data[srcOffset];
    }
  }
}

#define DECLARE_SMALL(KERNEL_NAME, INDEX_TYPE, DST_DIM, SRC_DIM, IDX_DIM) \
    extern ""C"" {\
        __global__ void KERNEL_NAME(\
                                          TensorInfo<INDEX_TYPE> dst,\
                                          TensorInfo<INDEX_TYPE> src,\
                                          TensorInfo<INDEX_TYPE> indices,\
                                          int dstSelectDim,\
                                          int srcSelectDim,\
                                          INDEX_TYPE innerSize,\
                                          __int64 srcSelectDimSize)\
        {\
            indexSelectSmallIndex<INDEX_TYPE, DST_DIM, SRC_DIM, IDX_DIM>(dst, src, indices, dstSelectDim, srcSelectDim, innerSize, srcSelectDimSize);\
        }\
    }

#define DECLARE_LARGE(KERNEL_NAME, INDEX_TYPE, DST_DIM, SRC_DIM, IDX_DIM) \
    extern ""C"" {\
        __global__ void KERNEL_NAME(\
                                          TensorInfo<INDEX_TYPE> dst,\
                                          TensorInfo<INDEX_TYPE> src,\
                                          TensorInfo<INDEX_TYPE> indices,\
                                          int dstSelectDim,\
                                          int srcSelectDim,\
                                          INDEX_TYPE totalSize,\
                                          INDEX_TYPE innerSize,\
                                          __int64 srcSelectDimSize)\
        {\
            indexSelectLargeIndex<INDEX_TYPE, DST_DIM, SRC_DIM, IDX_DIM>(dst, src, indices, dstSelectDim, srcSelectDim, totalSize, innerSize, srcSelectDimSize);\
        }\
    }

";

        private const string SmallBaseName = "kernel_indexSelectSmallIndex_";
        private const string LargeBaseName = "kernel_indexSelectLargeIndex_";

        public IndexSelectKernels() : base(GetCode(), "General", "ReduceApplyUtils")
        {
        }

        private static string GetCode()
        {
            var sb = new StringBuilder();
            sb.AppendLine(Code);
            sb.AppendLine(GetMacroInvocation(true, true, 1, 1, -2));
            sb.AppendLine(GetMacroInvocation(true, true, 2, 2, -2));
            sb.AppendLine(GetMacroInvocation(true, true, 3, 4, -2));
            sb.AppendLine(GetMacroInvocation(true, true, -1, -1, -1));

            sb.AppendLine(GetMacroInvocation(false, true, 1, 1, -2));
            sb.AppendLine(GetMacroInvocation(false, true, 2, 2, -2));
            sb.AppendLine(GetMacroInvocation(false, true, 3, 4, -2));
            sb.AppendLine(GetMacroInvocation(false, true, -1, -1, -1));

            sb.AppendLine(GetMacroInvocation(false, false, -1, -1, -1));
            return sb.ToString();
        }

        private static string GetMacroInvocation(bool isSmall, bool is32, int dstDims, int srcDims, int idxDims)
        {
            var kernelName = MakeKernelName(isSmall, is32, dstDims, srcDims, idxDims);
            return string.Format("{0}({1}, {2}, {3}, {4}, {5})",
                isSmall ? "DECLARE_SMALL" : "DECLARE_LARGE",
                kernelName,
                is32 ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64,
                dstDims,
                srcDims,
                idxDims);
        }

        public Tensor IndexSelect(Tensor result, Tensor src, int dim, Tensor indices)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var cudaContext = context.CudaContextForTensor(src);

            var requiredOutputSize = (long[])src.Sizes.Clone();
            requiredOutputSize[dim] = 1;
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, requiredOutputSize);


            // The `src` is partitioned into two parts:
            // -the size of each slice we are indexing, which is the
            // total size of the tensor ignoring dimension `dim`;
            // -the number of indices we are choosing, which is the total size
            // of the tensor `indices`.
            var numIndices = indices.ElementCount();
            var dstTotalSize = writeTarget.ElementCount();
            var srcSelectDimSize = src.Sizes[dim];
            var sliceSize = dstTotalSize / numIndices;

            var mpc = context.DeviceInfoForContext(cudaContext).MultiProcessorCount;
            var smallIndexGrid = new dim3((uint)Math.Min(ApplyUtils.CeilDiv(sliceSize, 128), (mpc * 8)));
            var smallIndexBlock = new dim3((uint)Math.Min(sliceSize, 128));

            var largeIndexGrid = new dim3((uint)Math.Min(ApplyUtils.CeilDiv(dstTotalSize, 128), (mpc * 8)));
            var largeIndexBlock = new dim3((uint)Math.Min(dstTotalSize, 128));


            var newResultSize = (long[])writeTarget.Sizes.Clone();
            newResultSize[dim] = 1;
            var resultFlat = new Tensor(newResultSize, writeTarget.Strides, writeTarget.Storage, writeTarget.StorageOffset);

            var newSrcSize = (long[])src.Sizes.Clone();
            newSrcSize[dim] = 1;
            var srcFlat = new Tensor(newSrcSize, src.Strides, src.Storage, src.StorageOffset);


            if (ApplyUtils.CanUse32BitIndexMath(writeTarget) &&
                ApplyUtils.CanUse32BitIndexMath(src) &&
                ApplyUtils.CanUse32BitIndexMath(indices))
            {
                // Threshold for small kernel
                var smallKernel = numIndices <= 16;
                string kernelName = "";
                var indContig = indices.IsContiguous();

                if (writeTarget.DimensionCount == src.DimensionCount &&
                    writeTarget.DimensionCount <= 3 &&
                    indContig)
                {
                    kernelName = MakeKernelName(smallKernel, true, writeTarget.DimensionCount, src.DimensionCount, -2);
                }
                else
                {
                    kernelName = MakeKernelName(smallKernel, true, -1, -1, -1);
                }

                var grid = smallKernel ? smallIndexGrid : largeIndexGrid;
                var block = smallKernel ? smallIndexBlock : largeIndexBlock;
                Invoke(context, cudaContext, kernelName, grid, block, 0, CUstream.NullStream, true,
                    writeTarget, src, indices, dim, dim, sliceSize, srcSelectDimSize);
            }
            else
            {
                var kernelName = MakeKernelName(false, false, -1, -1, -1);
                
                Invoke(context, cudaContext, kernelName, largeIndexGrid, largeIndexBlock, 0, CUstream.NullStream, false,
                    writeTarget, src, indices, dim, dim, dstTotalSize, sliceSize, srcSelectDimSize);
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

        private static string MakeKernelName(bool isSmall, bool is32, int dstDims, int srcDims, int idxDims)
        {
            return string.Format("{0}{1}_{2}_{3}_{4}",
                isSmall ? SmallBaseName : LargeBaseName,
                is32 ? "__int32" : "__int64",
                dstDims.ToString().Replace('-', 'M'),
                srcDims.ToString().Replace('-', 'M'),
                idxDims.ToString().Replace('-', 'M')
                );
        }
    }
}
