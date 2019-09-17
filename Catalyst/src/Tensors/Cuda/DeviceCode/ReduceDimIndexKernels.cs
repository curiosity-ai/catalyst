using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.Core;

namespace Catalyst.Tensors.CUDA.DeviceCode
{
    [Precompile]
    public class ReduceDimIndexKernels : CudaCode
    {
        private static readonly string Code = @"

REDUCE_INDEX_KERNELS(argmin, if (a.first < b.first) return a; else return b;)
REDUCE_INDEX_KERNELS(argmax, if (a.first > b.first) return a; else return b;)

";

        public ReduceDimIndexKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "ReduceBlock", "Reduce", "ReduceMacros", "ReduceIndex", "Math")
        {
        }

        private static string GetFullCode()
        {
            return Code;
        }

        private void ReduceIndexOuterDim(TSCudaContext context, Tensor resultValues, Tensor resultIndices, Tensor src, int dimension, Tuple<float, float> init, string baseKernelName)
        {
            var cudaContext = context.CudaContextForTensor(src);

            var ndim = src.DimensionCount;
            long num_orows = 1;
            for (int dim = 0; dim < dimension; dim++)
            {
                num_orows *= src.Sizes[dim];
            }
            var row_size = src.Sizes[dimension];
            long num_irows = 1;
            for (int dim = dimension + 1; dim < ndim; dim++)
            {
                num_irows *= src.Sizes[dim];
            }

            var threads = new dim3((uint)Math.Min(512, num_irows));
            var maxGridDim = 1024;
            var grid = new dim3((uint)Math.Min(maxGridDim, num_orows), (uint)Math.Min(maxGridDim, ApplyUtils.CeilDiv(num_irows, threads.x)));

            var resultValPtr = CudaHelpers.GetBufferStart(resultValues);
            var resultIdxPtr = CudaHelpers.GetBufferStart(resultIndices);
            var srcPtr = CudaHelpers.GetBufferStart(src);

            var kernelName = "outer_index_" + baseKernelName;

            Invoke(context, cudaContext, kernelName, grid, threads, 0, CUstream.NullStream, resultValPtr, resultIdxPtr, srcPtr, num_orows, num_irows, row_size, init.Item1, init.Item2);
        }

        private void ReduceIndexInnermostDim(TSCudaContext context, Tensor resultValues, Tensor resultIndices, Tensor src, Tuple<float, float> init, string baseKernelName)
        {
            var cudaContext = context.CudaContextForTensor(src);

            var ndim = src.DimensionCount;
            long num_rows = 1;
            for (int dim = 0; dim < ndim - 1; dim++)
            {
                num_rows *= src.Sizes[dim];
            }
            var row_size = src.Sizes[ndim - 1];

            var threads = new dim3(16, 32);
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(num_rows, threads.y)));

            var resultValPtr = CudaHelpers.GetBufferStart(resultValues);
            var resultIdxPtr = CudaHelpers.GetBufferStart(resultIndices);
            var srcPtr = CudaHelpers.GetBufferStart(src);

            var kernelName = "inner_index_" + baseKernelName;

            Invoke(context, cudaContext, kernelName, grid, threads, 0, CUstream.NullStream, resultValPtr, resultIdxPtr, srcPtr, num_rows, row_size, init.Item1, init.Item2);
        }

        private Tensor RunReduceIndexOp(Tensor resultIndices, Tensor src, int dimension, Tuple<float, float> init, string baseKernelName)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var requiredOutputSize = (long[])src.Sizes.Clone();
            requiredOutputSize[dimension] = 1;
            var writeTarget = TensorResultBuilder.GetWriteTarget(resultIndices, src.Allocator, DType.Float32, true, requiredOutputSize);

            using (var resultValueBuffer = new Tensor(src.Allocator, src.ElementType, requiredOutputSize))
            {
                if (dimension == src.DimensionCount - 1)
                {
                    ReduceIndexInnermostDim(context, resultValueBuffer, writeTarget, src, init, baseKernelName);
                }
                else
                {
                    ReduceIndexOuterDim(context, resultValueBuffer, writeTarget, src, dimension, init, baseKernelName);
                }

            }

            return writeTarget;
        }

        public Tensor ArgMin(Tensor result, Tensor src, int dimension)
        {
            return RunReduceIndexOp(result, src, dimension, Tuple.Create(float.MaxValue, 0.0f), "argmin");
        }

        public Tensor ArgMax(Tensor result, Tensor src, int dimension)
        {
            return RunReduceIndexOp(result, src, dimension, Tuple.Create(float.MinValue, 0.0f), "argmax");
        }

        private void Invoke(TSCudaContext context, CudaContext cudaContext, string kernelName, dim3 grid, dim3 block, uint smemSize, CUstream stream, params object[] args)
        {
            var ptx = GetPtx(context.Compiler);
            var kernel = context.KernelCache.Get(cudaContext, ptx, kernelName);
            kernel.GridDimensions = grid;
            kernel.BlockDimensions = block;
            kernel.DynamicSharedMemory = smemSize;
            kernel.RunAsync(stream, args);
        }
    }
}
