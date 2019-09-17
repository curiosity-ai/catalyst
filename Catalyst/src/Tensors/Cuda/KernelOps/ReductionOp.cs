using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Catalyst.Tensors.Core;
using Catalyst.Tensors.CUDA.DeviceCode;
using Catalyst.Tensors.CUDA.RuntimeCompiler;

namespace Catalyst.Tensors.CUDA.KernelOps
{
    public static class ReductionOp
    {
        public static Tensor Invoke(CudaReduceKernels reduceKernels, string kernelName, float init, ReduceInitType initType, Tensor result, Tensor src, int dim, object extraArg = null)
        {
            if (src.DimensionCount == 0)
                return result;

            var context = CudaHelpers.TSContextForTensor(src);
            var cudaContext = context.CudaContextForTensor(src);

            var requiredOutputSize = (long[])src.Sizes.Clone();
            requiredOutputSize[dim] = 1;
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, requiredOutputSize);
            ThrowIfAnyTensorInvalid(writeTarget, src);

            var inElements = src.ElementCount();
            var reductionSize = src.Sizes[dim];
            var reductionStride = src.Strides[dim];
            var outElements = inElements / reductionSize;
            var contigReduction = reductionStride == 1;


            // We must make sure that when the tensor is passed to the kernel, src.Sizes[dim] is set to 1
            // This includes for the purposes of determining which tensor specializations to use (changing
            // the dimension size to 1 may make the tensor non-contiguous
            var newSizes = (long[])src.Sizes.Clone();
            newSizes[dim] = 1;
            var srcSlim = new Tensor(newSizes, src.Strides, src.Storage, src.StorageOffset);

            var config = new ApplySpecialization(writeTarget, srcSlim);
            object totalSlices = config.Use32BitIndices ? (uint)outElements : (ulong)outElements;
            object reductionSizeTyped = config.Use32BitIndices ? (uint)reductionSize : (ulong)reductionSize;
            object reductionStrideTyped = config.Use32BitIndices ? (uint)reductionStride : (ulong)reductionStride;
            object initValueTyped = ReduceInitConverter.GetInitValue(init, initType, src.ElementType);

            var ptx = reduceKernels.GetPtx(context.Compiler);

            if (contigReduction)
            {
                var block = GetContigReduceBlock(cudaContext, outElements, reductionSize);
                var grid = GetContigReduceGrid(outElements);
                uint smemSize = (uint)src.ElementType.Size() * block.x;

                var fullName = "contig_" + PermutationGenerator.GetMangledName(kernelName, config);
                if(extraArg == null)
                    InvokeReduce(context, cudaContext, ptx, fullName, grid, block, smemSize, config, writeTarget, srcSlim, reductionSizeTyped, totalSlices, initValueTyped);
                else
                    InvokeReduce(context, cudaContext, ptx, fullName, grid, block, smemSize, config, writeTarget, srcSlim, reductionSizeTyped, totalSlices, initValueTyped, extraArg);
            }
            else
            {
                var deviceProps = context.DeviceInfoForContext(cudaContext);
                var block = GetNonContigReduceBlock(deviceProps);
                var grid = GetNoncontigReduceGrid(deviceProps, outElements);
                uint smemSize = 0;

                var fullName = "noncontig_" + PermutationGenerator.GetMangledName(kernelName, config);
                if(extraArg == null)
                    InvokeReduce(context, cudaContext, ptx, fullName, grid, block, smemSize, config, writeTarget, srcSlim, reductionStrideTyped, reductionSizeTyped, totalSlices, initValueTyped);
                else
                    InvokeReduce(context, cudaContext, ptx, fullName, grid, block, smemSize, config, writeTarget, srcSlim, reductionStrideTyped, reductionSizeTyped, totalSlices, initValueTyped, extraArg);
            }

            return writeTarget;
        }

        public static void InvokeReduce(TSCudaContext context, CudaContext cudaContext, byte[] ptx, string kernelName, dim3 grid, dim3 block, uint smemSize, ApplySpecialization spec, params object[] args)
        {
            ConvertTensorArgs.Convert(cudaContext, spec.Use32BitIndices, args);
            
            var kernel = context.KernelCache.Get(cudaContext, ptx, kernelName);

            kernel.GridDimensions = grid;
            kernel.BlockDimensions = block;
            kernel.DynamicSharedMemory = smemSize;

            kernel.Run(args);
            
        }

        private static int GetNonContigReduceBlockSize(CudaDeviceProperties deviceProps)
        {
            return Math.Min(DeviceCode.Headers.Reduce.NonContigReduceBlockSize, (int)deviceProps.MaxBlockDim.x);
        }

        private static dim3 GetNonContigReduceBlock(CudaDeviceProperties deviceProps)
        {
            return new dim3(GetNonContigReduceBlockSize(deviceProps));
        }

        private static dim3 GetContigReduceBlock(CudaContext cudaContext, long numSlices, long reductionSize)
        {
            // If the number of slices is low but the reduction dimension size
            // is high, then we should increase block size for greater parallelism.
            // Aim for at least 32 warps per SM (assume 15 SMs; don't bother
            // inquiring the real number for now).
            int smCount = 15;
            int maxWarps = 4; // better occupancy if many blocks are around
                              // For numSlices > smCount * 8, there are > 32 warps active per SM.
            if (numSlices < smCount * 8)
            {
                maxWarps = 8;
                if (numSlices < smCount * 4)
                {
                    maxWarps = 16;
                    if (numSlices < smCount * 2)
                    {
                        maxWarps = 32;
                    }
                }
            }

            // Scale up block size based on the reduction dimension size
            long warpsInReductionSize = ApplyUtils.CeilDiv(reductionSize, 32);
            int numWarps =
              warpsInReductionSize > (long)maxWarps ? maxWarps : (int)warpsInReductionSize;

            int targetSize = numWarps * 32;

            targetSize = Math.Min(targetSize, (int)cudaContext.GetDeviceInfo().MaxBlockDim.x);
            return new dim3(targetSize);
        }

        private static dim3 GetNoncontigReduceGrid(CudaDeviceProperties deviceProps, long elements)
        {
            // One output point per thread
            return GridFromTiles(ApplyUtils.CeilDiv(elements, (long)GetNonContigReduceBlockSize(deviceProps)));
        }

        private static dim3 GetContigReduceGrid(long elements)
        {
            // One output point per block
            return GridFromTiles(elements);
        }


        private const long MaxGridSize = 65535;

        private static dim3 GridFromTiles(long gridTiles)
        {
            if (gridTiles > MaxGridSize * MaxGridSize * MaxGridSize)
            {
                throw new ArgumentException("gridTiles exceeds the maximum allowed tile count", "gridTiles");
            }

            long gridX = gridTiles > MaxGridSize ? MaxGridSize : gridTiles;
            long gridY = 1;
            long gridZ = 1;

            if (gridTiles > MaxGridSize)
            {
                gridTiles = ApplyUtils.CeilDiv(gridTiles, MaxGridSize);
                gridY = gridTiles > MaxGridSize ? MaxGridSize : gridTiles;

                if (gridTiles > MaxGridSize)
                {
                    gridTiles = ApplyUtils.CeilDiv(gridTiles, MaxGridSize);
                    gridZ = gridTiles > MaxGridSize ? MaxGridSize : gridTiles;
                }
            }

            return new dim3((uint)gridX, (uint)gridY, (uint)gridZ);
        }



        private static void ThrowIfAnyTensorInvalid(params Tensor[] args)
        {
            foreach (var tensor in args)
            {
                if (tensor.DimensionCount > TSCudaContext.MaxDims)
                    throw new InvalidOperationException("Tensors with dimension count > " + TSCudaContext.MaxDims + " are not supported");
            }
        }
    }
}
