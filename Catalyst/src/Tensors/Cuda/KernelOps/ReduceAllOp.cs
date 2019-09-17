using ManagedCuda;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.Core;
using Catalyst.Tensors.CUDA.DeviceCode;
using Catalyst.Tensors.CUDA.RuntimeCompiler;

namespace Catalyst.Tensors.CUDA.KernelOps
{
    public static class ReduceAllOp
    {
        private const long ReduceAllBlockSize = 1024;
        private const long TwoPassReductionSize = 2048;

        
        public static Tensor Invoke(CudaReduceAllKernels reduceAllKernels, float init, ReduceInitType initType, string kernelName, Tensor result, Tensor src, object extraArg = null)
        {
            var deviceId = CudaHelpers.GetDeviceId(src);
            var context = CudaHelpers.TSContextForTensor(src);
            var cudaContext = context.CudaContextForDevice(deviceId);

            if (src.DimensionCount > TSCudaContext.MaxDims)
                throw new InvalidOperationException("Tensors with dimension count > " + TSCudaContext.MaxDims + " are not supported");

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);

            if (src.DimensionCount == 0)
            {
                return result;
            }

            var totalElements = src.ElementCount();
            var config = new ApplySpecialization(src);
            object totalElementsTyped = config.Use32BitIndices ? (uint)totalElements : (ulong)totalElements;
            object initValueTyped = ReduceInitConverter.GetInitValue(init, initType, src.ElementType);

            dim3 grid;
            dim3 block;

            var ptx = reduceAllKernels.GetPtx(context.Compiler);
            var fullKernelName = PermutationGenerator.GetMangledName(kernelName, config);

            var outputDevicePtr = CudaHelpers.GetBufferStart(writeTarget);

            if (isTwoPassReductionSize(totalElements))
            {
                getPass1ReduceBlockGrid(context, deviceId, totalElements, out grid, out block);
                uint smemSize = block.x * sizeof(float);

                var scratchSpace = context.ScratchSpaceForDevice(deviceId).buffer;

                if(extraArg == null)
                    InvokeReduceAll(context, cudaContext, ptx, "twoPassA_" + fullKernelName, grid, block, smemSize, config, src, totalElementsTyped, initValueTyped, scratchSpace);
                else
                    InvokeReduceAll(context, cudaContext, ptx, "twoPassA_" + fullKernelName, grid, block, smemSize, config, src, totalElementsTyped, initValueTyped, scratchSpace, extraArg);


                uint numPass1Blocks = grid.x;
                getPass2ReduceBlockGrid(context, deviceId, totalElements, out grid, out block);
                smemSize = block.x * sizeof(float);

                InvokeReduceAllPass2(context, cudaContext, ptx, "twoPassB_" + fullKernelName, grid, block, smemSize, config.Use32BitIndices, numPass1Blocks, initValueTyped, scratchSpace, outputDevicePtr);

            }
            else {
                getSinglePassReduceBlockGrid(totalElements, out grid, out block);
                uint smemSize = block.x * sizeof(float);

                if(extraArg == null)
                    InvokeReduceAll(context, cudaContext, ptx, "onePass_" + fullKernelName, grid, block, smemSize, config, src, totalElementsTyped, initValueTyped, outputDevicePtr);
                else
                    InvokeReduceAll(context, cudaContext, ptx, "onePass_" + fullKernelName, grid, block, smemSize, config, src, totalElementsTyped, initValueTyped, outputDevicePtr, extraArg);
            }

            return writeTarget;
        }

        public static void InvokeReduceAllPass2(TSCudaContext context, CudaContext cudaContext, byte[] ptx, string kernelName, dim3 grid, dim3 block, uint smemSize, bool index32, params object[] args)
        {
            var config = new ApplySpecialization(index32).GetConfig();

            
            var kernel = context.KernelCache.Get(cudaContext, ptx, kernelName);

            kernel.GridDimensions = grid;
            kernel.BlockDimensions = block;
            kernel.DynamicSharedMemory = smemSize;

            kernel.Run(args);            
        }

        public static void InvokeReduceAll(TSCudaContext context, CudaContext cudaContext, byte[] ptx, string kernelName, dim3 grid, dim3 block, uint smemSize, ApplySpecialization spec, params object[] args)
        {
            ConvertTensorArgs.Convert(cudaContext, spec.Use32BitIndices, args);
            
            var kernel = context.KernelCache.Get(cudaContext, ptx, kernelName);

            kernel.GridDimensions = grid;
            kernel.BlockDimensions = block;
            kernel.DynamicSharedMemory = smemSize;

            kernel.Run(args);
            
        }


        private static bool isTwoPassReductionSize(long elements)
        {
            return (elements > TwoPassReductionSize);
        }

        private static long getTwoPassBlocks(TSCudaContext context, int deviceId, long elements)
        {
            long numBlocks = ApplyUtils.CeilDiv(elements, ReduceAllBlockSize);

            // We can only have as many blocks as there is scratch space
            long scratchSpace =
              context.ScratchSpaceForDevice(deviceId).size / sizeof(float);
            if (scratchSpace <= 0)
                throw new ApplicationException("Device id " + deviceId + " has no scratch space");

            if (numBlocks > scratchSpace)
            {
                numBlocks = scratchSpace;
            }

            return numBlocks;
        }

        private static void getPass1ReduceBlockGrid(TSCudaContext context, int deviceId, long elements, out dim3 grid, out dim3 block)
        {
            grid = new dim3((uint)getTwoPassBlocks(context, deviceId, elements));
            block = new dim3((uint)ReduceAllBlockSize);
        }

        private static void getPass2ReduceBlockGrid(TSCudaContext context, int deviceId, long elements, out dim3 grid, out dim3 block)
        {
            grid = new dim3(1);
            // We only need as many threads as there were blocks originally
            block = new dim3((uint)getTwoPassBlocks(context, deviceId, elements));
        }

        private static void getSinglePassReduceBlockGrid(long elements, out dim3 grid, out dim3 block)
        {
            grid = new dim3(1);
            block = new dim3((uint)ReduceAllBlockSize);
        }
    }
}
