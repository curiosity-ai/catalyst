using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.Core;
using Catalyst.Tensors.CUDA.DeviceCode;
using Catalyst.Tensors.CUDA.RuntimeCompiler;

namespace Catalyst.Tensors.CUDA.KernelOps
{
    public static class ApplyOpInvoke
    {
        public static void Invoke(TSCudaContext context, CudaContext cudaContext, byte[] ptx, string baseName, params object[] args)
        {
            ThrowIfAnyTensorInvalid(args);

            cudaContext.SetCurrent();

            var deviceInfo = context.DeviceInfoForContext(cudaContext);

            var allTensors = args.OfType<Tensor>();
            var firstTensor = allTensors.First();
            var elementCount = firstTensor.ElementCount();
            var spec = new ApplySpecialization(allTensors.ToArray());

            ConvertTensorArgs.Convert(cudaContext, spec.Use32BitIndices, args);
            
            var block = ApplyUtils.GetApplyBlock();
            var grid = ApplyUtils.GetApplyGrid(deviceInfo, elementCount);

            var fullKernelName = PermutationGenerator.GetMangledName(baseName, spec);
            var kernel = context.KernelCache.Get(cudaContext, ptx, fullKernelName);

            kernel.GridDimensions = grid;
            kernel.BlockDimensions = block;
            kernel.RunAsync(CUstream.NullStream, args);
            
        }


        private static void ThrowIfAnyTensorInvalid(object[] args)
        {
            foreach (var tensor in args.OfType<Tensor>())
            {
                if (tensor.DimensionCount > TSCudaContext.MaxDims)
                    throw new InvalidOperationException("Tensors with dimension count > " + TSCudaContext.MaxDims + " are not supported");
            }
        }


        public static void ApplyPrecompile(CudaCompiler compiler, DeviceKernelTemplate template, int tensorCount)
        {
            foreach(var spec in ApplySpecialization.AllSpecializations(tensorCount))
            {
                template.PtxForConfig(compiler, spec.GetConfig());
            }
        }
    }

    public static class ElementwiseTTOp
    {
        public static Tensor Invoke(ElementwiseKernels kernels, string funcName, Tensor result, Tensor src)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var cudaContext = context.CudaContextForTensor(src);

            cudaContext.SetCurrent();

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            var elementCount = writeTarget.ElementCount();

            var ptx = kernels.GetPtx(context.Compiler);

            if (result == src)
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t1_" + funcName, writeTarget, elementCount);
            else
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t2_" + funcName, writeTarget, src, elementCount);
            
            return writeTarget;
        }
    }

    public static class ElementwiseTTSOp
    {
        public static Tensor Invoke(ElementwiseKernels kernels, string funcName, Tensor result, Tensor src, float value)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var cudaContext = context.CudaContextForTensor(src);

            cudaContext.SetCurrent();

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            var elementCount = writeTarget.ElementCount();

            var ptx = kernels.GetPtx(context.Compiler);

            if (result == src)
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t1_" + funcName, writeTarget, value, elementCount);
            else
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t2_" + funcName, writeTarget, src, value, elementCount);

            return writeTarget;
        }
    }

    public static class ElementwiseTTTSOp
    {
        public static Tensor Invoke(ElementwiseKernels kernels, string funcName, Tensor result, Tensor src, Tensor src2, float value)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var cudaContext = context.CudaContextForTensor(src);

            cudaContext.SetCurrent();

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            var elementCount = writeTarget.ElementCount();

            var ptx = kernels.GetPtx(context.Compiler);

            if (result == src)
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t1_" + funcName, writeTarget, src2, value, elementCount);
            else
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t2_" + funcName, writeTarget, src, src2, value, elementCount);

            return writeTarget;
        }
    }

    public static class ElementwiseTTTTSOp
    {
        public static Tensor Invoke(ElementwiseKernels kernels, string funcName, Tensor result, Tensor src, Tensor src2,Tensor src3, float value)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var cudaContext = context.CudaContextForTensor(src);

            cudaContext.SetCurrent();

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            var elementCount = writeTarget.ElementCount();

            var ptx = kernels.GetPtx(context.Compiler);

            if (result == src)
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t1_" + funcName, writeTarget, src2, src3, value, elementCount);
            else
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t2_" + funcName, writeTarget, src, src2, src3, value, elementCount);

            return writeTarget;
        }
    }

    public static class ElementwiseTTTTSSOp
    {
        public static Tensor Invoke(ElementwiseKernels kernels, string funcName, Tensor result, Tensor src, Tensor src2, Tensor src3, float step_size, float value)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var cudaContext = context.CudaContextForTensor(src);

            cudaContext.SetCurrent();

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            var elementCount = writeTarget.ElementCount();

            var ptx = kernels.GetPtx(context.Compiler);

            if (result == src)
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t1_" + funcName, writeTarget, src2, src3, step_size, value, elementCount);
            else
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t2_" + funcName, writeTarget, src, src2, src3, step_size, value, elementCount);

            return writeTarget;
        }
    }

    public static class ElementwiseTTTSSOp
    {
        public static Tensor Invoke(ElementwiseKernels kernels, string funcName, Tensor result, Tensor src, Tensor src2, float step_size, float value)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var cudaContext = context.CudaContextForTensor(src);

            cudaContext.SetCurrent();

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            var elementCount = writeTarget.ElementCount();

            var ptx = kernels.GetPtx(context.Compiler);

            if (result == src)
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t1_" + funcName, writeTarget, src2, step_size, value, elementCount);
            else
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t2_" + funcName, writeTarget, src, src2, step_size, value, elementCount);

            return writeTarget;
        }
    }

    public static class ElementwiseTTTOp
    {
        public static Tensor Invoke(ElementwiseKernels kernels, string funcName, Tensor result, Tensor lhs, Tensor rhs)
        {
            var context = CudaHelpers.TSContextForTensor(lhs);
            var cudaContext = context.CudaContextForTensor(lhs);

            cudaContext.SetCurrent();

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            var elementCount = writeTarget.ElementCount();

            var ptx = kernels.GetPtx(context.Compiler);

            if (result == lhs)
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t1_" + funcName, writeTarget, rhs, elementCount);
            else
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t2_" + funcName, writeTarget, lhs, rhs, elementCount);

            return writeTarget;
        }
    }

    public static class ElementwiseTTTTOp
    {
        public static Tensor Invoke(ElementwiseKernels kernels, string funcName, Tensor result, Tensor lhs, Tensor rhs, Tensor rhs2)
        {
            var context = CudaHelpers.TSContextForTensor(lhs);
            var cudaContext = context.CudaContextForTensor(lhs);

            cudaContext.SetCurrent();

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            var elementCount = writeTarget.ElementCount();

            var ptx = kernels.GetPtx(context.Compiler);

            if (result == lhs)
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t1_" + funcName, writeTarget, rhs, rhs2, elementCount);
            else
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t2_" + funcName, writeTarget, lhs, rhs, rhs2, elementCount);

            return writeTarget;
        }
    }

    public static class ElementwiseTTTTTOp
    {
        public static Tensor Invoke(ElementwiseKernels kernels, string funcName, Tensor result, Tensor lhs, Tensor rhs, Tensor rhs2, Tensor rhs3)
        {
            var context = CudaHelpers.TSContextForTensor(lhs);
            var cudaContext = context.CudaContextForTensor(lhs);

            cudaContext.SetCurrent();

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            var elementCount = writeTarget.ElementCount();

            var ptx = kernels.GetPtx(context.Compiler);

            if (result == lhs)
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t1_" + funcName, writeTarget, rhs, rhs2, rhs3, elementCount);
            else
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t2_" + funcName, writeTarget, lhs, rhs, rhs2, rhs3, elementCount);

            return writeTarget;
        }
    }

    public static class ClampOp
    {
        public static Tensor Invoke(ElementwiseKernels kernels, Tensor result, Tensor src, float min, float max)
        {
            var funcName = "clamp";
            var context = CudaHelpers.TSContextForTensor(src);
            var cudaContext = context.CudaContextForTensor(src);

            cudaContext.SetCurrent();

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            var elementCount = writeTarget.ElementCount();

            var ptx = kernels.GetPtx(context.Compiler);

            if (result == src)
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t1_" + funcName, writeTarget, min, max, elementCount);
            else
                ApplyOpInvoke.Invoke(context, cudaContext, ptx, "t2_" + funcName, writeTarget, src, min, max, elementCount);

            return writeTarget;
        }
    }

    public static class Atan2Op
    {
        public static Tensor Invoke(ElementwiseKernels kernels, Tensor result, Tensor srcY, Tensor srcX)
        {
            var context = CudaHelpers.TSContextForTensor(srcY);
            var cudaContext = context.CudaContextForTensor(srcY);

            cudaContext.SetCurrent();

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, srcY, false, srcY.Sizes);
            var elementCount = writeTarget.ElementCount();

            var ptx = kernels.GetPtx(context.Compiler);
            ApplyOpInvoke.Invoke(context, cudaContext, ptx, "atan2", writeTarget, srcY, srcX, elementCount);

            return writeTarget;
        }
    }

    public static class LerpOp
    {
        public static Tensor Invoke(ElementwiseKernels kernels, Tensor result, Tensor srcA, Tensor srcB, float weight)
        {
            var context = CudaHelpers.TSContextForTensor(srcA);
            var cudaContext = context.CudaContextForTensor(srcA);

            cudaContext.SetCurrent();

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, srcA, false, srcA.Sizes);
            var elementCount = writeTarget.ElementCount();

            var ptx = kernels.GetPtx(context.Compiler);
            ApplyOpInvoke.Invoke(context, cudaContext, ptx, "lerp", writeTarget, srcA, srcB, weight, elementCount);

            return writeTarget;
        }
    }

    public static class CopyOp
    {
        public static void Invoke(FillCopyKernels kernels, TSCudaContext context, CudaContext cudaContext, Tensor result, Tensor src)
        {
        //    cudaContext.SetCurrent();

            var ptx = kernels.GetPtx(context.Compiler);
            var elementCount = result.ElementCount();
            ApplyOpInvoke.Invoke(context, cudaContext, ptx, "copy", result, src, elementCount);
        }
    }

    public static class FillOp
    {
        public static void Invoke(FillCopyKernels kernels, Tensor result, float value)
        {
            var context = CudaHelpers.TSContextForTensor(result);
            var cudaContext = context.CudaContextForTensor(result);

            cudaContext.SetCurrent();

            var ptx = kernels.GetPtx(context.Compiler);
            var elementCount = result.ElementCount();
            ApplyOpInvoke.Invoke(context, cudaContext, ptx, "fill", result, value, elementCount);
        }
    }
}
