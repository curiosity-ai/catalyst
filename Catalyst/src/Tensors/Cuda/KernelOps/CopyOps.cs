using ManagedCuda;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using ManagedCuda.BasicTypes;
using Catalyst.Tensors.CUDA.KernelOps;
using System.Runtime.InteropServices;

namespace Catalyst.Tensors.CUDA.KernelOps
{
    public class CopyOps
    {
        private readonly Cpu.CpuAllocator cpuAllocator = new Cpu.CpuAllocator();
        private readonly DeviceCode.FillCopyKernels fillCopyKernels;


        public CopyOps(DeviceCode.FillCopyKernels fillCopyKernels)
        {
           this.fillCopyKernels = fillCopyKernels;
        }

        // Can memcpy if both tensors have the same element type, AND any of the following are true
        // - both tensors are contiguous
        // - there is only one element
        // It would also be possible to memcpy if tensors have matching size & stride
        // and there are no holes (ie. there is some permutation of dims such that the
        // tensors are contiguous). This is not currently checked for.
        private static bool CanMemcpy(Tensor result, Tensor src, long totalElements)
        {
            if(result.ElementType != src.ElementType)
            {
                return false;
            }


            return
                (result.IsContiguous() && src.IsContiguous()) || totalElements == 1;
        }


        public void CopyGpu(Tensor result, Tensor src, long totalElements)
        {
            // We assume here that we are using the default stream for both devices.
            var context = CudaHelpers.TSContextForTensor(src);

            var resultStorage = (CudaStorage)result.Storage;
            var resultContext = context.CudaContextForTensor(result);
            var resultPtr = resultStorage.DevicePtrAtElement(result.StorageOffset);

            var srcStorage = (CudaStorage)src.Storage;
            var srcContext = context.CudaContextForTensor(src);
            var srcPtr = srcStorage.DevicePtrAtElement(src.StorageOffset);


            if (CudaHelpers.GetDeviceId(result) != CudaHelpers.GetDeviceId(src))
            {
                // Cross-device copy. Perform two-way barrier between both devices' default streams.
                resultContext.SetCurrent();
                var dstReady = new CudaEvent(CUEventFlags.DisableTiming);
                dstReady.Record();

                srcContext.SetCurrent();
                var res = DriverAPINativeMethods.Streams.cuStreamWaitEvent(CUstream.NullStream, dstReady.Event, 0);
                if (res != CUResult.Success) throw new CudaException(res);
                dstReady.Dispose();
            }
            else
            {
                srcContext.SetCurrent();
            }

            var canMemcpy = CanMemcpy(result, src, totalElements);

            if (canMemcpy)
            {
                var res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAsync(
                    resultPtr, srcPtr, totalElements * src.ElementType.Size(), CUstream.NullStream);
                if (res != CUResult.Success) throw new CudaException(res);
            }
            else
            {
                if (result.ElementType != src.ElementType)
                {
                    CopyGpuConvertTypes(result, src, totalElements);
                }
                else if (context.CanAccessPeer(CudaHelpers.GetDeviceId(src), CudaHelpers.GetDeviceId(result)))
                {
                    CopyGpuDirect(result, src, srcContext);
                }
                else
                {
                    CopyGpuIndirect(result, src, totalElements);
                }
            }
        }

        private void CopyGpuDirect(Tensor result, Tensor src, CudaContext srcContext)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            CopyOp.Invoke(fillCopyKernels, context, srcContext, result, src);
        }

        private void CopyGpuIndirect(Tensor result, Tensor src, long totalElements)
        {
            // This is only called if the tensors have the same type, but memcpy cannot be used on the tensor pair,
            // and we can't get direct access to the other GPU's memory.

            // We will make contiguous proxy tensors as necessary, so we can use cuMemcpy to perform the copy.
            // If result needs to be proxied, we then copy back from the contiguous proxy to result on the same GPU

            var context = CudaHelpers.TSContextForTensor(src);
            var isResultContig = result.IsContiguous();
            var resultContig = result;

            using (var srcContig = Ops.AsContiguous(src))
            {
                if (!isResultContig)
                {
                    resultContig = new Tensor(result.Allocator, result.ElementType, result.Sizes);
                }

                var resultContigPtr = ((CudaStorage)resultContig.Storage).DevicePtrAtElement(resultContig.StorageOffset);
                var srcContigPtr = ((CudaStorage)srcContig.Storage).DevicePtrAtElement(srcContig.StorageOffset);

                var res = DriverAPINativeMethods.AsynchronousMemcpy_v2.cuMemcpyAsync(
                    resultContigPtr, srcContigPtr, totalElements * srcContig.ElementType.Size(), CUstream.NullStream);
                if (res != CUResult.Success) throw new CudaException(res);

                if (!isResultContig)
                {
                    CopyGpuDirect(result, resultContig, context.CudaContextForTensor(result));
                    resultContig.Dispose();
                }
            }
        }

        private void CopyGpuConvertTypes(Tensor result, Tensor src, long totalElements)
        {
            // Type conversions are currently done via CPU
            using (var srcCopy = new Tensor(cpuAllocator, src.ElementType, src.Sizes))
            using (var srcConverted = new Tensor(cpuAllocator, result.ElementType, src.Sizes))
            {
                CopyGpuToCpu(srcCopy, src, totalElements);
                Ops.Copy(srcConverted, srcCopy); // Do type conversion on CPU
                CopyCpuToGpu(result, srcConverted, totalElements);
            }
        }



        private Tensor AsTypeCpu(Tensor tensor, DType elementType, bool requireContig)
        {
            if (tensor.ElementType == elementType && (!requireContig || tensor.IsContiguous()))
            {
                return tensor.CopyRef();
            }
            else
            {
                var result = new Tensor(cpuAllocator, elementType, tensor.Sizes);
                Ops.Copy(result, tensor);
                return result;
            }
        }

        public void CopyCpuToGpu(Tensor result, Tensor src, long totalElements)
        {
            var context = CudaHelpers.TSContextForTensor(result);
            var resultContext = context.CudaContextForTensor(result);

            // If types of src and result are different, convert on the CPU first.
            using (var srcContig = AsTypeCpu(src, result.ElementType, true))
            using (var resultContig = Ops.AsContiguous(result))
            {
                var resultContigPtr = ((CudaStorage)resultContig.Storage).DevicePtrAtElement(resultContig.StorageOffset);
                var srcContigPtr = ((Cpu.CpuStorage)srcContig.Storage).PtrAtElement(srcContig.StorageOffset);

                resultContext.CopyToDevice(resultContigPtr, srcContigPtr, totalElements * srcContig.ElementType.Size());

                if (result.Storage != resultContig.Storage)
                {
                    CopyGpuDirect(result, resultContig, resultContext);
                }
            }
        }

        public void CopyGpuToCpu(Tensor result, Tensor src, long totalElements)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var srcContext = context.CudaContextForTensor(src);

            using (var srcContig = Ops.AsContiguous(src))
            using (var resultContig = AsTypeCpu(result, src.ElementType, true))
            {
                var resultContigPtr = ((Cpu.CpuStorage)resultContig.Storage).PtrAtElement(resultContig.StorageOffset);
                var srcContigPtr = ((CudaStorage)srcContig.Storage).DevicePtrAtElement(srcContig.StorageOffset);

                var totalBytes = totalElements * srcContig.ElementType.Size();

                // Use DriverAPINativeMethods directly here instead of CudaContext.CopyToHost, because CopyToHost only has an overload
                // for specifying totalBytes as a uint, but we may exceed the range of a uint here.
                var res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(resultContigPtr, srcContigPtr, totalBytes);
                if (res != CUResult.Success) throw new CudaException(res);

                if (result.Storage != resultContig.Storage)
                {
                    Ops.Copy(result, resultContig); // copy on CPU
                }
            }
        }

    }
}
