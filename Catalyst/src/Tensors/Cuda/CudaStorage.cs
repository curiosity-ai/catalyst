using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Catalyst.Tensors.CUDA.ContextState;
using Catalyst.Tensors;

namespace Catalyst.Tensors.CUDA
{
    [Serializable]
    public class CudaStorage : Storage
    {
        private readonly CudaContext context;

        private IDeviceMemory bufferHandle;
        private readonly CUdeviceptr deviceBuffer;


        public CudaStorage(IAllocator allocator, TSCudaContext tsContext, CudaContext context, DType ElementType, long elementCount)
            : base(allocator, ElementType, elementCount)
        {
            this.TSContext = tsContext;
            this.context = context;

            this.bufferHandle = tsContext.AllocatorForDevice(DeviceId).Allocate(this.ByteLength);
            this.deviceBuffer = this.bufferHandle.Pointer;
        }

        public TSCudaContext TSContext { get; private set; }

        protected override void Destroy()
        {
            if (bufferHandle != null)
            {
                bufferHandle.Free();
                bufferHandle = null;
            }
        }

        public override string LocationDescription()
        {
            return "CUDA:" + context.DeviceId;
        }

        public int DeviceId
        {
            get { return context.DeviceId; }
        }

        public CUdeviceptr DevicePtrAtElement(long index)
        {
            var offset = ElementType.Size() * index;
            return new CUdeviceptr(deviceBuffer.Pointer + offset);
        }

        public override float GetElementAsFloat(long index)
        {
            var ptr = DevicePtrAtElement(index);

            if(ElementType == DType.Float32) { var result = new float[1]; context.CopyToHost(result, ptr); return result[0]; }
            else if (ElementType == DType.Float64) { var result = new double[1]; context.CopyToHost(result, ptr); return (float)result[0]; }
            else if (ElementType == DType.Int32) { var result = new int[1]; context.CopyToHost(result, ptr); return result[0]; }
            else if (ElementType == DType.UInt8) { var result = new byte[1]; context.CopyToHost(result, ptr); return result[0]; }
            else
                throw new NotSupportedException("Element type " + ElementType + " not supported");
        }


        public override float[] GetElementsAsFloat(long index, int length)
        {
            var ptr = DevicePtrAtElement(index);

            if (ElementType == DType.Float32) { var result = new float[length]; context.CopyToHost(result, ptr); return result; }
            else
                throw new NotSupportedException("Element type " + ElementType + " not supported");
        }

        public override void SetElementAsFloat(long index, float value)
        {
            var ptr = DevicePtrAtElement(index);

            if (ElementType == DType.Float32) { context.CopyToDevice(ptr, (float)value); }
            else if (ElementType == DType.Float64) { context.CopyToDevice(ptr, (double)value); }
            else if (ElementType == DType.Int32) { context.CopyToDevice(ptr, (int)value); }
            else if (ElementType == DType.UInt8) { context.CopyToDevice(ptr, (byte)value); }
            else
                throw new NotSupportedException("Element type " + ElementType + " not supported");
        }

        public override void SetElementsAsFloat(long index, float[] value)
        {
            var ptr = DevicePtrAtElement(index);

            if (ElementType == DType.Float32) { context.CopyToDevice(ptr, value); }
            else
                throw new NotSupportedException("Element type " + ElementType + " not supported");
        }

        public override void CopyToStorage(long storageIndex, IntPtr src, long byteCount)
        {
            var dstPtr = DevicePtrAtElement(storageIndex);
            context.CopyToDevice(dstPtr, src, byteCount);
        }

        public override void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount)
        {
            var srcPtr = DevicePtrAtElement(storageIndex);

            // Call this method directly instead of CudaContext.CopyToHost because this method supports a long byteCount
            // CopyToHost only supports uint byteCount.
            var res = DriverAPINativeMethods.SynchronousMemcpy_v2.cuMemcpyDtoH_v2(dst, srcPtr, byteCount);
            if (res != CUResult.Success)
                throw new CudaException(res);
        }
    }
}
