using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors.CUDA.ContextState
{
    /// <summary>
    /// This allocator simply forwards all alloc/free requests to CUDA. This will generally be slow
    /// because calling cudaMalloc causes GPU synchronization
    /// </summary>
    public class BasicDeviceAllocator : IDeviceAllocator
    {
        private readonly CudaContext context;

        public BasicDeviceAllocator(CudaContext cudaContext)
        {
            context = cudaContext;
        }

        public void Dispose()
        {
        }


        public IDeviceMemory Allocate(long byteCount)
        {
            var buffer = context.AllocateMemory(byteCount);
            return new BasicDeviceMemory(buffer, () => context.FreeMemory(buffer));
        }

        public void FreeMemory(bool callGC = false)
        {

        }
    }

    public class BasicDeviceMemory : IDeviceMemory
    {
        private readonly CUdeviceptr pointer;
        private readonly Action freeHandler;

        public CUdeviceptr Pointer { get { return pointer; } }


        public BasicDeviceMemory(CUdeviceptr pointer, Action freeHandler)
        {
            this.pointer = pointer;
            this.freeHandler = freeHandler;
        }

        public void Free()
        {
            freeHandler();
        }
    }
}
