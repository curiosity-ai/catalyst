using ManagedCuda;
using ManagedCuda.CudaBlas;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.CUDA.ContextState;
using Catalyst.Tensors.CUDA.Util;

namespace Catalyst.Tensors.CUDA
{
    /// <summary>
    /// Used by TSCudaContext to maintain per-device state
    /// </summary>
    public class DeviceState : IDisposable
    {
        private const int ScratchSpacePerSMStream = 4 * sizeof(float);


        public readonly CudaContext CudaContext;
        public readonly CudaDeviceProperties DeviceInfo;

        public readonly ObjectPool<CudaBlas> BlasHandles;
       // public readonly ObjectPool<ManagedCuda.CudaDNN.CudaDNNContext> DnnHandles;

        public readonly IDeviceAllocator MemoryAllocator;
        public readonly ScratchSpace ScratchSpace;


        public DeviceState(int deviceId)
        {
            CudaContext = new CudaContext(deviceId);
            DeviceInfo = CudaContext.GetDeviceInfo();

            BlasHandles = new ObjectPool<CudaBlas>(1, () =>
            {
                CudaContext.SetCurrent();
                return new CudaBlas();
            },
                blas => blas.Dispose());

            MemoryAllocator = new PoolingDeviceAllocator(CudaContext);
            ScratchSpace = AllocScratchSpace(CudaContext, DeviceInfo);
        }


        public void FreeMemory(bool callGC = false)
        {
            MemoryAllocator.FreeMemory(callGC);
        }

        public void Dispose()
        {
            BlasHandles.Dispose();
            CudaContext.Dispose();
            MemoryAllocator.Dispose();
        }

        private static ScratchSpace AllocScratchSpace(CudaContext context, CudaDeviceProperties deviceProps)
        {
            var size = ScratchSpacePerSMStream * deviceProps.MultiProcessorCount;
            var buffer = context.AllocateMemory(size);
            return new ScratchSpace() { size = size, buffer = buffer };
        }
    }
}
