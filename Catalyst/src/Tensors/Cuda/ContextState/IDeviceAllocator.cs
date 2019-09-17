using ManagedCuda;
using ManagedCuda.BasicTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors.CUDA.ContextState
{
    public interface IDeviceMemory
    {
        CUdeviceptr Pointer { get; }

        void Free();
    }

    public interface IDeviceAllocator : IDisposable
    {
        IDeviceMemory Allocate(long byteCount);
        void FreeMemory(bool callGC = false);
    }
}
