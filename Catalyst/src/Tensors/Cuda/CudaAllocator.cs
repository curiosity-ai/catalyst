using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors.CUDA
{
    [Serializable]
    public class CudaAllocator : IAllocator
    {
        private readonly TSCudaContext context;
        private readonly int deviceId;

        public CudaAllocator(TSCudaContext context, int deviceId)
        {
            this.context = context;
            this.deviceId = deviceId;
        }

        public TSCudaContext Context { get { return context; } }
        public int DeviceId { get { return deviceId; } }

        public Storage Allocate(DType elementType, long elementCount)
        {
            return new CudaStorage(this, context, context.CudaContextForDevice(deviceId), elementType, elementCount);
        }
    }
}
