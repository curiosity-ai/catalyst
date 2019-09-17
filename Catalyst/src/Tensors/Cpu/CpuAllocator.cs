using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors.Cpu
{
    public class CpuAllocator : IAllocator
    {
        public CpuAllocator()
        {
        }

        public Storage Allocate(DType elementType, long elementCount)
        {
            return new CpuStorage(this, elementType, elementCount);
        }
    }
}
