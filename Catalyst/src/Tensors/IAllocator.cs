using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors
{
    public interface IAllocator
    {
        Storage Allocate(DType elementType, long elementCount);
    }
}
