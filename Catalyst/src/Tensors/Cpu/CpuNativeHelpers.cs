using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Catalyst.Tensors.Cpu
{
    public static class CpuNativeHelpers
    {
        public static IntPtr GetBufferStart(Tensor tensor)
        {
            var buffer = ((CpuStorage)tensor.Storage).buffer;
            return PtrAdd(buffer, tensor.StorageOffset * tensor.ElementType.Size());
        }

        private static IntPtr PtrAdd(IntPtr ptr, long offset)
        {
            return new IntPtr(ptr.ToInt64() + offset);
        }

    }
}
