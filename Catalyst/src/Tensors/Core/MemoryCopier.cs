using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Catalyst.Tensors.Core
{
    internal static class MemoryCopier
    {
        private static class NativeMethods32
        {
            [DllImport("kernel32.dll")]
            public static extern void CopyMemory(IntPtr destination, IntPtr source, uint length);
        }

        private static class NativeMethods64
        {
            [DllImport("kernel32.dll")]
            public static extern void CopyMemory(IntPtr destination, IntPtr source, ulong length);
        }

        public static void Copy(IntPtr destination, IntPtr source, ulong length)
        {
            var is32 = IntPtr.Size == 4;

            if (is32)
            {
                // Note: if this is run, length should always be in range of a uint
                // (it should be impossible to allocate a buffer bigger than that range
                // on a 32-bit system)
                NativeMethods32.CopyMemory(destination, source, (uint)length);
            }
            else
            {
                NativeMethods64.CopyMemory(destination, source, (ulong)length);
            }

        }
    }
}
