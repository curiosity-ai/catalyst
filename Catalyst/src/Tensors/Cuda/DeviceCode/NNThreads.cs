using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors.CUDA.DeviceCode
{
    public static class NNThreads
    {
        public const int NumThreads = 1024;

        public static int NumBlocks(int n)
        {
            return (n + NumThreads - 1) / NumThreads;
        }
    }
}
