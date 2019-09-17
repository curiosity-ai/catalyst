using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors.CUDA.RuntimeCompiler
{
    public class DeviceKernel
    {
        private readonly byte[] ptx;


        public DeviceKernel(byte[] ptx)
        {
            this.ptx = ptx;
        }


    }
}
