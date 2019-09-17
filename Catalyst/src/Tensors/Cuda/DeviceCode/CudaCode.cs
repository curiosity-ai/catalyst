using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.CUDA.RuntimeCompiler;

namespace Catalyst.Tensors.CUDA.DeviceCode
{
    public abstract class CudaCode : IPrecompilable
    {
        private readonly string code;
        private readonly string[] requiredHeaders;
        private byte[] ptx = null;

        protected CudaCode(string code, params string[] requiredHeaders)
        {
            this.code = code;
            this.requiredHeaders = requiredHeaders;
        }

        public byte[] GetPtx(CudaCompiler compiler)
        {
            if (ptx == null)
            {
                Precompile(compiler);
            }
            return ptx;
        }

        public void Precompile(CudaCompiler compiler)
        {
            ptx = compiler.CompileToPtx(code, requiredHeaders);
        }
    }

}
