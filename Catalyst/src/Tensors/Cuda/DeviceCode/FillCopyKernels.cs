using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors.CUDA.DeviceCode
{
    [Precompile]
    public class FillCopyKernels : CudaCode
    {
        public FillCopyKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "PointwiseApply", "ApplyMacros")
        {
        }

        private static string GetFullCode()
        {
            var result = new PermutationGenerator();
            result.AddApplyTS("fill", "*a = b;");

            result.AddApplyTT("copy", "*a = *b;");

            return result.ToString();
        }
    }

}
