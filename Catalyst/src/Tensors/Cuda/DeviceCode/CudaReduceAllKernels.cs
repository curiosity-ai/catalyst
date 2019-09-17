using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.CUDA.RuntimeCompiler;

namespace Catalyst.Tensors.CUDA.DeviceCode
{
    [Precompile]
    public class CudaReduceAllKernels : CudaCode
    {
        public CudaReduceAllKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "ReduceBlock", "ReduceAll", "ReduceAllMacros", "Math")
        {
        }

        private static string GetFullCode()
        {
            var identity = "return a;";

            var result = new PermutationGenerator();
            result.AddReduceAll("sumAll", identity, "return a + b;");
            result.AddReduceAll("prodAll", identity, "return a * b;");
            result.AddReduceAll("minAll", identity, "return min(a, b);");
            result.AddReduceAll("maxAll", identity, "return max(a, b);");

            result.AddReduceAll("e0_norm", "return a != 0 ? 1 : 0;", "return a + b;");
            result.AddReduceAll("e1_norm", "return fabsf(a);", "return a + b;");
            result.AddReduceAll("e2_norm", "return a * a;", "return a + b;");
            result.AddReduceAllNorm("en_norm");

            result.AddReduceAllSubSquare("subSquare");

            return result.ToString();
        }
    }
}
