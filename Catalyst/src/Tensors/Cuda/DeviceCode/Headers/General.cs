using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.CUDA.RuntimeCompiler;

namespace Catalyst.Tensors.CUDA.DeviceCode.Headers
{
    [CudaInclude("Code", "General")]
    public static class KernelGeneral
    {
        public static readonly string Code = @"

#define __int64 long long
#define __int32 int

#define MAX_CUTORCH_DIMS " + TSCudaContext.MaxDims + "\n" + @"

template <typename IndexType>
struct TensorInfo {
  float* data;
  IndexType sizes[MAX_CUTORCH_DIMS];
  IndexType strides[MAX_CUTORCH_DIMS];
  int dims;
};

";

    }
}
