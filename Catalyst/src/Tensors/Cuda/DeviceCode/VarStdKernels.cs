using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.Core;

namespace Catalyst.Tensors.CUDA.DeviceCode
{
    [Precompile]
    public class VarStdKernels : CudaCode
    {
        private static readonly string Code = @"
// Given the sum of values and the sum of squares, compute the variance or standard deviation.
template<bool flag, bool apply_sqrt>
__forceinline__ __device__ float computeVar(float sum, float sum2, unsigned row_size) {
  if (flag) {
    sum /= row_size;
    sum2 /= row_size;
    sum2 -= sum * sum;
    sum2 = (sum2 < 0 ? 0 : sum2);
  }
  else {
    sum /= row_size;
    sum2 /= row_size - 1;
    sum2 -= ((float)row_size) / ((float)(row_size - 1)) * sum * sum;
    sum2 = (sum2 < 0 ? 0 : sum2);
  }
  if (apply_sqrt)
    return sqrt(sum2);
  else
    return sum2;
}

/* Compute the variance (or standard deviation) along an outer dimension of a tensor.
 *
 * - num_orows is the size of the flattened outer dimensions;
 * - num_irows is the size of the flattened inner dimensions;
 * - row_size is the size of the dimension along which to compute the variance;
 * - if flag is set, normalize by `row_size` instead of `row_size - 1`
 * - if apply_sqrt is set, compute the standard deviation instead of variance
 *
 * The dimensions to the outside and inside of the specified dimension are considered as flattened.
 * Thread blocks with the same blockIdx.y process an 'outer row' (i.e. an element of the flattened
 * outer dimensions, which contains several 'inner rows').
 * Each thread processes a single inner row at a time.
 */
template<bool flag, bool apply_sqrt>
__device__ void kernel_varOuterDim(float* tgt, float* src_, unsigned num_orows, unsigned num_irows, unsigned row_size)
{
    for (unsigned orow = blockIdx.x; orow < num_orows; orow += gridDim.x)
    {
        for (unsigned irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows; irow += gridDim.y * blockDim.x)
        {
            float* src = src_ + orow * row_size * num_irows + irow;
            float sum = 0, sum2 = 0;

            for (unsigned col = 0; col < row_size; ++col)
            {
                float val = *src;
                sum += val;
                sum2 += val * val;

                src += num_irows;
            }

            tgt[orow * num_irows + irow] = computeVar<flag, apply_sqrt>(sum, sum2, row_size);
        }
    }
}





/* Compute the variance (or standard deviation) of the innermost dimension of a tensor.
 *
 * - num_rows is the size of the flattened outer dimensions;
 * - row_size is the size of the innermost dimension;
 * - if flag is set, normalize by `row_size` instead of `row_size - 1`
 * - if apply_sqrt is set, compute the standard deviation instead of variance
 *
 * The outer dimensions of the tensor are considered as a single dimension, i.e. the tensor is
 * considered as having 'num_rows' rows of size 'row_size'.
 * Each thread block processes one or more sets of contiguous rows (processing multiple rows
 * per thread block is quicker than processing a single row, especially for short rows).
 */
template<bool flag, bool apply_sqrt>
__device__ void kernel_varInnermostDim(float *tgt, float *src_, unsigned num_rows, unsigned row_size)
{
  __shared__ float ssum[32][16];
  __shared__ float ssum2[32][16];

  for (unsigned block_row = blockIdx.x * blockDim.y; block_row < num_rows; block_row += blockDim.y * gridDim.x) {
    unsigned row = block_row + threadIdx.y;
    float sum = 0, sum2 = 0;
    if (row < num_rows) {
      float *src = src_ + row * row_size;
      // Sequential reduction within a thread.
      for (unsigned col = threadIdx.x; col < row_size; col += blockDim.x) {
        float val = src[col];
        sum += val;
        sum2 += val * val;
      }
    }
    ssum[threadIdx.y][threadIdx.x] = sum;
    ssum2[threadIdx.y][threadIdx.x] = sum2;
    __syncthreads();

    // Reduce intermediate values to single value.
    for (unsigned s = 8; s > 1; s >>= 1) {
      if (row < num_rows && threadIdx.x < s) {
        ssum[threadIdx.y][threadIdx.x] += ssum[threadIdx.y][threadIdx.x + s];
        ssum2[threadIdx.y][threadIdx.x] += ssum2[threadIdx.y][threadIdx.x + s];
      }
      __syncthreads();
    }

    if (row < num_rows && threadIdx.x == 0) {
      sum = ssum[threadIdx.y][0] + ssum[threadIdx.y][1];
      sum2 = ssum2[threadIdx.y][0] + ssum2[threadIdx.y][1];
      tgt[row] = computeVar<flag, apply_sqrt>(sum, sum2, row_size);
    }
    __syncthreads();
  }
}

#define DECLARE_OUTERMOST(FLAG_VALUE, SQRT_VALUE) \
    __global__ void kernel_varOuterDim_##FLAG_VALUE##_##SQRT_VALUE(float* tgt, float* src_, unsigned num_orows, unsigned num_irows, unsigned row_size)\
    {\
        kernel_varOuterDim<FLAG_VALUE, SQRT_VALUE>(tgt, src_, num_orows, num_irows, row_size);\
    }

#define DECLARE_INNERMOST(FLAG_VALUE, SQRT_VALUE) \
    __global__ void kernel_varInnermostDim_##FLAG_VALUE##_##SQRT_VALUE(float *tgt, float *src_, unsigned num_rows, unsigned row_size)\
    {\
        kernel_varInnermostDim<FLAG_VALUE, SQRT_VALUE>(tgt, src_, num_rows, row_size);\
    }

extern ""C"" {
    DECLARE_OUTERMOST(true, true)
    DECLARE_OUTERMOST(true, false)
    DECLARE_OUTERMOST(false, true)
    DECLARE_OUTERMOST(false, false)

    DECLARE_INNERMOST(true, true)
    DECLARE_INNERMOST(true, false)
    DECLARE_INNERMOST(false, true)
    DECLARE_INNERMOST(false, false)
}

";

        public VarStdKernels()
            : base(GetFullCode(), "General", "ReduceApplyUtils", "ReduceBlock", "Reduce", "ReduceMacros", "Math")
        {
        }

        private static string GetFullCode()
        {
            return Code;
        }

        private static string GetMangledNameSuffix(bool normByN, bool applySqrt)
        {
            return string.Format("_{0}_{1}", normByN, applySqrt).ToLower();
        }


        private void VarOuterDim(TSCudaContext context, Tensor result, Tensor src, int dimension, bool normByN, bool applySqrt)
        {
            var cudaContext = context.CudaContextForTensor(src);

            int ndim = src.DimensionCount;
            long num_orows = 1;
            for (int dim = 0; dim < dimension; dim++)
            {
                num_orows *= src.Sizes[dim];
            }
            long row_size = src.Sizes[dimension];
            // Treat all inner dimensions (i.e. dim > dimension) as one.
            long num_irows = 1;
            for (int dim = dimension + 1; dim < ndim; dim++)
            {
                num_irows *= src.Sizes[dim];
            }

            var threads = new dim3((uint)Math.Min(512, num_irows));
            var maxGridDim = 1024;
            var grid = new dim3((uint)Math.Min(maxGridDim, num_orows), (uint)Math.Min(maxGridDim, ApplyUtils.CeilDiv(num_irows, threads.x)));

            var resultPtr = CudaHelpers.GetBufferStart(result);
            var srcPtr = CudaHelpers.GetBufferStart(src);
            var kernelName = "kernel_varOuterDim" + GetMangledNameSuffix(normByN, applySqrt);

            Invoke(context, cudaContext, kernelName, grid, threads, 0, CUstream.NullStream, resultPtr, srcPtr, num_orows, num_irows, row_size);
        }


        private void VarInnermostDim(TSCudaContext context, Tensor result, Tensor src, bool normByN, bool applySqrt)
        {
            var cudaContext = context.CudaContextForTensor(src);

            var ndim = src.DimensionCount;
            long num_rows = 1;
            for (var dim = 0; dim < ndim - 1; dim++)
            {
                num_rows *= src.Sizes[dim];
            }
            var row_size = src.Sizes[ndim - 1];

            // (Comment from cuTorch source): From limited testing, 16x32 seemed a good compromise for handling both long and short dimensions.
            var threads = new dim3(16, 32);
            var grid = new dim3((uint)Math.Min(1024, ApplyUtils.CeilDiv(num_rows, threads.y)));

            var resultPtr = CudaHelpers.GetBufferStart(result);
            var srcPtr = CudaHelpers.GetBufferStart(src);
            var kernelName = "kernel_varInnermostDim" + GetMangledNameSuffix(normByN, applySqrt);

            Invoke(context, cudaContext, kernelName, grid, threads, 0, CUstream.NullStream, resultPtr, srcPtr, num_rows, row_size);
        }

     
        private Tensor RunVarOp(Tensor result, Tensor src, int dimension, bool normByN, bool applySqrt)
        {
            var context = CudaHelpers.TSContextForTensor(src);
            var requiredOutputSize = (long[])src.Sizes.Clone();
            requiredOutputSize[dimension] = 1;
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, requiredOutputSize);

            if (dimension == src.DimensionCount - 1)
            {
                VarInnermostDim(context, writeTarget, src, normByN, applySqrt);
            }
            else
            {
                VarOuterDim(context, writeTarget, src, dimension, normByN, applySqrt);
            }

            return writeTarget;
        }

        public Tensor Var(Tensor result, Tensor src, int dimension, bool normByN)
        {
            return RunVarOp(result, src, dimension, normByN, false);
        }

        public Tensor Std(Tensor result, Tensor src, int dimension, bool normByN)
        {
            return RunVarOp(result, src, dimension, normByN, true);
        }

        private void Invoke(TSCudaContext context, CudaContext cudaContext, string kernelName, dim3 grid, dim3 block, uint smemSize, CUstream stream, params object[] args)
        {
            var ptx = GetPtx(context.Compiler);
            var kernel = context.KernelCache.Get(cudaContext, ptx, kernelName);
            kernel.GridDimensions = grid;
            kernel.BlockDimensions = block;
            kernel.DynamicSharedMemory = smemSize;
            kernel.RunAsync(stream, args);
        }
    }
}
