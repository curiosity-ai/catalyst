using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.CUDA.RuntimeCompiler;

namespace Catalyst.Tensors.CUDA.DeviceCode.Headers
{
    [CudaInclude("Code", "ReduceMacros")]
    public static class ReduceMacros
    {
        public static readonly string Code = @"
#define REDUCE_KERNELS(INDEX_TYPE, DIMSA, DIMSB, KERNEL_NAME, MODIFY_OP_CODE, REDUCE_OP_CODE) \
struct ModifyOp##KERNEL_NAME { __device__ __forceinline__ float operator()(const float a) const { MODIFY_OP_CODE } };\
struct ReduceOp##KERNEL_NAME { __device__ __forceinline__ float operator()(const float a, const float b) const { REDUCE_OP_CODE } };\
extern ""C"" {\
    __global__ void contig_##KERNEL_NAME(TensorInfo<INDEX_TYPE> out, TensorInfo<INDEX_TYPE> in, INDEX_TYPE reductionSize, INDEX_TYPE totalSlices, float init) {\
        reduceContigDim_apply<ModifyOp##KERNEL_NAME, ReduceOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB>(out, in, reductionSize, totalSlices, init, ModifyOp##KERNEL_NAME(), ReduceOp##KERNEL_NAME());\
    }\
    __global__ void noncontig_##KERNEL_NAME(TensorInfo<INDEX_TYPE> out,\
                                TensorInfo<INDEX_TYPE> in,\
                                INDEX_TYPE reductionStride,\
                                INDEX_TYPE reductionSize,\
                                INDEX_TYPE totalSlices,\
                                float init) {\
        reduceNoncontigDim_apply<ModifyOp##KERNEL_NAME, ReduceOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB>(out, in, reductionStride, reductionSize, totalSlices, init, ModifyOp##KERNEL_NAME(), ReduceOp##KERNEL_NAME());\
    }\
}

#define REDUCE_NORM_KERNELS(INDEX_TYPE, DIMSA, DIMSB, KERNEL_NAME) \
struct ModifyOp##KERNEL_NAME {\
    const float exponent;\
    __device__ ModifyOp##KERNEL_NAME(float exp) : exponent(exp) {}\
    __device__ __forceinline__ float operator()(const float a) const { return powf(fabsf(a), exponent);\
} };\
struct ReduceOp##KERNEL_NAME { __device__ __forceinline__ float operator()(const float a, const float b) const { return a + b; } };\
extern ""C"" {\
    __global__ void contig_##KERNEL_NAME(TensorInfo<INDEX_TYPE> out, TensorInfo<INDEX_TYPE> in, INDEX_TYPE reductionSize, INDEX_TYPE totalSlices, float init, float exponent) {\
        reduceContigDim_apply<ModifyOp##KERNEL_NAME, ReduceOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB>(out, in, reductionSize, totalSlices, init, ModifyOp##KERNEL_NAME(exponent), ReduceOp##KERNEL_NAME());\
    }\
    __global__ void noncontig_##KERNEL_NAME(TensorInfo<INDEX_TYPE> out,\
                                TensorInfo<INDEX_TYPE> in,\
                                INDEX_TYPE reductionStride,\
                                INDEX_TYPE reductionSize,\
                                INDEX_TYPE totalSlices,\
                                float init, float exponent) {\
        reduceNoncontigDim_apply<ModifyOp##KERNEL_NAME, ReduceOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB>(out, in, reductionStride, reductionSize, totalSlices, init, ModifyOp##KERNEL_NAME(exponent), ReduceOp##KERNEL_NAME());\
    }\
}

";
    }
}
