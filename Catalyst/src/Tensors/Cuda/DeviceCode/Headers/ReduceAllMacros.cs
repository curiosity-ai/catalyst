using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.CUDA.RuntimeCompiler;

namespace Catalyst.Tensors.CUDA.DeviceCode.Headers
{
    [CudaInclude("Code", "ReduceAllMacros")]
    public static class ReduceAllMacros
    {
        public static readonly string Code = @"

#define REDUCE_ALL_KERNELS(INDEX_TYPE, DIMSA, KERNEL_NAME, MODIFY_OP_CODE, REDUCE_OP_CODE) \
struct ModifyOp##KERNEL_NAME { __device__ __forceinline__ float operator()(const float a) const { MODIFY_OP_CODE } };\
struct ReduceOp##KERNEL_NAME { __device__ __forceinline__ float operator()(const float a, const float b) const { REDUCE_OP_CODE } };\
extern ""C"" {\
    __global__ void onePass_##KERNEL_NAME(TensorInfo<INDEX_TYPE> in,\
                       INDEX_TYPE totalElements,\
                       float init,\
                       float* out) {\
        reduceAll<ModifyOp##KERNEL_NAME, ReduceOp##KERNEL_NAME, INDEX_TYPE, DIMSA>(in, totalElements, init, ModifyOp##KERNEL_NAME(), ReduceOp##KERNEL_NAME(), out);\
    }\
    __global__ void twoPassA_##KERNEL_NAME(TensorInfo<INDEX_TYPE> in,\
                       INDEX_TYPE totalElements,\
                       float init,\
                       float* scratchSpace) {\
        reduceAllPass1<ModifyOp##KERNEL_NAME, ReduceOp##KERNEL_NAME, INDEX_TYPE, DIMSA>(in, totalElements, init, ModifyOp##KERNEL_NAME(), ReduceOp##KERNEL_NAME(), scratchSpace);\
    }\
    __global__ void twoPassB_##KERNEL_NAME(int numPass1Blocks,\
                float init,\
                float* scratchSpace,\
                float* out) {\
        reduceAllPass2<ReduceOp##KERNEL_NAME, INDEX_TYPE>(numPass1Blocks, init, ReduceOp##KERNEL_NAME(), scratchSpace, out);\
    }\
}

#define REDUCE_ALL_NORM_KERNELS(INDEX_TYPE, DIMSA, KERNEL_NAME) \
struct ModifyOp##KERNEL_NAME {\
    const float exponent;\
    __device__ ModifyOp##KERNEL_NAME(float exp) : exponent(exp) {}\
    __device__ __forceinline__ float operator()(const float a) const { return powf(fabsf(a), exponent);\
} };\
struct ReduceOp##KERNEL_NAME { __device__ __forceinline__ float operator()(const float a, const float b) const { return a + b; } };\
extern ""C"" {\
    __global__ void onePass_##KERNEL_NAME(TensorInfo<INDEX_TYPE> in,\
                       INDEX_TYPE totalElements,\
                       float init,\
                       float* out, float exponent) {\
        reduceAll<ModifyOp##KERNEL_NAME, ReduceOp##KERNEL_NAME, INDEX_TYPE, DIMSA>(in, totalElements, init, ModifyOp##KERNEL_NAME(exponent), ReduceOp##KERNEL_NAME(), out);\
    }\
    __global__ void twoPassA_##KERNEL_NAME(TensorInfo<INDEX_TYPE> in,\
                       INDEX_TYPE totalElements,\
                       float init,\
                       float* scratchSpace, float exponent) {\
        reduceAllPass1<ModifyOp##KERNEL_NAME, ReduceOp##KERNEL_NAME, INDEX_TYPE, DIMSA>(in, totalElements, init, ModifyOp##KERNEL_NAME(exponent), ReduceOp##KERNEL_NAME(), scratchSpace);\
    }\
    __global__ void twoPassB_##KERNEL_NAME(int numPass1Blocks,\
                float init,\
                float* scratchSpace,\
                float* out) {\
        reduceAllPass2<ReduceOp##KERNEL_NAME, INDEX_TYPE>(numPass1Blocks, init, ReduceOp##KERNEL_NAME(), scratchSpace, out);\
    }\
}

#define REDUCE_ALL_SUB_SQUARE_KERNELS(INDEX_TYPE, DIMSA, KERNEL_NAME) \
struct ModifyOp##KERNEL_NAME {\
    const float mean;\
    __device__ ModifyOp##KERNEL_NAME(float m) : mean(m) {}\
    __device__ __forceinline__ float operator()(const float a) const { return (a - mean) * (a - mean);\
} };\
struct ReduceOp##KERNEL_NAME { __device__ __forceinline__ float operator()(const float a, const float b) const { return a + b; } };\
extern ""C"" {\
    __global__ void onePass_##KERNEL_NAME(TensorInfo<INDEX_TYPE> in,\
                       INDEX_TYPE totalElements,\
                       float init,\
                       float* out, float mean) {\
        reduceAll<ModifyOp##KERNEL_NAME, ReduceOp##KERNEL_NAME, INDEX_TYPE, DIMSA>(in, totalElements, init, ModifyOp##KERNEL_NAME(mean), ReduceOp##KERNEL_NAME(), out);\
    }\
    __global__ void twoPassA_##KERNEL_NAME(TensorInfo<INDEX_TYPE> in,\
                       INDEX_TYPE totalElements,\
                       float init,\
                       float* scratchSpace, float mean) {\
        reduceAllPass1<ModifyOp##KERNEL_NAME, ReduceOp##KERNEL_NAME, INDEX_TYPE, DIMSA>(in, totalElements, init, ModifyOp##KERNEL_NAME(mean), ReduceOp##KERNEL_NAME(), scratchSpace);\
    }\
    __global__ void twoPassB_##KERNEL_NAME(int numPass1Blocks,\
                float init,\
                float* scratchSpace,\
                float* out) {\
        reduceAllPass2<ReduceOp##KERNEL_NAME, INDEX_TYPE>(numPass1Blocks, init, ReduceOp##KERNEL_NAME(), scratchSpace, out);\
    }\
}


";
    }
}
