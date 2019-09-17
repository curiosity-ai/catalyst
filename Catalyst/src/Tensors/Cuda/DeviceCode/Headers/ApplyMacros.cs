using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.CUDA.RuntimeCompiler;

namespace Catalyst.Tensors.CUDA.DeviceCode.Headers
{
    [CudaInclude("Code", "ApplyMacros")]
    public static class ApplyMacros
    {
        public static readonly string Code = @"

#define APPLY_T(INDEX_TYPE, DIMSA, KERNEL_NAME, OP_CODE) \
struct ConcreteOp##KERNEL_NAME { __device__ __forceinline__ void operator()(float* v) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> src, __int64 totalElements)\
    {\
        pointwiseApply1<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA>(src, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME());\
    }\
}

#define APPLY_TT(INDEX_TYPE, DIMSA, DIMSB, KERNEL_NAME, OP_CODE) \
struct ConcreteOp##KERNEL_NAME { __device__ __forceinline__ void operator()(float* a, float* b) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> tensorA, TensorInfo<INDEX_TYPE> tensorB, __int64 totalElements)\
    {\
        pointwiseApply2<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB>(tensorA, tensorB, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME());\
    }\
}

#define APPLY_TTT(INDEX_TYPE, DIMSA, DIMSB, DIMSC, KERNEL_NAME, OP_CODE) \
struct ConcreteOp##KERNEL_NAME { __device__ __forceinline__ void operator()(float* a, float* b, float *c) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> tensorA, TensorInfo<INDEX_TYPE> tensorB, TensorInfo<INDEX_TYPE> tensorC, __int64 totalElements)\
    {\
        pointwiseApply3<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB, DIMSC>(tensorA, tensorB, tensorC, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME());\
    }\
}

#define APPLY_TTTT(INDEX_TYPE, DIMSA, DIMSB, DIMSC, DIMSD, KERNEL_NAME, OP_CODE) \
struct ConcreteOp##KERNEL_NAME { __device__ __forceinline__ void operator()(float* a, float* b, float *c, float *d) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> tensorA, TensorInfo<INDEX_TYPE> tensorB, TensorInfo<INDEX_TYPE> tensorC, TensorInfo<INDEX_TYPE> tensorD, __int64 totalElements)\
    {\
        pointwiseApply4<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB, DIMSC, DIMSD>(tensorA, tensorB, tensorC, tensorD, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME());\
    }\
}


#define APPLY_TTTTT(INDEX_TYPE, DIMSA, DIMSB, DIMSC, DIMSD, DIMSE, KERNEL_NAME, OP_CODE) \
struct ConcreteOp##KERNEL_NAME { __device__ __forceinline__ void operator()(float* a, float* b, float *c, float *d, float *e) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> tensorA, TensorInfo<INDEX_TYPE> tensorB, TensorInfo<INDEX_TYPE> tensorC, TensorInfo<INDEX_TYPE> tensorD, TensorInfo<INDEX_TYPE> tensorE, __int64 totalElements)\
    {\
        pointwiseApply5<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB, DIMSC, DIMSD, DIMSE>(tensorA, tensorB, tensorC, tensorD, tensorE, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME());\
    }\
}



#define APPLY_TS(INDEX_TYPE, DIMSA, KERNEL_NAME, OP_CODE)\
struct ConcreteOp##KERNEL_NAME {\
    float b;\
    __device__ ConcreteOp##KERNEL_NAME(float bVal) { this->b = bVal; }\
    __device__ __forceinline__ void operator()(float* a) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> a, float b, __int64 totalElements)\
    {\
        pointwiseApply1<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA>(a, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME(b));\
    }\
}

#define APPLY_TSS(INDEX_TYPE, DIMSA, KERNEL_NAME, OP_CODE)\
struct ConcreteOp##KERNEL_NAME {\
    float b;\
    float c;\
    __device__ ConcreteOp##KERNEL_NAME(float bVal, float cVal) { this->b = bVal; this->c = cVal; }\
    __device__ __forceinline__ void operator()(float* a) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> a, float b, float c, __int64 totalElements)\
    {\
        pointwiseApply1<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA>(a, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME(b, c));\
    }\
}

#define APPLY_TTS(INDEX_TYPE, DIMSA, DIMSB, KERNEL_NAME, OP_CODE)\
struct ConcreteOp##KERNEL_NAME {\
    float c;\
    __device__ ConcreteOp##KERNEL_NAME(float cVal) { this->c = cVal; }\
    __device__ __forceinline__ void operator()(float* a, float* b) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> a, TensorInfo<INDEX_TYPE> b, float c, __int64 totalElements)\
    {\
        pointwiseApply2<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB>(a, b, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME(c));\
    }\
}

#define APPLY_TTSS(INDEX_TYPE, DIMSA, DIMSB, KERNEL_NAME, OP_CODE)\
struct ConcreteOp##KERNEL_NAME {\
    float c;\
    float d;\
    __device__ ConcreteOp##KERNEL_NAME(float cVal, float dVal) { this->c = cVal; this->d = dVal; }\
    __device__ __forceinline__ void operator()(float* a, float* b) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> a, TensorInfo<INDEX_TYPE> b, float c, float d, __int64 totalElements)\
    {\
        pointwiseApply2<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB>(a, b, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME(c, d));\
    }\
}


#define APPLY_TTTS(INDEX_TYPE, DIMSA, DIMSB, DIMSC, KERNEL_NAME, OP_CODE)\
struct ConcreteOp##KERNEL_NAME {\
    float d;\
    __device__ ConcreteOp##KERNEL_NAME(float dVal) { this->d = dVal; }\
    __device__ __forceinline__ void operator()(float* a, float* b, float* c) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> a, TensorInfo<INDEX_TYPE> b, TensorInfo<INDEX_TYPE> c, float d, __int64 totalElements)\
    {\
        pointwiseApply3<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB, DIMSC>(a, b, c, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME(d));\
    }\
}

/*
#define APPLY_TTTSS(INDEX_TYPE, DIMSA, DIMSB, DIMSC, KERNEL_NAME, OP_CODE)\
struct ConcreteOp##KERNEL_NAME {\
    float d;\
    float e;\
    __device__ ConcreteOp##KERNEL_NAME(float dVal, float eVal) { this->d = dVal; this->e = eVal; }\
    __device__ __forceinline__ void operator()(float* a, float* b, float* c) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> a, TensorInfo<INDEX_TYPE> b, TensorInfo<INDEX_TYPE> c, float d, float e, __int64 totalElements)\
    {\
        pointwiseApply3<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB, DIMSC>(a, b, c, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME(d, e));\
    }\
}

#define APPLY_TTTTS(INDEX_TYPE, DIMSA, DIMSB, DIMSC, DIMSD, KERNEL_NAME, OP_CODE)\
struct ConcreteOp##KERNEL_NAME {\
    float e;\
    __device__ ConcreteOp##KERNEL_NAME(float eVal) { this->e = eVal; }\
    __device__ __forceinline__ void operator()(float* a, float* b, float* c, float* d) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> a, TensorInfo<INDEX_TYPE> b, TensorInfo<INDEX_TYPE> c, TensorInfo<INDEX_TYPE> d, float e, __int64 totalElements)\
    {\
        pointwiseApply4<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB, DIMSC, DIMSD>(a, b, c, d, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME(e));\
    }\
}

#define APPLY_TTTTS(INDEX_TYPE, DIMSA, DIMSB, DIMSC, DIMSD, KERNEL_NAME, OP_CODE)\
struct ConcreteOp##KERNEL_NAME {\
    float e;\
    float f;\
    __device__ ConcreteOp##KERNEL_NAME(float eVal, float fVal) { this->e = eVal; this->f = fVal; }\
    __device__ __forceinline__ void operator()(float* a, float* b, float* c, float* d) const { OP_CODE } };\
extern ""C"" {\
    __global__ void KERNEL_NAME(TensorInfo<INDEX_TYPE> a, TensorInfo<INDEX_TYPE> b, TensorInfo<INDEX_TYPE> c, TensorInfo<INDEX_TYPE> d, float e, float f, __int64 totalElements)\
    {\
        pointwiseApply4<ConcreteOp##KERNEL_NAME, INDEX_TYPE, DIMSA, DIMSB, DIMSC, DIMSD>(a, b, c, d, (INDEX_TYPE)totalElements, ConcreteOp##KERNEL_NAME(e, f));\
    }\
}
*/

";
    }
}
