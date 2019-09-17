using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace Catalyst.Tensors.Cpu
{
    // When used with 64bit openblas, this interface requires that it is compiled with 32-bit ints
    public static class OpenBlasNative
    {
        private const string dll = "libopenblas.dll";
        private const CallingConvention cc = CallingConvention.Cdecl;

        [DllImport(dll, CallingConvention = cc)]
        public static unsafe extern void sgemm_(byte* transa, byte* transb, int *m, int *n, int *k,
            float *alpha, float *a, int *lda, float *b, int *ldb, float *beta, float *c, int *ldc);

        [DllImport(dll, CallingConvention = cc)]
        public static unsafe extern void dgemm_(byte* transa, byte* transb, int* m, int* n, int* k,
            double* alpha, double* a, int* lda, double* b, int* ldb, double* beta, double* c, int* ldc);


        [DllImport(dll, CallingConvention = cc)]
        public static unsafe extern void sgemv_(byte* trans, int* m, int* n,
            float* alpha, float* a, int* lda, float* x, int* incx, float* beta, float* y, int* incy);

        [DllImport(dll, CallingConvention = cc)]
        public static unsafe extern void dgemv_(byte* trans, int* m, int* n,
            double* alpha, double* a, int* lda, double* x, int* incx, double* beta, double* y, int* incy);


        [DllImport(dll, CallingConvention = cc)]
        public static unsafe extern float sdot_(int* n, float* x, int* incx, float* y, int* incy);

        [DllImport(dll, CallingConvention = cc)]
        public static unsafe extern double ddot_(int* n, double* x, int* incx, double* y, int* incy);
    }
}
