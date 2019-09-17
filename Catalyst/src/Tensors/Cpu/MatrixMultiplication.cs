using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.Core;

namespace Catalyst.Tensors.Cpu
{
    public enum BlasOp : byte
    {
        NonTranspose = (byte)'n',
        Transpose = (byte)'t',
        ConjugateTranspose = (byte)'c',
    }


    public static class MatrixMultiplication
    {
        public static Tensor Dot(Tensor result, Tensor lhs, Tensor rhs)
        {
            if (lhs.ElementType != rhs.ElementType || (result != null && result.ElementType != lhs.ElementType))
                throw new InvalidOperationException("All tensors must have the same element type");
            if (result != null && !(result.Storage is CpuStorage)) throw new ArgumentException("result must be a CPU tensor", "result");
            if (!(lhs.Storage is CpuStorage)) throw new ArgumentException("lhs must be a CPU tensor", "lhs");
            if (!(rhs.Storage is CpuStorage)) throw new ArgumentException("rhs must be a CPU tensor", "rhs");

            if (lhs.DimensionCount != 1) throw new ArgumentException("lhs must have 1 dimension (ie. be a vector)", "lhs");
            if (rhs.DimensionCount != 1) throw new ArgumentException("rhs must have 1 dimension (ie. be a vector)", "rhs");


            var writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, 1);

            if(writeTarget.ElementType == DType.Float32) Run_Dot_float(writeTarget, lhs, rhs);
            else if (writeTarget.ElementType == DType.Float64) Run_Dot_double(writeTarget, lhs, rhs);
            else
                throw new NotSupportedException("CPU vector dot product with element type " + result.ElementType + " not supported");

            return writeTarget;
        }

        private static void Run_Dot_float(Tensor result, Tensor lhs, Tensor rhs)
        {
            unsafe
            {
                var resultPtr = (float*)CpuNativeHelpers.GetBufferStart(result);
                var lhsPtr = (float*)CpuNativeHelpers.GetBufferStart(lhs);
                var rhsPtr = (float*)CpuNativeHelpers.GetBufferStart(rhs);

                int n = (int)lhs.Sizes[0];
                int incx = (int)lhs.Strides[0];
                int incy = (int)rhs.Strides[0];
                *resultPtr = OpenBlasNative.sdot_(&n, lhsPtr, &incx, rhsPtr, &incy);
            }
        }

        private static void Run_Dot_double(Tensor result, Tensor lhs, Tensor rhs)
        {
            unsafe
            {
                var resultPtr = (double*)CpuNativeHelpers.GetBufferStart(result);
                var lhsPtr = (double*)CpuNativeHelpers.GetBufferStart(lhs);
                var rhsPtr = (double*)CpuNativeHelpers.GetBufferStart(rhs);

                int n = (int)lhs.Sizes[0];
                int incx = (int)lhs.Strides[0];
                int incy = (int)rhs.Strides[0];
                *resultPtr = OpenBlasNative.ddot_(&n, lhsPtr, &incx, rhsPtr, &incy);
            }
        }

        public static Tensor Mul_M_V(Tensor result, Tensor lhs, Tensor rhs)
        {
            if (lhs.ElementType != rhs.ElementType || (result != null && result.ElementType != lhs.ElementType))
                throw new InvalidOperationException("All tensors must have the same element type");
            if (result != null && (result.Storage is CpuStorage)) throw new ArgumentException("result must be a CPU tensor", "result");
            if (!(lhs.Storage is CpuStorage)) throw new ArgumentException("lhs must be a CPU tensor", "lhs");
            if (!(rhs.Storage is CpuStorage)) throw new ArgumentException("rhs must be a CPU tensor", "rhs");

            if (lhs.DimensionCount != 2) throw new ArgumentException("lhs must have 2 dimensions", "lhs");
            if (rhs.DimensionCount != 1) throw new ArgumentException("rhs must have 1 dimension (ie. be a vector)", "rhs");

            Tensor lhsClone;
            if (lhs.Strides[1] == 1) // If lhs is already row-major, do nothing
            {
                lhsClone = lhs.CopyRef();
            }
            else if (lhs.Strides[0] == 1) // If lhs is column-major, transpose it
            {
                lhsClone = lhs.Transpose();
            }
            else // If lhs is not contiguous in either dimension, make a temporary contiguous copy
            {
                lhsClone = Ops.NewContiguous(lhs);
            }

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, rhs, false, lhs.Sizes[0]);

            try
            {
                if (writeTarget.ElementType == DType.Float32) Run_M_V_float(writeTarget, lhsClone, rhs);
                else if (writeTarget.ElementType == DType.Float64) Run_M_V_double(writeTarget, lhsClone, rhs);
                else
                    throw new NotSupportedException("CPU Matrix-Vector multiplication with element type " + result.ElementType + " not supported");
            }
            finally
            {
                lhsClone.Dispose();
            }

            return writeTarget;
        }

        private static void Run_M_V_float(Tensor result, Tensor mat, Tensor vec)
        {
            // Require lhs to be row-major. This means we must tell BLAS to transpose it (BLAS expects column-major matrices)
            if (mat.Strides[1] != 1) throw new ArgumentException("lhs must be contiguous in the last dimension");

            unsafe
            {
                var yPtr = (float*)CpuNativeHelpers.GetBufferStart(result);
                var aPtr = (float*)CpuNativeHelpers.GetBufferStart(mat);
                var xPtr = (float*)CpuNativeHelpers.GetBufferStart(vec);

                byte trans = (byte)'t';
                int m = (int)mat.Sizes[1];
                int n = (int)mat.Sizes[0];
                int incx = (int)vec.Strides[0];
                int lda = (int)mat.Strides[0];
                int incy = (int)result.Strides[0];
                float alpha = 1;
                float beta = 0;
                OpenBlasNative.sgemv_(&trans, &m, &n, &alpha, aPtr, &lda, xPtr, &incx, &beta, yPtr, &incy);
            }
        }

        private static void Run_M_V_double(Tensor result, Tensor lhs, Tensor rhs)
        {
            // Require lhs to be row-major. This means we must tell BLAS to transpose it (BLAS expects column-major matrices)
            if (lhs.Strides[1] != 1) throw new ArgumentException("lhs must be contiguous in the last dimension");

            unsafe
            {
                var resultPtr = (double*)CpuNativeHelpers.GetBufferStart(result);
                var lhsPtr = (double*)CpuNativeHelpers.GetBufferStart(lhs);
                var rhsPtr = (double*)CpuNativeHelpers.GetBufferStart(rhs);

                byte trans = (byte)'t';
                int m = (int)rhs.Sizes[1];
                int n = (int)lhs.Sizes[0];
                int lda = (int)rhs.Strides[0];
                int ldb = (int)lhs.Strides[0];
                int ldc = (int)result.Strides[0];
                double alpha = 1;
                double beta = 0;
                OpenBlasNative.dgemv_(&trans, &m, &n, &alpha, rhsPtr, &lda, lhsPtr, &ldb, &beta, resultPtr, &ldc);
            }
        }



        public static Tensor Mul_M_M(Tensor result, Tensor lhs, Tensor rhs)
        {
            if (lhs.ElementType != rhs.ElementType || (result != null && result.ElementType != lhs.ElementType))
                throw new InvalidOperationException("All tensors must have the same element type");
            if (result != null && !(result.Storage is CpuStorage)) throw new ArgumentException("result must be a CPU tensor", "result");
            if (!(lhs.Storage is CpuStorage)) throw new ArgumentException("lhs must be a CPU tensor", "lhs");
            if (!(rhs.Storage is CpuStorage)) throw new ArgumentException("rhs must be a CPU tensor", "rhs");

            var writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes[0], rhs.Sizes[1]);
            
            Gemm(1, lhs, rhs, 0, writeTarget);
            

            return writeTarget;
        }

        // Computes  c := alpha * a * b  +  beta * c
        public static void Gemm(float alpha, Tensor a, Tensor b, float beta, Tensor c)
        {
            if (a.Sizes[0] != c.Sizes[0] || b.Sizes[1] != c.Sizes[1] || a.Sizes[1] != b.Sizes[0])
                throw new InvalidOperationException("Size mismatch");

            BlasOp aOp = default(BlasOp);
            BlasOp bOp = default(BlasOp);
            bool copyC = false;

            Tensor aClone = null;
            Tensor bClone = null;
            Tensor cClone = null;


            if (c.Strides[0] == 1 &&
                c.Strides[1] != 0)
            {
                // If c is contiguous in dimension 0 (column-major)
                aClone = a.CopyRef();
                bClone = b.CopyRef();
                cClone = c.CopyRef();
            }
            else if (c.Strides[1] == 1 &&
                c.Strides[0] != 0)
            {
                // If c is contiguous in dimension 1 (row-major)
                // using (a * b)' == b' * a'
                // we can pass row-major matrices to BLAS functions that expect column-major by swapping A and B,
                // and transposing all 3 matrices

                cClone = c.Transpose();
                aClone = b.Transpose(); // Note swap of a and b
                bClone = a.Transpose();
            }
            else
            {
                var cNew = new Tensor(c.Allocator, c.ElementType, c.Sizes[1], c.Sizes[0]);
                cClone = cNew.Transpose();
                Ops.Copy(cClone, c);
                cNew.Dispose();
                copyC = true;

                aClone = a.CopyRef();
                bClone = b.CopyRef();
            }

            try
            {
                if (aClone.Strides[0] == 1 &&
                    aClone.Strides[1] != 0)
                {
                    // If a is contiguous in dimension 0 (column-major)
                    aOp = BlasOp.NonTranspose;
                }
                else if (aClone.Strides[1] == 1 &&
                    aClone.Strides[0] != 0)
                {
                    aOp = BlasOp.Transpose;
                    var aNew = aClone.Transpose();
                    aClone.Dispose();
                    aClone = aNew;
                }
                else
                {
                    var aNew = new Tensor(aClone.Allocator, aClone.ElementType, aClone.Sizes[1], aClone.Sizes[0]);
                    var aClone2 = aNew.Transpose();
                    Ops.Copy(aClone2, aClone);
                    aClone.Dispose();
                    aClone = aClone2;
                    aNew.Dispose();
                }

                if (bClone.Strides[0] == 1 &&
                    bClone.Strides[1] != 0)
                {
                    // If a is contiguous in dimension 0 (column-major)
                    bOp = BlasOp.NonTranspose;
                }
                else if (bClone.Strides[1] == 1 &&
                    bClone.Strides[0] != 0)
                {
                    bOp = BlasOp.Transpose;
                    var bNew = bClone.Transpose();
                    bClone.Dispose();
                    bClone = bNew;
                }
                else
                {
                    var bNew = new Tensor(bClone.Allocator, bClone.ElementType, bClone.Sizes[1], bClone.Sizes[0]);
                    var bClone2 = bNew.Transpose();
                    Ops.Copy(bClone2, bClone);
                    bClone.Dispose();
                    bClone = bClone2;
                    bNew.Dispose();
                }

                GemmOp(aOp, bOp, alpha, aClone, bClone, beta, cClone);

                if (copyC)
                {
                    Ops.Copy(c, cClone);
                }
            }
            finally
            {
                aClone.Dispose();
                bClone.Dispose();
                cClone.Dispose();
            }
        }


        private static void GemmOp(BlasOp transA, BlasOp transB, float alpha, Tensor a, Tensor b, float beta, Tensor c)
        {
            if (a.Strides[0] != 1) throw new ArgumentException("a must be contiguous in the first dimension (column major / fortran order)");
            if (b.Strides[0] != 1) throw new ArgumentException("b must be contiguous in the first dimension (column major / fortran order)");
            if (c.Strides[0] != 1) throw new ArgumentException("c must be contiguous in the first dimension (column major / fortran order)");

            unsafe
            {
                // dimensons: (m x k) * (k * n) = (m x n)
                bool nta = transA == BlasOp.NonTranspose;
                bool ntb = transB == BlasOp.NonTranspose;
                byte transa = (byte)transA;
                byte transb = (byte)transB;
                int m = (int)a.Sizes[nta ? 0 : 1];
                int k = (int)b.Sizes[ntb ? 0 : 1];
                int n = (int)b.Sizes[ntb ? 1 : 0];
                int lda = (int)a.Strides[1];
                int ldb = (int)b.Strides[1];
                int ldc = (int)c.Strides[1];

                if (c.ElementType == DType.Float32)
                {
                    var aPtrSingle = (float*)CpuNativeHelpers.GetBufferStart(a);
                    var bPtrSingle = (float*)CpuNativeHelpers.GetBufferStart(b);
                    var cPtrSingle = (float*)CpuNativeHelpers.GetBufferStart(c);

                    OpenBlasNative.sgemm_(&transa, &transb, &m, &n, &k, &alpha, aPtrSingle, &lda, bPtrSingle, &ldb, &beta, cPtrSingle, &ldc);
                }
                else if (c.ElementType == DType.Float64)
                {
                    var aPtrDouble = (double*)CpuNativeHelpers.GetBufferStart(a);
                    var bPtrDouble = (double*)CpuNativeHelpers.GetBufferStart(b);
                    var cPtrDouble = (double*)CpuNativeHelpers.GetBufferStart(c);
                    var alphaDouble = (double)alpha;
                    var betaDouble = (double)beta;
                    OpenBlasNative.dgemm_(&transa, &transb, &m, &n, &k, &alphaDouble, aPtrDouble, &lda, bPtrDouble, &ldb, &betaDouble, cPtrDouble, &ldc);
                }
                else
                {
                    throw new NotSupportedException("CPU GEMM with element type " + c.ElementType + " not supported");
                }
            }
        }
    }
}
