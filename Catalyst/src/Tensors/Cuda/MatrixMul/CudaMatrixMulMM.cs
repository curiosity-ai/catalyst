using ManagedCuda.BasicTypes;
using ManagedCuda.CudaBlas;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Catalyst.Tensors.Core;
using Catalyst.Tensors.Cpu;

namespace Catalyst.Tensors.CUDA.MatrixMul
{
    public static class CudaMatrixMulMM
    {
        // Computes  c := alpha * a * b  +  beta * c
        public static void Gemm(TSCudaContext context, float alpha, Tensor a, Tensor b, float beta, Tensor c)
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
                c.Strides[1] != 0 && c.Strides[1] != 1)
            {
                // If c is contiguous in dimension 0 (column-major)
                aClone = a.CopyRef();
                bClone = b.CopyRef();
                cClone = c.CopyRef();
            }
            else if (c.Strides[1] == 1 &&
                c.Strides[0] != 0 && c.Strides[0] != 1)
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
                    aClone.Strides[1] != 0 && aClone.Strides[1] != 1)
                {
                    // If a is contiguous in dimension 0 (column-major)
                    aOp = BlasOp.NonTranspose;
                }
                else if (aClone.Strides[1] == 1 &&
                    aClone.Strides[0] != 0 && aClone.Strides[0] != 1)
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

                    aOp = BlasOp.NonTranspose;
                }

                if (bClone.Strides[0] == 1 &&
                    bClone.Strides[1] != 0 && bClone.Strides[1] != 1)
                {
                    // If a is contiguous in dimension 0 (column-major)
                    bOp = BlasOp.NonTranspose;
                }
                else if (bClone.Strides[1] == 1 &&
                    bClone.Strides[0] != 0 && bClone.Strides[0] != 1)
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

                    bOp = BlasOp.NonTranspose;
                }

                GemmOp(context, aOp, bOp, alpha, aClone, bClone, beta, cClone);

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



        // Computes  c := alpha * a * b  +  beta * c
        public static void GemmBatch(TSCudaContext context, float alpha, Tensor a, Tensor b, float beta, Tensor c)
        {
            if (a.Sizes[1] != c.Sizes[1] || b.Sizes[2] != c.Sizes[2] || a.Sizes[2] != b.Sizes[1])
                throw new InvalidOperationException("Size mismatch");

            BlasOp aOp = default(BlasOp);
            BlasOp bOp = default(BlasOp);
            bool copyC = false;

            Tensor aClone = null;
            Tensor bClone = null;
            Tensor cClone = null;


            if (c.Strides[1] == 1 &&
                c.Strides[2] != 0 && c.Strides[2] != 1)
            {
                // If c is contiguous in dimension 0 (column-major)
                aClone = a.CopyRef();
                bClone = b.CopyRef();
                cClone = c.CopyRef();
            }
            else if (c.Strides[2] == 1 &&
                c.Strides[1] != 0 && c.Strides[1] != 1)
            {
                // If c is contiguous in dimension 1 (row-major)
                // using (a * b)' == b' * a'
                // we can pass row-major matrices to BLAS functions that expect column-major by swapping A and B,
                // and transposing all 3 matrices

                cClone = c.Transpose(1, 2);
                aClone = b.Transpose(1, 2); // Note swap of a and b
                bClone = a.Transpose(1, 2);
            }
            else
            {
                var cNew = new Tensor(c.Allocator, c.ElementType, c.Sizes[0], c.Sizes[2], c.Sizes[1]);
                cClone = cNew.Transpose(1, 2);
                Ops.Copy(cClone, c);
                cNew.Dispose();
                copyC = true;

                aClone = a.CopyRef();
                bClone = b.CopyRef();
            }

            try
            {
                if (aClone.Strides[1] == 1 &&
                    aClone.Strides[2] != 0 && aClone.Strides[2] != 1)
                {
                    // If a is contiguous in dimension 0 (column-major)
                    aOp = BlasOp.NonTranspose;
                }
                else if (aClone.Strides[2] == 1 &&
                    aClone.Strides[1] != 0 && aClone.Strides[1] != 1)
                {
                    aOp = BlasOp.Transpose;
                    var aNew = aClone.Transpose(1, 2);
                    aClone.Dispose();
                    aClone = aNew;
                }
                else
                {
                    var aNew = new Tensor(aClone.Allocator, aClone.ElementType, aClone.Sizes[0], aClone.Sizes[2], aClone.Sizes[1]);
                    var aClone2 = aNew.Transpose(1, 2);
                    Ops.Copy(aClone2, aClone);
                    aClone.Dispose();
                    aClone = aClone2;
                    aNew.Dispose();

                    aOp = BlasOp.NonTranspose;
                }

                if (bClone.Strides[1] == 1 &&
                    bClone.Strides[2] != 0 && bClone.Strides[2] != 1)
                {
                    // If a is contiguous in dimension 0 (column-major)
                    bOp = BlasOp.NonTranspose;
                }
                else if (bClone.Strides[2] == 1 &&
                    bClone.Strides[1] != 0 && bClone.Strides[1] != 1)
                {
                    bOp = BlasOp.Transpose;
                    var bNew = bClone.Transpose(1, 2);
                    bClone.Dispose();
                    bClone = bNew;
                }
                else
                {
                    var bNew = new Tensor(bClone.Allocator, bClone.ElementType, bClone.Sizes[0], bClone.Sizes[2], bClone.Sizes[1]);
                    var bClone2 = bNew.Transpose(1, 2);
                    Ops.Copy(bClone2, bClone);
                    bClone.Dispose();
                    bClone = bClone2;
                    bNew.Dispose();

                    bOp = BlasOp.NonTranspose;
                }

                GemmOpBatch(context, aOp, bOp, alpha, aClone, bClone, beta, cClone);

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



        private static Operation GetCudaBlasOp(BlasOp op)
        {
            switch(op)
            {
                case BlasOp.NonTranspose: return Operation.NonTranspose;
                case BlasOp.Transpose: return Operation.Transpose;
                case BlasOp.ConjugateTranspose: return Operation.ConjugateTranspose;
                default:
                    throw new InvalidOperationException("BlasOp not supported: " + op);
            }
        }

        private static void GemmOp(TSCudaContext context, BlasOp transA, BlasOp transB, float alpha, Tensor a, Tensor b, float beta, Tensor c)
        {
            if (a.Strides[0] != 1) throw new ArgumentException($"a must be contiguous in the first dimension (column major / fortran order). ({a.Strides[0]},{a.Strides[1]}) ({b.Strides[0]},{b.Strides[1]}) ({c.Strides[0]},{c.Strides[1]})");
            if (b.Strides[0] != 1) throw new ArgumentException("b must be contiguous in the first dimension (column major / fortran order)");
            if (c.Strides[0] != 1) throw new ArgumentException("c must be contiguous in the first dimension (column major / fortran order)");

            using (var blas = context.BlasForTensor(c))
            {
                bool nta = transA == BlasOp.NonTranspose;
                bool ntb = transB == BlasOp.NonTranspose;
                Operation transa = GetCudaBlasOp(transA);
                Operation transb = GetCudaBlasOp(transB);
                int m = (int)a.Sizes[nta ? 0 : 1];
                int k = (int)b.Sizes[ntb ? 0 : 1];
                int n = (int)b.Sizes[ntb ? 1 : 0];
                int lda = (int)a.Strides[1];
                int ldb = (int)b.Strides[1];
                int ldc = (int)c.Strides[1];



                if (c.ElementType == DType.Float32)
                {
                    var aPtrSingle = CudaHelpers.GetBufferStart(a);
                    var bPtrSingle = CudaHelpers.GetBufferStart(b);
                    var cPtrSingle = CudaHelpers.GetBufferStart(c);

                    var _statusF32 = CudaBlasNativeMethods.cublasSgemm_v2(blas.Value.CublasHandle,
                        transa, transb, m, n, k, ref alpha, aPtrSingle, lda, bPtrSingle, ldb, ref beta, cPtrSingle, ldc);
                    if (_statusF32 != CublasStatus.Success) throw new CudaBlasException(_statusF32);
                }
                else if (c.ElementType == DType.Float64)
                {
                    var aPtrDouble = CudaHelpers.GetBufferStart(a);
                    var bPtrDouble = CudaHelpers.GetBufferStart(b);
                    var cPtrDouble = CudaHelpers.GetBufferStart(c);
                    var alphaDouble = (double)alpha;
                    var betaDouble = (double)beta;
                    var _statusF64 = CudaBlasNativeMethods.cublasDgemm_v2(blas.Value.CublasHandle,
                        transa, transb, m, n, k, ref alphaDouble, aPtrDouble, lda, bPtrDouble, ldb, ref betaDouble, cPtrDouble, ldc);
                    if (_statusF64 != CublasStatus.Success) throw new CudaBlasException(_statusF64);
                }
                else
                {
                    throw new NotSupportedException("CUDA GEMM with element type " + c.ElementType + " not supported");
                }
            }
        }






        private static void GemmOpBatch(TSCudaContext context, BlasOp transA, BlasOp transB, float alpha, Tensor a, Tensor b, float beta, Tensor c)
        {
            if (a.Strides[1] != 1) throw new ArgumentException($"a must be contiguous in the first dimension (column major / fortran order). ({a.Strides[0]},{a.Strides[1]}) ({b.Strides[0]},{b.Strides[1]}) ({c.Strides[0]},{c.Strides[1]})");
            if (b.Strides[1] != 1) throw new ArgumentException("b must be contiguous in the first dimension (column major / fortran order)");
            if (c.Strides[1] != 1) throw new ArgumentException($"c must be contiguous in the first dimension (column major / fortran order) ({a.Strides[0]}, {a.Strides[1]}, {a.Strides[2]}) ({b.Strides[0]}, {b.Strides[1]}, {b.Strides[2]}) ({c.Strides[0]}, {c.Strides[1]}, {c.Strides[2]})");

            using (var blas = context.BlasForTensor(c))
            {
                bool nta = transA == BlasOp.NonTranspose;
                bool ntb = transB == BlasOp.NonTranspose;
                Operation transa = GetCudaBlasOp(transA);
                Operation transb = GetCudaBlasOp(transB);
                int m = (int)a.Sizes[nta ? 1 : 2];
                int k = (int)b.Sizes[ntb ? 1 : 2];
                int n = (int)b.Sizes[ntb ? 2 : 1];
                int lda = (int)a.Strides[2];
                int ldb = (int)b.Strides[2];
                int ldc = (int)c.Strides[2];

                int stra = (int)a.Strides[0];
                int strb = (int)b.Strides[0];
                int strc = (int)c.Strides[0];
                int batchSize = (int)c.Sizes[0];

                if (c.ElementType == DType.Float32)
                {
                    var aPtrSingle = CudaHelpers.GetBufferStart(a);
                    var bPtrSingle = CudaHelpers.GetBufferStart(b);
                    var cPtrSingle = CudaHelpers.GetBufferStart(c);

                    var _statusF32 = CudaBlasNativeMethods.cublasSgemmStridedBatched(blas.Value.CublasHandle,
                        transa, transb, m, n, k, ref alpha, aPtrSingle, lda, stra, bPtrSingle, ldb, strb, ref beta, cPtrSingle, ldc, strc, batchSize);
                    if (_statusF32 != CublasStatus.Success) throw new CudaBlasException(_statusF32);
                }
                else if (c.ElementType == DType.Float64)
                {
                    var aPtrDouble = CudaHelpers.GetBufferStart(a);
                    var bPtrDouble = CudaHelpers.GetBufferStart(b);
                    var cPtrDouble = CudaHelpers.GetBufferStart(c);
                    var alphaDouble = (double)alpha;
                    var betaDouble = (double)beta;
                    var _statusF64 = CudaBlasNativeMethods.cublasDgemmStridedBatched(blas.Value.CublasHandle,
                        transa, transb, m, n, k, ref alphaDouble, aPtrDouble, lda, stra, bPtrDouble, ldb, strb, ref betaDouble, cPtrDouble, ldc, strc, batchSize);
                    if (_statusF64 != CublasStatus.Success) throw new CudaBlasException(_statusF64);
                }
                else
                {
                    throw new NotSupportedException("CUDA GEMM with element type " + c.ElementType + " not supported");
                }
            }
        }



        public static Tensor Mul_M_M(TSCudaContext context, Tensor result, Tensor lhs, Tensor rhs)
        {
            if (lhs.ElementType != rhs.ElementType || (result != null && result.ElementType != lhs.ElementType))
                throw new InvalidOperationException("All tensors must have the same element type");
            CudaHelpers.ThrowIfDifferentDevices(result, lhs, rhs);
            if (result != null && !(result.Storage is CudaStorage)) throw new ArgumentException("result must be a CUDA tensor", "result");
            if (!(lhs.Storage is CudaStorage)) throw new ArgumentException("lhs must be a CUDA tensor", "lhs");
            if (!(rhs.Storage is CudaStorage)) throw new ArgumentException("rhs must be a CUDA tensor", "rhs");

            
            var writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes[0], rhs.Sizes[1]);
            
            Gemm(context, 1, lhs, rhs, 0, writeTarget);
            
            return writeTarget;
        }
    }
}
