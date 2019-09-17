using Catalyst.Tensors.Models.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Catalyst.Tensors
{
    /// <summary>
    /// The matrix data storage format.
    /// </summary>
    public enum Order
    {
        /// <summary>
        /// The matrix array uses a row-major layout.
        /// </summary>
        Row = 101,

        /// <summary>
        /// The matrix array uses a column-major layout.
        /// </summary>
        Column = 102
    }

    /// <summary>
    /// Matrix transpose type.
    /// </summary>
    public enum Transpose
    {
        /// <summary>
        /// Don't transpose the matrix.  Equivalent to trans='N'
        /// </summary>
        NoTrans = 111,

        /// <summary>
        /// Transpose the matrix.  Equivalent to trans='T'
        /// </summary>
        Trans = 112,

        /// <summary>
        /// Conjugate transpose the matrix. The only refers to complex matrices. Real matrices will just be transposed.  Equivalent to trans='C'
        /// </summary>
        ConjTrans = 113
    }

    public class ComputeGraphMKL : ComputeGraph
    {
        const string mklDllName = "mkl_rt.dll";
        [DllImport(mklDllName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void cblas_sgemm(Order order, Transpose transa, Transpose transb, int m, int n, int k, float alpha, float[] a, int lda, float[] b, int ldb, float beta, float[] c, int ldc);


        [DllImport(mklDllName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern void cblas_scopy(int n, float* x, int incX, float* y, int incY);

        [DllImport(mklDllName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern void vsAdd(int n, float* a, float* b, float* y);

        [DllImport(mklDllName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void vsMul(int n, float[] a, float[] b, float[] y);

        [DllImport(mklDllName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern void cblas_saxpy(int n, float alpha, float* x, int incX, float* y, int incY);

        [DllImport(mklDllName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void cblas_sscal(int n, float alpha, float[] x, int incX);

        [DllImport(mklDllName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        internal static unsafe extern float cblas_sdot(int n, float* x, int incX, float* y, int incY);

        [DllImport(mklDllName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern int cblas_isamax(int n, float[] x, int incX);

        [DllImport(mklDllName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void saxpy(int n, float alpha, float[] x, int incX, float[] y, int incY);

        [DllImport(mklDllName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
        internal static extern void vsDiv(int n, float[] a, float[] b, float[] y);


        public ComputeGraphMKL(IWeightFactory weightFactory, bool needBack = true)
            : base(weightFactory, needBack)
        {
        }

        public override IWeightMatrix Mul(IWeightMatrix w1, IWeightMatrix w2)
        {
            var m1 = w1 as WeightMatrix;
            var m2 = w2 as WeightMatrix;

            var n = m1.Rows;
            var d = m2.Columns;
            var res = weightMatrixFactory.CreateWeightMatrix(n, d);

            cblas_sgemm(Order.Row, Transpose.NoTrans, Transpose.NoTrans, m1.Rows, m2.Columns, m1.Columns, 1.0f, m1.Weight, m1.Columns, m2.Weight, m2.Columns, 0.0f, res.Weight, m2.Columns);

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    cblas_sgemm(Order.Row, Transpose.NoTrans, Transpose.Trans, m1.Rows, m1.Columns, res.Columns, 1.0f, res.Gradient, res.Columns, m2.Weight, res.Columns, 1.0f, m1.Gradient, m1.Columns);
                    cblas_sgemm(Order.Row, Transpose.Trans, Transpose.NoTrans, m2.Rows, m2.Columns, res.Rows, 1.0f, m1.Weight, m2.Rows, res.Gradient, m2.Columns, 1.0f, m2.Gradient, m2.Columns);
                };
                this.backprop.Add(backward);
            }
            return res;
        }


        public override IWeightMatrix Add(IWeightMatrix w1, IWeightMatrix w2)
        {
            var m1 = w1 as WeightMatrix;
            var m2 = w2 as WeightMatrix;

            var res = weightMatrixFactory.CreateWeightMatrix(m1.Rows, m1.Columns);

            unsafe
            {
                fixed (float* m1W = m1.Weight, m2W = m2.Weight, resW = res.Weight)
                {
                    vsAdd(res.Weight.Length, m1W, m2W, resW);
                }
            }

            if (this.needs_backprop)
            {

                Action backward = () =>
                {
                    unsafe
                    {
                        fixed (float* resG = res.Gradient, m1G = m1.Gradient, m2G = m2.Gradient)
                        {
                            vsAdd(res.Gradient.Length, resG, m1G, m1G);
                            vsAdd(res.Gradient.Length, resG, m2G, m2G);
                        }
                    }
                };
                this.backprop.Add(backward);
            }
            return res;

        }

        public override WeightMatrix MulAdd(WeightMatrix m1, WeightMatrix m2, WeightMatrix m3)
        {
            var n = m1.Rows;
            var d = m2.Columns;
            var res = weightMatrixFactory.CreateWeightMatrix(n, d);

            unsafe
            {
                fixed (float* m3W = m3.Weight, resW = res.Weight)
                {
                    cblas_scopy(m3.Weight.Length, m3W, 1, resW, 1);
                }
            }

            cblas_sgemm(Order.Row, Transpose.NoTrans, Transpose.NoTrans, m1.Rows, m2.Columns, m1.Columns, 1.0f, m1.Weight, m1.Columns, m2.Weight, m2.Columns, 1.0f, res.Weight, m2.Columns);

            if (this.needs_backprop)
            {
                Action backward = () =>
                {
                    unsafe
                    {
                        fixed (float* m3G = m3.Gradient, resG = res.Gradient)
                        {
                            vsAdd(m3.Gradient.Length, m3G, resG, m3G);
                        }
                    }

                    cblas_sgemm(Order.Row, Transpose.NoTrans, Transpose.Trans, m1.Rows, m1.Columns, res.Columns, 1.0f, res.Gradient, res.Columns, m2.Weight, res.Columns, 1.0f, m1.Gradient, m1.Columns);
                    cblas_sgemm(Order.Row, Transpose.Trans, Transpose.NoTrans, m2.Rows, m2.Columns, res.Rows, 1.0f, m1.Weight, m2.Rows, res.Gradient, m2.Columns, 1.0f, m2.Gradient, m2.Columns);
             
                };
                this.backprop.Add(backward);
            }
            return res;
        }       
    }
}
