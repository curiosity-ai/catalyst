using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Numerics;
using System.Runtime.CompilerServices;
using Catalyst.Tensors.Models.Tools;
using System.Collections.Concurrent;

namespace Catalyst.Tensors
{
    public class ComputeGraph : IComputeGraph
    {
        internal static WeightMatrixFactory weightMatrixFactory;


        public ConcurrentStack<Action> backprop = new ConcurrentStack<Action>();

        public bool needs_backprop { get; set; }
        public ComputeGraph(IWeightFactory weightFactory, bool needBack = true)
        {
            weightMatrixFactory = weightFactory as WeightMatrixFactory;


            needs_backprop = needBack;
        }



        const float d1024 = 1.0f / 1024;

        Vector<float> vecd1024 = new Vector<float>(d1024);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected float FastExp(float x)
        {
            x = 1.0f + x * d1024;
            x *= x; x *= x; x *= x; x *= x;
            x *= x; x *= x; x *= x; x *= x;
            x *= x; x *= x;

            return x;
        }



        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected Vector<float> FastExp(Vector<float> x)
        {
            x = Vector<float>.One + x * vecd1024;
            x *= x; x *= x; x *= x; x *= x;
            x *= x; x *= x; x *= x; x *= x;
            x *= x; x *= x;

            return x;
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private float FastTanh(float x)
        {
            double ax = Math.Abs(x);
            double x2 = x * x;

            return (float)(x * (2.45550750702956 + 2.45550750702956 * ax +
               (0.893229853513558 + 0.821226666969744 * ax) * x2) /
               (2.44506634652299 + (2.44506634652299 + x2) *
               Math.Abs(x + 0.814642734961073 * x * ax)));

        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Vector<float> FastTanh(Vector<float> x)
        {
            var ax = Vector.Abs<float>(x);
            var x2 = x * x;

            return (x * (new Vector<float>(2.45550750702956f) + new Vector<float>(2.45550750702956f) * ax +
               (new Vector<float>(0.893229853513558f) + new Vector<float>(0.821226666969744f) * ax) * x2) /
               (new Vector<float>(2.44506634652299f) + (new Vector<float>(2.44506634652299f) + x2) *
               Vector.Abs<float>(x + new Vector<float>(0.814642734961073f) * x * ax)));

        }

        public virtual IWeightMatrix Tanh(IWeightMatrix w, bool updateWeightsInPlace = false)
        {
            var m = w as WeightMatrix;

            // tanh nonlinearity
            var res = weightMatrixFactory.CreateWeightMatrix(m.Rows, m.Columns);
            var n = m.Weight.Length;
            var moreItems = (n % Vector<float>.Count);
            var i = 0;

            while (i < n - moreItems)
            {
                var vecMW = new Vector<float>(m.Weight, i);
                var vecSig = FastTanh(vecMW);
                vecSig.CopyTo(res.Weight, i);

                i += Vector<float>.Count;
            }

            while (i < n)
            {
                res.Weight[i] = FastTanh(m.Weight[i]);
                i++;
            }


            if (needs_backprop)
            {
                Action backward = () =>
                {
                    i = 0;
                    while (i < n - moreItems)
                    {
                        var vecResW = new Vector<float>(res.Weight, i);
                        var vecResGrad = new Vector<float>(res.Gradient, i);
                        var vecMGrad = new Vector<float>(m.Gradient, i);

                        vecMGrad = (vecMGrad + (Vector<float>.One - vecResW * vecResW) * vecResGrad);
                        vecMGrad.CopyTo(m.Gradient, i);

                        i += Vector<float>.Count;
                    }

                    while (i < n)
                    {
                        var mwi = res.Weight[i];
                        m.Gradient[i] = (float)(m.Gradient[i] + (1.0 - mwi * mwi) * res.Gradient[i]);
                        i++;
                    }


                };
                backprop.Push(backward);
            }
            return res;
        }

        public virtual IWeightMatrix MulAdd(IWeightMatrix m1, IWeightMatrix m2, IWeightMatrix m3)
        {
            return MulAdd(m1 as WeightMatrix, m2 as WeightMatrix, m3 as WeightMatrix);
        }

        public virtual IWeightMatrix AddTanh(IWeightMatrix w1, IWeightMatrix w2)
        {
            var m1 = w1 as WeightMatrix;
            var m2 = w2 as WeightMatrix;

            // tanh nonlinearity
            var res = weightMatrixFactory.CreateWeightMatrix(m1.Rows, m1.Columns);
            var n = m1.Weight.Length;
            var moreItems = (n % Vector<float>.Count);
            var i = 0;

            while (i < n - moreItems)
            {
                var vecM1W = new Vector<float>(m1.Weight, i);
                var vecM2W = new Vector<float>(m2.Weight, i);
                var vecSig = FastTanh(vecM1W + vecM2W);
                vecSig.CopyTo(res.Weight, i);

                i += Vector<float>.Count;
            }

            while (i < n)
            {
                res.Weight[i] = FastTanh(m1.Weight[i] + m2.Weight[i]);
                i++;
            }


            if (needs_backprop)
            {
                Action backward = () =>
                {
                    i = 0;
                    while (i < n - moreItems)
                    {
                        var vecResW = new Vector<float>(res.Weight, i);
                        var vecResGrad = new Vector<float>(res.Gradient, i);
                        var vecM1Grad = new Vector<float>(m1.Gradient, i);
                        var vecM2Grad = new Vector<float>(m2.Gradient, i);

                        var vecMGrad = (Vector<float>.One - vecResW * vecResW) * vecResGrad;

                        vecM1Grad += vecMGrad;
                        vecM1Grad.CopyTo(m1.Gradient, i);

                        vecM2Grad += vecMGrad;
                        vecM2Grad.CopyTo(m2.Gradient, i);

                        i += Vector<float>.Count;
                    }

                    while (i < n)
                    {
                        var mwi = res.Weight[i];
                        var grad = (1.0 - mwi * mwi) * res.Gradient[i];
                        m1.Gradient[i] += (float)grad;
                        m2.Gradient[i] += (float)grad;

                        i++;
                    }


                };
                backprop.Push(backward);
            }
            return res;
        }


        public virtual IWeightMatrix ConcatColumns(IWeightMatrix w1, IWeightMatrix w2)
        {
            var m1 = w1 as WeightMatrix;
            var m2 = w2 as WeightMatrix;

            int sx = 1;
            int sy = 1;

            sy = m1.Columns + m2.Columns;
            sx = m1.Rows;

            var res = weightMatrixFactory.CreateWeightMatrix(sx, sy);
            var n = m1.Weight.Length;


            for (var i = 0; i < m1.Rows; i++)
            {
                Array.Copy(m1.Weight, i * m1.Columns, res.Weight, i * res.Columns, m1.Columns);
            }

            for (var i = 0; i < m2.Rows; i++)
            {
                Array.Copy(m2.Weight, i * m2.Columns, res.Weight, i * res.Columns + m1.Columns, m2.Columns);
            }

            if (needs_backprop)
            {
                Action backward = () =>
                {

                    for (var i = 0; i < m1.Rows; i++)
                    {
                        var k = 0;
                        var moreItem = (m1.Columns % Vector<float>.Count);
                        var offsetM1 = i * m1.Columns;
                        var offsetRes = i * res.Columns;

                        while (k < m1.Columns - moreItem)
                        {
                            var vecResG = new Vector<float>(res.Gradient, offsetRes + k);
                            var vecM1G = new Vector<float>(m1.Gradient, offsetM1 + k);
                            vecM1G += vecResG;
                            vecM1G.CopyTo(m1.Gradient, offsetM1 + k);

                            k += Vector<float>.Count;
                        }

                        while (k < m1.Columns)
                        {
                            m1.Gradient[offsetM1 + k] += res.Gradient[offsetRes + k];
                            k++;
                        }
                    }

                    for (var i = 0; i < m2.Rows; i++)
                    {

                        var k = 0;
                        var moreItem = (m2.Columns % Vector<float>.Count);
                        var offsetM2 = i * m2.Columns;
                        var offsetRes = i * res.Columns + m1.Columns;

                        while (k < m2.Columns - moreItem)
                        {
                            var vecResG = new Vector<float>(res.Gradient, offsetRes + k);
                            var vecM2G = new Vector<float>(m2.Gradient, offsetM2 + k);
                            vecM2G += vecResG;
                            vecM2G.CopyTo(m2.Gradient, offsetM2 + k);

                            k += Vector<float>.Count;
                        }

                        while (k < m2.Columns)
                        {
                            m2.Gradient[offsetM2 + k] += res.Gradient[offsetRes + k];
                            k++;
                        }
                    }
                };
                backprop.Push(backward);
            }
            return res;
        }

        public virtual IWeightMatrix PeekRow(IWeightMatrix w, int ix, int num = 1)
        {
            WeightMatrix m = w as WeightMatrix;

            var d = m.Columns;
            var res = weightMatrixFactory.CreateWeightMatrix(num, d);

            Array.Copy(m.Weight, d * ix, res.Weight, 0, d * num);

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    var offset = d * ix;
                    var n = d * num;
                    var moreItems = (n % Vector<float>.Count);
                    var i = 0;
                    while (i < n - moreItems)
                    {
                        var vecMG = new Vector<float>(m.Gradient, offset + i);
                        var vecResG = new Vector<float>(res.Gradient, i);

                        vecMG += vecResG;
                        vecMG.CopyTo(m.Gradient, offset + i);

                        i += Vector<float>.Count;
                    }

                    while (i < n)
                    {
                        m.Gradient[offset + i] += res.Gradient[i];
                        i++;
                    }

                    for (int j = 0; j < num; j++)
                    {
                        if (m.RowToBeUpdated.ContainsKey(ix + j) == false)
                        {
                            m.RowToBeUpdated.Add(ix + j, 1);
                        }
                        else
                        {
                            m.RowToBeUpdated[ix + j]++;
                        }
                    }
                };
                backprop.Push(backward);
            }
            return res;
        }

        private float sig(float x)
        {
            // helper function for computing sigmoid
            return (float)(1.0 / (1 + FastExp(-x)));
        }

        private Vector<float> sig(Vector<float> x)
        {
            return (Vector<float>.One / (Vector<float>.One + FastExp(-x)));
        }


        Random ra = new Random();
        public IWeightMatrix Dropout(IWeightMatrix w, float drop_prob)
        {
            var V = w as WeightMatrix;

            var res = weightMatrixFactory.CreateWeightMatrix(V.Rows, V.Columns);
            var N = V.Weight.Length;
            bool[] dropped = new bool[V.Rows * V.Columns];
            var V2 = V.Clone();

            for (var i = 0; i < N; i++)
            {
                if (ra.NextDouble() < drop_prob) { V2.Weight[i] = 0; dropped[i] = true; } // drop!
                else { dropped[i] = false; }
            }

            res = V2;


            if (needs_backprop)
            {
                Action backward = () =>
                {
                    var chain_grad = res;
                    V.Gradient = new float[N]; // zero out gradient wrt data
                    for (var i = 0; i < N; i++)
                    {
                        if (!(dropped[i]))
                        {
                            V.Gradient[i] += chain_grad.Gradient[i]; // copy over the gradient
                        }
                    }

                };
                backprop.Push(backward);
            }
            return res;
        }

        //public virtual void DropoutPredict(IWeightMatrix V, float drop_prob)
        //{
        //    WeightMatrix m = V as WeightMatrix;

        //    for (int i = 0; i < m.Weight.Length; i++)
        //    {
        //        m.Weight[i] *= 0.2f;
        //    }

        //}

        public virtual IWeightMatrix Mul(IWeightMatrix w1, IWeightMatrix w2)
        {
            var m1 = w1 as WeightMatrix;
            var m2 = w2 as WeightMatrix;

            var n = m1.Rows;
            var d = m2.Columns;
            var res = weightMatrixFactory.CreateWeightMatrix(n, d);
            var moreItemsD = (d % Vector<float>.Count);

            //            Parallel.For(0, m1.Rows, i =>
            for (int i = 0; i < m1.Rows; i++)
            {
                // loop over rows of m1

                var m1BaseIndex = d * i;
                var m1ColBaseIndex = m1.Columns * i;

                for (var k = 0; k < m1.Columns; k++)
                { // dot product loop

                    var j = 0;
                    var m1w = m1.Weight[m1ColBaseIndex + k];
                    var m2BaseIndex = m2.Columns * k;

                    while (j < d - moreItemsD)
                    {
                        int offset = m1BaseIndex + j;

                        var vecM2W = new Vector<float>(m2.Weight, m2BaseIndex + j);
                        var vecResWeight = new Vector<float>(res.Weight, offset);

                        vecResWeight += m1w * vecM2W;

                        vecResWeight.CopyTo(res.Weight, offset);

                        j += Vector<float>.Count;
                    }

                    while (j < d)
                    {
                        var v = m1w * m2.Weight[m2BaseIndex + j];
                        res.Weight[m1BaseIndex + j] += v;

                        j++;
                    }

                }

            }//);

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    //                    Parallel.For(0, m1.Rows, i =>
                    for (int i = 0; i < m1.Rows; i++)
                    {

                        // loop over rows of m1

                        var resBaseIndex = d * i;
                        var m1BaseIndex = m1.Columns * i;

                        // loop over cols of m2
                        for (var k = 0; k < m1.Columns; k++)
                        {
                            var m1GIndex = m1BaseIndex + k;
                            var m2GBaseIndex = m2.Columns * k;
                            var m1G = 0.0f;
                            var m1W = m1.Weight[m1GIndex];

                            var j = 0;
                            while (j < d - moreItemsD)
                            {
                                int m2Index = m2GBaseIndex + j;
                                int offset = resBaseIndex + j;
                                var vecResG = new Vector<float>(res.Gradient, offset);
                                var vecM2W = new Vector<float>(m2.Weight, m2Index);
                                var vecM2G = new Vector<float>(m2.Gradient, m2Index);

                                m1G += Vector.Dot(vecM2W, vecResG);

                                vecM2G += m1W * vecResG;
                                vecM2G.CopyTo(m2.Gradient, m2Index);


                                j += Vector<float>.Count;
                            }

                            while (j < d)
                            {
                                int m2Index = m2GBaseIndex + j;
                                var b = res.Gradient[resBaseIndex + j];

                                m1G += m2.Weight[m2Index] * b;
                                m2.Gradient[m2Index] += m1W * b;
                                j++;
                            }

                            m1.Gradient[m1GIndex] += m1G;
                        }
                    }//);
                };
                backprop.Push(backward);
            }
            return res;
        }

        public virtual WeightMatrix MulAdd(WeightMatrix m1, WeightMatrix m2, WeightMatrix m3)
        {
            var n = m1.Rows;
            var d = m2.Columns;
            var res = weightMatrixFactory.CreateWeightMatrix(n, d);
            var moreItemsD = (d % Vector<float>.Count);

            //            Parallel.For(0, m1.Rows, i =>
            for (int i = 0; i < m1.Rows; i++)
            {
                // loop over rows of m1

                var m1BaseIndex = d * i;
                var m1ColBaseIndex = m1.Columns * i;

                Array.Copy(m3.Weight, m1BaseIndex, res.Weight, m1BaseIndex, d);

                for (var k = 0; k < m1.Columns; k++)
                { // dot product loop

                    var j = 0;
                    var m1w = m1.Weight[m1ColBaseIndex + k];
                    var m2BaseIndex = m2.Columns * k;

                    while (j < d - moreItemsD)
                    {
                        int offset = m1BaseIndex + j;

                        var vecM2W = new Vector<float>(m2.Weight, m2BaseIndex + j);
                        var vecResWeight = new Vector<float>(res.Weight, offset);
                        vecResWeight += m1w * vecM2W;

                        vecResWeight.CopyTo(res.Weight, offset);

                        j += Vector<float>.Count;
                    }

                    while (j < d)
                    {
                        res.Weight[m1BaseIndex + j] += m1w * m2.Weight[m2BaseIndex + j];
                        j++;
                    }

                }

            }//);

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    //                    Parallel.For(0, m1.Rows, i =>
                    for (int i = 0; i < m1.Rows; i++)
                    {

                        // loop over rows of m1

                        var resBaseIndex = d * i;
                        var m1BaseIndex = m1.Columns * i;

                        var j = 0;
                        while (j < d - moreItemsD)
                        {
                            int offset = resBaseIndex + j;
                            var vecResG = new Vector<float>(res.Gradient, offset);
                            var vecM3G = new Vector<float>(m3.Gradient, offset);
                            vecM3G += vecResG;
                            vecM3G.CopyTo(m3.Gradient, offset);

                            j += Vector<float>.Count;
                        }

                        while (j < d)
                        {
                            int offset = resBaseIndex + j;
                            m3.Gradient[offset] += res.Gradient[offset];

                            j++;
                        }

                        // loop over cols of m2
                        for (var k = 0; k < m1.Columns; k++)
                        {
                            var m1GIndex = m1BaseIndex + k;
                            var m2GBaseIndex = m2.Columns * k;
                            var m1G = 0.0f;
                            var m1W = m1.Weight[m1GIndex];

                            j = 0;
                            while (j < d - moreItemsD)
                            {
                                int m2Index = m2GBaseIndex + j;
                                int offset = resBaseIndex + j;
                                var vecResG = new Vector<float>(res.Gradient, offset);
                                var vecM2W = new Vector<float>(m2.Weight, m2Index);
                                var vecM2G = new Vector<float>(m2.Gradient, m2Index);

                                m1G += Vector.Dot(vecM2W, vecResG);

                                vecM2G += m1W * vecResG;
                                vecM2G.CopyTo(m2.Gradient, m2Index);


                                j += Vector<float>.Count;
                            }

                            while (j < d)
                            {
                                int m2Index = m2GBaseIndex + j;
                                var b = res.Gradient[resBaseIndex + j];

                                m1G += m2.Weight[m2Index] * b;
                                m2.Gradient[m2Index] += m1W * b;
                                j++;
                            }

                            m1.Gradient[m1GIndex] += m1G;

                        }

                    }//);
                };
                backprop.Push(backward);
            }
            return res;
        }


        public virtual IWeightMatrix Add(IWeightMatrix w1, IWeightMatrix w2)
        {
            var m1 = w1 as WeightMatrix;
            var m2 = w2 as WeightMatrix;

            var res = weightMatrixFactory.CreateWeightMatrix(m1.Rows, m1.Columns);

            var n = m1.Weight.Length;
            var moreItems = (n % Vector<float>.Count);
            var i = 0;
            while (i < n - moreItems)
            {
                var vecM1W = new Vector<float>(m1.Weight, i);
                var vecM2W = new Vector<float>(m2.Weight, i);
                var vecRW = new Vector<float>(res.Weight, i);

                vecRW = vecM1W + vecM2W;
                vecRW.CopyTo(res.Weight, i);

                i += Vector<float>.Count;
            }

            while (i < n)
            {
                res.Weight[i] = m1.Weight[i] + m2.Weight[i];
                i++;
            }


            if (needs_backprop)
            {

                Action backward = () =>
                {
                    n = m1.Weight.Length;
                    moreItems = (n % Vector<float>.Count);
                    i = 0;

                    while (i < n - moreItems)
                    {
                        var vecRG = new Vector<float>(res.Gradient, i);

                        if (vecRG != Vector<float>.Zero)
                        {
                            var vecM1G = new Vector<float>(m1.Gradient, i);
                            var vecM2G = new Vector<float>(m2.Gradient, i);

                            vecM1G += vecRG;
                            vecM2G += vecRG;

                            vecM1G.CopyTo(m1.Gradient, i);
                            vecM2G.CopyTo(m2.Gradient, i);
                        }

                        i += Vector<float>.Count;
                    }

                    while (i < n)
                    {
                        if (res.Gradient[i] != 0)
                        {
                            m1.Gradient[i] += res.Gradient[i];
                            m2.Gradient[i] += res.Gradient[i];
                        }

                        i++;
                    }

                };
                backprop.Push(backward);
            }
            return res;

        }

        
       


        public virtual IWeightMatrix Sigmoid(IWeightMatrix w, bool updateWeightsInPlace = false)
        {
            var m = w as WeightMatrix;

            var res = weightMatrixFactory.CreateWeightMatrix(m.Rows, m.Columns);

            var n = m.Weight.Length;
            var moreItems = (n % Vector<float>.Count);
            var i = 0;
            while (i < n - moreItems)
            {
                var vecMW = new Vector<float>(m.Weight, i);
                var vecRW = new Vector<float>(res.Weight, i);

                vecRW = sig(vecMW);
                vecRW.CopyTo(res.Weight, i);

                i += Vector<float>.Count;
            }

            while (i < n)
            {
                res.Weight[i] = sig(m.Weight[i]);
                i++;
            }


            if (needs_backprop)
            {

                Action backward = () =>
                {
                    i = 0;
                    while (i < n - moreItems)
                    {
                        var vecRW = new Vector<float>(res.Weight, i);
                        var vecRG = new Vector<float>(res.Gradient, i);
                        var vecMG = vecRW * (Vector<float>.One - vecRW) * vecRG;

                        var vecM1G = new Vector<float>(m.Gradient, i);

                        vecM1G += vecMG;

                        vecM1G.CopyTo(m.Gradient, i);


                        i += Vector<float>.Count;
                    }

                    while (i < n)
                    {
                        var mwi = res.Weight[i];
                        var g = mwi * (1.0 - mwi) * res.Gradient[i];

                        m.Gradient[i] += (float)g;

                        i++;
                    }

                };
                backprop.Push(backward);
            }
            return res;
        }

       
        public virtual IWeightMatrix EltMul(IWeightMatrix w1, IWeightMatrix w2)
        {
            var m1 = w1 as WeightMatrix;
            var m2 = w2 as WeightMatrix;

            var res = weightMatrixFactory.CreateWeightMatrix(m1.Rows, m1.Columns);
            var n = m1.Weight.Length;
            var i = 0;
            var moreItems = (n % Vector<float>.Count);
            while (i < n - moreItems)
            {
                var vecResW = new Vector<float>(res.Weight, i);
                var vecM1W = new Vector<float>(m1.Weight, i);
                var vecM2W = new Vector<float>(m2.Weight, i);

                vecResW = vecM1W * vecM2W;
                vecResW.CopyTo(res.Weight, i);

                i += Vector<float>.Count;
            }

            while (i < n)
            {
                res.Weight[i] = m1.Weight[i] * m2.Weight[i];
                i++;
            }


            if (needs_backprop)
            {

                Action backward = () =>
                {
                    i = 0;
                    while (i < n - moreItems)
                    {
                        var vecResGrad = new Vector<float>(res.Gradient, i);
                        var vecM1W = new Vector<float>(m1.Weight, i);
                        var vecM2W = new Vector<float>(m2.Weight, i);

                        var vecM1Grad = new Vector<float>(m1.Gradient, i);
                        var vecM2Grad = new Vector<float>(m2.Gradient, i);

                        vecM1Grad += vecM2W * vecResGrad;
                        vecM2Grad += vecM1W * vecResGrad;

                        vecM1Grad.CopyTo(m1.Gradient, i);
                        vecM2Grad.CopyTo(m2.Gradient, i);

                        i += Vector<float>.Count;
                    }

                    while (i < n)
                    {
                        m1.Gradient[i] += m2.Weight[i] * res.Gradient[i];
                        m2.Gradient[i] += m1.Weight[i] * res.Gradient[i];
                        i++;
                    }

                };
                backprop.Push(backward);
            }
            return res;
        }
           
        public void Backward()
        {
            while(backprop.TryPop(out var action))
            {
                action();
            }
        }

        public void RunTopBackward()
        {
            if(backprop.TryPop(out var action))
            {
                action();
            }
        }

        public IWeightMatrix CreatePositionMatrix(int dimWords, int dimEmb)
        {
            double numTimescales = (float)dimEmb / 2;
            double logTimescaleIncrement = Math.Log(10000.0f) / (numTimescales - 1.0f);

            var res = weightMatrixFactory.CreateWeightMatrix(dimWords, dimEmb);

            for (int p = 0; p < dimWords; ++p)
            {
                for (int i = 0; i < numTimescales; ++i)
                {
                    float v = (float)(p * Math.Exp(i * -logTimescaleIncrement));

                    res.Weight[p * dimEmb + i] = (float)Math.Sin(v);
                    res.Weight[p * dimEmb + (int)numTimescales + i] = (float)Math.Cos(v);
                }
            }


            return res;
        }

        public IWeightMatrix LayerNorm(IWeightMatrix src, IWeightMatrix alpha, IWeightMatrix beta, float eps = 1e-09f)
        {
            WeightMatrix srcM = src as WeightMatrix;
            WeightMatrix alphaM = alpha as WeightMatrix;
            WeightMatrix betaM = beta as WeightMatrix;

            int rows = srcM.Rows;
            int cols = srcM.Columns;

            var res = weightMatrixFactory.CreateWeightMatrix(rows, cols);

            for (int j = 0; j < rows; j++)
            {
                int baseIdx = j * cols;

                var sum = 0.0f;
                for (int i = 0; i < cols; i++)
                {
                    sum += srcM.Weight[baseIdx + i];
                }

                float mean = sum / cols;
                float sqSum = 0.0f;

                for (int i = 0; i < cols; i++)
                {
                    float ex = srcM.Weight[baseIdx + i] - mean;
                    sqSum += ex * ex;
                }

                float sigma = (float)Math.Sqrt(eps + sqSum / cols);

                for (int i = 0; i < cols; i++)
                {
                    float t = alphaM.Weight[baseIdx + i] * ((srcM.Weight[baseIdx + i] - mean) / sigma);
                    t += betaM.Weight[baseIdx + i];

                    res.Weight[baseIdx + i] = t;
                }
            }


            if (needs_backprop)
            {
                Action backward = () =>
                {
                    for (int j = 0; j < rows; j++)
                    {
                        float sum_x = 0.0f;
                        float sum_adj = 0.0f;
                        float sum_adj_x = 0.0f;
                        float sum_sqr = 0.0f;
                        int baseIdx = j * cols;

                        for (int i = 0; i < cols; i++)
                        {
                            sum_x += srcM.Weight[baseIdx + i];
                            sum_adj_x += res.Gradient[baseIdx + i] * (res.Weight[baseIdx + i] - betaM.Weight[baseIdx + i]);
                            sum_adj += res.Gradient[baseIdx + i];
                        }

                        float mean = sum_x / cols;

                        for (int i = 0; i < cols; i++)
                        {
                            float ex = srcM.Weight[baseIdx + i] - mean;
                            sum_sqr += ex * ex;
                        }

                        float sigma = (float)Math.Sqrt(eps + sum_sqr / cols);

                        for (int i = 0; i < cols; i++)
                        {
                            float grad_x = 0.0f;

                            float x_hat = (res.Weight[baseIdx + i] - betaM.Weight[baseIdx + i]) / alphaM.Weight[baseIdx + i];

                            grad_x += cols * res.Gradient[baseIdx + i];
                            grad_x -= sum_adj;
                            grad_x -= sum_adj_x * x_hat;
                            grad_x /= cols * sigma;

                            srcM.Gradient[baseIdx + i] += alphaM.Weight[baseIdx + i] * grad_x;
                            alphaM.Gradient[baseIdx + i] += res.Gradient[baseIdx + i] * x_hat;
                            betaM.Gradient[baseIdx + i] += res.Gradient[baseIdx + i];
                        }
                    }
                };
                backprop.Push(backward);
            }


            return res;
        }


        public virtual IWeightMatrix Softmax(IWeightMatrix src, bool bp = true)
        {
            WeightMatrix m = src as WeightMatrix;

            var res = weightMatrixFactory.CreateWeightMatrix(m.Rows, m.Columns); // probability volume

            var maxval = -999999.0f;
            var n = m.Weight.Length;
            var moreItem = (n % Vector<float>.Count);

            var k = 0;
            var vecMaxVal = new Vector<float>(maxval);
            while (k < n - moreItem)
            {
                var vecMW = new Vector<float>(m.Weight, k);
                vecMaxVal = Vector.Max(vecMW, vecMaxVal);

                k += Vector<float>.Count;
            }

            for (int i = 0; i < Vector<float>.Count; i++)
            {
                if (vecMaxVal[i] > maxval)
                {
                    maxval = vecMaxVal[i];
                }
            }


            while (k < n)
            {
                if (m.Weight[k] > maxval) maxval = m.Weight[k];

                k++;
            }


            double s = 0.0;
            k = 0;
            vecMaxVal = new Vector<float>(maxval);
            while (k < n - moreItem)
            {
                var vecMW = new Vector<float>(m.Weight, k);
                var vecV = FastExp(vecMW - vecMaxVal);
                vecV.CopyTo(res.Weight, k);

                s += Vector.Dot(vecV, Vector<float>.One);

                k += Vector<float>.Count;
            }

            k = n - moreItem;
            while (k < n)
            {
                float v = FastExp(m.Weight[k] - maxval);
                res.Weight[k] = (float)v;
                s += v;

                k++;
            }

            k = 0;
            var vecS = new Vector<float>((float)s);
            while (k < n - moreItem)
            {
                var vecResW = new Vector<float>(res.Weight, k);
                vecResW = vecResW / vecS;
                vecResW.CopyTo(res.Weight, k);

                k += Vector<float>.Count;
            }

            while (k < n)
            {
                float v = (float)(res.Weight[k] / s);
                res.Weight[k] = v;
                k++;
            }

            if (needs_backprop && bp)
            {
                Action backward = () =>
                {
                    double ss = 0.0;
                    for (int i = 0; i < n; i++)
                    {
                        var v = res.Gradient[i] * res.Weight[i];
                        m.Gradient[i] += v;

                        ss += v;
                    }
                    for (int i = 0; i < n; i++)
                    {
                        m.Gradient[i] = (float)(m.Gradient[i] - ss * res.Weight[i]);

                    }
                };
                backprop.Push(backward);
            }

            return res;
        }




        public virtual IWeightMatrix ConcatColumns(params IWeightMatrix[] wl)
        {
            List<WeightMatrix> twl = new List<WeightMatrix>();
            int sx = 0;
            int sy = 0;

            foreach (IWeightMatrix item in wl)
            {
                WeightMatrix m = item as WeightMatrix;
                sx = m.Rows;
                sy += m.Columns;

                twl.Add(m);
            }

            var res = weightMatrixFactory.CreateWeightMatrix(sx, sy);

            for (var i = 0; i < sx; i++)
            {
                int startIdx = 0;
                for (var j = 0; j < twl.Count; j++)
                {
                    Array.Copy(twl[j].Weight, i * twl[j].Columns, res.Weight, i * res.Columns + startIdx, twl[j].Columns);
                    startIdx += twl[j].Columns;
                }
            }
                      
            if (needs_backprop)
            {
                Action backward = () =>
                {
                    for (var i = 0; i < sx; i++)
                    {
                        int startIdx = 0;
                        for (var j = 0; j < twl.Count; j++)
                        {
                            var k = 0;
                            var tw_j = twl[j];
                            var moreItem = (tw_j.Columns % Vector<float>.Count);
                            var offsetM1 = i * tw_j.Columns;
                            var offsetRes = i * res.Columns + startIdx;

                            while (k < tw_j.Columns - moreItem)
                            {
                                var vecResG = new Vector<float>(res.Gradient, offsetRes + k);
                                var vecM1G = new Vector<float>(tw_j.Gradient, offsetM1 + k);
                                vecM1G += vecResG;
                                vecM1G.CopyTo(tw_j.Gradient, offsetM1 + k);

                                k += Vector<float>.Count;
                            }

                            while (k < twl[j].Columns)
                            {
                                tw_j.Gradient[offsetM1 + k] += res.Gradient[offsetRes + k];
                                k++;
                            }

                            startIdx += tw_j.Columns;
                        }
                    }                   
                };
                backprop.Push(backward);
            }
            return res;
        }      

        public virtual List<IWeightMatrix> SplitColumns2(IWeightMatrix w, params int[] sizes)
        {
            var m = w as WeightMatrix;
            List<IWeightMatrix> resList = new List<IWeightMatrix>();

            int x = 0;
            foreach (int size in sizes)
            {
                WeightMatrix res = weightMatrixFactory.CreateWeightMatrix(m.Rows, size);

                for (int i = 0; i < m.Rows; i++)
                {
                    Array.Copy(m.Weight, i * m.Columns + x, res.Weight, i * res.Columns, size);

                }

                x += size;

                resList.Add(res);
            }


            if (needs_backprop)
            {
                Action backward = () =>
                {
                    int startIdx = 0;
                    for (int i = 0; i < resList.Count; i++)
                    {
                        WeightMatrix r = resList[i] as WeightMatrix;
                        for (int j = 0; j < r.Rows; j++)
                        {
                            Array.Copy(r.Gradient, j * r.Columns, m.Gradient, j * m.Columns + startIdx, r.Columns);
                        }

                        startIdx += r.Columns;
                    }
                };
                backprop.Push(backward);
            }


            return resList;
        }

        public (IWeightMatrix r1, IWeightMatrix r2) SplitColumns(IWeightMatrix w, int size1, int size2)
        {
            var res = SplitColumns2(w, size1, size2);

            return (res[0], res[1]);
        }

        public (IWeightMatrix r1, IWeightMatrix r2, IWeightMatrix r3) SplitColumns(IWeightMatrix w, int size1, int size2, int size3)
        {
            var res = SplitColumns2(w, size1, size2, size3);

            return (res[0], res[1], res[2]);
        }

        public virtual IWeightMatrix ConcatRows(List<IWeightMatrix> wl)
        {
            List<WeightMatrix> twl = new List<WeightMatrix>();
            int sx = 0;
            int sy = 0;

            foreach (IWeightMatrix item in wl)
            {
                WeightMatrix m = item as WeightMatrix;
                sx += m.Rows;
                sy = m.Columns;

                twl.Add(m);
            }

            var res = weightMatrixFactory.CreateWeightMatrix(sx, sy);

            int startIdx = 0;
            for (var i = 0; i < twl.Count; i++)
            {
                Array.Copy(twl[i].Weight, 0, res.Weight, startIdx, twl[i].Weight.Length);
                startIdx += twl[i].Weight.Length;
            }

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    startIdx = 0;
                    for (var i = 0; i < twl.Count; i++)
                    {
                        var k = 0;
                        var n = twl[i].Gradient.Length;
                        var moreItem = (n % Vector<float>.Count);
                        var Gradient = twl[i].Gradient;
                     
                        while (k < n - moreItem)
                        {
                            var vecResG = new Vector<float>(res.Gradient, startIdx + k);
                            var vecM1G = new Vector<float>(Gradient, k);
                            vecM1G += vecResG;
                            vecM1G.CopyTo(Gradient, k);

                            k += Vector<float>.Count;
                        }

                        while (k < n)
                        {
                            Gradient[k] += res.Gradient[startIdx + k];
                            k++;
                        }

                        startIdx += n;
                    }                
                };
                backprop.Push(backward);
            }

            return res;
        }

        public virtual IWeightMatrix RepeatRows(IWeightMatrix w, int n)
        {
            List<IWeightMatrix> wl = new List<IWeightMatrix>();
            for (int i = 0; i < n; i++)
            {
                wl.Add(w);
            }

            return ConcatRows(wl);
        }


        public virtual IWeightMatrix Transpose2(IWeightMatrix w)
        {
            var m = w as WeightMatrix;

            var res = weightMatrixFactory.CreateWeightMatrix(m.Columns, m.Rows);

            for (var i = 0; i < m.Rows; i++)
            {
                for (var j = 0; j < m.Columns; j++)
                {
                    res.Weight[j * res.Columns + i] = m.Weight[i * m.Columns + j];
                }
            }

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    for (var i = 0; i < m.Rows; i++)
                    {
                        for (var j = 0; j < m.Columns; j++)
                        {
                            m.Gradient[i * m.Columns + j] += res.Gradient[j * res.Columns + i];
                        }
                    }
                };
                backprop.Push(backward);
            }

            return res;
        }

        public IWeightMatrix PermuteBatch(IWeightMatrix m, int batchSize)
        {
            return m;
        }

        public IWeightMatrix View(IWeightMatrix m, int r, int c)
        {
            return m;
        }

        public List<IWeightMatrix> UnFolderRow(IWeightMatrix m, int n, bool gradient = true)
        {
            if (n != 1)
            {
                throw new InvalidOperationException($"n must be 1 when using CPU arch type.");
            }

            List<IWeightMatrix> result = new List<IWeightMatrix>();
            result.Add(m);

            return result;
        }
		
		public IWeightMatrix Mul(IWeightMatrix w, float v)
        {
            var m = w as WeightMatrix;

            var res = weightMatrixFactory.CreateWeightMatrix(m.Rows, m.Columns);
            var n = m.Weight.Length;
            var i = 0;
            var moreItems = (n % Vector<float>.Count);
            while (i < n - moreItems)
            {
                var vecResW = new Vector<float>(res.Weight, i);
                var vecM1W = new Vector<float>(m.Weight, i);

                vecResW = vecM1W * v;
                vecResW.CopyTo(res.Weight, i);

                i += Vector<float>.Count;
            }

            while (i < n)
            {
                res.Weight[i] = m.Weight[i] * v;
                i++;
            }


            if (needs_backprop)
            {

                Action backward = () =>
                {
                    i = 0;
                    while (i < n - moreItems)
                    {
                        var vecResGrad = new Vector<float>(res.Gradient, i);
                        var vecM1W = new Vector<float>(m.Weight, i);

                        var vecM1Grad = new Vector<float>(m.Gradient, i);

                        vecM1Grad += v * vecResGrad;

                        vecM1Grad.CopyTo(m.Gradient, i);

                        i += Vector<float>.Count;
                    }

                    while (i < n)
                    {
                        m.Gradient[i] += v * res.Gradient[i];
                        i++;
                    }

                };
                backprop.Push(backward);
            }
            return res;
        }
       
        public IWeightMatrix MulBatch(IWeightMatrix m1, IWeightMatrix m2, int batchSize)
        {
            if (batchSize != 1)
            {
                throw new ArgumentException($"For CPU operations, its batch size must be 1.");
            }

            return Mul(m1, m2);
        }

        public IWeightMatrix ConcatRowColumn(List<IWeightMatrix> wl1, List<IWeightMatrix> wl2)
        {
            var r1 = ConcatRows(wl1);
            var r2 = ConcatRows(wl2);

            return ConcatColumns(r1, r2);
        }

        public IWeightMatrix EltMulMulAdd(IWeightMatrix w1, IWeightMatrix w2, IWeightMatrix w3, IWeightMatrix w4)
        {
            return Add(EltMul(w1, w2), EltMul(w3, w4));
        }
    }     
}
