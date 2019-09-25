using Microsoft.Extensions.Logging;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Catalyst.Tensors.Models.Tools
{
    public class ComputeGraphTensor : IComputeGraph
    {
        internal WeightTensorFactory weightTensorFactory;
        public ConcurrentStack<Action> backprop = new ConcurrentStack<Action>();
        public bool needs_backprop { get; set; }
        private int deviceId;

        public ComputeGraphTensor(IWeightFactory weightFactory, int deviceId, bool needBack = true)
        {
            weightTensorFactory = weightFactory as WeightTensorFactory;

            needs_backprop = needBack;
            this.deviceId = deviceId;
        }

        public void Backward()
        {
            while (backprop.TryPop(out var action))
            {
                action();
            }
        }

        public void RunTopBackward()
        {
            if (backprop.TryPop(out var action))
            {
                action();
            }
        }
        
        public IWeightMatrix Sigmoid(IWeightMatrix w, bool updateWeightsInPlace = false)
        {
            var m = w as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(m.Rows, m.Columns, deviceId);
            if (updateWeightsInPlace)
            {
                res.TWeight = m.TWeight.CopyRef();
            }

            Ops.Sigmoid(res.TWeight, m.TWeight);

            if (needs_backprop)
            {
                Action backward = () =>
                {
                   // Ops.AddSigmoidD(m.TGradient, m.TGradient, res.TWeight, res.TGradient);

                    m.AddSigmoidGradient(res);

                    res.Dispose();
                };
                backprop.Push(backward);
            }

            return res;

        }
      
            
        public IWeightMatrix AddTanh(IWeightMatrix w1, IWeightMatrix w2)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(m1.Rows, m1.Columns, deviceId);
            Ops.AddTanh(res.TWeight, m1.TWeight, m2.TWeight);

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    //Ops.AddTanhD(m1.TGradient, m1.TGradient, res.TWeight, res.TGradient);
                    //Ops.AddTanhD(m2.TGradient, m2.TGradient, res.TWeight, res.TGradient);

                    m1.AddTanhGradient(res);
                    m2.AddTanhGradient(res);


                    res.Dispose();
                };
                backprop.Push(backward);
            }

            return res;

        }



        public IWeightMatrix Mul(IWeightMatrix w, float v)
        {
            var m = w as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(m.Rows, m.Columns, deviceId);

            Ops.Mul(res.TWeight, m.TWeight, v);

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    Ops.AddMulV(m.TGradient, m.TGradient, res.TGradient, v);
                };
                backprop.Push(backward);
            }

            return res;
        }
		

        public IWeightMatrix EltMulMulAdd(IWeightMatrix w1, IWeightMatrix w2, IWeightMatrix w3, IWeightMatrix w4)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;
            var m3 = w3 as WeightTensor;
            var m4 = w4 as WeightTensor;

            var res = weightTensorFactory.CreateWeightTensor(m1.Rows, m1.Columns, deviceId);

            Ops.MulMulAdd(res.TWeight, m1.TWeight, m2.TWeight, m3.TWeight, m4.TWeight);

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    //Ops.AddMul(m1.TGradient, m1.TGradient, m2.TWeight, res.TGradient);
                    //Ops.AddMul(m2.TGradient, m2.TGradient, m1.TWeight, res.TGradient);

                    //Ops.AddMul(m3.TGradient, m3.TGradient, m4.TWeight, res.TGradient);
                    //Ops.AddMul(m4.TGradient, m4.TGradient, m3.TWeight, res.TGradient);

                    m1.AddMulGradient(m2.TWeight, res.TGradient);
                    m2.AddMulGradient(m1.TWeight, res.TGradient);

                    m3.AddMulGradient(m4.TWeight, res.TGradient);
                    m4.AddMulGradient(m3.TWeight, res.TGradient);

                    res.Dispose();
                };
                backprop.Push(backward);
            }

            return res;
        }
       
        public IWeightMatrix EltMul(IWeightMatrix w1, IWeightMatrix w2)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(m1.Rows, m1.Columns, deviceId);

            Ops.Mul(res.TWeight, m1.TWeight, m2.TWeight);

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    //Ops.AddMul(m1.TGradient, m1.TGradient, m2.TWeight, res.TGradient);
                    //Ops.AddMul(m2.TGradient, m2.TGradient, m1.TWeight, res.TGradient);

                    m1.AddMulGradient(m2.TWeight, res.TGradient);
                    m2.AddMulGradient(m1.TWeight, res.TGradient);

                    res.Dispose();
                };
                backprop.Push(backward);
            }

            return res;
        }

        public IWeightMatrix Add(IWeightMatrix w1, IWeightMatrix w2)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;          
            var res = weightTensorFactory.CreateWeightTensor(m1.Rows, m1.Columns, deviceId);

            Ops.Add(res.TWeight, m1.TWeight, m2.TWeight);

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    //Ops.Add(m1.TGradient, res.TGradient, m1.TGradient);
                    //Ops.Add(m2.TGradient, res.TGradient, m2.TGradient);

                    m1.CopyOrAddGradient(res);
                    m2.CopyOrAddGradient(res);

                    res.Dispose();
                };
                backprop.Push(backward);
            }

            return res;
        }

        public IWeightMatrix Tanh(IWeightMatrix w, bool updateWeightsInPlace = false)
        {
            var m = w as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(m.Rows, m.Columns, deviceId);
            if (updateWeightsInPlace)
            {
                res.TWeight = m.TWeight.CopyRef();
            }

            Ops.Tanh(res.TWeight, m.TWeight);

            if (needs_backprop)
            {
                Action backward = () =>
                {
                   // Ops.AddTanhD(m.TGradient, m.TGradient, res.TWeight, res.TGradient);

                    m.AddTanhGradient(res);

                    res.Dispose();
                };
                backprop.Push(backward);
            }

            return res;
        }

        public IWeightMatrix MulBatch(IWeightMatrix m1, IWeightMatrix m2, int batchSize)
        {
            WeightTensor t1 = m1 as WeightTensor;
            WeightTensor t2 = m2 as WeightTensor;
            var n = t1.Rows;
            var d = t2.Columns;
            WeightTensor res = weightTensorFactory.CreateWeightTensor(n, d, deviceId);

            Tensor t1W = t1.TWeight.View(batchSize, t1.Rows / batchSize, t1.Columns);
            Tensor t2W = t2.TWeight.View(batchSize, t2.Rows / batchSize, t2.Columns);
            Tensor rW = res.TWeight.View(batchSize, n / batchSize, d);

            Ops.AddmmBatch(rW, 0.0f, rW, 1.0f, t1W, t2W);
            rW.Dispose();

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();
                    
                    Tensor t1G = t1.TGradient.View(batchSize, t1.Rows / batchSize, t1.Columns);
                    Tensor t2G = t2.TGradient.View(batchSize, t2.Rows / batchSize, t2.Columns);
                    Tensor rG = res.TGradient.View(batchSize, n / batchSize, d);

                    var tW2 = t2W.Transpose(1, 2);
                    Ops.AddmmBatch(t1G, 1.0f, t1G, 1.0f, rG, tW2);

                    var tW1 = t1W.Transpose(1, 2);
                    Ops.AddmmBatch(t2G, 1.0f, t2G, 1.0f, tW1, rG);

                    tW1.Dispose();
                    tW2.Dispose();

                    t1W.Dispose();
                    t2W.Dispose();
                    t1G.Dispose();
                    t2G.Dispose();

                    rG.Dispose();

                    res.Dispose();

                };
                backprop.Push(backward);
            }
            else
            {
                t1W.Dispose();
                t2W.Dispose();
            }

            return res;
        }

        public IWeightMatrix Mul(IWeightMatrix m1, IWeightMatrix m2)
        {
            WeightTensor t1 = m1 as WeightTensor;
            WeightTensor t2 = m2 as WeightTensor;
            var n = t1.Rows;
            var d = t2.Columns;
            WeightTensor res;

            res = weightTensorFactory.CreateWeightTensor(n, d, deviceId);
            Ops.Addmm(res.TWeight, 0.0f, res.TWeight, 1.0f, t1.TWeight, t2.TWeight);

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    var tW2 = t2.TWeight.Transpose();
                    Ops.Addmm(t1.TGradient, 1.0f, t1.TGradient, 1.0f, res.TGradient, tW2);               

                    var tW1 = t1.TWeight.Transpose();
                    Ops.Addmm(t2.TGradient, 1.0f, t2.TGradient, 1.0f, tW1, res.TGradient);

                    tW1.Dispose();
                    tW2.Dispose();

                    res.Dispose();
                };
                backprop.Push(backward);
            }

            return res;
        }


        public IWeightMatrix MulAdd(IWeightMatrix m1, IWeightMatrix m2, IWeightMatrix m3)
        {            
            WeightTensor t1 = m1 as WeightTensor;
            WeightTensor t2 = m2 as WeightTensor;
            WeightTensor t3 = m3 as WeightTensor;

            var n = t1.Rows;
            var d = t2.Columns;

            WeightTensor res = weightTensorFactory.CreateWeightTensor(n, d, deviceId);
            Ops.Addmm(res.TWeight, 1.0f, t3.TWeight, 1.0f, t1.TWeight, t2.TWeight);

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                  //  Ops.Add(t3.TGradient, t3.TGradient, res.TGradient);

                    t3.CopyOrAddGradient(res);

                    var tW2 = t2.TWeight.Transpose();
                    Ops.Addmm(t1.TGradient, 1.0f, t1.TGradient, 1.0f, res.TGradient, tW2);


                    var tW1 = t1.TWeight.Transpose();
                    Ops.Addmm(t2.TGradient, 1.0f, t2.TGradient, 1.0f, tW1, res.TGradient);

                    tW1.Dispose();
                    tW2.Dispose();

                    res.Dispose();
                };
                backprop.Push(backward);
            }

            return res;
        }
           
        public IWeightMatrix Transpose2(IWeightMatrix w)
        {
            WeightTensor m = w as WeightTensor;

            var wT = m.TWeight.Transpose();
            var gT = m.TGradient.Transpose();

            var res = weightTensorFactory.CreateWeightTensor(m.Columns, m.Rows, wT, gT);

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    res.Dispose();
                };
                backprop.Push(backward);
            }

            return res;
        }

      

        public IWeightMatrix Softmax(IWeightMatrix w, bool bp = true)
        {
            WeightTensor m = w as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(m.Rows, m.Columns, deviceId);
            Ops.Softmax(res.TWeight, m.TWeight);

            if (needs_backprop && bp)
            {
                Action backward = () =>
                {
                   // Ops.SoftmaxGrad(m.TGradient, res.TGradient, res.TWeight);

                    m.AddSoftmaxGradient(res);

                    res.Dispose();
                };
                backprop.Push(backward);
            }

            return res;
        }


        private static object locker = new object();

        public IWeightMatrix PeekRow(IWeightMatrix w, int ix, int num = 1)
        {
            WeightTensor m = w as WeightTensor;
            var tw = m.TWeight.Narrow(0, ix, num);
            var tg = m.TGradient != null ? m.TGradient.Narrow(0, ix, num) : null;

            var res = weightTensorFactory.CreateWeightTensor(num, m.Columns, tw, tg);

            lock (locker)
            {
                for (int i = 0; i < num; i++)
                {
                    if (m.RowToBeUpdated.ContainsKey(ix + i) == false)
                    {
                        m.RowToBeUpdated.Add(ix + i, 1);
                    }
                    else
                    {
                        m.RowToBeUpdated[ix + i]++;
                    }
                }
            }

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    res.Dispose();
                };
                backprop.Push(backward);
            }

            return res;
        }
    
        public IWeightMatrix ConcatColumns(IWeightMatrix w1, IWeightMatrix w2)
        {
            var m1 = w1 as WeightTensor;
            var m2 = w2 as WeightTensor;

            int sx = m1.Rows;
            int sy = m1.Columns + m2.Columns;

            var res = weightTensorFactory.CreateWeightTensor(sx, sy, deviceId);

            Ops.Concat(res.TWeight, 1, m1.TWeight, m2.TWeight);

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    Tensor tTmp1 = res.TGradient.Narrow(1, 0, m1.Columns);
                 //   Ops.Add(m1.TGradient, m1.TGradient, tTmp1);

                    m1.CopyOrAddGradient(tTmp1);

                    Tensor tTmp2 = res.TGradient.Narrow(1, m1.Columns, m2.Columns);
                  //  Ops.Add(m2.TGradient, m2.TGradient, tTmp2);

                    m2.CopyOrAddGradient(tTmp2);

                    tTmp1.Dispose();
                    tTmp2.Dispose();

                    res.Dispose();
                };
                backprop.Push(backward);
            }
            return res;
        }

        public IWeightMatrix RepeatRows(IWeightMatrix w, int n)
        {
            var m = w as WeightTensor;
            if (m.Rows == 1)
            {
                var res = weightTensorFactory.CreateWeightTensor(m.Rows * n, m.Columns, m.TWeight.Expand(n, m.Columns), m.TGradient.Expand(n, m.Columns));

                if (needs_backprop)
                {
                    Action backward = () =>
                    {
                        res.Dispose();
                    };
                    backprop.Push(backward);
                }

                return res;
            }
            else
            {
                List<IWeightMatrix> ws = new List<IWeightMatrix>();
                for (int i = 0; i < n; i++)
                {
                    ws.Add(w);
                }

                return ConcatRows(ws);
            }
        }


        public IWeightMatrix ConcatRows(List<IWeightMatrix> wl)
        {
            if (wl.Count == 1)
            {
                return wl[0];
            }

            List<Tensor> twl = new List<Tensor>();
            int sx = 0;
            int sy = 0;
            foreach (IWeightMatrix item in wl)
            {
                WeightTensor m = item as WeightTensor;
                sx += m.Rows;
                sy = m.Columns;

                twl.Add(m.TWeight);
            }

            var res = weightTensorFactory.CreateWeightTensor(sx, sy, deviceId);
            Ops.Concat(res.TWeight, 0, twl.ToArray());


            if (needs_backprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    sx = 0;
                    foreach (IWeightMatrix item in wl)
                    {
                        WeightTensor m = item as WeightTensor;

                        Tensor tTmp = res.TGradient.Narrow(0, sx, m.Rows);
                   //     Ops.Add(m.TGradient, m.TGradient, tTmp);

                        m.CopyOrAddGradient(tTmp);

                        sx += m.Rows;

                        tTmp.Dispose();
                    }

                    res.Dispose();
                };
                backprop.Push(backward);
            }
            return res;

        }

        public IWeightMatrix ConcatRowColumn(List<IWeightMatrix> wl1, List<IWeightMatrix> wl2)
        {
            int sx = wl1[0].Rows * wl1.Count;
            int sy = wl1[0].Columns + wl2[0].Columns;

            var res = weightTensorFactory.CreateWeightTensor(sx, sy, deviceId);

            var resTWC1 = res.TWeight.Narrow(1, 0, wl1[0].Columns);
            var resTWC2 = res.TWeight.Narrow(1, wl1[0].Columns, wl2[0].Columns);

            for (int i = 0; i < wl1.Count; i++)
            {
                WeightTensor m1 = wl1[i] as WeightTensor;
                WeightTensor m2 = wl2[i] as WeightTensor;

                var resTWC1R = resTWC1.Narrow(0, i * m1.Rows, m1.Rows);
                Ops.Copy(resTWC1R, m1.TWeight);

                var resTWC2R = resTWC2.Narrow(0, i * m2.Rows, m2.Rows);
                Ops.Copy(resTWC2R, m2.TWeight);

                resTWC1R.Dispose();
                resTWC2R.Dispose();
            }

            resTWC1.Dispose();
            resTWC2.Dispose();

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    var res1 = res.TGradient.Narrow(1, 0, wl1[0].Columns);
                    var res2 = res.TGradient.Narrow(1, wl1[0].Columns, wl2[0].Columns);

                    for (int i = 0; i < wl1.Count; i++)
                    {
                        WeightTensor m1 = wl1[i] as WeightTensor;
                        WeightTensor m2 = wl2[i] as WeightTensor;

                        var resTGC1R = res1.Narrow(0, i * m1.Rows, m1.Rows);
                        var resTGC2R = res2.Narrow(0, i * m1.Rows, m1.Rows);

                        //Ops.Add(m1.TGradient, m1.TGradient, resTGC1R);
                        //Ops.Add(m2.TGradient, m2.TGradient, resTGC2R);

                        m1.CopyOrAddGradient(resTGC1R);
                        m2.CopyOrAddGradient(resTGC2R);

                        resTGC1R.Dispose();
                        resTGC2R.Dispose();
                    }

                    res1.Dispose();
                    res2.Dispose();
                    res.Dispose();
                };
                backprop.Push(backward);
            }

            return res;
        }

        public IWeightMatrix ConcatColumns(params IWeightMatrix[] wl)
        {
            if (wl.Length == 1)
            {
                return wl[0];
            }

            List<Tensor> twl = new List<Tensor>();
            int sx = 0;
            int sy = 0;

            foreach (IWeightMatrix item in wl)
            {
                WeightTensor m = item as WeightTensor;
                sx = m.Rows;
                sy += m.Columns;

                twl.Add(m.TWeight);
            }


            var res = weightTensorFactory.CreateWeightTensor(sx, sy, deviceId);
            Ops.Concat(res.TWeight, 1, twl.ToArray());


            if (needs_backprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                    sy = 0;
                    foreach (IWeightMatrix item in wl)
                    {
                        WeightTensor m = item as WeightTensor;

                        Tensor tTmp = res.TGradient.Narrow(1, sy, m.Columns);
                      //  Ops.Add(m.TGradient, m.TGradient, tTmp);

                        m.CopyOrAddGradient(tTmp);

                        sy += m.Columns;

                        tTmp.Dispose();
                    }

                    res.Dispose();
                };
                backprop.Push(backward);
            }
            return res;
        }

        public List<IWeightMatrix> SplitColumns2(IWeightMatrix w, params int[] sizes)
        {
            var m = w as WeightTensor;
            List<IWeightMatrix> resList = new List<IWeightMatrix>();

            int x = 0;
            foreach (int size in sizes)
            {
                WeightTensor res = weightTensorFactory.CreateWeightTensor(m.Rows, size, deviceId);// m.TWeight.Narrow(1, x, size), m.TGradient.Narrow(1, x, size));
                res.TWeight = m.TWeight.Narrow(1, x, size);

                resList.Add(res);

                x += size;
            }


            if (needs_backprop)
            {
                Action backward = () =>
                {
                    x = 0;
                    int i = 0;
                    foreach (var item in resList)
                    {
                        var item_i = item as WeightTensor;
                        var mG = m.TGradient.Narrow(1, x, sizes[i]);

                        Ops.Add(mG, mG, item_i.TGradient);

                        mG.Dispose();
                        item.Dispose();

                        x += sizes[i];
                        i++;
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


        public IWeightMatrix View(IWeightMatrix m, int r, int c)
        {
            WeightTensor t = m as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(r, c, deviceId);
            res.TWeight = t.TWeight.View(r, c);

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    var g = res.TGradient.View(m.Rows, m.Columns);
                    t.CopyOrAddGradient(g);
                };
                backprop.Push(backward);
            }


            return res;
        }

        public IWeightMatrix PermuteBatch(IWeightMatrix m, int batchSize)
        {
            WeightTensor t = m as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(m.Rows, m.Columns, deviceId);
            int sizeEveryBatch = m.Rows / batchSize;

            res.TWeight = Ops.AsContiguous(t.TWeight.View(sizeEveryBatch, batchSize, m.Columns).Permute(1, 0, 2)).View(m.Rows, m.Columns);

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    var g = t.TGradient.View(sizeEveryBatch, batchSize, m.Columns);
                    var t2 = res.TGradient.View(batchSize, sizeEveryBatch, m.Columns).Permute(1, 0, 2);
                    Ops.Add(g, g, t2);
                   
                    g.Dispose();
                    t2.Dispose();
                    res.Dispose();
                };
                backprop.Push(backward);
            }


            return res;
        }



        public List<IWeightMatrix> UnFolderRow(IWeightMatrix m, int n, bool gradient = true)
        {
            List<IWeightMatrix> resList = new List<IWeightMatrix>();

            WeightTensor t = m as WeightTensor;

            if (gradient)
            {
                Tensor tW = t.TWeight.Unfold(0, n, n);
                Tensor tG = t.TGradient.Unfold(0, n, n);

                for (int i = 0; i < n; i++)
                {
                    WeightTensor res = weightTensorFactory.CreateWeightTensor(m.Rows / n, m.Columns, tW.Select(2, i), tG.Select(2, i));

                    if (res.Rows != res.TWeight.Sizes[0] || res.Rows != res.TGradient.Sizes[0])
                    {
                        throw new InvalidOperationException("Invalide unfolder");
                    }

                    resList.Add(res);
                }

                tW.Dispose();
                tG.Dispose();
            }
            else
            {
                Tensor tw = t.TWeight.Unfold(0, n, n);
                for (int i = 0; i < n; i++)
                {
                    WeightTensor res = weightTensorFactory.CreateWeightTensor(m.Rows / n, m.Columns, tw.Select(2, i), null);

                    if (res.Rows != res.TWeight.Sizes[0])
                    {
                        throw new InvalidOperationException("Invalide unfolder");
                    }

                    resList.Add(res);
                }

                tw.Dispose();
            }

            if (needs_backprop && gradient)
            {
                Action backward = () =>
                {
                    foreach (var item in resList)
                    {
                        item.Dispose();
                    }
                };
                backprop.Push(backward);
            }


            return resList;
        }

        Random rnd = new Random(DateTime.Now.Millisecond);
        private Tensor BuildRandomTensor(int rows, int columns, double prob)
        {
            float[] weights = new float[rows * columns];
            for (int i = 0; i < weights.Length; i++)
            {
                double r = rnd.NextDouble();
                if (r < prob)
                {
                    weights[i] = 1.0f;
                }
            }

            Tensor noise = new Tensor(TensorAllocator.Allocator(deviceId), DType.Float32, rows, columns);
            noise.SetElementsAsFloat(weights);

            return noise;
        }


        public IWeightMatrix CreatePositionMatrix(int dimWords, int dimEmb)
        {
            double numTimescales = (float)dimEmb / 2;
            double logTimescaleIncrement = Math.Log(10000.0f) / (numTimescales - 1.0f);
            float[] weights = new float[dimWords * dimEmb];

            var res = weightTensorFactory.CreateWeightTensor(dimWords, dimEmb, deviceId);

            for (int p = 0; p < dimWords; ++p)
            {
                for (int i = 0; i < numTimescales; ++i)
                {
                    float v = (float)(p * Math.Exp(i * -logTimescaleIncrement));

                    weights[p * dimEmb + i] = (float)Math.Sin(v);
                    weights[p * dimEmb + (int)numTimescales + i] = (float)Math.Cos(v);
                }
            }

            res.TWeight.CopyFrom(weights);

            return res;
        }

        public IWeightMatrix LayerNorm(IWeightMatrix src, IWeightMatrix alpha, IWeightMatrix beta, float eps = 1e-09f)
        {
            var srcT = src as WeightTensor;
            var alphaT = alpha as WeightTensor;
            var betaT = beta as WeightTensor;

            var res = weightTensorFactory.CreateWeightTensor(src.Rows, src.Columns, deviceId);

            Ops.LayerNorm(res.TWeight, srcT.TWeight, alphaT.TWeight, betaT.TWeight, eps);


            if (needs_backprop)
            {
                Action backward = () =>
                {
                    Ops.LayerNormGrad(srcT.TGradient, alphaT.TGradient, betaT.TGradient, res.TGradient, res.TWeight, srcT.TWeight, alphaT.TWeight, betaT.TWeight, eps);
                };
                backprop.Push(backward);
            }

            return res;
        }

        public IWeightMatrix Dropout(IWeightMatrix V, float drop_prob)
        {
            float p = 1.0f - drop_prob;
            var w = V as WeightTensor;
            var res = weightTensorFactory.CreateWeightTensor(V.Rows, V.Columns, deviceId);

            Tensor noise = BuildRandomTensor(V.Rows, V.Columns, p);
            Ops.Mul(res.TWeight, w.TWeight, noise);

            if (needs_backprop)
            {
                Action backward = () =>
                {
                    res.ReleaseWeight();

                  //  Ops.AddMul(w.TGradient, w.TGradient, res.TGradient, noise);

                    w.AddMulGradient(noise, res.TGradient);

                    noise.Dispose();

                    res.Dispose();
                };
                backprop.Push(backward);
            }

            return res;
        }



    }
}
