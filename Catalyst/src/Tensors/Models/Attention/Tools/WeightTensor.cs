using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization;
using System.Security.Permissions;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace Catalyst.Tensors.Models.Tools
{
    [Serializable]
    public class WeightTensor : IWeightMatrix,  IDisposable
    {
        public int Rows { get; set; }
        public int Columns { get; set; }

        public Dictionary<int, int> RowToBeUpdated { get; set; } = new Dictionary<int, int>();

        public int DeviceId { get; set; }
        IAllocator allocator;

        private Tensor m_TWeight = null;
        private Tensor m_TGradient = null;

        private bool releasedTWeight = false;
        private bool releasedTGradient = false;

        public Tensor TWeight
        {
            get
            {
                if (releasedTWeight)
                {
                    return null;
                }

                if (m_TWeight == null)
                {                    
                    m_TWeight = new Tensor(allocator, DType.Float32, Rows, Columns);
                }

                return m_TWeight;
            }
            set
            {
                m_TWeight = value;
                releasedTWeight = false;
            }
        }

        public Tensor TGradient
        {
            get
            {
                if (releasedTGradient)
                {
                    return null;
                }

                if (m_TGradient == null)
                {
                    m_TGradient = new Tensor(allocator, DType.Float32, Rows, Columns);
                    Ops.Fill(m_TGradient, 0.0f);
                }

                return m_TGradient;
            }

            set
            {
                m_TGradient = value;
                releasedTGradient = false;
            }
        }

      //  private Tensor m_TLrW;
        private Tensor m_TCache;

       // private bool releasedTLrW = false;
        private bool releasedTCache = false;

        //public Tensor TLrW
        //{
        //    get
        //    {
        //        if (releasedTLrW)
        //        {
        //            return null;
        //        }

        //        if (m_TLrW == null)
        //        {
        //            m_TLrW = new Tensor(allocator, DType.Float32, Rows, Columns);
        //            Ops.Fill(m_TLrW, 0.0f);
        //        }

        //        return m_TLrW;
        //    }
        //    set
        //    {
        //        m_TLrW = value;
        //        releasedTLrW = false;
        //    }
        //}

        public Tensor TCache
        {
            get
            {
                if (releasedTCache)
                {
                    return null;
                }

                if (m_TCache == null)
                {
                    m_TCache = new Tensor(allocator, DType.Float32, Rows, Columns);
                    Ops.Fill(m_TCache, 0.0f);
                }

                return m_TCache;
            }
            set
            {
                m_TCache = value;
                releasedTCache = false;
            }


        }




        public WeightTensor(int rows, int columns, int deviceId, bool normal = false)
        {
            DeviceId = deviceId;
            allocator = TensorAllocator.Allocator(DeviceId);

            Rows = rows;
            Columns = columns;
            var n = rows * columns;

            float[] weight = new float[n];


            var scale = (float)Math.Sqrt(1.0 / (rows * columns));
            if (normal)
            {
                scale = 0.08f;
            }
            for (int i = 0; i < n; i++)
            {
                weight[i] = RandomGenerator.NormalRandom(0.0f, scale);
            }

            TGradient = new Tensor(allocator, DType.Float32, Rows, Columns);
            Ops.Fill(TGradient, 0.0f);

            TWeight = Tensor.FromArray(allocator, weight).View(Rows, Columns);
        }

        public WeightTensor(int rows, int columns, int deviceId)
        {
            DeviceId = deviceId;
            allocator = TensorAllocator.Allocator(DeviceId);

            Rows = rows;
            Columns = columns;

            //var allocator = TensorAllocator.Allocator(deviceId);

            //TGradient = new Tensor(allocator, DType.Float32, Rows, Columns);
            //Ops.Fill(TGradient, 0.0f);

            //TWeight = new Tensor(allocator, DType.Float32, Rows, Columns);
        }


        //public WeightTensor(int rows, int columns, Tensor weight, int deviceId, bool graident = true)
        //{
        //    DeviceId = deviceId;
        //    Rows = rows;
        //    Columns = columns;

        //    TWeight = weight;

        //    //if (graident)
        //    //{
        //    //    var allocator = TensorAllocator.Allocator(deviceId);

        //    //    TGradient = new Tensor(allocator, DType.Float32, Rows, Columns);
        //    //    Ops.Fill(TGradient, 0.0f);
        //    //}
        //}

        public WeightTensor(int rows, int columns, Tensor weight, Tensor gradient)
        {
            Rows = rows;
            Columns = columns;

            m_TGradient = gradient;
            m_TWeight = weight;
        }


        public WeightTensor(int rows, int columns, float c, int deviceId)
        {
            DeviceId = deviceId;
            allocator = TensorAllocator.Allocator(DeviceId);

            Rows = rows;
            Columns = columns;

            var n = rows * columns;

            TGradient = new Tensor(allocator, DType.Float32, Rows, Columns);
            Ops.Fill(TGradient, 0.0f);

            TWeight = new Tensor(allocator, DType.Float32, Rows, Columns);
            Ops.Fill(TWeight, c);
        }


        public void CleanCache()
        {
            Ops.Fill(TCache, 0.0f);
        //    Ops.Fill(TLrW, 0.0f);
        }

        public void ClearGradient()
        {
            Ops.Fill(TGradient, 0.0f);
        }

        public void ClearWeight()
        {
            Ops.Fill(TWeight, 0.0f);
        }

        public float GetWeightAt(int offset)
        {
            return TWeight.GetElementAsFloat(0, offset);
        }


        public float GetGradientAt(int offset)
        {
            return TGradient.GetElementAsFloat(0, offset);
        }

        public void SetGradientAt(float val, int offset)
        {
            TGradient.SetElementAsFloat(val, 0, offset);
        }

        public void SetWeightAt(float val, int offset)
        {
            TWeight.SetElementAsFloat(val, 0, offset);
        }

        public void SetWeightAtRow(int row, float[] val)
        {
            TWeight.SetElementsAsFloat(val, row, 0);
        }


        public void SetGradientByWeight(IWeightMatrix src)
        {
            WeightTensor m = src as WeightTensor;

            //  Ops.Copy(TGradient, m.TWeight);

            if (m_TGradient != null)
            {
                m_TGradient.Dispose();
            }
            m_TGradient = m.TWeight;

            m.m_TWeight = null;
        }

        public void CopyWeights(IWeightMatrix src)
        {
            WeightTensor m = src as WeightTensor;

            Ops.Copy(TWeight, m.TWeight);
        }

        private object locker = new object();
        public void AddGradient(IWeightMatrix src)
        {
            WeightTensor m = src as WeightTensor;

            lock (locker)
            {
                Tensor t = new Tensor(TGradient.Allocator, DType.Float32, Rows, Columns);
                Ops.Copy(t, m.TGradient);

                Ops.Add(TGradient, TGradient, t);
                foreach (var kv in m.RowToBeUpdated)
                {
                    if (RowToBeUpdated.ContainsKey(kv.Key) == false)
                    {
                        RowToBeUpdated.Add(kv.Key, kv.Value);
                    }
                    else
                    {
                        RowToBeUpdated[kv.Key] += kv.Value;
                    }
                }

                t.Dispose();
            }           
        }

        public float[] ToWeightArray()
        {
            return TWeight.GetElementsAsFloat(Rows * Columns);
        }

        public void AddSoftmaxGradient(WeightTensor src)
        {
            if (m_TGradient == null)
            {
                allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(allocator, DType.Float32, Rows, Columns);
                Ops.SoftmaxGrad(m_TGradient, src.TGradient, src.TWeight, false);
            }
            else
            {
                Ops.SoftmaxGrad(m_TGradient, src.TGradient, src.TWeight);
            }
        }


        public void CopyOrAddGradient(WeightTensor src)
        {
            if (m_TGradient == null)
            {
                allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(allocator, DType.Float32, Rows, Columns);
                Ops.Copy(m_TGradient, src.TGradient);
            }
            else
            {
                Ops.Add(m_TGradient, m_TGradient, src.TGradient);
            }
        }

        public void CopyOrAddGradient(Tensor src)
        {
            if (m_TGradient == null)
            {
                allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(allocator, DType.Float32, Rows, Columns);
                Ops.Copy(m_TGradient, src);
            }
            else
            {
                Ops.Add(m_TGradient, m_TGradient, src);
            }
        }

        public void AddMulGradient(Tensor w, Tensor g)
        {
            if (m_TGradient == null)
            {
                allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(allocator, DType.Float32, Rows, Columns);
                Ops.Mul(m_TGradient, w, g);
            }
            else
            {
                Ops.AddMul(m_TGradient, m_TGradient, w, g);
            }
        }


        public void AddSigmoidGradient(WeightTensor src)
        {
            if (m_TGradient == null)
            {
                allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(allocator, DType.Float32, Rows, Columns);
                Ops.SigmoidD(m_TGradient, src.TWeight, src.TGradient);
            }
            else
            {
                Ops.AddSigmoidD(m_TGradient, m_TGradient, src.TWeight, src.TGradient);
            }
        }


        public void AddTanhGradient(WeightTensor src)
        {
            if (m_TGradient == null)
            {
                allocator = TensorAllocator.Allocator(DeviceId);
                m_TGradient = new Tensor(allocator, DType.Float32, Rows, Columns);

                Ops.TanhD(m_TGradient, src.TWeight, src.TGradient);
            }
            else
            {
                Ops.AddTanhD(m_TGradient, m_TGradient, src.TWeight, src.TGradient);
            }
        }

        public int GetMaxWeightIdx()
        {
            float[] weights = ToWeightArray();
            var maxv = weights[0];
            var maxi = 0;
            for (int i = 1; i < weights.Length; i++)
            {
                if (weights[i] > maxv)
                {
                    maxv = weights[i];
                    maxi = i;
                }
            }

            return maxi;
        }

        public List<int> GetTopNMaxWeightIdx(int topN)
        {
            float[] weights = ToWeightArray();
            FixedSizePriorityQueue<ComparableItem<int>> q = new FixedSizePriorityQueue<ComparableItem<int>>(topN, new ComparableItemComparer<int>(true));

            for (int i = 0; i < weights.Length; i++)
            {
                q.Enqueue(new ComparableItem<int>(weights[i], i));
            }

            return q.Select(x => x.Value).ToList();
        }        

        public void SetWeightArray(float[] v)
        {
            TWeight.SetElementsAsFloat(v);
        }

        public void Dispose()
        {
            ReleaseWeight();
            ReleaseGradient();
          //  ReleaseLrW();
            ReleaseCache();
        }

        public void ReleaseWeight()
        {
            if (m_TWeight != null)
            {
                m_TWeight.Dispose();
                m_TWeight = null;
                releasedTWeight = true;
            }
        }

        private void ReleaseGradient()
        {
            if (m_TGradient != null)
            {
                m_TGradient.Dispose();
                m_TGradient = null;
                releasedTGradient = true;
            }
        }

        //private void ReleaseLrW()
        //{
        //    if (m_TLrW != null)
        //    {
        //        m_TLrW.Dispose();
        //        m_TLrW = null;
        //        releasedTLrW = true;
        //    }
        //}

        private void ReleaseCache()
        {
            if (m_TCache != null)
            {
                m_TCache.Dispose();
                m_TCache = null;
                releasedTCache = true;
            }

        }


        public void Save(Stream stream)
        {
            var floatArray1 = ToWeightArray();

            // create a byte array and copy the floats into it...
            var byteArray = new byte[floatArray1.Length * 4];
            Buffer.BlockCopy(floatArray1, 0, byteArray, 0, byteArray.Length);

            stream.Write(byteArray, 0, byteArray.Length);
        }

        public void Load(Stream stream)
        {
            int size = Rows * Columns;
            var byteArray = new byte[size * 4];
            stream.Read(byteArray, 0, byteArray.Length);

            var floatArray2 = new float[byteArray.Length / 4];
            Buffer.BlockCopy(byteArray, 0, floatArray2, 0, byteArray.Length);

            SetWeightArray(floatArray2);
        }
    }
}
