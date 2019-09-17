using Catalyst.Tensors.Models.Tools;
using Mosaik.Core;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Catalyst.Tensors
{
    [Serializable]
    public class WeightMatrix : IWeightMatrix
    {
        internal float[] weight;
        internal float[] gradient;

        public int Rows { get; set; }
        public int Columns { get; set; }
        public float[] Weight { get => weight; set => weight = value; }
        public float[] Gradient { get => gradient; set => gradient = value; }
        public float[] Cache { get; set; }
        public float[] LrW { get; set; }
        public Dictionary<int, int> RowToBeUpdated { get; set; } = new Dictionary<int, int>();

        public int DeviceId { get; set; }

        public WeightMatrix()
        {

        }

        public float[] ToWeightArray()
        {
            return weight;
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
            weight = v;
        }

        public void SetGradientFromArray(float[] array)
        {
            gradient = array;
        }

        public void ClearGradient()
        {
            SIMD.Zero(ref gradient);
        }

        public void ClearWeight()
        {
            SIMD.Zero(ref weight);
        }

        public WeightMatrix(int rows, int columns, bool normal = false)
        {
            Rows = rows;
            Columns = columns;
            var n = rows * columns;
            weight = new float[n];
            gradient = new float[n];
            Cache = new float[n];
            LrW = new float[n];

            var scale = (float)Math.Sqrt(1.0 / (rows * columns));
            if (normal)
            {
                scale = 0.08f;
            }

            float a2 = 2 * scale;
            float an = -scale;

            ThreadSafeFastRandom.NextFloats(weight);
            SIMD.Multiply(ref weight, a2);
            SIMD.Add(ref weight, an);
        }

        public WeightMatrix(int rows, int columns)
        {
            Rows = rows;
            Columns = columns;
            var n = rows * columns;
            weight = new float[n];
            gradient = new float[n];
        }

        public WeightMatrix(int rows, int columns, float c)
        {
            Rows = rows;
            Columns = columns;
            var n = rows * columns;
            weight = new float[n];
            gradient = new float[n];
            Cache = new float[n];
            LrW = new float[n];

            if (c != 0.0)
            {
                SIMD.Add(ref weight, c);
            }
        }

        public void SetWeightAtRow(int row, float[] val)
        {
            var offset = Columns * row;
            Array.Copy(val, 0, weight, offset, val.Length);
        }

        public WeightMatrix Clone()
        {
            var v = new WeightMatrix(Rows, Columns, 0);
            weight.AsSpan().CopyTo(v.weight.AsSpan());
            return v;
        }

        public void Dispose()
        {

        }

        public void CleanCache()
        {
            Cache = new float[Cache.Length];
            LrW = new float[LrW.Length];
        }


        public float GetWeightAt(int offset)
        {
            return weight[offset];
        }

        public void SetGradientAt(float val, int offset)
        {
            gradient[offset] = val;
        }

        public void SetWeightAt(float val, int offset)
        {
            weight[offset] = val;
        }

        public void SetGradientByWeight(IWeightMatrix src)
        {
            WeightMatrix m = src as WeightMatrix;
            Array.Copy(m.weight, gradient, m.weight.Length);
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

        public void ReleaseWeight()
        {

        }

        public void CopyWeights(IWeightMatrix src)
        {
            WeightMatrix m = src as WeightMatrix;

            Array.Copy(m.Weight, Weight, m.Weight.Length);
        }

        public void AddGradient(IWeightMatrix src)
        {
            WeightMatrix m = src as WeightMatrix;

            SIMD.Add(ref gradient, ref m.gradient);

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
        }
    }
}
