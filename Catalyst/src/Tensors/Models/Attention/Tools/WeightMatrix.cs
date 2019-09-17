using Catalyst.Tensors.Models.Tools;
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
        public int Rows { get; set; }
        public int Columns { get; set; } 
        public float[] Weight { get; set; }
        public float[] Gradient { get; set; }
        public float[] Cash { get; set; }
        public float[] LrW { get; set; }
        public Dictionary<int, int> RowToBeUpdated { get; set; } = new Dictionary<int, int>();

        public int DeviceId { get; set; }

        public WeightMatrix( )
        {
          
        }

        public float[] ToWeightArray()
        {
            return Weight;
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
            Weight = v;
        }

        public void SetGradientFromArray(float[] array)
        {
            Gradient = array;
        }

        public void ClearGradient()
        {
            Array.Clear(Gradient, 0, Gradient.Length);
        }

        public void ClearWeight()
        {
            Array.Clear(Weight, 0, Weight.Length);
        }

        public WeightMatrix(int rows, int columns,  bool normal=false)
        {
            this.Rows = rows;
            this.Columns = columns; 
            var n = rows * columns  ;
            this.Weight = new float[n];
            this.Gradient = new float[n];
            this.Cash = new float[n];
            this.LrW = new float[n];

            var scale = (float)Math.Sqrt(1.0 / (rows * columns ));
            if (normal)
            {
                scale = 0.08f;
            }
            for (int i = 0; i < n; i++)
            {
                this.Weight[i] = RandomGenerator.NormalRandom(0.0f, scale);  
            }

        }

        public WeightMatrix(int rows, int columns)
        {
            this.Rows = rows;
            this.Columns = columns;
            var n = rows * columns;
            this.Weight = new float[n];
            this.Gradient = new float[n];
        }

        public WeightMatrix(int rows, int columns, float c)
        {
            this.Rows = rows;
            this.Columns = columns; 
            var n = rows * columns  ;
            this.Weight = new float[n];
            this.Gradient = new float[n];
            this.Cash = new float[n];
            this.LrW = new float[n];

            if (c != 0.0)
            {
                for (int i = 0; i < n; i++)
                {
                    this.Weight[i] = c;
                }
            }        
        }

        public void SetWeightAtRow(int row, float[] val)
        {
            var offset = this.Columns * row;
            Array.Copy(val, 0, Weight, offset, val.Length);
        }

        public WeightMatrix Clone()
        {
            var v= new WeightMatrix(this.Rows, this.Columns, 0);
            var n = this.Weight.Length;
            for (int i = 0; i < n; i++)
            {
                v.Weight[i] = this.Weight[i];
            }
            return v;
        }

        public void Dispose()
        {
        }

        public void CleanCache()
        {
            Cash = new float[Cash.Length];
            LrW = new float[LrW.Length];
        }


        public float GetWeightAt(int offset)
        {
            return Weight[offset];
        }

        public void SetGradientAt(float val, int offset)
        {
            Gradient[offset] = val;
        }

        public void SetWeightAt(float val, int offset)
        {
            Weight[offset] = val;
        }

        public void SetGradientByWeight(IWeightMatrix src)
        {
            WeightMatrix m = src as WeightMatrix;
//            Gradient = m.Weight;

            Array.Copy(m.Weight, Gradient, m.Weight.Length);
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
            throw new NotImplementedException();
        }
    }
}
