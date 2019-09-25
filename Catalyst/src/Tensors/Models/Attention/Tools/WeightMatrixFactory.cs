using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Catalyst.Tensors.Models.Tools
{
    public class WeightMatrixPool
    {
        private int Current=0;

        private List<WeightMatrix> Matrices = new List<WeightMatrix>();

        internal WeightMatrix Borrow(int row, int column)
        {

            WeightMatrix m;
            lock (Matrices)
            {
                if(Current < Matrices.Count-1)
                {
                    m = Matrices[Current];
                    Current++;
                }
                else
                {
                    m = new WeightMatrix(row, column);
                    Matrices.Add(m);
                    Current++;
                }
            }
            return m;
        }

        internal void ReturnAll()
        {
            lock (Matrices)
            {
                Current = 0;
            }
        }
    }

    public class WeightMatrixFactory : IWeightFactory
    {
        public static ConcurrentDictionary<int, ConcurrentDictionary<int, WeightMatrixPool>> Pool = new ConcurrentDictionary<int, ConcurrentDictionary<int, WeightMatrixPool>>();

        public WeightMatrix CreateWeightMatrix(int row, int column, bool cleanWeights = false)
        {
            var cols = Pool.GetOrAdd(row, k => new ConcurrentDictionary<int, WeightMatrixPool>());
            var mat = cols.GetOrAdd(column, k => new WeightMatrixPool());

            var m = mat.Borrow(row, column);
            if (cleanWeights)
            {
                m.ClearWeight();
            }
            return m;
        }

        public void Clear()
        {
            foreach(var cols in Pool.Values)
            {
                foreach(var mat in cols.Values)
                {
                    mat.ReturnAll();
                }
            }
        }

        public IWeightMatrix CreateWeights(int row, int column, int deviceId)
        {
            return CreateWeightMatrix(row, column);
        }

        public IWeightMatrix CreateWeights(int row, int column, int deviceId, bool cleanWeights)
        {
            return CreateWeightMatrix(row, column, cleanWeights);
        }
    }
}
