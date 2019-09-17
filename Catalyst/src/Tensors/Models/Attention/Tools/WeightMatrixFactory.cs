using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Catalyst.Tensors.Models.Tools
{
    public class WeightMatrixFactory : IWeightFactory
    {
        public WeightMatrix CreateWeightMatrix(int row, int column, bool cleanWeights = false)
        {
            return new WeightMatrix(row, column);

            //TODO: refactor this to use an object pool instead
        }

        public void Clear()
        {
            //TODO: refactor this to use an object pool instead
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
