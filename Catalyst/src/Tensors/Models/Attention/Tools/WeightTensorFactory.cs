using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Catalyst.Tensors.Models.Tools
{
    public class WeightTensorFactory : IWeightFactory
    {
        List<WeightTensor> weights = new List<WeightTensor>();

        public WeightTensor CreateWeightTensor(int row, int column, Tensor w, Tensor g)
        {
            WeightTensor t = new WeightTensor(row, column, w, g);
            weights.Add(t);

            return t;
        }

        //public WeightTensor CreateWeightTensor(int row, int column, int deviceId, Tensor w, bool gradient = true)
        //{
        //    WeightTensor t = new WeightTensor(row, column, w, deviceId, gradient);
        //    weights.Add(t);

        //    return t;
        //}


        public WeightTensor CreateWeightTensor(int row, int column, int deviceId, bool cleanWeights = false)
        {
            WeightTensor r = new WeightTensor(row, column, deviceId);

            if (cleanWeights)
            {
                r.ClearWeight();
            }

            weights.Add(r);

            return r;
        }

        public void Clear()
        {
            foreach (var item in weights)
            {
                item.Dispose();
            }
            weights.Clear();

        }

        public IWeightMatrix CreateWeights(int row, int column, int deviceId)
        {
            return CreateWeightTensor(row, column, deviceId);
        }

        public IWeightMatrix CreateWeights(int row, int column, int deviceId, bool cleanWeights)
        {
            return CreateWeightTensor(row, column, deviceId, cleanWeights);
        }
    }
}
