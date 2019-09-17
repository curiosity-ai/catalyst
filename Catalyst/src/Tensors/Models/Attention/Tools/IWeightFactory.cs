using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Catalyst.Tensors.Models.Tools
{
    public interface IWeightFactory
    {
        IWeightMatrix CreateWeights(int row, int column, int deviceId);
        IWeightMatrix CreateWeights(int row, int column, int deviceId, bool cleanWeights);

        void Clear();
    }
}
