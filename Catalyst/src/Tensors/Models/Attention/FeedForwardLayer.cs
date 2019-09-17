using Catalyst.Tensors.Models.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Catalyst.Tensors.Models
{
    class FeedForwardLayer
    {
        private IWeightMatrix m_Whd;
        private IWeightMatrix m_Bd;

        public FeedForwardLayer(int inputDim, int outputDim, ArchTypeEnums archType, int deviceId)
        {
            if (archType == ArchTypeEnums.GPU_CUDA)
            {
                m_Whd = new WeightTensor(inputDim, outputDim, deviceId, true);
                m_Bd = new WeightTensor(1, outputDim, 0, deviceId);
            }
            else
            {
                m_Whd = new WeightMatrix(inputDim, outputDim, true);
                m_Bd = new WeightMatrix(1, outputDim, 0);
            }
        }

        public IWeightMatrix Process(IWeightMatrix inputT, IComputeGraph g)
        {
            var bds = g.RepeatRows(m_Bd, inputT.Rows);
            return g.MulAdd(inputT, m_Whd, bds);
        }

        public virtual List<IWeightMatrix> getParams()
        {
            List<IWeightMatrix> response = new List<IWeightMatrix>();
            response.Add(m_Whd);
            response.Add(m_Bd);

            return response;
        }

        public void Save(Stream stream)
        {
            m_Whd.Save(stream);
            m_Bd.Save(stream);
        }


        public void Load(Stream stream)
        {
            m_Whd.Load(stream);
            m_Bd.Load(stream);
        }
    }
}
