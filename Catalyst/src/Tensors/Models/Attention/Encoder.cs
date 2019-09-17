
using Catalyst.Tensors.Models.Tools;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Catalyst.Tensors;

namespace Catalyst.Tensors.Models
{

    [Serializable]
    public class Encoder
    {
        public List<LSTMCell> encoders = new List<LSTMCell>();
        public int hdim { get; set; }
        public int dim { get; set; }
        public int depth { get; set; }

        public Encoder(int batchSize, int hdim, int dim, int depth, ArchTypeEnums archType, int deviceId)
        {
            encoders.Add(new LSTMCell(batchSize, hdim, dim, archType, deviceId));

            for (int i = 1; i < depth; i++)
            {
                encoders.Add(new LSTMCell(batchSize, hdim, hdim, archType, deviceId));

            }
            this.hdim = hdim;
            this.dim = dim;
            this.depth = depth;
        }

        public void SetBatchSize(IWeightFactory weightFactory, int batchSize)
        {
            foreach (var item in encoders)
            {
                item.SetBatchSize(weightFactory, batchSize);
            }
        }

        public void Reset(IWeightFactory weightFactory)
        {
            foreach (var item in encoders)
            {
                item.Reset(weightFactory);
            }

        }

        public IWeightMatrix Encode(IWeightMatrix V, IComputeGraph g)
        {
            foreach (var encoder in encoders)
            {
                var e = encoder.Step(V, g);
                V = e;
            }

            return V;
        }


        public List<IWeightMatrix> getParams()
        {
            List<IWeightMatrix> response = new List<IWeightMatrix>();

            foreach (var item in encoders)
            {
                response.AddRange(item.getParams());

            }

            return response;
        }

        public void Save(Stream stream)
        {
            foreach (var item in encoders)
            {
                item.Save(stream);
            }
        }

        public void Load(Stream stream)
        {
            foreach (var item in encoders)
            {
                item.Load(stream);
            }
        }
    }
}
