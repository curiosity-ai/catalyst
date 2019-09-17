
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
    public class BiEncoder
    {
        public List<LSTMCell> forwardEncoders = new List<LSTMCell>();
        public List<LSTMCell> backwardEncoders = new List<LSTMCell>();

        public int hdim { get; set; }
        public int dim { get; set; }
        public int depth { get; set; }

        public BiEncoder(int batchSize, int hdim, int dim, int depth, ArchTypeEnums archType, int deviceId)
        {
            forwardEncoders.Add(new LSTMCell(batchSize, hdim, dim, archType, deviceId));
            backwardEncoders.Add(new LSTMCell(batchSize, hdim, dim, archType, deviceId));

            for (int i = 1; i < depth; i++)
            {
                forwardEncoders.Add(new LSTMCell(batchSize, hdim, hdim * 2, archType, deviceId));
                backwardEncoders.Add(new LSTMCell(batchSize, hdim, hdim * 2, archType, deviceId));
            }

            this.hdim = hdim;
            this.dim = dim;
            this.depth = depth;
        }

        public void SetBatchSize(IWeightFactory weightFactory, int batchSize)
        {
            foreach (var item in forwardEncoders)
            {
                item.SetBatchSize(weightFactory, batchSize);
            }

            foreach (var item in backwardEncoders)
            {
                item.SetBatchSize(weightFactory, batchSize);
            }
        }

        public void Reset(IWeightFactory weightFactory)
        {
            foreach (var item in forwardEncoders)
            {
                item.Reset(weightFactory);
            }

            foreach (var item in backwardEncoders)
            {
                item.Reset(weightFactory);
            }
        }

        public List<IWeightMatrix> Encode(List<IWeightMatrix> inputs, IComputeGraph g)
        {
            List<IWeightMatrix> forwardOutputs = new List<IWeightMatrix>();
            List<IWeightMatrix> backwardOutputs = new List<IWeightMatrix>();

            List<IWeightMatrix> layerOutputs = inputs.ToList();
            int seqLen = inputs.Count;

            for (int i = 0; i < depth; i++)
            {
                for (int j = 0; j < seqLen; j++)
                {
                    var forwardOutput = forwardEncoders[i].Step(layerOutputs[j], g);
                    forwardOutputs.Add(forwardOutput);

                    var backwardOutput = backwardEncoders[i].Step(layerOutputs[inputs.Count - j - 1], g);
                    backwardOutputs.Add(backwardOutput);
                }

                backwardOutputs.Reverse();
                layerOutputs.Clear();
                for (int j = 0; j < seqLen; j++)
                {
                    var concatW = g.ConcatColumns(forwardOutputs[j], backwardOutputs[j]);
                    layerOutputs.Add(concatW);
                }

            }

            return layerOutputs;
        }


        public List<IWeightMatrix> getParams()
        {
            List<IWeightMatrix> response = new List<IWeightMatrix>();

            foreach (var item in forwardEncoders)
            {
                response.AddRange(item.getParams());
            }


            foreach (var item in backwardEncoders)
            {
                response.AddRange(item.getParams());
            }

            return response;
        }

        public void Save(Stream stream)
        {
            foreach (var item in forwardEncoders)
            {
                item.Save(stream);
            }

            foreach (var item in backwardEncoders)
            {
                item.Save(stream);
            }
        }

        public void Load(Stream stream)
        {
            foreach (var item in forwardEncoders)
            {
                item.Load(stream);
            }

            foreach (var item in backwardEncoders)
            {
                item.Load(stream);
            }
        }
    }
}
