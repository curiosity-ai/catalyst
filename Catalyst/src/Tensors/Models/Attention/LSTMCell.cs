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
    public class LSTMCell 
    {
        public IWeightMatrix Wxh { get; set; }

        public IWeightMatrix b { get; set; }

        public IWeightMatrix ht { get; set; }
        public IWeightMatrix ct { get; set; }

        public int hdim { get; set; }
        public int dim { get; set; }

        private int batchSize;
        private int deviceId;

        private LayerNormalization layerNorm1;
        private LayerNormalization layerNorm2;

        public LSTMCell(int batchSize, int hdim, int dim, ArchTypeEnums archType, int deviceId)
        {
            if (archType == ArchTypeEnums.GPU_CUDA)
            {
                Wxh = new WeightTensor(dim + hdim, hdim * 4, deviceId, true);
                b = new WeightTensor(1, hdim * 4, 0, deviceId);
            }
            else
            {
                Wxh = new WeightMatrix(dim + hdim, hdim * 4, true);
                b = new WeightMatrix(1, hdim * 4, 0);
            }

            this.hdim = hdim;
            this.dim = dim;
            this.batchSize = batchSize;
            this.deviceId = deviceId;

            layerNorm1 = new LayerNormalization(hdim * 4, archType, deviceId);
            layerNorm2 = new LayerNormalization(hdim, archType, deviceId);
        }

        public IWeightMatrix Step(IWeightMatrix input, IComputeGraph innerGraph)
        {
            var hidden_prev = ht;
            var cell_prev = ct;
        
            var inputs = innerGraph.ConcatColumns(input, hidden_prev);
            var bs = innerGraph.RepeatRows(b, input.Rows);
            var hhSum = innerGraph.MulAdd(inputs, Wxh, bs);
            var hhSum2 = layerNorm1.Process(hhSum, innerGraph);

            (var gates_raw, var cell_write_raw) = innerGraph.SplitColumns(hhSum2, hdim * 3, hdim);
            var gates = innerGraph.Sigmoid(gates_raw);
            var cell_write = innerGraph.Tanh(cell_write_raw);

            (var input_gate, var forget_gate, var output_gate) = innerGraph.SplitColumns(gates, hdim, hdim, hdim);

            // compute new cell activation: ct = forget_gate * cell_prev + input_gate * cell_write
            ct = innerGraph.EltMulMulAdd(forget_gate, cell_prev, input_gate, cell_write);
            var ct2 = layerNorm2.Process(ct, innerGraph);

            // compute hidden state as gated, saturated cell activations
            ht = innerGraph.EltMul(output_gate, innerGraph.Tanh(ct2));

            return ht;
        }

        public virtual List<IWeightMatrix> getParams()
        {
            List<IWeightMatrix> response = new List<IWeightMatrix>();
            response.Add(Wxh);
            response.Add(b);

            response.AddRange(layerNorm1.getParams());
            response.AddRange(layerNorm2.getParams());

            return response;
        }

        public void SetBatchSize(IWeightFactory weightFactory, int batchSize)
        {
            this.batchSize = batchSize;
            Reset(weightFactory);
        }

        public void Reset(IWeightFactory weightFactory)
        {
            ht = weightFactory.CreateWeights(batchSize, hdim, deviceId, true);
            ct = weightFactory.CreateWeights(batchSize, hdim, deviceId, true);
        }

        public void Save(Stream stream)
        {
            Wxh.Save(stream);
            b.Save(stream);

            layerNorm1.Save(stream);
            layerNorm2.Save(stream);

        }


        public void Load(Stream stream)
        {
            Wxh.Load(stream);
            b.Load(stream);

            layerNorm1.Load(stream);
            layerNorm2.Load(stream);
        }
    }
     
}
