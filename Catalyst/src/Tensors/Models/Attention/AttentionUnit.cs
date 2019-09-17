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

    public class AttentionPreProcessResult
    {
        public IWeightMatrix uhs;
        public IWeightMatrix inputs;

    }

    [Serializable]
    public class AttentionUnit
    {

        public IWeightMatrix V { get; set; }
        public IWeightMatrix Ua { get; set; }
        public IWeightMatrix bUa { get; set; }
        public IWeightMatrix Wa { get; set; }
        public IWeightMatrix bWa { get; set; }

        int m_batchSize;

        public AttentionUnit(int batchSize, int size, int context, ArchTypeEnums archType, int deviceId)
        {
            m_batchSize = batchSize;

            if (archType == ArchTypeEnums.GPU_CUDA)
            {
                this.Ua = new WeightTensor(context, size, deviceId, true);
                this.Wa = new WeightTensor(size, size, deviceId, true);
                this.bUa = new WeightTensor(1, size, 0, deviceId);
                this.bWa = new WeightTensor(1, size, 0, deviceId);
                this.V = new WeightTensor(size, 1, deviceId, true);
            }
            else
            {
                this.Ua = new WeightMatrix((size * 2), size, true);
                this.Wa = new WeightMatrix(size, size, true);
                this.bUa = new WeightMatrix(1, size, 0);
                this.bWa = new WeightMatrix(1, size, 0);
                this.V = new WeightMatrix(size, 1, true);
            }
        }



        public AttentionPreProcessResult PreProcess(IWeightMatrix inputs, IComputeGraph g)
        {
            AttentionPreProcessResult r = new AttentionPreProcessResult();

            IWeightMatrix bUas = g.RepeatRows(bUa, inputs.Rows);
            r.uhs = g.MulAdd(inputs, Ua, bUas);
            r.inputs = g.PermuteBatch(inputs, m_batchSize);

            return r;
        }

      

        public IWeightMatrix Perform(IWeightMatrix state, AttentionPreProcessResult attenPreProcessResult, IComputeGraph g)
        {
            var bWas = g.RepeatRows(bWa, state.Rows);
            var wc = g.MulAdd(state, Wa, bWas);
            var wcs = g.RepeatRows(wc, attenPreProcessResult.inputs.Rows / m_batchSize);
            var ggs = g.AddTanh(attenPreProcessResult.uhs, wcs);
            var atten = g.Mul(ggs, V);

            var atten2 = g.PermuteBatch(atten, m_batchSize);
            var attenT = g.Transpose2(atten2);
            var attenT2 = g.View(attenT, m_batchSize, attenPreProcessResult.inputs.Rows / m_batchSize);

            var attenSoftmax = g.Softmax(attenT2);

            IWeightMatrix contexts = g.MulBatch(attenSoftmax, attenPreProcessResult.inputs, m_batchSize);


            return contexts;
        }

      

        public virtual List<IWeightMatrix> getParams()
        {
            List<IWeightMatrix> response = new List<IWeightMatrix>();

            response.Add(Ua);
            response.Add(Wa);
            response.Add(bUa);
            response.Add(bWa);
            response.Add(V);

            return response;
        }

        public void SetBatchSize(int batchSize)
        {
            m_batchSize = batchSize;
        }

        public void Save(Stream stream)
        {
            Ua.Save(stream);
            Wa.Save(stream);
            bUa.Save(stream);
            bWa.Save(stream);
            V.Save(stream);
        }


        public void Load(Stream stream)
        {
            Ua.Load(stream);
            Wa.Load(stream);
            bUa.Load(stream);
            bWa.Load(stream);
            V.Load(stream);
        }
    }
}



