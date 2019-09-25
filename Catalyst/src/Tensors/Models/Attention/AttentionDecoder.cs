
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
    public class AttentionDecoder
    {
        public List<LSTMAttentionDecoderCell> decoders = new List<LSTMAttentionDecoderCell>();
        public int hdim { get; set; }
        public int dim { get; set; }
        public int depth { get; set; }
        public AttentionUnit attentionLayer { get; set; }

        public AttentionDecoder(int batchSize, int hdim, int dim, int context, int depth, ArchTypeEnums archType, int deviceId)
        {
            attentionLayer = new AttentionUnit(batchSize, hdim, context, archType, deviceId);
            this.hdim = hdim;
            this.dim = dim;
            this.depth = depth;

            decoders.Add(new LSTMAttentionDecoderCell(batchSize, hdim, dim, archType, deviceId));
            for (int i = 1; i < depth; i++)
            {
                decoders.Add(new LSTMAttentionDecoderCell(batchSize, hdim, hdim, archType, deviceId));
            }
        }

        public void SetBatchSize(IWeightFactory weightFactory, int batchSize)
        {
            attentionLayer.SetBatchSize(batchSize);
            foreach (var item in decoders)
            {
                item.SetBatchSize(weightFactory, batchSize);
            }
        }

        public void Reset(IWeightFactory weightFactory)
        {
            foreach (var item in decoders)
            {
                item.Reset(weightFactory);
            }

        }

        public AttentionPreProcessResult PreProcess(IWeightMatrix encoderOutput, IComputeGraph g)
        {
            return attentionLayer.PreProcess(encoderOutput, g);
        }


        public IWeightMatrix Decode(IWeightMatrix input, AttentionPreProcessResult attenPreProcessResult, IComputeGraph g)
        {
            var V = input;
            var lastStatus = decoders.LastOrDefault().ct;
            var context = attentionLayer.Perform(lastStatus, attenPreProcessResult, g);

            foreach (var decoder in decoders)
            {
                var e = decoder.Step(context, V, g);
                V = e;
            }

            return V;
        }

        public List<IWeightMatrix> GetCTs()
        {
            List<IWeightMatrix> res = new List<IWeightMatrix>();
            foreach (var decoder in decoders)
            {
                res.Add(decoder.ct);
            }

            return res;
        }

        public List<IWeightMatrix> GetHTs()
        {
            List<IWeightMatrix> res = new List<IWeightMatrix>();
            foreach (var decoder in decoders)
            {
                res.Add(decoder.ht);
            }

            return res;
        }

        public void SetCTs(List<IWeightMatrix> l)
        {
            for (int i = 0; i < l.Count; i++)
            {
                decoders[i].ct = l[i];
            }
        }

        public void SetHTs(List<IWeightMatrix> l)
        {
            for (int i = 0; i < l.Count; i++)
            {
                decoders[i].ht = l[i];
            }
        }

        public List<IWeightMatrix> getParams()
        {
            List<IWeightMatrix> response = new List<IWeightMatrix>();

            foreach (var item in decoders)
            {
                response.AddRange(item.getParams());
            }
            response.AddRange(attentionLayer.getParams());

            return response;
        }

        public void Save(Stream stream)
        {
            attentionLayer.Save(stream);
            foreach (var item in decoders)
            {
                item.Save(stream);
            }
        }

        public void Load(Stream stream)
        {
            attentionLayer.Load(stream);
            foreach (var item in decoders)
            {
                item.Load(stream);
            }
        }
    }
}


//using Catalyst.Tensors.Models.Tools;
//using System;
//using System.Collections.Generic;
//using System.IO;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;
//using Catalyst.Tensors;

//namespace Catalyst.Tensors.Models
//{


//    [Serializable]
//    public class AttentionDecoder
//    {
//        public List<LSTMAttentionDecoderCell> decoders = new List<LSTMAttentionDecoderCell>();
//        public int hdim { get; set; }
//        public int dim { get; set; }
//        public int depth { get; set; }
//        public AttentionUnit attentionLayer { get; set; }

//        public AttentionDecoder(int hdim, int dim, int depth, bool isGPU)
//        {
//            attentionLayer = new AttentionUnit(hdim, isGPU);
//            this.hdim = hdim;
//            this.dim = dim;
//            this.depth = depth;

//            decoders.Add(new LSTMAttentionDecoderCell(hdim, dim, isGPU));
//            for (int i = 1; i < depth; i++)
//            {
//                decoders.Add(new LSTMAttentionDecoderCell(hdim, hdim, isGPU));
//            }
//        }
//        public void Reset(IWeightFactory weightFactory)
//        {
//            foreach (var item in decoders)
//            {
//                item.Reset(weightFactory);
//            }

//        }

//        public void PreProcess(IWeightMatrix encoderOutput, IComputeGraph g)
//        {
//            attentionLayer.PreProcess(encoderOutput, g);
//        }


//        public IWeightMatrix Decode(IWeightMatrix input, IWeightMatrix encoderOutput, IComputeGraph g)
//        {
//            var V = input;
//            var lastStatus = this.decoders.FirstOrDefault().ct;
//            var context = attentionLayer.Perform(encoderOutput, lastStatus, g);

//            foreach (var decoder in decoders)
//            {
//                var e = decoder.Step(context, V, g);
//                V = e;
//            }

//            return V;
//        }

//        public List<IWeightMatrix> getParams()
//        {
//            List<IWeightMatrix> response = new List<IWeightMatrix>();

//            foreach (var item in decoders)
//            {
//                response.AddRange(item.getParams());
//            }
//            response.AddRange(attentionLayer.getParams());

//            return response;
//        }

//        public void Save(Stream stream)
//        {
//            attentionLayer.Save(stream);
//            foreach (var item in decoders)
//            {
//                item.Save(stream);
//            }
//        }

//        public void Load(Stream stream)
//        {
//            attentionLayer.Load(stream);
//            foreach (var item in decoders)
//            {
//                item.Load(stream);
//            }
//        }
//    }
//}
