using Microsoft.Extensions.Logging;
using Catalyst.Tensors.Models.Tools;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;
using Catalyst.Tensors;
using Catalyst.Tensors.CUDA;

namespace Catalyst.Tensors.Models
{
    public class Optimizer
    {
        public static float decay_rate = 0.999f;
        public static float smooth_eps = 1e-8f;
        public static float lr_decay_rate = 0.999f;

        public Vector<float> vecDecayRate = new Vector<float>(decay_rate);
        public Vector<float> vecSmoothEPS = new Vector<float>(smooth_eps);

        public float UpdateWeights(List<IWeightMatrix> model, int batchSize, float step_size, float regc, float clipval, ArchTypeEnums archType)
        {
            var vecMaxClipval = new Vector<float>(clipval);
            var vecMinClipval = new Vector<float>(-clipval);

            if (archType == ArchTypeEnums.GPU_CUDA)
            {
                UpdateWeightsTensors(model, batchSize, step_size, regc, clipval);
            }
            else
            {
                foreach (var m in model)
                {
                    UpdateWeightsCPU(step_size, regc, clipval, vecMaxClipval, vecMinClipval, m as WeightMatrix);
                }
            }

            return step_size;
        }

        private void UpdateWeightsTensors(List<IWeightMatrix> model, int batchSize, float step_size, float regc, float clipval)
        {
            Dictionary<int, List<IWeightMatrix>> id2Models = new Dictionary<int, List<IWeightMatrix>>();
            foreach (var item in model)
            {
                if (id2Models.ContainsKey(item.DeviceId) == false)
                {
                    id2Models.Add(item.DeviceId, new List<IWeightMatrix>());
                }
                id2Models[item.DeviceId].Add(item);
            }


            Parallel.ForEach(id2Models, kv => 
            {
                foreach (var item in kv.Value)
                {
                    var m = item as WeightTensor;

                    UpdateWeightsTensor(m, batchSize, step_size, clipval, regc);
                    m.RowToBeUpdated.Clear();
                }
            });
        }

        private void UpdateWeightsCPU(float step_size, float regc, float clipval, Vector<float> vecMaxClipval, Vector<float> vecMinClipval, WeightMatrix m)
        {
            if (m.RowToBeUpdated.Count == 0)
            {
                UpdateWeights(step_size, regc, clipval, m, vecMaxClipval, vecMinClipval, m.Weight.Length, 0);
            }
            else
            {
                foreach (var kv in m.RowToBeUpdated)
                {
                    int rowId = kv.Key;
                    UpdateWeights(step_size, regc, clipval, m, vecMaxClipval, vecMinClipval, m.Columns, rowId * m.Columns);
                }

                m.RowToBeUpdated.Clear();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateWeightsTensor(WeightTensor m, int batchSize, float step_size, float clipval, float regc)
        {
            Ops.RMSProp(m.TWeight, m.TGradient, m.TCache, batchSize, step_size, clipval, regc, decay_rate, smooth_eps);
        }


        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void UpdateWeights(float step_size, float regc, float clipval, WeightMatrix m, Vector<float> vecMaxClipval, Vector<float> vecMinClipval, int n, int i)
        {
            var s = m.Cache;
            var l = m.LrW;
            var vecBaseLR = new Vector<float>(step_size);

            var moreItems = (n % Vector<float>.Count);
            while (i < n - moreItems)
            {
                var vecMDWI = new Vector<float>(m.Gradient, i);

                vecMDWI = Vector.Min(vecMDWI, vecMaxClipval);
                vecMDWI = Vector.Max(vecMDWI, vecMinClipval);

                var vecS = new Vector<float>(s, i);
                vecS = vecS * vecDecayRate + (Vector<float>.One - vecDecayRate) * vecMDWI * vecMDWI;
                vecS.CopyTo(s, i);

                var vecMDWIDelta = vecMDWI / Vector.SquareRoot(vecS + vecSmoothEPS);
                var vecLRWeight = new Vector<float>(l, i);
                var vecLR = ComputeLearningRate(vecMDWIDelta, ref vecLRWeight, vecBaseLR);
                vecLRWeight.CopyTo(l, i);

                var vecMW = new Vector<float>(m.Weight, i);
                var vecDelta = -vecLR * vecMDWIDelta - regc * vecMW;

                vecMW += vecDelta;
                vecMW.CopyTo(m.Weight, i);

                i += Vector<float>.Count;
            }

            while (i < n)
            {
                // rmsprop adaptive learning rate
                var mdwi = m.Gradient[i];
                // gradient clip
                if (mdwi > clipval)
                {
                    mdwi = clipval;
                }
                if (mdwi < -clipval)
                {
                    mdwi = -clipval;
                }

                s[i] = (float)(s[i] * decay_rate + (1.0 - decay_rate) * mdwi * mdwi);

                var wDelta = (float)(mdwi / Math.Sqrt(s[i] + smooth_eps));
                var lr = ComputeLearningRate(wDelta, l, i, step_size);

                var delta = (float)(-lr * wDelta - regc * m.Weight[i]);

                // update (and regularize)
                m.Weight[i] += delta;


                i++;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float ComputeLearningRate(float delta, float[] m, int i, float baseLR)
        {
            var dg = m[i] + delta * delta;
            m[i] = dg;

            return (float)(baseLR / (1.0 + Math.Sqrt(dg)));
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static Vector<float> ComputeLearningRate(Vector<float> vecDelta, ref Vector<float> vecWeightLearningRate, Vector<float> vecBaseLR)
        {
            var dg = vecWeightLearningRate + vecDelta * vecDelta;
            vecWeightLearningRate = dg;

            return vecBaseLR / (Vector.SquareRoot(dg) + Vector<float>.One);

        }

        public void CleanCache(List<IWeightMatrix> model)
        {
            foreach (var k in model)
            {
                k.CleanCache();
            }
        }
    }
}
