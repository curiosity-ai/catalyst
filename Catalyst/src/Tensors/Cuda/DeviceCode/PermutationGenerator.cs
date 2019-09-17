using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst.Tensors.CUDA.DeviceCode
{
    public class PermutationGenerator
    {
        public readonly StringBuilder sb = new StringBuilder();

        public PermutationGenerator()
        {
        }

        public override string ToString()
        {
            return sb.ToString();
        }
        
        public void AddApplyT(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(1))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                sb.AppendFormat("APPLY_T({0}, {1}, {2}, {3})\n", indexType, dimsA, kernelName, operatorCode);
            }
        }

        public void AddApplyTT(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(2))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                sb.AppendFormat("APPLY_TT({0}, {1}, {2}, {3}, {4})\n", indexType, dimsA, dimsB, kernelName, operatorCode);
            }
        }

        public void AddApplyTTT(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(3))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                var dimsC = spec.TensorDims[2].ToString();
                sb.AppendFormat("APPLY_TTT({0}, {1}, {2}, {3}, {4}, {5})\n", indexType, dimsA, dimsB, dimsC, kernelName, operatorCode);
            }
        }

        public void AddApplyTTTT(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(4))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                var dimsC = spec.TensorDims[2].ToString();
                var dimsD = spec.TensorDims[3].ToString();
                sb.AppendFormat("APPLY_TTTT({0}, {1}, {2}, {3}, {4}, {5}, {6})\n", indexType, dimsA, dimsB, dimsC, dimsD, kernelName, operatorCode);
            }
        }

        public void AddApplyTTTTT(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(5))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                var dimsC = spec.TensorDims[2].ToString();
                var dimsD = spec.TensorDims[3].ToString();
                var dimsE = spec.TensorDims[4].ToString();
                sb.AppendFormat("APPLY_TTTTT({0}, {1}, {2}, {3}, {4}, {5}, {6}, {7})\n", indexType, dimsA, dimsB, dimsC, dimsD, dimsE, kernelName, operatorCode);
            }
        }
        public void AddApplyTS(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(1))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                sb.AppendFormat("APPLY_TS({0}, {1}, {2}, {3})\n", indexType, dimsA, kernelName, operatorCode);
            }
        }

        public void AddApplyTSS(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(1))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                sb.AppendFormat("APPLY_TSS({0}, {1}, {2}, {3})\n", indexType, dimsA, kernelName, operatorCode);
            }
        }

        public void AddApplyTTS(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(2))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                sb.AppendFormat("APPLY_TTS({0}, {1}, {2}, {3}, {4})\n", indexType, dimsA, dimsB, kernelName, operatorCode);
            }
        }

        public void AddApplyTTSS(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(2))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                sb.AppendFormat("APPLY_TTSS({0}, {1}, {2}, {3}, {4})\n", indexType, dimsA, dimsB, kernelName, operatorCode);
            }
        }

        public void AddApplyTTTS(string kernelBaseName, string operatorCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(3))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                var dimsC = spec.TensorDims[2].ToString();
                sb.AppendFormat("APPLY_TTTS({0}, {1}, {2}, {3}, {4}, {5})\n", indexType, dimsA, dimsB, dimsC, kernelName, operatorCode);
            }
        }

        //public void AddApplyTTTSS(string kernelBaseName, string operatorCode)
        //{
        //    foreach (var spec in ApplySpecialization.AllSpecializations(3))
        //    {
        //        var kernelName = GetMangledName(kernelBaseName, spec);
        //        var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
        //        var dimsA = spec.TensorDims[0].ToString();
        //        var dimsB = spec.TensorDims[1].ToString();
        //        var dimsC = spec.TensorDims[2].ToString();
        //        sb.AppendFormat("APPLY_TTTSS({0}, {1}, {2}, {3}, {4}, {5})\n", indexType, dimsA, dimsB, dimsC, kernelName, operatorCode);
        //    }
        //}

        //public void AddApplyTTTTSS(string kernelBaseName, string operatorCode)
        //{
        //    foreach (var spec in ApplySpecialization.AllSpecializations(4))
        //    {
        //        var kernelName = GetMangledName(kernelBaseName, spec);
        //        var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
        //        var dimsA = spec.TensorDims[0].ToString();
        //        var dimsB = spec.TensorDims[1].ToString();
        //        var dimsC = spec.TensorDims[2].ToString();
        //        var dimsD = spec.TensorDims[3].ToString();
        //        sb.AppendFormat("APPLY_TTTTS({0}, {1}, {2}, {3}, {4}, {5}, {6})\n", indexType, dimsA, dimsB, dimsC, dimsD, kernelName, operatorCode);
        //    }
        //}

        //public void AddApplyTTTTS(string kernelBaseName, string operatorCode)
        //{
        //    foreach (var spec in ApplySpecialization.AllSpecializations(4))
        //    {
        //        var kernelName = GetMangledName(kernelBaseName, spec);
        //        var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
        //        var dimsA = spec.TensorDims[0].ToString();
        //        var dimsB = spec.TensorDims[1].ToString();
        //        var dimsC = spec.TensorDims[2].ToString();
        //        var dimsD = spec.TensorDims[3].ToString();
        //        sb.AppendFormat("APPLY_TTTTS({0}, {1}, {2}, {3}, {4}, {5}, {6})\n", indexType, dimsA, dimsB, dimsC, dimsD, kernelName, operatorCode);
        //    }
        //}


        public void AddReduce(string kernelBaseName, string modifyOpCode, string reduceOpCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(2))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                sb.AppendFormat("REDUCE_KERNELS({0}, {1}, {2}, {3}, {4}, {5})\n", indexType, dimsA, dimsB, kernelName, modifyOpCode, reduceOpCode);
            }
        }

        public void AddReduceNorm(string kernelBaseName)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(2))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                var dimsB = spec.TensorDims[1].ToString();
                sb.AppendFormat("REDUCE_NORM_KERNELS({0}, {1}, {2}, {3})\n", indexType, dimsA, dimsB, kernelName);
            }
        }

        public void AddReduceAll(string kernelBaseName, string modifyOpCode, string reduceOpCode)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(1))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                sb.AppendFormat("REDUCE_ALL_KERNELS({0}, {1}, {2}, {3}, {4})\n", indexType, dimsA, kernelName, modifyOpCode, reduceOpCode);
            }
        }

        public void AddReduceAllNorm(string kernelBaseName)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(1))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                sb.AppendFormat("REDUCE_ALL_NORM_KERNELS({0}, {1}, {2})\n", indexType, dimsA, kernelName);
            }
        }

        public void AddReduceAllSubSquare(string kernelBaseName)
        {
            foreach (var spec in ApplySpecialization.AllSpecializations(1))
            {
                var kernelName = GetMangledName(kernelBaseName, spec);
                var indexType = spec.Use32BitIndices ? ApplySpecialization.IndexType32 : ApplySpecialization.IndexType64;
                var dimsA = spec.TensorDims[0].ToString();
                sb.AppendFormat("REDUCE_ALL_SUB_SQUARE_KERNELS({0}, {1}, {2})\n", indexType, dimsA, kernelName);
            }
        }


        // TODO make member of ApplySpecialization
        public static string GetMangledName(string baseName, ApplySpecialization spec)
        {
            var sb = new StringBuilder();

            sb.Append(baseName);
            sb.Append(spec.Use32BitIndices ? "__int32" : "__int64");
            foreach (var dimSize in spec.TensorDims)
            {
                sb.Append("_").Append(dimSize.ToString().Replace('-', 'M'));
            }
            return sb.ToString();
        }
    }
}
