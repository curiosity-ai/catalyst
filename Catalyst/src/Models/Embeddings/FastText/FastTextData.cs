using System;
using System.Collections.Generic;
using System.Threading;
using Mosaik.Core;

namespace Catalyst.Models
{
    [FormerName("Mosaik.NLU.Models", "VectorizerModel")]
    public class FastTextData : StorableObjectData
    {
        public int Dimensions = 200;
        public uint Buckets = 2_000_000;
        public int ContextWindow = 5;
        public int MinimumCount = 5;
        public int MinimumNgrams = 3;
        public int MaximumNgrams = 6;
        public int MaximumWordNgrams = 1;
        public int MinimumWordNgramsCounts = 100;
        public bool StoreTrainingData = false;
        public double ReusePreviousCorpusFactor;

        public bool CBowUseWordNgrams = false;

        public int Epoch = 5;
        public float LearningRate = 0.05f;
        public long LearningRateUpdateRate = 100;
        public int Threads = Environment.ProcessorCount;
        public int NegativeSamplingCount = 10;
        public double SamplingThreshold = 1e-4;

        public FastText.ModelType Type;
        public FastText.LossType Loss;
        public QuantizationType VectorQuantization;

        #region IgnoreCaseFix

        // This fixes the mistake made in the naming of this variable (invariant case != ignore case).
        // As we cannot rename here (due to the serialization using keyAsPropertyName:true), we add a second property
        // that refers to the same underlying variable. As MessagePack reads properties in the order of GetProperties,
        // this ensures the new one (IgnoreCase) is set before the old one (InvariantCase), so we don't the stored value
        private bool ignoreCase = true;

        public bool IgnoreCase { get { return ignoreCase; } set { ignoreCase = value; } }

        [Obsolete("Wrong property name, use IgnoreCase instead", true)]
        public bool InvariantCase { get { return ignoreCase; } set { ignoreCase = value; } }

        #endregion IgnoreCaseFix

        public int EntryCount = 0;
        public int LabelCount = 0;
        public int SubwordCount = 0;

        public bool IsTrained = false;

        public Dictionary<int, FastText.Entry> Entries;
        public Dictionary<int, FastText.Entry> Labels;
        public Dictionary<uint, int> EntryHashToIndex;
        public Dictionary<uint, int> LabelHashToIndex;
        public Dictionary<uint, int> SubwordHashToIndex;

        public ThreadPriority ThreadPriority = ThreadPriority.Normal;

        public TrainingHistory TrainingHistory;

        //public Dictionary<Language, int> LanguageOffset;
        //public Dictionary<int, int> TranslateTable;
    }
}