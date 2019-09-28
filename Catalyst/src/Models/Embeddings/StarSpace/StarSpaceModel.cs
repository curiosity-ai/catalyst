using Mosaik.Core;
using System;
using System.Collections.Generic;
using System.Threading;

namespace Catalyst.Models
{
    public class StarSpaceModel : StorableObjectData
    {
        public StarSpace.ModelType Type { get; set; }
        public int ContextWindow { get; set; }
        public bool InvariantCase { get; set; }
        public long Epoch { get; set; } = 5;
        public bool IsTrained { get; set; }
        public ThreadPriority ThreadPriority { get; internal set; } = ThreadPriority.Normal;
        public Dictionary<uint, int> EntryHashToIndex { get; set; } = new Dictionary<uint, int>();
        public Dictionary<int, FastText.Entry> Entries { get; set; } = new Dictionary<int, FastText.Entry>();
        public int MinimumCount { get; set; } = 1;
        public int EntryCount { get; set; }
        public int ThreadCount { get; set; } = Environment.ProcessorCount;
        public bool TrainWordEmbeddings { get; set; }
        public int BatchSize { get; set; } = 5;
        public StarSpace.LossType LossType { get; set; } = StarSpace.LossType.Hinge;
        public int NegativeSamplingSearchLimit { get; set; } = 50;
        public float WordWeight { get; set; } = 0.5f;
        public StarSpace.SimilarityType Similarity { get; set; } = StarSpace.SimilarityType.Cosine;
        public float Margin { get; set; } = 0.05f;
        public float LearningRate { get; set; } = 0.01f;
        public int Dimensions { get; set; } = 128;
        public int MaximumNegativeSamples { get; set; } = 10;
        public bool AdaGrad { get; set; } = true;
        public double P { get; set; } = 0.5;
        public float InitializationStandardDeviation { get; set; } = 0.001f;
        public bool ShareEmbeddings { get; set; } = true;
        public int Buckets { get; internal set; } = 2_000_000;
        public int WordNGrams { get; set; } = 1;
        public StarSpace.InputType InputType { get; set; }

        public QuantizationType VectorQuantization { get; set; } = QuantizationType.None;

        public TrainingHistory TrainingHistory { get; set; }
    }
}