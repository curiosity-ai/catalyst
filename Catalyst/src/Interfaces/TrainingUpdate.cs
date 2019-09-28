using System;
using System.Collections.Generic;
using MessagePack;

namespace Catalyst
{
    [MessagePackObject(keyAsPropertyName:true)]
    public class TrainingHistory
    {
        public List<TrainingUpdate> History { get; set; } = new List<TrainingUpdate>();
        public TimeSpan ElapsedTime { get; set; }

        public void Append(TrainingUpdate update)
        {
            History.Add(update);
        }
    }

    [MessagePackObject]
    public class TrainingUpdate : EventArgs
    {
        [Key(0)] public float Progress { get; set; }
        [Key(1)] public float Epoch { get; set; }
        [Key(2)] public float Loss { get; set; }
        [Key(3)] public float ItemsPerSecond { get; set; }
        [Key(4)] public TimeSpan ElapsedTime { get; set; }

        [IgnoreMember] public TimeSpan EstimatedRemainingTime => TimeSpan.FromSeconds((1 - Progress) * ElapsedTime.TotalSeconds / Progress);

        public TrainingUpdate At(float epoch, float maxEpoch, float loss)
        {
            Epoch = epoch;
            Progress = epoch / maxEpoch;
            Loss = loss;
            return this;
        }

        public TrainingUpdate Processed(float items, TimeSpan elapsed)
        {
            ItemsPerSecond = items / (float)elapsed.TotalSeconds;
            ElapsedTime = elapsed;
            return this;
        }
    }
}