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

        [IgnoreMember] public TimeSpan EstimatedRemainingTime => Progress > 0 ? TimeSpan.FromSeconds((1 - Progress) * ElapsedTime.TotalSeconds / Progress) : TimeSpan.Zero;

        public TrainingUpdate At(float epoch, float maxEpoch, float loss)
        {
            Epoch = epoch;
            if(maxEpoch > 0)
            {
                Progress = epoch / maxEpoch;
            }
            else
            {
                Progress = epoch;
            }
            Loss = loss;
            return this;
        }

        public TrainingUpdate Processed(float items, TimeSpan elapsed)
        {
            if(elapsed.TotalSeconds > 0)
            {
                ItemsPerSecond = items / (float)elapsed.TotalSeconds;
            }
            else
            {
                ItemsPerSecond = 0;
            }
            ElapsedTime = elapsed;
            return this;
        }
    }
}