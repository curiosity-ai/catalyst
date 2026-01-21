using System;
using System.Collections.Generic;
using MessagePack;

namespace Catalyst
{
    /// <summary>
    /// Represents the training history of a model.
    /// </summary>
    [MessagePackObject(keyAsPropertyName:true)]
    public class TrainingHistory
    {
        /// <summary>
        /// Gets or sets the list of training updates.
        /// </summary>
        public List<TrainingUpdate> History { get; set; } = new List<TrainingUpdate>();

        /// <summary>
        /// Gets or sets the total elapsed time for training.
        /// </summary>
        public TimeSpan ElapsedTime { get; set; }

        /// <summary>
        /// Appends a training update to the history.
        /// </summary>
        /// <param name="update">The training update to append.</param>
        public void Append(TrainingUpdate update)
        {
            History.Add(update);
        }
    }

    /// <summary>
    /// Represents a single update during the training of a model.
    /// </summary>
    [MessagePackObject]
    public class TrainingUpdate : EventArgs
    {
        /// <summary>
        /// Gets or sets the overall progress of the training (0 to 1).
        /// </summary>
        [Key(0)] public float Progress { get; set; }

        /// <summary>
        /// Gets or sets the current epoch.
        /// </summary>
        [Key(1)] public float Epoch { get; set; }

        /// <summary>
        /// Gets or sets the loss value at the current update.
        /// </summary>
        [Key(2)] public float Loss { get; set; }

        /// <summary>
        /// Gets or sets the number of items processed per second.
        /// </summary>
        [Key(3)] public float ItemsPerSecond { get; set; }

        /// <summary>
        /// Gets or sets the elapsed time since the start of training.
        /// </summary>
        [Key(4)] public TimeSpan ElapsedTime { get; set; }

        /// <summary>
        /// Gets the estimated remaining time for the training.
        /// </summary>
        [IgnoreMember] public TimeSpan EstimatedRemainingTime => Progress > 0 ? TimeSpan.FromSeconds((1 - Progress) * ElapsedTime.TotalSeconds / Progress) : TimeSpan.Zero;

        /// <summary>
        /// Updates the training progress and loss at a specific epoch.
        /// </summary>
        /// <param name="epoch">The current epoch.</param>
        /// <param name="maxEpoch">The maximum number of epochs.</param>
        /// <param name="loss">The current loss value.</param>
        /// <returns>The updated <see cref="TrainingUpdate"/> instance.</returns>
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

        /// <summary>
        /// Updates the processing speed based on the number of items and elapsed time.
        /// </summary>
        /// <param name="items">The number of items processed.</param>
        /// <param name="elapsed">The elapsed time.</param>
        /// <returns>The updated <see cref="TrainingUpdate"/> instance.</returns>
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