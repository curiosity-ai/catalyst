using System;

namespace Catalyst
{
    public interface ITrainableModel
    {
        event EventHandler<TrainingUpdate> TrainingStatus;
        TrainingHistory TrainingHistory { get; }
    }
}