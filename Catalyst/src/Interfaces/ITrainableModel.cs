using System;

namespace Catalyst
{
    public interface ITrainableModel
    {
        TrainingHistory TrainingHistory { get; }
    }
}