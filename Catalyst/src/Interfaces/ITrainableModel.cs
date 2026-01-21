using System;

namespace Catalyst
{
    /// <summary>
    /// Defines an interface for a model that can be trained.
    /// </summary>
    public interface ITrainableModel
    {
        /// <summary>
        /// Gets the training history of the model.
        /// </summary>
        TrainingHistory TrainingHistory { get; }
    }
}