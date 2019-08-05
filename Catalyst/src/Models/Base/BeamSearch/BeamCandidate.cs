// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

namespace Catalyst.Models
{
    internal class BeamCandidate<T>
    {
        internal T State;
        internal BeamAction Action;
        internal float Score;

        public BeamCandidate(T state, BeamAction action, float score)
        {
            State = state; Action = action; Score = score;
        }

        public BeamCandidate(BeamState<T> state)
        {
            State = state.State; Action = null; Score = state.Score;
        }
    }
}