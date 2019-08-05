namespace Catalyst.Models
{
    internal class BeamState<T>
    {
        internal T State;
        internal float Score;

        public BeamState(T state, float score)
        {
            State = state; Score = score;
        }

        public BeamState(BeamCandidate<T> candidate)
        {
            State = candidate.State; Score = candidate.Score;
        }
    }
}