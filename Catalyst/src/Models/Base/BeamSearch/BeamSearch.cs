using System;
using System.Collections.Generic;
using System.Linq;

namespace Catalyst.Models
{
    public class BeamSearch<T> where T : ICloneable
    {
        private readonly int Beams;

        private List<BeamState<T>> States = new List<BeamState<T>>();
        private List<BeamCandidate<T>> Candidates = new List<BeamCandidate<T>>();

        private readonly Func<T, IList<BeamAction>> Generator;
        private readonly Action<T, int> Mover;

        public bool Finished;
        public int BadEpochs;

        public BeamSearch(T initialState, int beams, Func<T, IList<BeamAction>> generator, Action<T, int> mover)
        {
            Beams = beams; Generator = generator; Mover = mover;
            States.Add(new BeamState<T>((T)initialState.Clone(), 0));
            Finished = false;
            BadEpochs = 0;
        }

        public void Search()
        {
            bool positiveActions = false;
            if (!Finished)
            {
                int actionsCount = 0;

                foreach (var s in States)
                {
                    var possibleActions = Generator(s.State);
                    if (possibleActions is object)
                    {
                        actionsCount += possibleActions.Count;

                        foreach (var pa in possibleActions)
                        {
                            if (pa.DeltaScore > 0) { positiveActions = true; }
                            Candidates.Add(new BeamCandidate<T>(s.State, pa, (s.Score + pa.DeltaScore)));
                        }
                    }
                    else
                    {
                        Finished = true;
                        return;
                    }
                }
                if (!positiveActions) { BadEpochs++; }
                Finished = (actionsCount == 0) || BadEpochs > 3;
            }
        }

        public void AdvanceBeam()
        {
            if (Finished) { return; }
            Candidates = Candidates.OrderByDescending(s => s.Score).ToList();

            for (int i = 0; i < Math.Min(Candidates.Count, Beams); i++) //Only need to process top K elements
            {
                var candidate = Candidates[i];
                if (candidate.Action is object)
                {
                    candidate.State = (T)candidate.State.Clone();
                    Mover(candidate.State, candidate.Action.Index);
                }
            }

            States = Candidates.Select(c => new BeamState<T>(c)).ToList();
            if (States.Count > Beams) { States.RemoveRange(Beams, States.Count - Beams); }
            Candidates.Clear();
        }

        public T GetResult()
        {
            return States.OrderByDescending(s => s.Score).First().State;
        }
    }
}