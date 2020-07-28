using System.Collections.Generic;
using System.Linq;

namespace Catalyst
{
    public class MatchingPatternPrototype : IMatchingPattern
    {
        public List<PatternUnitPrototype[]> Patterns { get; set; } = new List<PatternUnitPrototype[]>();
        public string Name { get; set; }

        public MatchingPatternPrototype(string name)
        {
            Name = name;
        }

        public IMatchingPattern Add(params IPatternUnit[] units)
        {
            Patterns.Add(units.Select(u => (PatternUnitPrototype)u).ToArray());
            return this;
        }
    }
}
