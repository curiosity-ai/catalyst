using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Catalyst
{
    /// <summary>
    /// Represents a prototype for a matching pattern.
    /// </summary>
    public class MatchingPatternPrototype : IMatchingPattern
    {
        /// <summary>Gets or sets the list of pattern unit sequences.</summary>
        public List<PatternUnitPrototype[]> Patterns { get; set; } = new List<PatternUnitPrototype[]>();

        /// <summary>Gets or sets the name of the matching pattern prototype.</summary>
        public string Name { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="MatchingPatternPrototype"/> class.
        /// </summary>
        /// <param name="name">The name of the prototype.</param>
        public MatchingPatternPrototype(string name)
        {
            Name = name;
        }

        /// <summary>
        /// Adds a sequence of pattern units to the prototype.
        /// </summary>
        /// <param name="units">The pattern units to add.</param>
        /// <returns>The updated matching pattern prototype.</returns>
        public IMatchingPattern Add(params IPatternUnit[] units)
        {
            Patterns.Add(units.Select(u => (PatternUnitPrototype)u).ToArray());
            return this;
        }
    }
}
