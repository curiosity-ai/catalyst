namespace Catalyst
{
    /// <summary>
    /// Defines an interface for a matching pattern composed of one or more pattern units.
    /// </summary>
    public interface IMatchingPattern
    {
        /// <summary>
        /// Adds one or more pattern units to the matching pattern.
        /// </summary>
        /// <param name="units">The pattern units to add.</param>
        /// <returns>The updated matching pattern.</returns>
        IMatchingPattern Add(params IPatternUnit[] units);
    }
}