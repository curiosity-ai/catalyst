namespace Catalyst
{
    /// <summary>
    /// Specifies the matching mode for a pattern unit.
    /// </summary>
    public enum PatternMatchingMode
    {
        /// <summary>The unit should not match.</summary>
        ShouldNotMatch,
        /// <summary>The unit should match exactly once.</summary>
        Single,
        /// <summary>The unit should match one or more times.</summary>
        Multiple,
        /// <summary>The unit should match using an AND condition (internal use).</summary>
        And,
        /// <summary>The unit should match using an OR condition (internal use).</summary>
        Or
    }
}
