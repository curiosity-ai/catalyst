using System.Collections.Generic;

namespace Catalyst
{
    /// <summary>
    /// Defines an interface for a group of tokens, typically representing an entity.
    /// </summary>
    public interface ITokens : IToken
    {
        /// <summary>
        /// Gets the entity type associated with this group of tokens.
        /// </summary>
        EntityType EntityType { get; }

        /// <summary>
        /// Gets the child tokens in this group.
        /// </summary>
        IEnumerable<IToken> Children { get; }
    }
}