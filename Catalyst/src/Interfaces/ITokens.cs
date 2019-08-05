using System.Collections.Generic;

namespace Catalyst
{
    public interface ITokens : IToken
    {
        EntityType EntityType { get; }
        IEnumerable<IToken> Children { get; }
    }
}