// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

using System.Collections.Generic;

namespace Catalyst
{
    public interface ITokens : IToken
    {
        EntityType EntityType { get; }
        IEnumerable<IToken> Children { get; }
    }
}