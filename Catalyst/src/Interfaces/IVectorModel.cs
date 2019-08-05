// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

using System.Collections.Generic;
using System.Threading.Tasks;

namespace Catalyst
{
    public interface IVectorModel
    {
        IEnumerable<TokenVector> GetTokenVectors();

        TokenVector GetTokenVector(string token);

        IEnumerable<TokenVector> GetTokenVector(IEnumerable<string> tokens);

        IEnumerable<MostSimilar> GetMostSimilar(TokenVector token, int k);

        Task<IEnumerable<MostSimilar>> GetMostSimilarAsync(TokenVector token, int k);
    }
}