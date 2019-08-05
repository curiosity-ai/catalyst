// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

using MessagePack;

namespace Catalyst
{
    public struct MostSimilar
    {
        [Key(0)] public TokenVector Token { get; set; }
        [Key(1)] public float Score { get; set; }

        public MostSimilar(TokenVector token, float score)
        {
            Token = token; Score = score;
        }
    }
}