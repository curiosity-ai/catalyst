// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

using MessagePack;
using Mosaik.Core;

namespace Catalyst
{
    public struct TokenVector
    {
        [Key(0)] public string Token { get; set; }
        [Key(1)] public float[] Vector { get; set; }
        [Key(2)] public int Hash { get; set; }
        [Key(3)] public PartOfSpeech POS;
        [Key(4)] public Language Language;
        [Key(5)] public float Frequency;

        public TokenVector(string token, float[] vector, int hash, PartOfSpeech pos, Language language, float frequency) : this()
        {
            Token = token;
            Vector = vector;
            Hash = hash;
            POS = pos;
            Language = language;
            Frequency = frequency;
        }
    }
}