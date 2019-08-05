// Copyright (c) Curiosity GmbH - All Rights Reserved. Proprietary and Confidential.
// Unauthorized copying of this file, via any medium is strictly prohibited.

using MessagePack;

namespace Catalyst
{
    [MessagePackObject]
    public struct TokenData
    {
        [IgnoreMember] public int LowerBound;
        [IgnoreMember] public int UpperBound;

        [Key(0)] public int[] Bounds { get { return new int[2] { LowerBound, UpperBound }; } }
        [Key(1)] public PartOfSpeech Tag;
        [Key(2)] public int Hash;
        [Key(3)] public int IgnoreCaseHash;
        [Key(4)] public int Head;
        [Key(5)] public float Frequency;
        [Key(6)] public string DependencyType;
        [Key(7)] public string Replacement;

        [SerializationConstructor]
        public TokenData(int[] bounds, PartOfSpeech tag, int hash, int ignoreCaseHash, int head, float frequency, string dependencyType, string replacement)
        {
            LowerBound = bounds[0];
            UpperBound = bounds[1];
            Tag = tag;
            Replacement = replacement;
            Hash = hash;
            IgnoreCaseHash = ignoreCaseHash;
            Head = head;
            DependencyType = dependencyType;
            Frequency = frequency;
        }

        public TokenData(int lowerBound, int upperBound, PartOfSpeech tag, int hash, int ignoreCaseHash, int head, float frequency, string dependencyType, string replacement)
        {
            LowerBound = lowerBound;
            UpperBound = upperBound;
            Tag = tag;
            Replacement = replacement;
            Hash = hash;
            IgnoreCaseHash = ignoreCaseHash;
            Head = head;
            DependencyType = dependencyType;
            Frequency = frequency;
        }

        public TokenData(int[] bounds)
        {
            LowerBound = bounds[0];
            UpperBound = bounds[1];
            Tag = PartOfSpeech.NONE;
            Replacement = null;
            Hash = 0;
            IgnoreCaseHash = 0;
            Head = -1;
            DependencyType = null;
            Frequency = 0f;
        }
    }
}