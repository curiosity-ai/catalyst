using UID;
using MessagePack;
using Mosaik.Core;
using System;
using System.Collections.Generic;

namespace Catalyst
{
    [MessagePackObject]
    public class SingleToken : IToken
    {
        [Key(0)] public string Value { get; set; }
        [Key(1)] public int Hash { get; set; }
        [Key(2)] public int IgnoreCaseHash { get; set; }
        [Key(3)] public Dictionary<string, string> Metadata { get; set; }
        [Key(4)] public EntityType[] EntityTypes { get; set; }
        [Key(5)] public int Length => Value.Length;
        [Key(6)] public PartOfSpeech POS { get; set; }
        [Key(7)] public Language Language { get; set; }
        [Key(8)] public string Lemma { get; set; }

        [IgnoreMember] public int Begin { get; set; }
        [IgnoreMember] public int End { get; set; }
        [IgnoreMember] public ReadOnlySpan<char> ValueAsSpan => Value.AsSpan();
        [IgnoreMember] public int Index => throw new NotImplementedException();
        [IgnoreMember] public string Replacement { get; set; }

        [IgnoreMember] public int Head { get; set; }
        [IgnoreMember] public string DependencyType { get; set; }
        [IgnoreMember] public float Frequency { get; set; }
        [IgnoreMember] public ReadOnlySpan<char> LemmaAsSpan => Lemma.AsSpan();

        public SingleToken(IToken source, Language language)
        {
            Value = source.Value;
            Hash = source.Hash;
            IgnoreCaseHash = source.IgnoreCaseHash;
            //Metadata      = (source.Metadata is object && source.Metadata.Count > 0) ? source.Metadata.ToDictionary(kv => kv.Key, kv => kv.Value) : null;
            //EntityTypes = source.EntityTypes.ToArray();
            POS = source.POS;
            Language = language;
        }

        public SingleToken(string value, Language language)
        {
            Value = value;
            Hash = value.CaseSensitiveHash32();
            IgnoreCaseHash = value.IgnoreCaseHash32();
            POS = PartOfSpeech.NONE;
            Language = language;
        }

        public SingleToken(string value, PartOfSpeech pos, int hash, int ignoreCaseHash, Language language, string lemma)
        {
            Value = value;
            Hash = hash;
            IgnoreCaseHash = ignoreCaseHash;
            POS = pos;
            Language = language;
            Lemma = lemma;
        }

        public void AddEntityType(EntityType entityType) => throw new NotImplementedException();

        public void UpdateEntityType(int ix, ref EntityType entityType) => throw new NotImplementedException();

        public void RemoveEntityType(string entityType) => throw new NotImplementedException();

        public void RemoveEntityType(int ix) => throw new NotImplementedException();

        public void ClearEntities() => throw new NotImplementedException();
    }
}