using UID;
using MessagePack;
using Mosaik.Core;
using System;
using System.Collections.Generic;

namespace Catalyst
{
    /// <summary>
    /// Represents a single token that is not part of a document.
    /// </summary>
    [MessagePackObject(AllowPrivate = true)]
    public class SingleToken : IToken
    {
        /// <inheritdoc />
        [Key(0)] public string Value { get; set; }
        /// <inheritdoc />
        [Key(1)] public int Hash { get; set; }
        /// <inheritdoc />
        [Key(2)] public int IgnoreCaseHash { get; set; }
        /// <inheritdoc />
        [Key(3)] public Dictionary<string, string> Metadata { get; set; }
        /// <inheritdoc />
        [Key(4)] public IReadOnlyList<EntityType> EntityTypes { get; set; }
        /// <inheritdoc />
        [Key(5)] public int Length => Value.Length;
        /// <inheritdoc />
        [Key(6)] public PartOfSpeech POS { get; set; }

        /// <summary>Gets or sets the language of the token.</summary>
        [Key(7)] public Language Language { get; set; }

        /// <inheritdoc />
        [Key(8)] public string Lemma { get; set; }

        /// <inheritdoc />
        [IgnoreMember] public string OriginalValue => Value;

        /// <inheritdoc />
        [IgnoreMember] public int Begin { get; set; }
        /// <inheritdoc />
        [IgnoreMember] public int End { get; set; }
        /// <inheritdoc />
        [IgnoreMember] public ReadOnlySpan<char> ValueAsSpan => Value.AsSpan();
        /// <inheritdoc />
        [IgnoreMember] public ReadOnlySpan<char> OriginalValueAsSpan => Value.AsSpan();
        /// <inheritdoc />
        [IgnoreMember] public int Index => throw new NotImplementedException();
        /// <inheritdoc />
        [IgnoreMember] public string Replacement { get; set; }

        /// <inheritdoc />
        [IgnoreMember] public int Head { get; set; }
        /// <inheritdoc />
        [IgnoreMember] public string DependencyType { get; set; }
        /// <inheritdoc />
        [IgnoreMember] public float Frequency { get; set; }
        /// <inheritdoc />
        [IgnoreMember] public ReadOnlySpan<char> LemmaAsSpan => Lemma.AsSpan();

        /// <inheritdoc />
        [IgnoreMember] public char? PreviousChar => null;
        /// <inheritdoc />
        [IgnoreMember] public char? NextChar => null;

        internal SingleToken()
        {

        }

        /// <summary>
        /// Initializes a new instance of the <see cref="SingleToken"/> class from an existing token.
        /// </summary>
        /// <param name="source">The source token.</param>
        /// <param name="language">The language.</param>
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

        /// <summary>
        /// Initializes a new instance of the <see cref="SingleToken"/> class with a value and language.
        /// </summary>
        /// <param name="value">The token value.</param>
        /// <param name="language">The language.</param>
        public SingleToken(string value, Language language)
        {
            Value = value;
            Hash = value.CaseSensitiveHash32();
            IgnoreCaseHash = value.IgnoreCaseHash32();
            POS = PartOfSpeech.NONE;
            Language = language;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="SingleToken"/> class with full details.
        /// </summary>
        /// <param name="value">The token value.</param>
        /// <param name="pos">The part-of-speech tag.</param>
        /// <param name="hash">The hash.</param>
        /// <param name="ignoreCaseHash">The case-insensitive hash.</param>
        /// <param name="language">The language.</param>
        /// <param name="lemma">The lemma.</param>
        public SingleToken(string value, PartOfSpeech pos, int hash, int ignoreCaseHash, Language language, string lemma)
        {
            Value = value;
            Hash = hash;
            IgnoreCaseHash = ignoreCaseHash;
            POS = pos;
            Language = language;
            Lemma = lemma;
        }

        /// <inheritdoc />
        public void AddEntityType(EntityType entityType) => throw new NotImplementedException();

        /// <inheritdoc />
        public void UpdateEntityType(int ix, ref EntityType entityType) => throw new NotImplementedException();

        /// <inheritdoc />
        public void RemoveEntityType(string entityType) => throw new NotImplementedException();

        /// <inheritdoc />
        public void RemoveEntityType(int ix) => throw new NotImplementedException();

        /// <inheritdoc />
        public void ClearEntities() => throw new NotImplementedException();
    }
}