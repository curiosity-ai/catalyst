using UID;
using System;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;

//using MessagePack;

namespace Catalyst
{
    public struct Token : IToken
    {
        public static readonly string BOS = "<BOS>";
        public static readonly string EOS = "<EOS>";
        public static readonly string ROOT = "<ROOT>";
        public static readonly string NULL = "<NULL>";

        private Token(string value, PartOfSpeech tag)
        {
            _value = value;
            _posOverride = tag;
            _begin = -1;
            _end = -1;
            _replacement = value;
            Parent = null;
            _index = -1;
            SpanIndex = -1;
            _hasReplacement = true;
        }

        internal bool IsBeginToken => _posOverride.HasValue && _value == Token.BOS;
        internal bool IsEndToken => _posOverride.HasValue && _value == Token.EOS;

        public static readonly Token BeginToken = new Token(Token.BOS, PartOfSpeech.NONE);
        public static readonly Token EndToken   = new Token(Token.EOS, PartOfSpeech.NONE);
        public static readonly Token RootToken  = new Token(Token.ROOT, PartOfSpeech.NONE);
        public static readonly Token NullToken  = new Token(Token.NULL, PartOfSpeech.NONE);

        public Token(Document parent, int index, int spanIndex, bool hasReplacement, int lowerBound, int upperBound)
        {
            Parent = parent;
            _index = index;
            SpanIndex = spanIndex;

            if (Parent is object)
            {
                _begin = lowerBound; _end = upperBound;
            }
            else
            {
                _begin = -1; _end = -1;
            }

            _replacement = null;
            _hasReplacement = hasReplacement;
            _value = null;
            _posOverride = null;
        }

        private int _begin;
        private int _end;
        private string _value;
        private string _replacement;
        private Document Parent;
        private int _index;
        private int SpanIndex;
        private bool _hasReplacement;
        private PartOfSpeech? _posOverride;

        public int Begin { get { if (_begin < 0) { _begin = Parent.TokensData[SpanIndex][_index].LowerBound; } return _begin; } set { throw new InvalidOperationException(); } }

        public int End { get { if (_end < 0) { _end = Parent.TokensData[SpanIndex][_index].UpperBound; } return _end; } set { throw new InvalidOperationException(); } }

        public int Length { get { return End - Begin + 1; } }

        public int Index { get { return _index; } }

        public string Value { get { if (_hasReplacement) { return Replacement; } if (_value is null) { _value = Parent.GetTokenValue(_index, SpanIndex); } return _value; } }

        public ReadOnlySpan<char> ValueAsSpan { get { if (_hasReplacement) { return Replacement.AsSpan(); } return Parent.GetTokenValueAsSpan(_index, SpanIndex); } }

        public int Hash { get { if (_hasReplacement) { return Replacement.CaseSensitiveHash32(); } return Parent.GetTokenHash(_index, SpanIndex); } set { throw new NotImplementedException(); } }

        public int IgnoreCaseHash { get { if (_hasReplacement) { return Replacement.IgnoreCaseHash32(); } return Parent.GetTokenIgnoreCaseHash(_index, SpanIndex); } set { throw new NotImplementedException(); } }

        public PartOfSpeech POS { get { return _posOverride ?? Parent.GetTokenTag(_index, SpanIndex); } set { if (_posOverride.HasValue) { throw new InvalidOperationException("Can't write when overloaded"); }  else { Parent.SetTokenTag(_index, SpanIndex, value); } } }

        public EntityType[] EntityTypes { get { return Parent.GetTokenEntityTypes(_index, SpanIndex); } }

        public void AddEntityType(EntityType entityType) => Parent.AddEntityTypeToToken(_index, SpanIndex, entityType);

        public void UpdateEntityType(int ix, ref EntityType entityType) => Parent.UpdateEntityTypeFromToken(_index, SpanIndex, ix, ref entityType);

        public void RemoveEntityType(int ix) => Parent.RemoveEntityTypeFromToken(_index, SpanIndex, ix);

        public void RemoveEntityType(string entityType) => Parent.RemoveEntityTypeFromToken(_index, SpanIndex, entityType);

        public void ClearEntities() => Parent.ClearEntityTypesFromToken(_index, SpanIndex);

        public Dictionary<string, string> Metadata { get { return Parent.GetTokenMetadata(_index, SpanIndex); } }

        public string Replacement { get { if (_hasReplacement && _replacement is null) { _replacement = Parent.GetTokenReplacement(_index, SpanIndex); } return _replacement; } set { Parent.SetTokenReplacement(_index, SpanIndex, value); _replacement = value; _hasReplacement = true; } }

        public int Head { get { return Parent.GetTokenHead(SpanIndex, _index); } set { Parent.SetTokenHead(SpanIndex, _index, value); } }

        public string DependencyType { get { return Parent.GetTokenDependencyType(SpanIndex, _index); } set { Parent.SetTokenDependencyType(SpanIndex, _index, value); } }

        public float Frequency { get { return Parent.GetTokenFrequency(SpanIndex, _index); } set { Parent.SetTokenFrequency(SpanIndex, _index, value); } }

        public string Lemma => LemmatizerStore.Get(Parent.Language).GetLemma(this);

        public ReadOnlySpan<char> LemmaAsSpan => LemmatizerStore.Get(Parent.Language).GetLemmaAsSpan(this);

        public override string ToString()
        {
            return $"[{Begin}->{End}]" + Value;
        }

        internal static Token Fake(string value)
        {
            return new Token(value, PartOfSpeech.X);
        }
    }
}