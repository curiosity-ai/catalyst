using UID;
using System;
using System.Collections.Generic;

//using MessagePack;

namespace Catalyst
{
    public struct Token : IToken
    {
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
        }

        private int _begin;
        private int _end;
        private string _value;
        private string _replacement;
        private readonly Document Parent;
        private readonly int _index;
        private readonly int SpanIndex;
        private bool _hasReplacement;

        public int Begin { get { if (_begin < 0) { _begin = Parent.TokensData[SpanIndex][_index].LowerBound; } return _begin; } set { throw new InvalidOperationException(); } }

        public int End { get { if (_end < 0) { _end = Parent.TokensData[SpanIndex][_index].UpperBound; } return _end; } set { throw new InvalidOperationException(); } }

        public int Length { get { return End - Begin + 1; } }

        public int Index { get { return _index; } }

        public string Value { get { if (_hasReplacement) { return Replacement; } if (_value is null) { _value = Parent.GetTokenValue(_index, SpanIndex); } return _value; } }

        public ReadOnlySpan<char> ValueAsSpan { get { if (_hasReplacement) { return Replacement.AsSpan(); } return Parent.GetTokenValueAsSpan(_index, SpanIndex); } }

        public int Hash { get { if (_hasReplacement) { return Replacement.CaseSensitiveHash32(); } return Parent.GetTokenHash(_index, SpanIndex); } set { throw new NotImplementedException(); } }

        public int IgnoreCaseHash { get { if (_hasReplacement) { return Replacement.IgnoreCaseHash32(); } return Parent.GetTokenIgnoreCaseHash(_index, SpanIndex); } set { throw new NotImplementedException(); } }

        public PartOfSpeech POS { get { return Parent.GetTokenTag(_index, SpanIndex); } set { Parent.SetTokenTag(_index, SpanIndex, value); } }

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

        public override string ToString()
        {
            return $"[{Begin}->{End}]" + Value;
        }
    }

    public struct SpecialToken : IToken
    {
        public static readonly string BOS = "<BOS>";
        public static readonly string EOS = "<EOS>";
        public static readonly string ROOT = "<ROOT>";
        public static readonly string NULL = "<NULL>";

        public SpecialToken(string value, PartOfSpeech tag)
        {
            Value = value;
            POS = tag;
        }

        public static IToken BeginToken { get; private set; } = new SpecialToken(SpecialToken.BOS, PartOfSpeech.NONE);
        public static IToken EndToken { get; private set; } = new SpecialToken(SpecialToken.EOS, PartOfSpeech.NONE);
        public static IToken RootToken { get; private set; } = new SpecialToken(SpecialToken.ROOT, PartOfSpeech.NONE);
        public static IToken NullToken { get; private set; } = new SpecialToken(SpecialToken.NULL, PartOfSpeech.NONE);

        public int Begin { get => -1; set => throw new NotImplementedException(); }
        public int End { get => -1; set => throw new NotImplementedException(); }

        public int Length => 0;
        public int Index => -1;

        public string Value { get; private set; }
        public ReadOnlySpan<char> ValueAsSpan { get => Value.AsSpan(); }

        public string Stem { get => throw new NotImplementedException(); set => throw new NotImplementedException(); }
        public int Hash { get => Value.CaseSensitiveHash32(); set => throw new NotImplementedException(); }
        public int IgnoreCaseHash { get => Value.IgnoreCaseHash32(); set => throw new NotImplementedException(); }

        public Dictionary<string, string> Metadata => throw new NotImplementedException();

        public PartOfSpeech POS { get; set; }

        public EntityType[] EntityTypes => throw new NotImplementedException();

        public string Replacement { get => Value; set => throw new NotImplementedException(); }

        public int Head { get { return -1; } set { } }
        public string DependencyType { get { return ""; } set { } }

        public float Frequency { get { return 0; } set { return; } }

        public void AddEntityType(EntityType entityType) => throw new NotImplementedException();

        public void RemoveEntityType(string entityType) => throw new NotImplementedException();

        public void RemoveEntityType(int ix) => throw new NotImplementedException();

        public void UpdateEntityType(int ix, ref EntityType entityType) => throw new NotImplementedException();

        public void ClearEntities() => throw new NotImplementedException();
    }
}