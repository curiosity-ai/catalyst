using Mosaik.Core;
using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;

//using MessagePack;

namespace Catalyst
{
    public struct Span : ISpan
    {
        public Span(Document parent, int index)
        {
            Parent = parent;
            Index = index;
            _begin = Parent.SpanBounds[Index][0];
            _end = Parent.SpanBounds[Index][1];
            _value = "";
        }

        public IToken this[int key]
        {
            get
            {
                if (key >= 0 && key < TokensCount)
                {
                    var td = Parent.TokensData[Index][key];
                    return new Token(Parent, key, Index, hasReplacement: td.Replacement is object, td.LowerBound, td.UpperBound);
                }
                else { throw new Exception("Invalid token index"); }
            }
            set { throw new InvalidOperationException(); }
        }

        public Language Language { get { return Parent.Language; } }

        private int _begin;
        private int _end;
        private string _value;
        private Document Parent;
        private int Index;

        public int Begin { get { if (_begin < 0) { _begin = Parent.SpanBounds[Index][0]; } return _begin; } set { Parent.SpanBounds[Index][0] = value; _begin = value; } }
        public int End { get { if (_end < 0) { _end = Parent.SpanBounds[Index][1]; } return _end; } set { Parent.SpanBounds[Index][1] = value; _end = value; } }

        public int Length { get { return End - Begin + 1; } }

        public string Value { get { if (string.IsNullOrEmpty(_value)) { _value = Parent.GetSpanValue(Index); } return _value; } }
        public ReadOnlySpan<char> ValueAsSpan { get { return Parent.GetSpanValue2(Index); } }

        public int TokensCount { get { return Parent.TokensData[Index].Count; } }

        public IEnumerable<IToken> Tokens
        {
            get
            {
                var sd = Parent.TokensData[Index];
                int count = sd.Count;
                for (int i = 0; i < count; i++)
                {
                    var td = sd[i];
                    yield return new Token(Parent, i, Index, hasReplacement: td.Replacement is object, td.LowerBound, td.UpperBound);
                }
            }
        }

        public IToken AddToken(int begin, int end)
        {
            return Parent.AddToken(Index, begin, end);
        }

        //Used by the sentence detector
        public IToken AddToken(IToken token)
        {
            var newtoken = Parent.AddToken(Index, token.Begin, token.End);
            if (token.Replacement is object)
            {
                newtoken.Replacement = token.Replacement;
            }
            return token;
        }

        public void ReserveTokens(int expectedTokenCount)
        {
            Parent.ReserveTokens(Index, expectedTokenCount);
        }

        public IEnumerator<IToken> GetEnumerator()
        {
            return Tokens.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return Tokens.GetEnumerator();
        }

        public void SetTokenTag(int tokenIndex, PartOfSpeech tag)
        {
            Parent.SetTokenTag(tokenIndex, Index, tag);
        }

        [JsonIgnore]
        public string TokenizedValue
        {
            get
            {
                var sb = new StringBuilder(Length + TokensCount + 100);
                foreach (var token in this)
                {
                    if (token.EntityTypes.Any(et => et.Tag == EntityTag.Begin || et.Tag == EntityTag.Inside))
                    {
                        bool isHyphen = token.ValueAsSpan.IsHyphen();
                        bool isNormalToken = !isHyphen && !token.ValueAsSpan.IsSentencePunctuation();
                        if (!isNormalToken)
                        {
                            if (sb[sb.Length - 1] == '_')
                            {
                                sb.Length--; //if we have a punctuation or hyphen, and the previous token added a '_', remove it here
                            }
                        }
                        if (!isHyphen)
                        {
                            sb.Append(token.Value);
                        }
                        else
                        {
                            sb.Append("_");
                        }
                        if (isNormalToken) { sb.Append("_"); } //don't add _ when the token is already a hyphen
                    }
                    else
                    {
                        sb.Append(token.Value).Append(" ");
                    }
                }

                return Regex.Replace(sb.ToString(), @"\s+", " ").TrimEnd(); //Remove the last space added during the loop
            }
        }

        public IEnumerable<IToken> GetTokenized()
        {
            var tc = TokensCount;
            for (int i = 0; i < tc; i++)
            {
                var t = this[i];

                var entityTypes = t.EntityTypes;
                if (entityTypes.Any())
                {
                    bool foundEntity = false;
                    foreach (var et in entityTypes.Where(et => et.Tag == EntityTag.Begin || et.Tag == EntityTag.Single).OrderBy(et => (char)et.Tag).ThenBy(et => et.Type.StartsWith("_") ? 1 : -1)) //Sort entity types by Begin then Single, ignore rest
                    {
                        if (et.Tag == EntityTag.Single)
                        {
                            foundEntity = true;
                            yield return new Tokens(Parent, Index, new int[] { t.Index }, entityType: et) { Frequency = t.Frequency };
                            break;
                        }
                        else if (et.Tag == EntityTag.Begin)
                        {
                            var tokens = new List<int>(3) { t.Index };
                            bool foundEnd = false;
                            for (int j = t.Index + 1; j < TokensCount; j++)
                            {
                                var other = this[j];
                                var otherET = other.EntityTypes.Where(oet => (oet.Tag == EntityTag.Inside || oet.Tag == EntityTag.End) && (oet.Type == et.Type));

                                if (otherET.Any()) //TODO: why not .Single()
                                {
                                    tokens.Add(other.Index);
                                    if (otherET.First().Tag == EntityTag.End)
                                    {
                                        foundEnd = true;
                                        i += j - t.Index;
                                        break;
                                    }
                                }
                                else
                                {
                                    break; // not the same type anymore
                                }
                            }

                            if (foundEnd)
                            {
                                foundEntity = true;
                                yield return new Tokens(Parent, Index, tokens.ToArray(), entityType: et) { Frequency = t.Frequency };
                                break;
                            }
                        }
                    }
                    if (!foundEntity)
                    {
                        yield return t;
                    }
                }
                else
                {
                    yield return t;
                }
            }
        }

        public IEnumerable<ITokens> GetEntities(string filter = null)
        {
            int tokensCount = TokensCount;
            bool hasFilter = !string.IsNullOrWhiteSpace(filter);
            foreach (var token in this)
            {
                var entityTypes = token.EntityTypes;
                if (entityTypes.Length > 0)
                {
                    foreach (var et in entityTypes.OrderBy(et => (char)et.Tag))
                    {
                        if (hasFilter && et.Type != filter)
                        {
                            continue; // Skip unwanted entities
                        }

                        if (et.Tag == EntityTag.Single)
                        {
                            yield return new Tokens(Parent, Index, new int[] { token.Index }, entityType: et) { Frequency = token.Frequency };
                        }
                        else if (et.Tag == EntityTag.Begin)
                        {
                            var tokens = new List<int>(3) { token.Index };
                            bool foundEnd = false;
                            for (int j = token.Index + 1; j < tokensCount; j++)
                            {
                                var other = this[j];
                                var otherET = other.EntityTypes.Where(oet => (oet.Tag == EntityTag.Inside || oet.Tag == EntityTag.End) & (oet.Type == et.Type));

                                if (otherET.Any()) //TODO: why not .Single()
                                {
                                    tokens.Add(other.Index);
                                    if (otherET.First().Tag == EntityTag.End)
                                    {
                                        foundEnd = true;
                                        //Don't break, as there might be overlaping entities that will end after this token
                                    }
                                }
                                else
                                {
                                    break; // not the same type anymore
                                }
                            }

                            if (foundEnd)
                            {
                                yield return new Tokens(Parent, Index, tokens.ToArray(), entityType: et) { Frequency = token.Frequency };
                            }
                        }
                    }
                }
            }
        }

        public Span<Token> ToTokenSpan()
        {
            var tkc = TokensCount;
            var tokens = new Token[tkc];
            var sd = Parent.TokensData[Index];
            for (int i = 0; i < tkc; i++)
            {
                var td = sd[i];
                tokens[i] = new Token(Parent, i, Index, td.Replacement is object, td.LowerBound, td.UpperBound);
            }
            return tokens;
        }
    }
}