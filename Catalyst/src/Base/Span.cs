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

        /// <summary>
        /// Return the tokenized text. Entities will be returned as a single Tokens instance with the inner tokens as children. This method will always prefer to return the longest possible entity match.
        /// </summary>
        /// <returns></returns>
        public IEnumerable<IToken> GetTokenized()
        {
            var tokensCount = TokensCount; //Cache the property to avoid fetching the value on every iteration
            for (int i = 0; i < tokensCount; i++)
            {
                var token = this[i];

                var entityTypes = token.EntityTypes;
                if (entityTypes.Any())
                {
                    bool foundEntity = false;
                    foreach (var et in FilterAndOrderByBeginThenSingle(entityTypes))
                    {
                        if (et.Tag == EntityTag.Single)
                        {
                            foundEntity = true;
                            yield return new Tokens(Parent, Index, new int[] { token.Index }, entityType: et) { Frequency = token.Frequency };
                            break;
                        }
                        else if (et.Tag == EntityTag.Begin)
                        {
                            var entityEnd = FindEntityEnd(tokensCount, token.Index, token.Frequency, entityTypes);

                            if(entityEnd.index > token.Index)
                            {
                                i = entityEnd.index;
                                yield return new Tokens(Parent, Index, Enumerable.Range(token.Index, entityEnd.index - token.Index + 1).ToArray(), entityType: entityEnd.entityType) { Frequency = entityEnd.lowestTokenFrequency };
                                break;
                            }
                        }
                    }
                    if (!foundEntity)
                    {
                        yield return token;
                    }
                }
                else
                {
                    yield return token;
                }
            }
        }

        /// <summary>
        /// Return only tokens that have entities attached to them. This method will always prefer to return the longest possible entity match.
        /// </summary>
        /// <param name="filter">Optional function to decide which entities to return. Should return true if you want to return the entity, or false if you want to skip it.</param>
        /// <returns></returns>
        public IEnumerable<ITokens> GetEntities(Func<EntityType, bool> filter = null)
        {
            int tokensCount = TokensCount;
            bool hasFilter = filter is object;

            for (int i = 0; i < tokensCount; i++)
            {
                var token = this[i];
                var entityTypes = token.EntityTypes;
                if (entityTypes.Length > 0)
                {
                    foreach (var et in FilterAndOrderByBeginThenSingle(entityTypes))
                    {
                        if (hasFilter && !filter(et))
                        {
                            continue; // Skip unwanted entities
                        }

                        if (et.Tag == EntityTag.Single)
                        {
                            yield return new Tokens(Parent, Index, new int[] { token.Index }, entityType: et) { Frequency = token.Frequency };
                        }
                        else if (et.Tag == EntityTag.Begin)
                        {
                            var entityEnd = FindEntityEnd(tokensCount, token.Index, token.Frequency, entityTypes);

                            if (entityEnd.index > token.Index)
                            {
                                i = entityEnd.index;
                                yield return new Tokens(Parent, Index, Enumerable.Range(token.Index, entityEnd.index - token.Index + 1).ToArray(), entityType: entityEnd.entityType) { Frequency = entityEnd.lowestTokenFrequency };
                                break;
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Attempts to find the end of an entity marked by an initial token with EntityType.Tag == EntityTag.Begin. 
        /// Will return the first longest possible match for a given [Begin] token.
        /// </summary>
        private (int index, EntityType entityType, float lowestTokenFrequency) FindEntityEnd(int tokenCount, int currentIndex, float tokenFrequency, EntityType[] entityTypes)
        {
            EntityType longestEntityType = default;
            int finalIndex = -1;
            float finalFrequency = tokenFrequency;

            foreach (var beginEntityType in entityTypes.Where(et => et.Tag == EntityTag.Begin))
            {
                int possibleFinal = -1;
                float possibleFrequency = tokenFrequency;
                bool foundEnd = false;

                for (int j = currentIndex + 1; j < tokenCount; j++)
                {
                    var other = this[j];
                    var otherET = other.EntityTypes.Where(oet => (oet.Tag == EntityTag.Inside || oet.Tag == EntityTag.End) && oet.Type == beginEntityType.Type);
                    if (otherET.Any())
                    {
                        possibleFinal = j;
                        possibleFrequency = Math.Min(possibleFrequency, other.Frequency);
                        foundEnd |= otherET.Any(oet => oet.Tag == EntityTag.End);
                    }
                    else
                    {
                        break;
                    }
                }

                if (foundEnd)
                {
                    if (possibleFinal > finalIndex)
                    {
                        finalIndex = possibleFinal;
                        finalFrequency = possibleFrequency;
                        longestEntityType = beginEntityType;
                    }
                }
            }

            return (finalIndex, longestEntityType, finalFrequency);
        }

        private static IOrderedEnumerable<EntityType> FilterAndOrderByBeginThenSingle(EntityType[] entityTypes)
        {
            //This method ensures we first try to enumerate the longest entities (i.e. starting with Begin), followed by Single entities
            //Note: Entities with name starting with _ are ordered last - this is a convention we depend on other Curiosity code-base, and need to keep here till we refactor that code
            return entityTypes.Where(et => et.Tag == EntityTag.Begin || et.Tag == EntityTag.Single).OrderBy(et => (char)et.Tag).ThenBy(et => et.Type.StartsWith("_") ? 1 : -1);
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