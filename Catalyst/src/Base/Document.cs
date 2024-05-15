using UID;
using MessagePack;
using Mosaik.Core;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

//using MessagePack;
using System.Text;
using System.Text.RegularExpressions;
using System.IO;

namespace Catalyst
{

    [JsonObject]
    [MessagePackObject]
    public class Document : IDocument
    {
        [Key(0)] public Language Language { get; set; }
        [Key(1)] public string Value { get; set; }
        [Key(2)] public List<List<TokenData>> TokensData { get; set; }
        [Key(3)] public List<int[]> SpanBounds { get; set; }
        [Key(4)] public Dictionary<string, string> Metadata { get; set; }
        [Key(5)] public UID128 UID { get; set; }
        [Key(6)] public List<string> Labels { get; set; }
        [Key(7)] public Dictionary<long, List<EntityType>> EntityData { get; set; }
        [Key(8)] public Dictionary<long, Dictionary<string, string>> TokenMetadata { get; set; }

        [IgnoreMember] public int Length { get { return Value.Length; } }
        [IgnoreMember] public bool IsParsed { get { return SpansCount > 0 && TokensCount  > 0; } }

        [JsonIgnore] [IgnoreMember] public int SpansCount { get { return SpanBounds.Count; } }

        [JsonIgnore]
        [IgnoreMember]
        public int TokensCount
        {
            get
            {
                int count = 0;
                for (int i = 0; i < TokensData.Count; i++)
                {
                    count += TokensData[i].Count;
                }
                return count;
            }
        }

        public Document()
        {
            TokensData = new List<List<TokenData>>();
            SpanBounds = new List<int[]>();
            Metadata = new Dictionary<string, string>();
            Language = Language.Unknown;
            Labels = new List<string>();
            EntityData = new Dictionary<long, List<EntityType>>();
            TokenMetadata = new Dictionary<long, Dictionary<string, string>>();
        }

        public Document(string doc, Language language = Language.Unknown) : this()
        {
            Value = string.IsNullOrWhiteSpace(doc) ? "" : doc.RemoveControlCharacters();
            Language = language;
        }

        public Document(Document doc)
        {
            Language = doc.Language;
            Value = doc.Value;
            TokensData = doc.TokensData.Select(tds => tds.Select(td => new TokenData(td.LowerBound, td.UpperBound, td.Tag, td.Hash, td.IgnoreCaseHash, td.Head, td.Frequency, td.DependencyType, td.Replacement)).ToList()).ToList();
            SpanBounds = doc.SpanBounds.Select(sb => sb.ToArray()).ToList();
            Metadata = doc.Metadata?.ToDictionary(kv => kv.Key, kv => kv.Value);
            UID = doc.UID;
            Labels = doc.Labels?.ToList();
            EntityData = doc.EntityData?.ToDictionary(kv => kv.Key, kv => kv.Value.Select(et => new EntityType(et.Type, et.Tag)).ToList());
            TokenMetadata = doc.TokenMetadata?.ToDictionary(kv => kv.Key, kv => kv.Value.ToDictionary(kv2 => kv2.Key, kv2 => kv2.Value));
        }

        [SerializationConstructor]
        public Document(Language language, string value, List<List<TokenData>> tokensData, List<int[]> spanBounds, Dictionary<string, string> metadata, UID128 uid, List<string> labels, Dictionary<long, List<EntityType>> entityData, Dictionary<long, Dictionary<string, string>> tokenMetadata)
        {
            Language = language;
            Value = value;
            TokensData = tokensData;
            SpanBounds = spanBounds;
            Metadata = (metadata is null || metadata.Count == 0) ? null : metadata;
            UID = uid;
            Labels = labels;
            EntityData = entityData;
            TokenMetadata = (tokenMetadata is null || tokenMetadata.Count == 0) ? null : tokenMetadata;
        }

        public Document Clone()
        {
            return new Document(this);
        }

        public void Clear()
        {
            SpanBounds.Clear();
            TokensData.Clear();
        }

        internal IToken AddToken(int spanIndex, int begin, int end)
        {
            if (end < begin)
            {
                throw new InvalidOperationException();
            }
            var sd = TokensData[spanIndex];
            int index = sd.Count;

            sd.Add(new TokenData(begin, end));

            return new Token(this, index, spanIndex, hasReplacement: false, begin, end);
        }

        internal void ReserveTokens(int spanIndex, int expectedTokenCount)
        {
            var sd = TokensData[spanIndex];
            sd.Capacity = Math.Max(1, Math.Max(sd.Capacity, expectedTokenCount));
        }

        public List<IToken> ToTokenList()
        {
            var list = new List<IToken>(TokensCount);
            foreach (var s in this)
            {
                list.AddRange(s);
            }
            return list;
        }

        internal int GetTokenHead(int spanIndex, int tokenIndex)
        {
            return TokensData[spanIndex][tokenIndex].Head;
        }

        internal void SetTokenHead(int spanIndex, int tokenIndex, int head)
        {
            var spanData = TokensData[spanIndex];
            var tmp = spanData [tokenIndex];
            tmp.Head = head;
            spanData[tokenIndex] = tmp;
        }

        internal string GetTokenDependencyType(int spanIndex, int tokenIndex)
        {
            return TokensData[spanIndex][tokenIndex].DependencyType;
        }

        internal void SetTokenDependencyType(int spanIndex, int tokenIndex, string type)
        {
            var spanData = TokensData[spanIndex];
            var tmp = spanData[tokenIndex];
            tmp.DependencyType = type;
            spanData[tokenIndex] = tmp;
        }

        internal float GetTokenFrequency(int spanIndex, int tokenIndex)
        {
            return TokensData[spanIndex][tokenIndex].Frequency;
        }

        internal void SetTokenFrequency(int spanIndex, int tokenIndex, float frequency)
        {
            var tmp = TokensData[spanIndex][tokenIndex];
            tmp.Frequency = frequency;
            TokensData[spanIndex][tokenIndex] = tmp;
        }

        internal IEnumerable<IToken> GetTokenDependencies(int spanIndex, int tokenIndex)
        {
            throw new NotImplementedException();
            //var arcs = DependencyArcs[spanIndex].Where(arc => arc.HeadIndex == tokenIndex);
            //foreach(var a in arcs )
            //{
            //    yield return new TokenSlim(this, a.DependencyIndex, spanIndex);
            //}
        }

        public Span AddSpan(int begin, int end)
        {
            SpanBounds.Add(new int[] { begin, end });
            TokensData.Add(new List<TokenData>());
            return new Span(this, SpanBounds.Count - 1);
        }

        public string TokenizedValue(bool mergeEntities = false)
        {
            var sb = new StringBuilder(Value.Length + TokensCount * 10 + 100);
            for (int i = 0; i < SpanBounds.Count(); i++)
            {
                foreach (var token in this[i])
                {
                    if (mergeEntities && token.EntityTypes.Any(et => et.Tag == EntityTag.Begin || et.Tag == EntityTag.Inside))
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
            }
            return Regex.Replace(sb.ToString(), @"\s+", " ").TrimEnd(); //Remove the last space added during the loop
        }

        [JsonIgnore]
        [IgnoreMember]
        public Span this[int key] { get { return Spans.ElementAt(key); } set { throw new InvalidOperationException(); } }

        [JsonIgnore]
        [IgnoreMember]
        public IEnumerable<Span> Spans
        {
            get
            {
                for (int i = 0; i < SpanBounds.Count; i++)
                {
                    yield return new Span(this, i);
                }
            }
        }

        [JsonIgnore] [IgnoreMember] public int EntitiesCount => EntityData is object ? EntityData.Values.Sum(l => l.Count) : 0;

        internal string GetSpanValue(int index)
        {
            var span = SpanBounds[index];
            return Value.Substring(span[0], span[1] - span[0] + 1);
        }

        internal ReadOnlySpan<char> GetSpanValue2(int index)
        {
            var span = SpanBounds[index];
            return Value.AsSpan().Slice(span[0], span[1] - span[0] + 1);
        }

        internal IReadOnlyList<EntityType> GetTokenEntityTypes(int tokenIndex, int spanIndex)
        {
            long ix = GetTokenIndex(spanIndex, tokenIndex);
            List<EntityType> entityList;
            if (EntityData is object && EntityData.TryGetValue(ix, out entityList))
            {
                return entityList;
            }
            else
            {
                return Array.Empty<EntityType>();
            }
        }

        internal void AddEntityTypeToToken(int tokenIndex, int spanIndex, EntityType entityType)
        {
            if (EntityData is null) { EntityData = new Dictionary<long, List<EntityType>>(); }

            long ix = GetTokenIndex(spanIndex, tokenIndex);
            List<EntityType> entityList;
            if (!EntityData.TryGetValue(ix, out entityList))
            {
                entityList = new List<EntityType>();
                EntityData.Add(ix, entityList);
            }
            entityList.Add(entityType);
        }

        internal void UpdateEntityTypeFromToken(int tokenIndex, int spanIndex, int entityIndex, ref EntityType entityType)
        {
            if (EntityData is null) { EntityData = new Dictionary<long, List<EntityType>>(); }
            long ix = GetTokenIndex(spanIndex, tokenIndex);
            List<EntityType> entityList;
            if (EntityData.TryGetValue(ix, out entityList))
            {
#if NET6_0_OR_GREATER
                // We use CollectionsMarshal.AsSpan directly so that we don't trigger increasing the _version field of the List, which would cause anything enumerating the list to throw.
                Span<EntityType> entityListAsSpan = System.Runtime.InteropServices.CollectionsMarshal.AsSpan(entityList);
                entityListAsSpan[entityIndex] = entityType;
#else
                entityList[entityIndex] = entityType;
#endif
            }
            else
            {
                throw new Exception("No entities to update");
            }
        }

        internal void RemoveEntityTypeFromToken(int tokenIndex, int spanIndex, int entityIndex)
        {
            long ix = GetTokenIndex(spanIndex, tokenIndex);
            List<EntityType> entityList;
            if (EntityData.TryGetValue(ix, out entityList))
            {
                entityList.RemoveAt(entityIndex);
                if (entityList.Count == 0) { EntityData.Remove(ix); }
            }
            else
            {
                throw new Exception("No entities to update");
            }
        }

        internal void RemoveEntityTypeFromToken(int tokenIndex, int spanIndex, string entityType)
        {
            if (EntityData is null) { return; } //nothing to do
            long ix = GetTokenIndex(spanIndex, tokenIndex);
            List<EntityType> entityList;
            if (EntityData.TryGetValue(ix, out entityList))
            {
                entityList.RemoveAll(et => et.Type == entityType);
                if (entityList.Count == 0) { EntityData.Remove(ix); }
            }
            else
            {
                throw new Exception("No entities to update");
            }
        }

        internal void ClearEntityTypesFromToken(int tokenIndex, int spanIndex)
        {
            if (EntityData is null) { return; } //nothing to do
            long ix = GetTokenIndex(spanIndex, tokenIndex);
            EntityData.Remove(ix);
        }

        private static long GetTokenIndex(int spanIndex, int tokenIndex)
        {
            return (long)spanIndex << 32 | (long)(uint)tokenIndex;
        }

        internal PartOfSpeech GetTokenTag(int tokenIndex, int spanIndex)
        {
            return TokensData[spanIndex][tokenIndex].Tag;
        }

        internal void SetTokenTag(int tokenIndex, int spanIndex, PartOfSpeech tag)
        {
            var spanData = TokensData[spanIndex];
            var tmp = spanData[tokenIndex];
            tmp.Tag = tag;
            spanData[tokenIndex] = tmp;
        }

        internal int GetTokenHash(int tokenIndex, int spanIndex)
        {
            var spanData = TokensData[spanIndex];
            var tmp = spanData[tokenIndex];

            if (tmp.Hash == 0) { tmp.Hash = GetTokenValueAsSpan(tokenIndex, spanIndex).CaseSensitiveHash32(); spanData[tokenIndex] = tmp; }

            return tmp.Hash;
        }

        internal int GetTokenIgnoreCaseHash(int tokenIndex, int spanIndex)
        {
            var spanData = TokensData[spanIndex];
            var tmp = spanData[tokenIndex];

            if (tmp.IgnoreCaseHash == 0) { tmp.IgnoreCaseHash = GetTokenValueAsSpan(tokenIndex, spanIndex).IgnoreCaseHash32(); spanData[tokenIndex] = tmp; }

            return tmp.IgnoreCaseHash;
        }

        public IEnumerator<Span> GetEnumerator()
        {
            return Spans.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return Spans.GetEnumerator();
        }

        internal string GetTokenReplacement(int tokenIndex, int spanIndex)
        {
            if (TokensData.Count == 0) { return null; } //If called before being tokenized, return no replacement - this was breaking SentenceDetector
            Debug.Assert(spanIndex >= 0 && spanIndex < TokensData.Count, $"Invalid span index {tokenIndex}");
            Debug.Assert(tokenIndex >= 0 && tokenIndex < TokensData[spanIndex].Count, $"Invalid token index {tokenIndex}");
            return TokensData[spanIndex][tokenIndex].Replacement;
        }

        internal void SetTokenReplacement(int tokenIndex, int spanIndex, string value)
        {
            var spanData = TokensData[spanIndex];
            var tmp = spanData[tokenIndex];
            tmp.Replacement = value;
            spanData[tokenIndex] = tmp;
        }

        internal bool TokenHasReplacement(int tokenIndex, int spanIndex)
        {
            return TokensData[spanIndex][tokenIndex].Replacement is object;
        }

        internal string GetTokenValue(int index, int spanIndex)
        {
            var td = TokensData[spanIndex][index];

            int b = td.LowerBound;
            int e = td.UpperBound;
            return Value.Substring(b, e - b + 1);
        }

        internal ReadOnlySpan<char> GetTokenValueAsSpan(int index, int spanIndex)
        {
            var td = TokensData[spanIndex][index];
            int b = td.LowerBound;
            int e = td.UpperBound;
            return Value.AsSpan(b, e - b + 1);
        }

        public string ToStringWithReplacements(Func<ITokens, string> replacement)
        {
            var sb = new StringBuilder();

            sb.Append(Value);

            foreach(var span in this.Reverse())
            {
                foreach(var entity in span.GetEntities().Reverse())
                {
                    var replaceWith = replacement(entity);
                    if (replaceWith != null)
                    {
                        sb.Remove(entity.Begin, entity.End - entity.Begin + 1);
                        sb.Insert(entity.Begin, replaceWith);
                    }
                }
            }

            return sb.ToString();
        }

        internal Dictionary<string, string> GetTokenMetadata(int tokenIndex, int spanIndex)
        {
            if (TokenMetadata is null) { TokenMetadata = new Dictionary<long, Dictionary<string, string>>(); }

            long ix = GetTokenIndex(spanIndex, tokenIndex);
            
            if (TokenMetadata.TryGetValue(ix, out var dict))
            {
                return dict;
            }
            else
            {
                dict = new Dictionary<string, string>();
                TokenMetadata.Add(ix, dict);
                return dict;
            }
            //TODO: REMOVE JsonIgnore from Token and from SingleToken when this is implemented
        }

        public void RemoveOverlapingTokens()
        {
            for (int i = 0; i < TokensData.Count; i++)
            {
                var tb = TokensData[i].OrderBy(t => t.LowerBound)
                                       .ThenByDescending(t => t.UpperBound)
                                       .GroupBy(t => t.LowerBound)
                                       .Select(g => g.First())
                                       .ToList();
                TokensData[i] = tb;
            }
        }

        public void TrimTokens()
        {
            for (int i = 0; i < TokensData.Count; i++)
            {
                var tokens = TokensData[i];
                for (int j = 0; j < tokens.Count; j++)
                {
                    var token = tokens[j];
                    int begin = token.LowerBound, end = token.UpperBound;
                    if (char.IsWhiteSpace(Value[begin]) || char.IsWhiteSpace(Value[end]))
                    {
                        while (char.IsWhiteSpace(Value[begin]) && begin < end) { begin++; }
                        while (char.IsWhiteSpace(Value[end]) && end > begin) { end--; }
                        token.LowerBound = begin;
                        token.UpperBound = end;

                        tokens[j] = token;
                    }
                }
            }
        }

        public string ToJson()
        {
            var sb = StringExtensions.StringBuilderPool.Rent();
            using(var tw = new StringWriter(sb))
            using (var jw = new JsonTextWriter(tw).AsIJsonWriter())
            {
                WriteAsJson(jw);
            }
            var json = sb.ToString();
            if(json.Length < 1024*1024) StringExtensions.StringBuilderPool.Return(sb); //let the StringBuilder be collected in case it grows too large
            return json;
        }

        public static IDocument FromJson(string json)
        {
            return FromJObject(JObject.Parse(json));
        }

        public void WriteAsJson(IJsonWriter jw)
        {
            jw.WriteStartObject();

            jw.WritePropertyName(nameof(Language)); jw.WriteStringValue(Languages.EnumToCode(Language));
            jw.WritePropertyName(nameof(Length)); jw.WriteNumberValue(Length);
            jw.WritePropertyName(nameof(Value)); jw.WriteStringValue(Value);

            if (UID.IsNotNull())
            {
                jw.WritePropertyName(nameof(UID)); jw.WriteStringValue(UID);
            }

            if (Metadata is object && Metadata.Any())
            {
                jw.WritePropertyName(nameof(Metadata));
                jw.WriteStartObject();
                foreach (var kv in Metadata)
                {
                    jw.WritePropertyName(kv.Key); jw.WriteStringValue(kv.Value);
                }
                jw.WriteEndObject();
            }

            if (Labels is object && Labels.Count > 0)
            {
                jw.WritePropertyName(nameof(Labels));
                jw.WriteStartArray();
                foreach (var l in Labels)
                {
                    jw.WriteStringValue(l);
                }
                jw.WriteEndArray();
            }

            jw.WritePropertyName(nameof(TokensData));
            jw.WriteStartArray();
            for (int i = 0; i < TokensData.Count; i++)
            {
                var spanData = TokensData[i];
                jw.WriteStartArray();
                for (int j = 0; j < spanData.Count; j++)
                {
                    var tk = spanData[j];
                    long ix = GetTokenIndex(i, j);

                    jw.WriteStartObject();
                    jw.WritePropertyName(nameof(TokenData.Bounds));
                    jw.WriteStartArray();
                    jw.WriteNumberValue(tk.LowerBound);
                    jw.WriteNumberValue(tk.UpperBound);
                    jw.WriteEndArray();

                    if (tk.Tag != PartOfSpeech.NONE)
                    {
                        jw.WritePropertyName(nameof(TokenData.Tag)); jw.WriteStringValue(tk.Tag.ToString());
                    }

                    if (tk.Head >= 0)
                    {
                        jw.WritePropertyName(nameof(TokenData.Head)); jw.WriteNumberValue(tk.Head);
                    }

                    if (tk.Frequency != 0)
                    {
                        jw.WritePropertyName(nameof(TokenData.Frequency)); jw.WriteNumberValue(tk.Frequency);
                    }

                    if (!string.IsNullOrEmpty(tk.Replacement))
                    {
                        jw.WritePropertyName(nameof(TokenData.Replacement)); jw.WriteStringValue(tk.Replacement);
                    }

                    if (TokenMetadata is object)
                    {
                        if (TokenMetadata.TryGetValue(ix, out var tokenMetadata))
                        {
                            if (!(tokenMetadata is null) && tokenMetadata.Any())
                            {
                                jw.WritePropertyName(nameof(Metadata));
                                jw.WriteStartObject();
                                foreach (var kv in tokenMetadata)
                                {
                                    jw.WritePropertyName(kv.Key); jw.WriteStringValue(kv.Value);
                                }
                                jw.WriteEndObject();
                            }
                        }
                    }

                    if (EntityData is object)
                    {
                        if (EntityData.TryGetValue(ix, out var entities))
                        {
                            if (!(entities is null) && entities.Any())
                            {
                                jw.WritePropertyName(nameof(EntityType));
                                jw.WriteStartArray();
                                for (int k = 0; k < entities.Count; k++)
                                {
                                    jw.WriteStartObject();
                                    jw.WritePropertyName(nameof(EntityType.Type)); jw.WriteStringValue(entities[k].Type);
                                    jw.WritePropertyName(nameof(EntityType.Tag)); jw.WriteStringValue(entities[k].Tag.ToString());

                                    if (entities[k].TargetUID.IsNotNull())
                                    {
                                        jw.WritePropertyName(nameof(EntityType.TargetUID)); jw.WriteStringValue(entities[k].TargetUID);
                                    }

                                    if (!(entities[k].Metadata is null) && entities[k].Metadata.Any())
                                    {
                                        jw.WritePropertyName(nameof(EntityType.Metadata));
                                        jw.WriteStartObject();
                                        foreach (var kv in entities[k].Metadata)
                                        {
                                            jw.WritePropertyName(kv.Key); jw.WriteStringValue(kv.Value);
                                        }
                                        jw.WriteEndObject();
                                    }

                                    jw.WriteEndObject();
                                }
                                jw.WriteEndArray();
                            }
                        }
                    }
                    jw.WriteEndObject();
                }
                jw.WriteEndArray();
            }
            jw.WriteEndArray();

            jw.WriteEndObject();
        }

        public static Document FromJObject(JObject jo)
        {
            var emptyEntityTypes = new List<EntityType>();

            var doc = new Document();
            doc.Language = Languages.CodeToEnum((string)jo[nameof(Language)]);
            doc.Value = (string)jo[nameof(Value)];
            doc.UID = UID128.TryParse((string)(jo[nameof(UID)]), out var uid) ? uid : default(UID128);

            var docmtd = jo[nameof(Metadata)];

            if (!(docmtd is null) && docmtd.HasValues)
            {
                doc.Metadata = new Dictionary<string, string>();
                foreach (JProperty md in docmtd)
                {
                    doc.Metadata.Add(md.Name, (string)md.Value);
                }
            }

            if (jo.ContainsKey(nameof(Labels)))
            {
                doc.Labels = jo[nameof(Labels)].Select(jt => (string)jt).ToList();
            }

            var td = jo[nameof(TokensData)];

            foreach (var sp in td)
            {
                var tokens = new List<(int begin, int end, PartOfSpeech tag, int head, float frequency, List<EntityType> entityType, IDictionary<string, string> metadata, string replacement)>();

                foreach (var tk in sp)
                {
                    var ets = tk[nameof(EntityType)];
                    var entityTypes = emptyEntityTypes;
                    if (!(ets is null) && ets.HasValues)
                    {
                        entityTypes = new List<EntityType>();
                        foreach (var et in ets)
                        {
                            Dictionary<string, string> entityMetadata = null;
                            var etmtd = et[nameof(Metadata)];
                            if (!(etmtd is null) && etmtd.HasValues)
                            {
                                entityMetadata = new Dictionary<string, string>();
                                foreach (JProperty md in etmtd)
                                {
                                    entityMetadata.Add(md.Name, (string)md.Value);
                                }
                            }

                            entityTypes.Add(new EntityType((string)(et[nameof(EntityType.Type)]),
                                                           (EntityTag)Enum.Parse(typeof(EntityTag), (string)(et[nameof(EntityType.Tag)])),
                                                           entityMetadata,
                                                           UID128.TryParse((string)(et[nameof(EntityType.TargetUID)]), out var uid2) ? uid2 : default(UID128)));
                        }
                    }

                    IDictionary<string, string> metadata = null;

                    var mtd = tk[nameof(Metadata)];
                    if (!(mtd is null) && mtd.HasValues)
                    {
                        metadata = new Dictionary<string, string>();
                        foreach (JProperty md in mtd)
                        {
                            metadata.Add(md.Name, (string)md.Value);
                        }
                    }

                    tokens.Add((((int)(tk[nameof(TokenData.Bounds)][0])),
                                ((int)(tk[nameof(TokenData.Bounds)][1])),
                                (PartOfSpeech)Enum.Parse(typeof(PartOfSpeech), (string)(tk[nameof(TokenData.Tag)] ?? nameof(PartOfSpeech.NONE))),
                                ((int)(tk[nameof(TokenData.Head)] ?? "-1")),
                                (((float)(tk[nameof(TokenData.Frequency)] ?? 0f))),
                                entityTypes,
                                metadata,
                                (string)tk[nameof(TokenData.Replacement)]));
                }

                if (tokens.Any())
                {
                    var span = doc.AddSpan(tokens.First().begin, tokens.Last().end);

                    foreach (var tk in tokens)
                    {
                        var token = span.AddToken(tk.begin, tk.end);
                        token.POS = tk.tag;
                        token.Head = tk.head;
                        token.Frequency = tk.frequency;
                        foreach (var et in tk.entityType)
                        {
                            token.AddEntityType(et);
                        }

                        if (tk.metadata is object)
                        {
                            foreach (var kv in tk.metadata)
                            {
                                token.Metadata.Add(kv.Key, kv.Value);
                            }
                        }
                    }
                }
            }

            return doc;
        }

        public static Document FromImmutable(ImmutableDocument imDoc)
        {
            var tokensData = imDoc.TokensData.Select(a => a.ToList()).ToList();
            var spanBounds = imDoc.SpanBounds.ToList();
            var metadata = imDoc.Metadata?.ToDictionary(kv => kv.Key, kv => kv.Value);
            var labels = imDoc.Labels?.ToList();
            var entityData = imDoc.EntityData?.ToDictionary(kv => kv.Key, kv => kv.Value.ToList());
            var tokenMetadata = imDoc.TokenMetadata?.ToDictionary(kv => kv.Key, kv => kv.Value.ToDictionary(kv2 => kv2.Key, kv2 => kv2.Value));
            return new Document(imDoc.Language, imDoc.Value, tokensData, spanBounds.Select(l => new int[] { (int)(l >> 32), (int)(l & 0xFFFF_FFFFL) }).ToList(), metadata, imDoc.UID, labels, entityData, tokenMetadata);
        }

        public ImmutableDocument ToImmutable()
        {
            return new ImmutableDocument(Language, Value,
                                         TokensData.Select(td => td.ToArray()).ToArray(),
                                         SpanBounds.Select(sb => (long)sb[0] << 32 | (uint)sb[1]).ToArray(),
                                         Metadata?.ToDictionary(kv => kv.Key, kv => kv.Value),
                                         UID,
                                         Labels?.ToArray(),
                                         EntityData?.ToDictionary(kv => kv.Key, kv => kv.Value.ToArray()),
                                         TokenMetadata?.ToDictionary(kv => kv.Key, kv => kv.Value.ToDictionary(kv2 => kv2.Key, kv2 => kv2.Value))
                                        );
        }

        internal char? GetNextChar(int index, int spanIndex)
        {
            var td = TokensData[spanIndex][index];
            int e = td.UpperBound;
            if (e == Value.Length - 1) return null;
            return Value[e];
        }

        internal char? GetPreviousChar(int index, int spanIndex)
        {
            var td = TokensData[spanIndex][index];
            int b = td.LowerBound;
            if (b == 0) return null;
            return Value[b - 1];
        }
    }
}