using MessagePack;
using Mosaik.Core;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;

namespace Catalyst
{
    /// <summary>
    /// A <see cref="Document"/> whose backing collections are rented from a <see cref="DocumentPool"/>
    /// instead of being allocated per document. It behaves exactly like a <see cref="Document"/> - the
    /// same <see cref="Span"/> / <see cref="IToken"/> API reads it, and any code taking a
    /// <see cref="Document"/> or an <see cref="IDocument"/> accepts one - but the caller owns its
    /// lifetime: once the document has been consumed (or serialized with
    /// <see cref="SerializeAsMessagePack(Stream, MessagePackSerializerOptions)"/>) it must go back to
    /// the pool through <see cref="Return"/>.
    /// </summary>
    /// <remarks>
    /// After <see cref="Return"/> the instance and everything reachable from it belong to the pool
    /// again, so reading a returned document - or a <see cref="Span"/>/<see cref="IToken"/> still
    /// pointing at it - reads whatever the next renter wrote. Keep something past the return by
    /// materializing it first: <see cref="Document.Clone"/> for a plain <see cref="Document"/>, or
    /// <see cref="Document.ToImmutable"/> for an <see cref="ImmutableDocument"/>.
    /// A pooled document is deliberately *not* part of the <c>IDocument</c> MessagePack union, so
    /// <c>MessagePackSerializer.Serialize&lt;IDocument&gt;(pooled)</c> is not supported - use
    /// <see cref="SerializeAsMessagePack(Stream, MessagePackSerializerOptions)"/>, which writes the
    /// same bytes the <see cref="Document"/> formatter would, without allocating on the way.
    /// </remarks>
    public sealed class PooledDocument : Document
    {
        private readonly DocumentPool          m_pool;
        private readonly TokenBoundsComparer   m_overlapComparer;

        internal PooledDocument(DocumentPool pool) : base(initializeCollections: false)
        {
            m_pool            = pool;
            m_overlapComparer = new TokenBoundsComparer();
        }

        /// <summary>Gets the pool this document was rented from.</summary>
        public DocumentPool Pool => m_pool;

        /// <summary>
        /// Returns this document and every collection it holds to the pool it came from. Equivalent to
        /// <c>Pool.Return(this)</c>.
        /// </summary>
        public void Return() => m_pool.Return(this);

        /// <summary>
        /// Reserves room for <paramref name="expectedSpanCount"/> spans, so a document deserialized from
        /// storage - where the span count is known up front - does not re-grow its two span lists on the way in.
        /// </summary>
        /// <param name="expectedSpanCount">The number of spans about to be added.</param>
        public void ReserveSpans(int expectedSpanCount)
        {
            if (expectedSpanCount <= 0) return;

            if (SpanBounds.Capacity < expectedSpanCount) { SpanBounds.Capacity = expectedSpanCount; }
            if (TokensData.Capacity < expectedSpanCount) { TokensData.Capacity = expectedSpanCount; }
        }

        /// <inheritdoc />
        /// <remarks>
        /// An empty list the pool handed back too small is swapped for one of the right size rather than
        /// grown - growing it allocates the array the pool exists to hand over.
        /// </remarks>
        public override void ReserveTokens(int spanIndex, int expectedTokenCount)
        {
            var current = TokensData[spanIndex];

            if (current.Count == 0 && current.Capacity < expectedTokenCount)
            {
                TokensData[spanIndex] = m_pool.RentTokenData(expectedTokenCount);
                m_pool.ReturnTokenData(current);
                return;
            }

            base.ReserveTokens(spanIndex, expectedTokenCount);
        }

        /// <inheritdoc />
        public override Span AddSpan(int begin, int end)
        {
            SpanBounds.Add(m_pool.RentSpanBounds(begin, end));
            TokensData.Add(m_pool.RentTokenData());

            return new Span(this, SpanBounds.Count - 1);
        }

        /// <inheritdoc />
        public override void Clear()
        {
            for (int i = 0; i < SpanBounds.Count; i++)
            {
                m_pool.ReturnSpanBounds(SpanBounds[i]);
            }

            for (int i = 0; i < TokensData.Count; i++)
            {
                m_pool.ReturnTokenData(TokensData[i]);
            }

            SpanBounds.Clear();
            TokensData.Clear();
        }

        /// <summary>
        /// Same result as <see cref="Document.RemoveOverlapingTokens"/> - one token per distinct lower
        /// bound, the longest one, earliest wins on a tie - reached with a rented index array and a
        /// rented replacement list so a re-tokenization costs no allocation.
        /// </summary>
        public override void RemoveOverlapingTokens()
        {
            for (int i = 0; i < TokensData.Count; i++)
            {
                var tokens = TokensData[i];
                int count  = tokens.Count;

                if (count < 2) continue;

                var order = ArrayPool<int>.Shared.Rent(count);

                try
                {
                    for (int j = 0; j < count; j++)
                    {
                        order[j] = j;
                    }

                    m_overlapComparer.Tokens = tokens;
                    Array.Sort(order, 0, count, m_overlapComparer);
                    m_overlapComparer.Tokens = null;

                    var kept         = m_pool.RentTokenData();
                    int lastKeptLow  = int.MinValue;

                    for (int j = 0; j < count; j++)
                    {
                        var token = tokens[order[j]];

                        if (j > 0 && token.LowerBound == lastKeptLow) continue;

                        lastKeptLow = token.LowerBound;
                        kept.Add(token);
                    }

                    m_pool.ReturnTokenData(tokens);
                    TokensData[i] = kept;
                }
                finally
                {
                    ArrayPool<int>.Shared.Return(order);
                }
            }
        }

        internal override void AddEntityTypeToToken(int tokenIndex, int spanIndex, EntityType entityType)
        {
            long ix = GetPooledTokenIndex(spanIndex, tokenIndex);

            if (!EntityData.TryGetValue(ix, out var entityList))
            {
                entityList = m_pool.RentEntityTypes();
                EntityData.Add(ix, entityList);
            }

            entityList.Add(entityType);
        }

        internal override void RemoveEntityTypeFromToken(int tokenIndex, int spanIndex, int entityIndex)
        {
            long ix = GetPooledTokenIndex(spanIndex, tokenIndex);

            if (!EntityData.TryGetValue(ix, out var entityList)) throw new Exception("No entities to update");

            entityList.RemoveAt(entityIndex);

            if (entityList.Count == 0) { RemoveEntityList(ix, entityList); }
        }

        internal override void RemoveEntityTypeFromToken(int tokenIndex, int spanIndex, string entityType)
        {
            long ix = GetPooledTokenIndex(spanIndex, tokenIndex);

            if (!EntityData.TryGetValue(ix, out var entityList)) throw new Exception("No entities to update");

            entityList.RemoveAll(et => et.Type == entityType);

            if (entityList.Count == 0) { RemoveEntityList(ix, entityList); }
        }

        internal override void ClearEntityTypesFromToken(int tokenIndex, int spanIndex)
        {
            long ix = GetPooledTokenIndex(spanIndex, tokenIndex);

            if (EntityData.TryGetValue(ix, out var entityList)) { RemoveEntityList(ix, entityList); }
        }

        internal override Dictionary<string, string> GetTokenMetadata(int tokenIndex, int spanIndex)
        {
            long ix = GetPooledTokenIndex(spanIndex, tokenIndex);

            if (TokenMetadata.TryGetValue(ix, out var dict)) return dict;

            dict = m_pool.RentMetadata();
            TokenMetadata.Add(ix, dict);

            return dict;
        }

        private void RemoveEntityList(long ix, List<EntityType> entityList)
        {
            EntityData.Remove(ix);
            m_pool.ReturnEntityTypes(entityList);
        }

        private static long GetPooledTokenIndex(int spanIndex, int tokenIndex)
        {
            return (long)spanIndex << 32 | (long)(uint)tokenIndex;
        }

        /// <summary>
        /// Writes this document to <paramref name="stream"/> in the same MessagePack layout the
        /// <see cref="Document"/> formatter produces, so it can be read back with
        /// <c>MessagePackSerializer.Deserialize&lt;Document&gt;</c> or with
        /// <see cref="DocumentPool.RentFromMessagePack(Stream, MessagePackSerializerOptions)"/>.
        /// </summary>
        /// <param name="stream">The stream to write to.</param>
        /// <param name="options">Serializer options; <see cref="MessagePackSerializerOptions.Standard"/> when null.</param>
        public void SerializeAsMessagePack(Stream stream, MessagePackSerializerOptions options = null)
        {
            var buffer = m_pool.RentBufferWriter();

            try
            {
                SerializeAsMessagePack(buffer, options);

                stream.Write(buffer.WrittenSpan);
            }
            finally
            {
                m_pool.ReturnBufferWriter(buffer);
            }
        }

        /// <summary>
        /// Writes this document to <paramref name="buffer"/> in the same MessagePack layout the
        /// <see cref="Document"/> formatter produces.
        /// </summary>
        /// <param name="buffer">The buffer to write to.</param>
        /// <param name="options">Serializer options; <see cref="MessagePackSerializerOptions.Standard"/> when null.</param>
        public void SerializeAsMessagePack(IBufferWriter<byte> buffer, MessagePackSerializerOptions options = null)
        {
            var writer = new MessagePackWriter(buffer);

            SerializeAsMessagePack(ref writer, options);

            writer.Flush();
        }

        /// <summary>
        /// Writes this document with an existing <see cref="MessagePackWriter"/>, for callers embedding the
        /// document inside a larger message.
        /// </summary>
        /// <param name="writer">The writer to write to.</param>
        /// <param name="options">Serializer options; <see cref="MessagePackSerializerOptions.Standard"/> when null.</param>
        public void SerializeAsMessagePack(ref MessagePackWriter writer, MessagePackSerializerOptions options = null)
        {
            options ??= MessagePackSerializerOptions.Standard;

            var resolver          = options.Resolver;
            var languageFormatter = resolver.GetFormatterWithVerify<Language>();
            var uidFormatter      = resolver.GetFormatterWithVerify<UID.UID128>();
            var posFormatter      = resolver.GetFormatterWithVerify<PartOfSpeech>();
            var tagFormatter      = resolver.GetFormatterWithVerify<EntityTag>();

            writer.WriteArrayHeader(9);

            languageFormatter.Serialize(ref writer, Language, options);
            writer.Write(Value);

            writer.WriteArrayHeader(TokensData.Count);
            for (int i = 0; i < TokensData.Count; i++)
            {
                var spanData = TokensData[i];

                writer.WriteArrayHeader(spanData.Count);
                for (int j = 0; j < spanData.Count; j++)
                {
                    var td = spanData[j];

                    //Written field by field rather than through the TokenData formatter, whose Bounds
                    //getter allocates an int[2] per token.
                    writer.WriteArrayHeader(8);
                    writer.WriteArrayHeader(2);
                    writer.Write(td.LowerBound);
                    writer.Write(td.UpperBound);
                    posFormatter.Serialize(ref writer, td.Tag, options);
                    writer.Write(td.Hash);
                    writer.Write(td.IgnoreCaseHash);
                    writer.Write(td.Head);
                    writer.Write(td.Frequency);
                    writer.Write(td.DependencyType);
                    writer.Write(td.Replacement);
                }
            }

            writer.WriteArrayHeader(SpanBounds.Count);
            for (int i = 0; i < SpanBounds.Count; i++)
            {
                var bounds = SpanBounds[i];

                writer.WriteArrayHeader(bounds.Length);
                for (int j = 0; j < bounds.Length; j++)
                {
                    writer.Write(bounds[j]);
                }
            }

            WriteStringMap(ref writer, Metadata);

            uidFormatter.Serialize(ref writer, UID, options);

            if (Labels is null)
            {
                writer.WriteNil();
            }
            else
            {
                writer.WriteArrayHeader(Labels.Count);
                for (int i = 0; i < Labels.Count; i++)
                {
                    writer.Write(Labels[i]);
                }
            }

            if (EntityData is null)
            {
                writer.WriteNil();
            }
            else
            {
                writer.WriteMapHeader(EntityData.Count);
                foreach (var kv in EntityData)
                {
                    writer.Write(kv.Key);

                    if (kv.Value is null)
                    {
                        writer.WriteNil();
                        continue;
                    }

                    writer.WriteArrayHeader(kv.Value.Count);
                    for (int i = 0; i < kv.Value.Count; i++)
                    {
                        var et = kv.Value[i];

                        writer.WriteArrayHeader(4);
                        writer.Write(et.Type);
                        tagFormatter.Serialize(ref writer, et.Tag, options);
                        WriteStringMap(ref writer, et.Metadata);
                        uidFormatter.Serialize(ref writer, et.TargetUID, options);
                    }
                }
            }

            if (TokenMetadata is null)
            {
                writer.WriteNil();
            }
            else
            {
                writer.WriteMapHeader(TokenMetadata.Count);
                foreach (var kv in TokenMetadata)
                {
                    writer.Write(kv.Key);
                    WriteStringMap(ref writer, kv.Value);
                }
            }
        }

        private static void WriteStringMap(ref MessagePackWriter writer, Dictionary<string, string> map)
        {
            if (map is null)
            {
                writer.WriteNil();
                return;
            }

            writer.WriteMapHeader(map.Count);
            foreach (var kv in map)
            {
                writer.Write(kv.Key);
                writer.Write(kv.Value);
            }
        }

        /// <summary>
        /// Orders token indexes by lower bound ascending, then upper bound descending, then original
        /// position - the total order <see cref="Document.RemoveOverlapingTokens"/> reaches with a stable
        /// LINQ sort. Kept as an instance so the comparison costs no closure per call.
        /// </summary>
        private sealed class TokenBoundsComparer : IComparer<int>
        {
            internal List<TokenData> Tokens;

            public int Compare(int x, int y)
            {
                var a = Tokens[x];
                var b = Tokens[y];

                if (a.LowerBound != b.LowerBound) return a.LowerBound < b.LowerBound ? -1 : 1;
                if (a.UpperBound != b.UpperBound) return a.UpperBound > b.UpperBound ? -1 : 1;

                return x.CompareTo(y);
            }
        }
    }
}
