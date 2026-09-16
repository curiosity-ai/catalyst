using MessagePack;
using Mosaik.Core;
using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using UID;

namespace Catalyst
{
    /// <summary>
    /// Hands out <see cref="PooledDocument"/> instances built from recycled collections, so a pipeline
    /// that parses documents in a loop stops paying for a fresh <see cref="Document"/>, span list, token
    /// list and dictionary set on every one of them.
    /// </summary>
    /// <remarks>
    /// Every pool here is bounded twice over: by how many instances it keeps, and by how many elements those
    /// instances retain between them. The second bound is what matters, because a cleared
    /// <see cref="List{T}"/> still holds its backing array - so counting instances alone would let a burst of
    /// unusually deep documents pin gigabytes for the rest of the process. Whatever does not fit either bound
    /// is dropped and collected normally, and a pool that has run dry simply allocates, which is what makes
    /// <see cref="Rent(string, Language)"/> safe to call from any number of threads.
    /// </remarks>
    public sealed class DocumentPool
    {
        /// <summary>The pool used when a caller does not bring its own.</summary>
        public static DocumentPool Shared { get; } = new DocumentPool();

        /// <summary>
        /// Documents kept by default. Sized for a caller that rents a whole indexing batch at once rather than
        /// one document at a time, which is what a batch-shaped pipeline does.
        /// </summary>
        public const int DEFAULT_POOLED_DOCUMENTS = 10_000;

        /// <summary>Spans a pooled document is assumed to carry, for sizing the per-span pools.</summary>
        private const int ASSUMED_SPANS_PER_DOCUMENT = 64;

        private const int MAXIMUM_POOLED_BUFFER_SIZE = 1024 * 1024;

        //Element budgets, in elements rather than bytes, which is what the pools can actually count. A TokenData
        //is ~48 bytes and an EntityType ~32, so these come to roughly 200 MB and 30 MB held at the extreme -
        //reached only by a workspace that really does keep that many spans in flight.
        private const long TOKEN_DATA_ELEMENT_BUDGET   = 4_000_000;
        private const long ENTITY_TYPE_ELEMENT_BUDGET  = 1_000_000;
        private const long SPAN_ELEMENT_BUDGET         = 4_000_000;
        private const long METADATA_ELEMENT_BUDGET     = 1_000_000;

        //A single collection grown past this is dropped rather than kept: one pathological document must not
        //leave an outsized collection parked for the lifetime of the process.
        private const int MAXIMUM_POOLED_COLLECTION_SIZE = 65_536;

        //Token lists get their own, much higher ceiling. A long document legitimately has a lot of tokens, and
        //its token list is both the largest thing the pool holds and the most expensive to rebuild - so the
        //general cap excluded precisely the documents worth pooling. Measured against a production
        //distribution of _Document.TokensCount, the half a percent of documents above 65,536 tokens
        //re-allocated on every pass and accounted for 1,035 MiB of the 1,055 MiB a pass allocated; at 2^20
        //that falls to 508 MiB, all of it one ten-million-token document the element budget rightly refuses.
        //What bounds retention is TOKEN_DATA_ELEMENT_BUDGET either way - this only decides which lists may
        //occupy it.
        private const int MAXIMUM_POOLED_TOKEN_LIST_SIZE = 1 << 20;

        private readonly BoundedPool<PooledDocument>                               m_documents;
        private readonly BoundedPool<List<List<TokenData>>>                        m_tokensDataLists;
        private readonly SizeClassedPool<TokenData>                                m_tokenDataLists;
        private readonly BoundedPool<List<int[]>>                                  m_spanBoundsLists;
        private readonly BoundedPool<int[]>                                        m_spanBounds;
        private readonly BoundedPool<List<string>>                                 m_labelsLists;
        private readonly BoundedPool<List<EntityType>>                             m_entityTypeLists;
        private readonly BoundedPool<Dictionary<string, string>>                   m_metadataMaps;
        private readonly BoundedPool<Dictionary<long, List<EntityType>>>           m_entityDataMaps;
        private readonly BoundedPool<Dictionary<long, Dictionary<string, string>>> m_tokenMetadataMaps;
        private readonly BoundedPool<ArrayBufferWriter<byte>>                      m_bufferWriters;

        /// <summary>
        /// Initializes a new pool.
        /// </summary>
        /// <param name="maximumPooledDocuments">
        /// How many documents - and how many of each of their document-level collections - are kept. The
        /// per-span pools are sized from this. Zero uses <see cref="DEFAULT_POOLED_DOCUMENTS"/>.
        /// </param>
        public DocumentPool(int maximumPooledDocuments = 0)
        {
            if (maximumPooledDocuments <= 0) { maximumPooledDocuments = DEFAULT_POOLED_DOCUMENTS; }

            //The per-span collections are rented once per span, so they are pooled that much more deeply
            long perSpanCapacity = (long)maximumPooledDocuments * ASSUMED_SPANS_PER_DOCUMENT;

            m_documents         = new BoundedPool<PooledDocument>(maximumPooledDocuments,                           long.MaxValue);
            m_tokensDataLists   = new BoundedPool<List<List<TokenData>>>(maximumPooledDocuments,                    SPAN_ELEMENT_BUDGET);
            m_spanBoundsLists   = new BoundedPool<List<int[]>>(maximumPooledDocuments,                              SPAN_ELEMENT_BUDGET);
            m_labelsLists       = new BoundedPool<List<string>>(maximumPooledDocuments,                             METADATA_ELEMENT_BUDGET);
            m_entityDataMaps    = new BoundedPool<Dictionary<long, List<EntityType>>>(maximumPooledDocuments,       ENTITY_TYPE_ELEMENT_BUDGET);
            m_tokenMetadataMaps = new BoundedPool<Dictionary<long, Dictionary<string, string>>>(maximumPooledDocuments, METADATA_ELEMENT_BUDGET);
            m_tokenDataLists    = new SizeClassedPool<TokenData>(perSpanCapacity,                                   TOKEN_DATA_ELEMENT_BUDGET);
            m_spanBounds        = new BoundedPool<int[]>(perSpanCapacity,                                           SPAN_ELEMENT_BUDGET);
            m_entityTypeLists   = new BoundedPool<List<EntityType>>(perSpanCapacity,                                ENTITY_TYPE_ELEMENT_BUDGET);
            m_metadataMaps      = new BoundedPool<Dictionary<string, string>>(perSpanCapacity,                      METADATA_ELEMENT_BUDGET);

            //Serialization buffers are held for the length of one call, so what bounds them is how many threads
            //serialize at once - not how many documents a batch carries.
            m_bufferWriters     = new BoundedPool<ArrayBufferWriter<byte>>(Math.Max(8, Environment.ProcessorCount * 4), long.MaxValue);
        }

        /// <summary>
        /// Drops everything this pool is holding, so a process under memory pressure can get the retained
        /// collections back. Renting keeps working - it just allocates until the pool refills.
        /// </summary>
        public void Trim()
        {
            m_documents.Trim();
            m_tokensDataLists.Trim();
            m_spanBoundsLists.Trim();
            m_labelsLists.Trim();
            m_entityDataMaps.Trim();
            m_tokenMetadataMaps.Trim();
            m_tokenDataLists.Trim();
            m_spanBounds.Trim();
            m_entityTypeLists.Trim();
            m_metadataMaps.Trim();
            m_bufferWriters.Trim();
        }

        /// <summary>
        /// Rents an empty document for the given text.
        /// </summary>
        /// <param name="text">The document's text. Control characters are removed, as in <see cref="Document"/>.</param>
        /// <param name="language">The document's language.</param>
        /// <returns>A document the caller must hand back with <see cref="Return"/>.</returns>
        public PooledDocument Rent(string text, Language language = Language.Unknown)
        {
            var document = RentEmpty();

            document.Value    = string.IsNullOrWhiteSpace(text) ? "" : text.RemoveControlCharacters();
            document.Language = language;

            return document;
        }

        /// <summary>
        /// Rents a document holding a copy of <paramref name="source"/>: same text, spans, tokens, entities
        /// and metadata, none of it shared with the source.
        /// </summary>
        /// <param name="source">The document to copy.</param>
        /// <returns>A document the caller must hand back with <see cref="Return"/>.</returns>
        public PooledDocument Rent(Document source)
        {
            if (source is null) throw new ArgumentNullException(nameof(source));

            var document = RentEmpty();

            document.Language = source.Language;
            document.Value    = source.Value;
            document.UID      = source.UID;

            for (int i = 0; i < source.SpanBounds.Count; i++)
            {
                var bounds = source.SpanBounds[i];
                document.AddSpan(bounds[0], bounds[1]);

                var from = source.TokensData[i];

                document.ReserveTokens(i, from.Count);

                var to = document.TokensData[i];

                for (int j = 0; j < from.Count; j++)
                {
                    to.Add(from[j]);
                }
            }

            if (source.Labels is object)
            {
                for (int i = 0; i < source.Labels.Count; i++)
                {
                    document.Labels.Add(source.Labels[i]);
                }
            }

            if (source.Metadata is object)
            {
                foreach (var kv in source.Metadata)
                {
                    document.Metadata[kv.Key] = kv.Value;
                }
            }

            if (source.EntityData is object)
            {
                foreach (var kv in source.EntityData)
                {
                    var entities = RentEntityTypes();

                    for (int i = 0; i < kv.Value.Count; i++)
                    {
                        entities.Add(kv.Value[i]);
                    }

                    document.EntityData[kv.Key] = entities;
                }
            }

            if (source.TokenMetadata is object)
            {
                foreach (var kv in source.TokenMetadata)
                {
                    var metadata = RentMetadata();

                    foreach (var kv2 in kv.Value)
                    {
                        metadata[kv2.Key] = kv2.Value;
                    }

                    document.TokenMetadata[kv.Key] = metadata;
                }
            }

            return document;
        }

        /// <summary>
        /// Rents a document holding a copy of <paramref name="source"/>, without going through
        /// <see cref="ImmutableDocument.ToMutable"/> - which would allocate a whole second document on the way.
        /// </summary>
        /// <param name="source">The document to copy.</param>
        /// <returns>A document the caller must hand back with <see cref="Return"/>.</returns>
        public PooledDocument Rent(ImmutableDocument source)
        {
            if (source is null) throw new ArgumentNullException(nameof(source));

            var document = RentEmpty();

            document.Language = source.Language;
            document.Value    = source.Value;
            document.UID      = source.UID;

            if (source.SpanBounds is object)
            {
                document.ReserveSpans(source.SpanBounds.Length);

                for (int i = 0; i < source.SpanBounds.Length; i++)
                {
                    var packed = source.SpanBounds[i];

                    document.AddSpan((int)(packed >> 32), (int)(packed & 0xFFFF_FFFFL));
                }
            }

            if (source.TokensData is object)
            {
                for (int i = 0; i < source.TokensData.Length; i++)
                {
                    //An immutable whose two span arrays disagree is malformed, but copying it must still not walk
                    //off the end of the list the spans above created.
                    if (i >= document.TokensData.Count) { document.TokensData.Add(RentTokenData(source.TokensData[i].Length)); }

                    var from = source.TokensData[i];

                    document.ReserveTokens(i, from.Length);

                    var to = document.TokensData[i];

                    for (int j = 0; j < from.Length; j++)
                    {
                        to.Add(from[j]);
                    }
                }
            }

            if (source.Labels is object)
            {
                for (int i = 0; i < source.Labels.Length; i++)
                {
                    document.Labels.Add(source.Labels[i]);
                }
            }

            if (source.Metadata is object)
            {
                foreach (var kv in source.Metadata)
                {
                    document.Metadata[kv.Key] = kv.Value;
                }
            }

            if (source.EntityData is object)
            {
                foreach (var kv in source.EntityData)
                {
                    var entities = RentEntityTypes();

                    if (entities.Capacity < kv.Value.Length) { entities.Capacity = kv.Value.Length; }

                    for (int i = 0; i < kv.Value.Length; i++)
                    {
                        entities.Add(kv.Value[i]);
                    }

                    document.EntityData[kv.Key] = entities;
                }
            }

            if (source.TokenMetadata is object)
            {
                foreach (var kv in source.TokenMetadata)
                {
                    var metadata = RentMetadata();

                    foreach (var kv2 in kv.Value)
                    {
                        metadata[kv2.Key] = kv2.Value;
                    }

                    document.TokenMetadata[kv.Key] = metadata;
                }
            }

            //A document whose spans were declared but whose token data was not still needs the two lists to line up
            while (document.TokensData.Count < document.SpanBounds.Count)
            {
                document.TokensData.Add(RentTokenData());
            }

            return document;
        }

        /// <summary>
        /// Reads a document written by <see cref="PooledDocument.SerializeAsMessagePack(Stream, MessagePackSerializerOptions)"/>
        /// (or by the <see cref="Document"/> formatter) into a pooled document.
        /// </summary>
        /// <param name="stream">The stream to read from.</param>
        /// <param name="options">Serializer options; <see cref="MessagePackSerializerOptions.Standard"/> when null.</param>
        /// <returns>A document the caller must hand back with <see cref="Return"/>.</returns>
        public PooledDocument RentFromMessagePack(Stream stream, MessagePackSerializerOptions options = null)
        {
            if (stream is null) throw new ArgumentNullException(nameof(stream));

            var buffer = RentBufferWriter();

            try
            {
                int read;

                while ((read = stream.Read(buffer.GetSpan(16 * 1024))) > 0)
                {
                    buffer.Advance(read);
                }

                return RentFromMessagePack(buffer.WrittenMemory, options);
            }
            finally
            {
                ReturnBufferWriter(buffer);
            }
        }

        /// <summary>
        /// Reads a document written by <see cref="PooledDocument.SerializeAsMessagePack(Stream, MessagePackSerializerOptions)"/>
        /// (or by the <see cref="Document"/> formatter) into a pooled document.
        /// </summary>
        /// <param name="bytes">The bytes to read from.</param>
        /// <param name="options">Serializer options; <see cref="MessagePackSerializerOptions.Standard"/> when null.</param>
        /// <returns>A document the caller must hand back with <see cref="Return"/>.</returns>
        public PooledDocument RentFromMessagePack(ReadOnlyMemory<byte> bytes, MessagePackSerializerOptions options = null)
        {
            var reader = new MessagePackReader(bytes);

            return RentFromMessagePack(ref reader, options);
        }

        /// <summary>
        /// Reads a document with an existing <see cref="MessagePackReader"/>, for callers reading it out of a
        /// larger message.
        /// </summary>
        /// <param name="reader">The reader to read from.</param>
        /// <param name="options">Serializer options; <see cref="MessagePackSerializerOptions.Standard"/> when null.</param>
        /// <returns>A document the caller must hand back with <see cref="Return"/>.</returns>
        public PooledDocument RentFromMessagePack(ref MessagePackReader reader, MessagePackSerializerOptions options = null)
        {
            options ??= MessagePackSerializerOptions.Standard;

            var resolver          = options.Resolver;
            var languageFormatter = resolver.GetFormatterWithVerify<Language>();
            var uidFormatter      = resolver.GetFormatterWithVerify<UID128>();
            var posFormatter      = resolver.GetFormatterWithVerify<PartOfSpeech>();
            var tagFormatter      = resolver.GetFormatterWithVerify<EntityTag>();

            var document = RentEmpty();

            try
            {
                int fields = reader.ReadArrayHeader();

                for (int field = 0; field < fields; field++)
                {
                    switch (field)
                    {
                        case 0: document.Language = languageFormatter.Deserialize(ref reader, options); break;
                        case 1: document.Value = reader.ReadString(); break;

                        case 2:
                        {
                            //Spans are added from the bounds in field 3, so the token lists are filled here and
                            //matched up below - the two fields always describe the same number of spans.
                            int spans = reader.ReadArrayHeader();

                            for (int i = 0; i < spans; i++)
                            {
                                int count  = reader.ReadArrayHeader();
                                var tokens = RentTokenData(count);

                                if (tokens.Capacity < count) { tokens.Capacity = count; }

                                for (int j = 0; j < count; j++)
                                {
                                    int members = reader.ReadArrayHeader();
                                    var token   = new TokenData();

                                    for (int m = 0; m < members; m++)
                                    {
                                        switch (m)
                                        {
                                            case 0:
                                            {
                                                int bounds = reader.ReadArrayHeader();
                                                if (bounds > 0) { token.LowerBound = reader.ReadInt32(); }
                                                if (bounds > 1) { token.UpperBound = reader.ReadInt32(); }
                                                for (int b = 2; b < bounds; b++) { reader.Skip(); }
                                                break;
                                            }
                                            case 1: token.Tag = posFormatter.Deserialize(ref reader, options); break;
                                            case 2: token.Hash = reader.ReadInt32(); break;
                                            case 3: token.IgnoreCaseHash = reader.ReadInt32(); break;
                                            case 4: token.Head = reader.ReadInt32(); break;
                                            case 5: token.Frequency = reader.ReadSingle(); break;
                                            case 6: token.DependencyType = reader.ReadString(); break;
                                            case 7: token.Replacement = reader.ReadString(); break;
                                            default: reader.Skip(); break;
                                        }
                                    }

                                    tokens.Add(token);
                                }

                                document.TokensData.Add(tokens);
                            }
                            break;
                        }

                        case 3:
                        {
                            int spans = reader.ReadArrayHeader();

                            for (int i = 0; i < spans; i++)
                            {
                                int count = reader.ReadArrayHeader();
                                int begin = count > 0 ? reader.ReadInt32() : 0;
                                int end   = count > 1 ? reader.ReadInt32() : 0;

                                for (int b = 2; b < count; b++) { reader.Skip(); }

                                document.SpanBounds.Add(RentSpanBounds(begin, end));
                            }
                            break;
                        }

                        case 4: ReadStringMap(ref reader, document.Metadata); break;
                        case 5: document.UID = uidFormatter.Deserialize(ref reader, options); break;

                        case 6:
                        {
                            if (!reader.TryReadNil())
                            {
                                int count = reader.ReadArrayHeader();

                                for (int i = 0; i < count; i++)
                                {
                                    document.Labels.Add(reader.ReadString());
                                }
                            }
                            break;
                        }

                        case 7:
                        {
                            if (!reader.TryReadNil())
                            {
                                int count = reader.ReadMapHeader();

                                for (int i = 0; i < count; i++)
                                {
                                    long key      = reader.ReadInt64();
                                    var  entities = RentEntityTypes();

                                    if (!reader.TryReadNil())
                                    {
                                        int entityCount = reader.ReadArrayHeader();

                                        for (int j = 0; j < entityCount; j++)
                                        {
                                            int members = reader.ReadArrayHeader();
                                            var entity  = new EntityType();

                                            for (int m = 0; m < members; m++)
                                            {
                                                switch (m)
                                                {
                                                    case 0: entity.Type = reader.ReadString(); break;
                                                    case 1: entity.Tag = tagFormatter.Deserialize(ref reader, options); break;
                                                    case 2:
                                                    {
                                                        if (reader.TryReadNil()) { entity.Metadata = null; }
                                                        else
                                                        {
                                                            var metadata = RentMetadata();
                                                            ReadStringMapEntries(ref reader, metadata);
                                                            entity.Metadata = metadata;
                                                        }
                                                        break;
                                                    }
                                                    case 3: entity.TargetUID = uidFormatter.Deserialize(ref reader, options); break;
                                                    default: reader.Skip(); break;
                                                }
                                            }

                                            entities.Add(entity);
                                        }
                                    }

                                    document.EntityData[key] = entities;
                                }
                            }
                            break;
                        }

                        case 8:
                        {
                            if (!reader.TryReadNil())
                            {
                                int count = reader.ReadMapHeader();

                                for (int i = 0; i < count; i++)
                                {
                                    long key      = reader.ReadInt64();
                                    var  metadata = RentMetadata();

                                    ReadStringMap(ref reader, metadata);

                                    document.TokenMetadata[key] = metadata;
                                }
                            }
                            break;
                        }

                        default: reader.Skip(); break;
                    }
                }

                //A document written with no spans still needs its two lists to line up
                while (document.TokensData.Count < document.SpanBounds.Count)
                {
                    document.TokensData.Add(RentTokenData());
                }

                return document;
            }
            catch
            {
                Return(document);
                throw;
            }
        }

        /// <summary>
        /// Returns <paramref name="document"/> and every collection it holds to this pool. The document must
        /// not be read afterwards - see the remarks on <see cref="PooledDocument"/>.
        /// </summary>
        /// <param name="document">The document to return. Null is ignored.</param>
        public void Return(PooledDocument document)
        {
            if (document is null) return;

            if (!ReferenceEquals(document.Pool, this)) throw new InvalidOperationException("Trying to return a pooled document to a pool it was not rented from");

            var tokensData    = document.TokensData;
            var spanBounds    = document.SpanBounds;
            var metadata      = document.Metadata;
            var labels        = document.Labels;
            var entityData    = document.EntityData;
            var tokenMetadata = document.TokenMetadata;

            document.TokensData    = null;
            document.SpanBounds    = null;
            document.Metadata      = null;
            document.Labels        = null;
            document.EntityData    = null;
            document.TokenMetadata = null;
            document.Value         = null;
            document.UID           = default;
            document.Language      = Language.Unknown;

            if (tokensData is object)
            {
                for (int i = 0; i < tokensData.Count; i++)
                {
                    ReturnTokenData(tokensData[i]);
                }

                tokensData.Clear();
                m_tokensDataLists.Return(tokensData, tokensData.Capacity);
            }

            if (spanBounds is object)
            {
                for (int i = 0; i < spanBounds.Count; i++)
                {
                    ReturnSpanBounds(spanBounds[i]);
                }

                spanBounds.Clear();
                m_spanBoundsLists.Return(spanBounds, spanBounds.Capacity);
            }

            if (entityData is object)
            {
                foreach (var kv in entityData)
                {
                    ReturnEntityTypes(kv.Value);
                }

                entityData.Clear();
                m_entityDataMaps.Return(entityData, entityData.Count);
            }

            if (tokenMetadata is object)
            {
                foreach (var kv in tokenMetadata)
                {
                    ReturnMetadata(kv.Value);
                }

                tokenMetadata.Clear();
                m_tokenMetadataMaps.Return(tokenMetadata, tokenMetadata.Count);
            }

            ReturnMetadata(metadata);

            if (labels is object)
            {
                int capacity = labels.Capacity;
                labels.Clear();
                m_labelsLists.Return(labels, capacity);
            }

            m_documents.Return(document, 0);
        }

        private PooledDocument RentEmpty()
        {
            var document = m_documents.Rent() ?? new PooledDocument(this);

            document.TokensData    = m_tokensDataLists.Rent()   ?? new List<List<TokenData>>();
            document.SpanBounds    = m_spanBoundsLists.Rent()   ?? new List<int[]>();
            document.Metadata      = RentMetadata();
            document.Labels        = m_labelsLists.Rent()       ?? new List<string>();
            document.EntityData    = m_entityDataMaps.Rent()    ?? new Dictionary<long, List<EntityType>>();
            document.TokenMetadata = m_tokenMetadataMaps.Rent() ?? new Dictionary<long, Dictionary<string, string>>();
            document.Language      = Language.Unknown;
            document.Value         = "";
            document.UID           = default;

            return document;
        }

        /// <summary>
        /// Rents a token list of the kind <see cref="PooledDocument.AddSpan"/> uses. For a caller filling a
        /// pooled document from its own storage format rather than through the span API.
        /// </summary>
        /// <returns>An empty list the pool takes back with the document.</returns>
        public List<TokenData> RentTokenData() => RentTokenData(0);

        /// <summary>
        /// Rents a token list that can already hold <paramref name="minimumCapacity"/> tokens. A caller that
        /// knows how many tokens it is about to write - a deserializer reading a span's token count, a
        /// tokenizer that has counted its split points - should say so: growing a list the pool handed back
        /// too small is one allocation the pool exists to avoid.
        /// </summary>
        /// <param name="minimumCapacity">The number of tokens the caller is about to add.</param>
        /// <returns>An empty list the pool takes back with the document.</returns>
        public List<TokenData> RentTokenData(int minimumCapacity)
        {
            var tokens = m_tokenDataLists.Rent(minimumCapacity);

            if (tokens is object) return tokens;

            return minimumCapacity > 0 ? new List<TokenData>(minimumCapacity) : new List<TokenData>();
        }

        /// <summary>Gives a token list back. Returning the document it belongs to does this for you.</summary>
        /// <param name="tokens">The list to return. Null is ignored.</param>
        public void ReturnTokenData(List<TokenData> tokens)
        {
            if (tokens is null) return;

            int capacity = tokens.Capacity;
            tokens.Clear();
            m_tokenDataLists.Return(tokens, capacity);
        }

        internal int[] RentSpanBounds(int begin, int end)
        {
            var bounds = m_spanBounds.Rent() ?? new int[2];

            bounds[0] = begin;
            bounds[1] = end;

            return bounds;
        }

        internal void ReturnSpanBounds(int[] bounds)
        {
            if (bounds is null || bounds.Length != 2) return;

            m_spanBounds.Return(bounds, 0);
        }

        /// <summary>
        /// Rents an entity list of the kind a token's entities are held in, for a caller writing straight into
        /// <see cref="Document.EntityData"/>.
        /// </summary>
        /// <returns>An empty list the pool takes back with the document.</returns>
        public List<EntityType> RentEntityTypes() => m_entityTypeLists.Rent() ?? new List<EntityType>();

        /// <summary>Gives an entity list back. Returning the document it belongs to does this for you.</summary>
        /// <param name="entities">The list to return. Null is ignored.</param>
        public void ReturnEntityTypes(List<EntityType> entities)
        {
            if (entities is null) return;

            int capacity = entities.Capacity;
            entities.Clear();
            m_entityTypeLists.Return(entities, capacity);
        }

        /// <summary>
        /// Rents a metadata dictionary, for a caller writing straight into <see cref="Document.TokenMetadata"/>
        /// or into an <see cref="EntityType.Metadata"/>.
        /// </summary>
        /// <returns>An empty dictionary the pool takes back with the document.</returns>
        public Dictionary<string, string> RentMetadata() => m_metadataMaps.Rent() ?? new Dictionary<string, string>();

        /// <summary>Gives a metadata dictionary back. Returning the document it belongs to does this for you.</summary>
        /// <param name="metadata">The dictionary to return. Null is ignored.</param>
        public void ReturnMetadata(Dictionary<string, string> metadata)
        {
            if (metadata is null) return;

            int count = metadata.Count;
            metadata.Clear();
            m_metadataMaps.Return(metadata, count);
        }

        internal ArrayBufferWriter<byte> RentBufferWriter() => m_bufferWriters.Rent() ?? new ArrayBufferWriter<byte>(16 * 1024);

        internal void ReturnBufferWriter(ArrayBufferWriter<byte> buffer)
        {
            if (buffer is null) return;

            int written = buffer.WrittenCount;
            buffer.Clear();
            m_bufferWriters.Return(buffer, written > MAXIMUM_POOLED_BUFFER_SIZE ? int.MaxValue : 0);
        }

        private void ReadStringMap(ref MessagePackReader reader, Dictionary<string, string> map)
        {
            if (reader.TryReadNil()) return;

            ReadStringMapEntries(ref reader, map);
        }

        private static void ReadStringMapEntries(ref MessagePackReader reader, Dictionary<string, string> map)
        {
            int count = reader.ReadMapHeader();

            for (int i = 0; i < count; i++)
            {
                var key   = reader.ReadString();
                var value = reader.ReadString();

                map[key] = value;
            }
        }

        /// <summary>
        /// A lock-free pool bounded by instance count and by the elements those instances retain between
        /// them, dropping anything bigger than <see cref="MAXIMUM_POOLED_TOKEN_LIST_SIZE"/> on its own.
        /// </summary>
        /// <remarks>
        /// The size a caller hands to <see cref="Return"/> is what the instance keeps hold of - a cleared
        /// list's <c>Capacity</c>, a dictionary's entry count - and it travels with the instance so
        /// <see cref="Rent"/> can give the budget back. Counting instances alone would be no bound at all:
        /// ten thousand token lists are nothing, and ten thousand token lists that each grew to a hundred
        /// thousand tokens are tens of gigabytes.
        /// </remarks>
        /// <summary>
        /// Keeps recycled lists in buckets by capacity, so a caller that knows how many elements it is about
        /// to write gets a list that already holds them.
        /// </summary>
        /// <remarks>
        /// <para>
        /// Bucket <c>k</c> holds lists whose capacity is in <c>[2^k, 2^(k+1))</c>, so <b>every member of
        /// bucket <c>k</c> holds at least <c>2^k</c></b>. That invariant is the whole point: <see cref="Rent"/>
        /// starts at the first bucket whose floor covers the request, which is what lets it promise the
        /// capacity rather than hope for it. A bucketing that only bounds a list from above cannot - the
        /// bucket holding a request for forty also holds lists of eight - and a rent that answers too small
        /// is worse than no bucketing at all, because the caller grows it after believing it was served.
        /// Powers of two keep the over-provisioning under 2x, the same slack <see cref="List{T}"/> gives
        /// itself when it grows.
        /// </para>
        /// <para>
        /// A rent that finds every large enough bucket empty falls back to a smaller list rather than
        /// allocating: growing that list costs one array - what an unbucketed pool cost every time - and
        /// keeps the <see cref="List{T}"/> itself out of the garbage. So the bucketing can win, and cannot
        /// lose, against handing out whichever list came back last.
        /// </para>
        /// <para>
        /// The element budget is shared across the buckets rather than divided between them. Dividing it caps
        /// each shape at a fraction of what the pool may hold, so the one shape a workload actually produces
        /// runs dry while the rest sit empty.
        /// </para>
        /// </remarks>
        private sealed class SizeClassedPool<T>
        {
            //Bucket 0 takes everything below 2, so a list that came back with no capacity has somewhere to go;
            //every bucket above it holds [2^k, 2^(k+1)). The last ends where a list is dropped rather than kept,
            //so it has to track the ceiling: a bucket that clamps below it stops promising the capacity it was
            //asked for, and Rent starts answering too small for exactly the largest callers.
            private const int MAXIMUM_BUCKET = 20; //2^20 == MAXIMUM_POOLED_TOKEN_LIST_SIZE

            private readonly ConcurrentQueue<Pooled>[] m_buckets;
            private readonly int                       m_capacity;
            private readonly long                      m_elementBudget;
            private          int                       m_count;
            private          long                      m_retainedElements;

            internal SizeClassedPool(long capacity, long elementBudget)
            {
                m_capacity      = (int)Math.Min(capacity, int.MaxValue);
                m_elementBudget = elementBudget;
                m_buckets       = new ConcurrentQueue<Pooled>[MAXIMUM_BUCKET + 1];

                for (int i = 0; i < m_buckets.Length; i++)
                {
                    m_buckets[i] = new ConcurrentQueue<Pooled>();
                }
            }

            /// <summary>Where a list of this capacity belongs - floor(log2), so the bucket's members all hold at least 2^k.</summary>
            private static int BucketHolding(int capacity)
            {
                int bucket = 0;

                while (bucket < MAXIMUM_BUCKET && (1 << (bucket + 1)) <= capacity) { bucket++; }

                return bucket;
            }

            /// <summary>The first bucket whose members are guaranteed to hold this many - ceil(log2).</summary>
            private static int BucketFor(int minimumCapacity)
            {
                int bucket = 0;

                while (bucket < MAXIMUM_BUCKET && (1 << bucket) < minimumCapacity) { bucket++; }

                return bucket;
            }

            /// <summary>
            /// Takes a list holding at least <paramref name="minimumCapacity"/> when the pool has one, a
            /// smaller one when it does not, and null when it is empty.
            /// </summary>
            internal List<T> Rent(int minimumCapacity)
            {
                var wanted = BucketFor(minimumCapacity);

                for (int bucket = wanted; bucket <= MAXIMUM_BUCKET; bucket++)
                {
                    if (TryTake(bucket, out var item)) return item;
                }

                for (int bucket = wanted - 1; bucket >= 0; bucket--)
                {
                    if (TryTake(bucket, out var item)) return item;
                }

                return null;
            }

            private bool TryTake(int bucket, out List<T> item)
            {
                if (m_buckets[bucket].TryDequeue(out var pooled))
                {
                    Interlocked.Decrement(ref m_count);
                    Interlocked.Add(ref m_retainedElements, -pooled.Size);

                    item = pooled.Item;
                    return true;
                }

                item = null;
                return false;
            }

            internal void Return(List<T> item, int capacity)
            {
                if (capacity > MAXIMUM_POOLED_TOKEN_LIST_SIZE) return;

                if (Interlocked.Add(ref m_retainedElements, capacity) > m_elementBudget)
                {
                    Interlocked.Add(ref m_retainedElements, -capacity);
                    return;
                }

                if (Interlocked.Increment(ref m_count) > m_capacity)
                {
                    Interlocked.Decrement(ref m_count);
                    Interlocked.Add(ref m_retainedElements, -capacity);
                    return;
                }

                m_buckets[BucketHolding(capacity)].Enqueue(new Pooled(item, capacity));
            }

            internal void Trim()
            {
                for (int bucket = 0; bucket <= MAXIMUM_BUCKET; bucket++)
                {
                    while (TryTake(bucket, out _)) { }
                }
            }

            private readonly struct Pooled
            {
                internal readonly List<T> Item;
                internal readonly int     Size;

                internal Pooled(List<T> item, int size)
                {
                    Item = item;
                    Size = size;
                }
            }
        }

        private sealed class BoundedPool<T> where T : class
        {
            private readonly ConcurrentQueue<Pooled> m_items = new ConcurrentQueue<Pooled>();
            private readonly int                     m_capacity;
            private readonly long                    m_elementBudget;
            private          int                     m_count;
            private          long                    m_retainedElements;

            internal BoundedPool(long capacity, long elementBudget)
            {
                m_capacity      = (int)Math.Min(capacity, int.MaxValue);
                m_elementBudget = elementBudget;
            }

            internal T Rent()
            {
                if (m_items.TryDequeue(out var pooled))
                {
                    Interlocked.Decrement(ref m_count);
                    Interlocked.Add(ref m_retainedElements, -pooled.Size);
                    return pooled.Item;
                }

                return null;
            }

            internal void Return(T item, int size)
            {
                if (size > MAXIMUM_POOLED_COLLECTION_SIZE) return;

                if (Interlocked.Add(ref m_retainedElements, size) > m_elementBudget)
                {
                    Interlocked.Add(ref m_retainedElements, -size);
                    return;
                }

                if (Interlocked.Increment(ref m_count) > m_capacity)
                {
                    Interlocked.Decrement(ref m_count);
                    Interlocked.Add(ref m_retainedElements, -size);
                    return;
                }

                m_items.Enqueue(new Pooled(item, size));
            }

            internal void Trim()
            {
                while (m_items.TryDequeue(out var pooled))
                {
                    Interlocked.Decrement(ref m_count);
                    Interlocked.Add(ref m_retainedElements, -pooled.Size);
                }
            }

            private readonly struct Pooled
            {
                internal readonly T   Item;
                internal readonly int Size;

                internal Pooled(T item, int size)
                {
                    Item = item;
                    Size = size;
                }
            }
        }
    }
}
