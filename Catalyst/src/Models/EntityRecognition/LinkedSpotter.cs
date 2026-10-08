using Mosaik.Core;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using UID;

namespace Catalyst.Models
{
    public class LinkedSpotterModel : StorableObjectData
    {
        public string CaptureTag { get; set; }
        public bool IgnoreOnlyNumeric { get; set; }
        public bool IgnoreCase { get; set; }

        /// <summary>Smallest character length among the individual words stored in the model, or 0 when the model is empty.</summary>
        public int MinTokenLength { get; set; }
        /// <summary>Largest character length among the individual words stored in the model, or 0 when the model is empty.</summary>
        public int MaxTokenLength { get; set; }

        // The entries as one sorted prefix-compressed dictionary, plus the values laid out densely by the rank
        // a lookup returns - so nothing stores a key alongside its value.
        public byte[] EntriesPayload { get; set; }
        public byte[] EntriesBlockOffsets { get; set; }
        public int EntriesCount { get; set; }
        public int EntriesBlockSize { get; set; }
        public int EntriesMaxBytes { get; set; }
        public UID128[] EntryValues { get; set; }

        public byte[] ExceptionBuckets { get; set; }
        public byte[] ExceptionLows { get; set; }
        public int ExceptionCount { get; set; }

        // Superseded by the entry dictionary, kept so models stored before it still load and match. Never written.
        public Dictionary<ulong, UID128> Hashes { get; set; } = new Dictionary<ulong, UID128>();
        public List<HashSet<ulong>> MultiGramHashes { get; set; } = new List<HashSet<ulong>>();
        public HashSet<int> TokenizerExceptionsSet { get; set; } = new HashSet<int>();
    }

    public class LinkedSpotter : StorableObjectV2<LinkedSpotter, LinkedSpotterModel>, IEntityRecognizer, IProcess, IHasSimpleSpecialCases, ICanOptimizeMemory
    {
        public string CaptureTag => Data.CaptureTag;

        public bool IgnoreCase
        {
            get { return Data.IgnoreCase; }
            set { Data.IgnoreCase = value; if (_engine is object) { _engine.IgnoreCase = value; } }
        }

        public const string Separator = "_";

        private readonly object _syncRoot = new object();
        private SpotterEngine   _engine;
        private bool            _initialized;

        private LegacyLinkedSpotterTables _legacy;

        /// <summary>True once everything added to the model has been flushed into its read-only in-memory form.</summary>
        public bool IsMemoryOptimized
        {
            get { Initialize(); return _legacy is object ? _legacy.IsFrozen : _engine.IsFrozen; }
        }

        /// <summary>Estimated bytes held by the read-only tables, or 0 while entries are waiting to be flushed.</summary>
        public long OptimizedMemoryBytes
        {
            get
            {
                Initialize();
                return _legacy is object ? _legacy.EstimatedBytes : _engine.EstimatedBytes;
            }
        }

        /// <summary>True when this model was loaded from a store written before the entry dictionary existed.</summary>
        public bool IsLegacyModel { get { Initialize(); return _legacy is object; } }

        /// <summary>Number of segments the entries are currently spread over. See <see cref="Flush"/>.</summary>
        public int SegmentCount { get { Initialize(); return _legacy is object ? 1 : _engine.SegmentCount; } }

        private LinkedSpotter(Language language, int version, string tag) : base(language, version, tag, compress: false)
        {
        }

        public LinkedSpotter(Language language, int version, string tag, string captureTag) : this(language, version, tag)
        {
            Data.CaptureTag = captureTag;
        }

        public new static async Task<LinkedSpotter> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new LinkedSpotter(language, version, tag);
            await a.LoadDataAsync();
            a.TrimExcess();
            return a;
        }

        private void Initialize()
        {
            if (_initialized) { return; }

            lock (_syncRoot)
            {
                if (_initialized) { return; }

                _engine = new SpotterEngine(withValues: true) { Language = Language, IgnoreCase = Data.IgnoreCase, MinTokenLength = Data.MinTokenLength, MaxTokenLength = Data.MaxTokenLength };

                if (Data.EntriesCount > 0 && Data.EntriesPayload is object)
                {
                    _engine.LoadFrom(Data.EntriesPayload, Data.EntriesBlockOffsets, Data.EntriesCount, Data.EntriesBlockSize, Data.EntriesMaxBytes,
                                     Data.ExceptionBuckets, Data.ExceptionLows, Data.ExceptionCount, Data.EntryValues);
                }
                else if ((Data.Hashes?.Count ?? 0) > 0 || (Data.MultiGramHashes?.Count ?? 0) > 0)
                {
                    _legacy = new LegacyLinkedSpotterTables(Data);
                }

                _initialized = true;
            }
        }

        private void EnsureFrozen()
        {
            if (_legacy is object)
            {
                if (!_legacy.IsFrozen) { lock (_syncRoot) { _legacy.Freeze(); } }
                return;
            }

            _engine.Flush();
        }

        public void TrimExcess()
        {
            if (Data is null) { return; }
            Initialize();
            EnsureFrozen();
        }

        /// <summary>Flushes what was added into the read-only tables. Idempotent with what <see cref="TrimExcess"/> does on load.</summary>
        public void OptimizeMemory() => TrimExcess();

        /// <summary>
        /// Makes every entry added or removed so far visible to recognition. What was buffered becomes a new
        /// segment next to the ones the model holds, and segments are merged only once they are of comparable
        /// size - so applying a few changes to a model of millions of entries allocates for the few, not the
        /// millions. Until this is first called, recognition flushes on its own when nothing else is writing to the
        /// model. Once it is called, the caller owns flushing: a batch of changes - a removal and the addition that
        /// replaces it - becomes visible at once, when the caller says it is complete, and never half applied.
        /// Returns false when there was nothing to apply.
        /// </summary>
        public bool Flush()
        {
            Initialize();
            if (_legacy is object) { EnsureFrozen(); return false; }
            return _engine.Flush(byOwner: true);
        }

        /// <summary>Merges every segment into one. Storing a model does this; nothing else needs to.</summary>
        public void Compact()
        {
            Initialize();
            if (_legacy is object) { EnsureFrozen(); return; }
            _engine.Compact();
        }

        public override async Task StoreAsync(System.IO.Stream stream)
        {
            Initialize();

            if (_legacy is object)
            {
                bool wasFrozen = _legacy.IsFrozen;
                _legacy.Unfreeze();
                await base.StoreAsync(stream);
                if (wasFrozen) { _legacy.Freeze(); }
                return;
            }

            var segment    = _engine.CompactedSegment();
            var dictionary = segment?.Dictionary ?? EntryDictionary.Empty;

            var (payload, blockOffsets, count, blockSize, maxEntryBytes) = dictionary.ToBlobs();
            var (buckets, lows, exceptionCount)                         = _engine.Exceptions.ToBlobs();

            Data.EntriesPayload          = count > 0 ? payload : null;
            Data.EntriesBlockOffsets     = count > 0 ? blockOffsets : null;
            Data.EntriesCount            = count;
            Data.EntriesBlockSize        = blockSize;
            Data.EntriesMaxBytes         = maxEntryBytes;
            Data.EntryValues             = count > 0 ? segment.Values : null;
            Data.ExceptionBuckets        = exceptionCount > 0 ? buckets : null;
            Data.ExceptionLows           = exceptionCount > 0 ? lows : null;
            Data.ExceptionCount          = exceptionCount;
            Data.MinTokenLength          = _engine.MinTokenLength;
            Data.MaxTokenLength          = _engine.MaxTokenLength;

            Data.Hashes                  = null;
            Data.MultiGramHashes         = null;
            Data.TokenizerExceptionsSet  = null;

            await base.StoreAsync(stream);
        }

        public void Process(IDocument document, CancellationToken cancellationToken = default)
        {
            RecognizeEntities(document);
        }

        public string[] Produces()
        {
            return new[] { CaptureTag };
        }

        public bool RecognizeEntities(IDocument document)
        {
            var foundAny = false;
            foreach (var span in document)
            {
                foundAny |= RecognizeEntities(span);
            }
            return foundAny;
        }

        public bool HasAnyEntity(IDocument document)
        {
            foreach (var span in document)
            {
                if (RecognizeEntities(span, stopOnFirstFound: true))
                {
                    return true;
                }
            }
            return false;
        }

        private readonly struct Sink : ISpotterMatchSink
        {
            private readonly string _captureTag;

            public Sink(string captureTag) { _captureTag = captureTag; }

            public void OnSingle(ref Token token, EntrySegment segment, int rank) => token.AddEntityType(new EntityType(_captureTag, EntityTag.Single, segment.ValueAt(rank)));

            public void OnMultiGram(Span<Token> tokens, int begin, int end, EntrySegment segment, int rank)
            {
                var value = segment.ValueAt(rank);

                tokens[begin].AddEntityType(new EntityType(_captureTag, EntityTag.Begin, value));
                tokens[end].AddEntityType(new EntityType(_captureTag, EntityTag.End, value));

                for (int m = begin + 1; m < end; m++)
                {
                    tokens[m].AddEntityType(new EntityType(_captureTag, EntityTag.Inside, value));
                }
            }
        }

        public bool RecognizeEntities(Span ispan, bool stopOnFirstFound = false)
        {
            Initialize();
            if (_legacy is object) { EnsureFrozen(); } else { _engine.FlushIfIdle(); }

            var pooledTokens = ispan.ToTokenSpanPolled(out var actualLength);

            try
            {
                var tokens = pooledTokens.AsSpan(0, actualLength);

                if (_legacy is object) { return _legacy.Match(tokens, CaptureTag, stopOnFirstFound); }

                var sink = new Sink(CaptureTag);
                return _engine.Match(tokens, stopOnFirstFound, ref sink);
            }
            finally
            {
                ArrayPool<Token>.Shared.Return(pooledTokens);
            }
        }

        public CompactHash32Set GetSimpleSpecialCases()
        {
            Initialize();
            EnsureFrozen();
            return _legacy is object ? _legacy.Exceptions : _engine.Exceptions;
        }

        /// <summary>The entries this model recognises, in sorted order, paired with what they link to.</summary>
        public IEnumerable<KeyValuePair<string, UID128>> GetEntries()
        {
            Initialize();
            if (_legacy is object) { return Enumerable.Empty<KeyValuePair<string, UID128>>(); }

            return _engine.Entries().Select(e => new KeyValuePair<string, UID128>(e.entry, e.value));
        }

        /// <summary>What <paramref name="entry"/> links to, as recognition would see it now - see <see cref="Flush"/>.</summary>
        public bool TryGetValue(string entry, out UID128 uid)
        {
            Initialize();

            if (_legacy is object) { uid = default; return false; }

            _engine.IgnoreCase = Data.IgnoreCase;
            _engine.FlushIfIdle();
            return _engine.TryGetValue(entry, out uid);
        }

        public void ClearModel()
        {
            Initialize();

            lock (_syncRoot)
            {
                _legacy = null;
                _engine.Clear();
                _engine.IgnoreCase = Data.IgnoreCase;

                Data.Hashes                 = null;
                Data.MultiGramHashes        = null;
                Data.TokenizerExceptionsSet = null;
                Data.EntriesPayload         = null;
                Data.EntriesBlockOffsets    = null;
                Data.EntriesCount           = 0;
                Data.EntryValues            = null;
                Data.ExceptionBuckets       = null;
                Data.ExceptionLows          = null;
                Data.ExceptionCount         = 0;
                Data.MinTokenLength         = 0;
                Data.MaxTokenLength         = 0;
            }
        }

        /// <summary>
        /// Links <paramref name="entry"/> to <paramref name="uid"/>, replacing whatever it linked to before. The entry
        /// is buffered and matched from the next <see cref="Flush"/>, as a small segment next to what the model
        /// already holds: adding to a model of millions of entries does not rebuild it.
        /// </summary>
        public void AddEntry(string entry, UID128 uid)
        {
            Initialize();

            if (_legacy is object) { _legacy.AddEntry(entry, uid, Data, Language); return; }

            _engine.IgnoreCase = Data.IgnoreCase;
            _engine.Add(entry, Data.IgnoreOnlyNumeric, uid);

            Data.MinTokenLength = _engine.MinTokenLength;
            Data.MaxTokenLength = _engine.MaxTokenLength;
        }

        /// <summary>
        /// Stops matching <paramref name="entry"/> if, when the removal is applied, it still links to
        /// <paramref name="uid"/>: an entry that has meanwhile been linked to something else - another node holding
        /// the same name - is left alone. Applied at the next <see cref="Flush"/>, in order with the additions around
        /// it. Returns false for a model stored before the entry dictionary existed, which cannot remove entries.
        /// </summary>
        public bool RemoveEntry(string entry, UID128 uid)
        {
            Initialize();

            if (_legacy is object) { return false; }

            _engine.IgnoreCase = Data.IgnoreCase;
            return _engine.Remove(entry, uid, onlyIfLinkedTo: true);
        }

        public void AppendList(IEnumerable<(string word, UID128 uid)> words)
        {
            foreach (var (word, uid) in words)
            {
                AddEntry(word, uid);
            }
        }
    }
}
