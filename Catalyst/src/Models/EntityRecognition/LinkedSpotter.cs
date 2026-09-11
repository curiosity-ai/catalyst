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

        private UID128[] _values;        // by rank, once frozen
        private UID128[] _pending;       // by insertion index, while building
        private int      _pendingCount;

        private LegacyLinkedSpotterTables _legacy;

        /// <summary>True once the model's entries have been compacted into their read-only in-memory form.</summary>
        public bool IsMemoryOptimized
        {
            get { Initialize(); return _legacy is object ? _legacy.IsFrozen : _engine.IsFrozen; }
        }

        /// <summary>Estimated bytes held by the compacted tables, or 0 when the model is not compacted.</summary>
        public long OptimizedMemoryBytes
        {
            get
            {
                Initialize();
                if (_legacy is object) { return _legacy.EstimatedBytes; }
                if (!_engine.IsFrozen) { return 0; }
                return _engine.EstimatedBytes + 24L + (long)(_values?.Length ?? 0) * 16;
            }
        }

        /// <summary>True when this model was loaded from a store written before the entry dictionary existed.</summary>
        public bool IsLegacyModel { get { Initialize(); return _legacy is object; } }

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

                if (Data.EntriesCount > 0 && Data.EntriesPayload is object)
                {
                    _engine = new SpotterEngine { Language = Language, IgnoreCase = Data.IgnoreCase, MinTokenLength = Data.MinTokenLength, MaxTokenLength = Data.MaxTokenLength };
                    _engine.LoadFrom(Data.EntriesPayload, Data.EntriesBlockOffsets, Data.EntriesCount, Data.EntriesBlockSize, Data.EntriesMaxBytes,
                                     Data.ExceptionBuckets, Data.ExceptionLows, Data.ExceptionCount);
                    _values = Data.EntryValues ?? Array.Empty<UID128>();
                }
                else if ((Data.Hashes?.Count ?? 0) > 0 || (Data.MultiGramHashes?.Count ?? 0) > 0)
                {
                    _legacy = new LegacyLinkedSpotterTables(Data);
                }
                else
                {
                    _engine = new SpotterEngine { Language = Language, IgnoreCase = Data.IgnoreCase, MinTokenLength = Data.MinTokenLength, MaxTokenLength = Data.MaxTokenLength };
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

            if (!_engine.IsFrozen)
            {
                lock (_syncRoot)
                {
                    if (_engine.IsFrozen) { return; }

                    var order = _engine.Freeze();

                    if (order is object)
                    {
                        var values = new UID128[order.Length];
                        for (int rank = 0; rank < order.Length; rank++)
                        {
                            int insertion = order[rank];
                            values[rank]  = insertion < _pendingCount ? _pending[insertion] : default;
                        }
                        _values = values;
                    }

                    _values ??= Array.Empty<UID128>();
                    _pending      = null;
                    _pendingCount = 0;

                    Data.MinTokenLength = _engine.MinTokenLength;
                    Data.MaxTokenLength = _engine.MaxTokenLength;
                }
            }
        }

        // Entries come back from the dictionary in rank order, so the values array - which is indexed by rank -
        // becomes the by-insertion-index array the builder needs, unchanged.
        private void Reopen()
        {
            if (!_engine.IsFrozen) { return; }

            _engine.Unfreeze();
            _pending      = _values ?? Array.Empty<UID128>();
            _pendingCount = _pending.Length;
            _values       = null;
        }

        public void TrimExcess()
        {
            if (Data is null) { return; }
            Initialize();
            EnsureFrozen();
        }

        /// <summary>Compacts the in-memory tables. Idempotent with the compaction <see cref="TrimExcess"/> does on load.</summary>
        public void OptimizeMemory() => TrimExcess();

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

            EnsureFrozen();

            var (payload, blockOffsets, count, blockSize, maxEntryBytes) = _engine.Dictionary.ToBlobs();
            var (buckets, lows, exceptionCount)                         = _engine.Exceptions.ToBlobs();

            Data.EntriesPayload          = count > 0 ? payload : null;
            Data.EntriesBlockOffsets     = count > 0 ? blockOffsets : null;
            Data.EntriesCount            = count;
            Data.EntriesBlockSize        = blockSize;
            Data.EntriesMaxBytes         = maxEntryBytes;
            Data.EntryValues             = count > 0 ? _values : null;
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
            private readonly string   _captureTag;
            private readonly UID128[] _values;

            public Sink(string captureTag, UID128[] values) { _captureTag = captureTag; _values = values; }

            private UID128 ValueOf(int rank) => rank >= 0 && rank < _values.Length ? _values[rank] : default;

            public void OnSingle(ref Token token, int rank) => token.AddEntityType(new EntityType(_captureTag, EntityTag.Single, ValueOf(rank)));

            public void OnMultiGram(Span<Token> tokens, int begin, int end, int rank)
            {
                var value = ValueOf(rank);

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
            EnsureFrozen();

            var pooledTokens = ispan.ToTokenSpanPolled(out var actualLength);

            try
            {
                var tokens = pooledTokens.AsSpan(0, actualLength);

                if (_legacy is object) { return _legacy.Match(tokens, CaptureTag, stopOnFirstFound); }

                var sink = new Sink(CaptureTag, _values ?? Array.Empty<UID128>());
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
            if (_legacy is object) { yield break; }

            EnsureFrozen();
            int rank = 0;
            foreach (var entry in _engine.Entries())
            {
                yield return new KeyValuePair<string, UID128>(entry, rank < _values.Length ? _values[rank] : default);
                rank++;
            }
        }

        public void ClearModel()
        {
            Initialize();

            lock (_syncRoot)
            {
                _legacy = null;
                _engine ??= new SpotterEngine { Language = Language };
                _engine.Clear();
                _engine.IgnoreCase = Data.IgnoreCase;

                _values       = null;
                _pending      = null;
                _pendingCount = 0;

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

        public void AddEntry(string entry, UID128 uid)
        {
            Initialize();

            if (_legacy is object) { _legacy.AddEntry(entry, uid, Data, Language); return; }

            lock (_syncRoot)
            {
                if (_engine.IsFrozen) { Reopen(); }

                _engine.IgnoreCase = Data.IgnoreCase;
                int index = _engine.Add(entry, Data.IgnoreOnlyNumeric);
                if (index < 0) { return; }

                if (_pending is null) { _pending = new UID128[Math.Max(16, index + 1)]; }
                if (index >= _pending.Length) { Array.Resize(ref _pending, Math.Max(_pending.Length * 2, index + 1)); }

                _pending[index] = uid;
                if (index >= _pendingCount) { _pendingCount = index + 1; }

                Data.MinTokenLength = _engine.MinTokenLength;
                Data.MaxTokenLength = _engine.MaxTokenLength;
            }
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
