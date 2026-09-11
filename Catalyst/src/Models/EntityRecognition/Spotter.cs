using Microsoft.Extensions.Logging;
using Mosaik.Core;
using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using UID;

namespace Catalyst.Models
{
    public class SpotterModel : StorableObjectData
    {
        public string CaptureTag { get; set; }
        public bool IgnoreOnlyNumeric { get; set; }
        public bool IgnoreCase { get; set; }

        /// <summary>Smallest character length among the individual words stored in the model, or 0 when the model is empty.</summary>
        public int MinTokenLength { get; set; }
        /// <summary>Largest character length among the individual words stored in the model, or 0 when the model is empty.</summary>
        public int MaxTokenLength { get; set; }

        // The entries, as one sorted prefix-compressed dictionary of their surface forms. These blobs are the
        // in-memory structure verbatim, so loading a model is a read rather than a rebuild.
        public byte[] EntriesPayload { get; set; }
        public byte[] EntriesBlockOffsets { get; set; }
        public int EntriesCount { get; set; }
        public int EntriesBlockSize { get; set; }
        public int EntriesMaxBytes { get; set; }

        // The words the tokenizer would otherwise split, as a compact set of their case-sensitive 32-bit hashes.
        public byte[] ExceptionBuckets { get; set; }
        public byte[] ExceptionLows { get; set; }
        public int ExceptionCount { get; set; }

        // Superseded by the entry dictionary, kept so models stored before it still load and match. Never written.
        public HashSet<ulong> Hashes { get; set; } = new HashSet<ulong>();
        public List<HashSet<ulong>> MultiGramHashes { get; set; } = new List<HashSet<ulong>>();
        public Dictionary<int, TokenizationException> TokenizerExceptions { get; set; } = new Dictionary<int, TokenizationException>();
    }

    public class Spotter : StorableObjectV2<Spotter, SpotterModel>, IEntityRecognizer, IProcess, IHasSimpleSpecialCases, ICanOptimizeMemory
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

        // Only reached by a model stored before the entry dictionary existed; see LegacySpotterTables.
        private LegacySpotterTables _legacy;

        /// <summary>True once the model's entries have been compacted into their read-only in-memory form.</summary>
        public bool IsMemoryOptimized
        {
            get { Initialize(); return _legacy is object ? _legacy.IsFrozen : _engine.IsFrozen; }
        }

        /// <summary>Estimated bytes held by the compacted tables, or 0 when the model is not compacted.</summary>
        public long OptimizedMemoryBytes
        {
            get { Initialize(); return _legacy is object ? _legacy.EstimatedBytes : _engine.EstimatedBytes; }
        }

        /// <summary>True when this model was loaded from a store written before the entry dictionary existed.</summary>
        public bool IsLegacyModel { get { Initialize(); return _legacy is object; } }

        private Spotter(Language language, int version, string tag) : base(language, version, tag, compress: false)
        {
        }

        public Spotter(Language language, int version, string tag, string captureTag) : this(language, version, tag)
        {
            Data.CaptureTag = captureTag;
        }

        public new static async Task<Spotter> FromStoreAsync(Language language, int version, string tag)
        {
            var a = new Spotter(language, version, tag);
            await a.LoadDataAsync();
            a.TrimExcess();
            return a;
        }

        // Decides once, after Data is in place, whether this model speaks the entry dictionary or the older
        // hash tables. Everything public funnels through here so the choice cannot be missed.
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
                }
                else if ((Data.Hashes?.Count ?? 0) > 0 || (Data.MultiGramHashes?.Count ?? 0) > 0)
                {
                    _legacy = new LegacySpotterTables(Data);
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
                    if (!_engine.IsFrozen)
                    {
                        _engine.Freeze();
                        Data.MinTokenLength = _engine.MinTokenLength;
                        Data.MaxTokenLength = _engine.MaxTokenLength;
                    }
                }
            }
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
            WriteEngineToData();
            await base.StoreAsync(stream);
        }

        private void WriteEngineToData()
        {
            var (payload, blockOffsets, count, blockSize, maxEntryBytes) = _engine.Dictionary.ToBlobs();
            var (buckets, lows, exceptionCount)                         = _engine.Exceptions.ToBlobs();

            Data.EntriesPayload      = count > 0 ? payload : null;
            Data.EntriesBlockOffsets = count > 0 ? blockOffsets : null;
            Data.EntriesCount        = count;
            Data.EntriesBlockSize    = blockSize;
            Data.EntriesMaxBytes     = maxEntryBytes;
            Data.ExceptionBuckets    = exceptionCount > 0 ? buckets : null;
            Data.ExceptionLows       = exceptionCount > 0 ? lows : null;
            Data.ExceptionCount      = exceptionCount;
            Data.MinTokenLength      = _engine.MinTokenLength;
            Data.MaxTokenLength      = _engine.MaxTokenLength;

            Data.Hashes              = null;
            Data.MultiGramHashes     = null;
            Data.TokenizerExceptions = null;
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

            public void OnSingle(ref Token token, int rank) => token.AddEntityType(new EntityType(_captureTag, EntityTag.Single));

            public void OnMultiGram(Span<Token> tokens, int begin, int end, int rank)
            {
                tokens[begin].AddEntityType(new EntityType(_captureTag, EntityTag.Begin));
                tokens[end].AddEntityType(new EntityType(_captureTag, EntityTag.End));

                for (int m = begin + 1; m < end; m++)
                {
                    tokens[m].AddEntityType(new EntityType(_captureTag, EntityTag.Inside));
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

                var sink = new Sink(CaptureTag);
                return _engine.Match(tokens, stopOnFirstFound, ref sink);
            }
            finally
            {
                ArrayPool<Token>.Shared.Return(pooledTokens);
            }
        }

        public bool IsEquivalentTo(Spotter other)
        {
            Initialize();
            other.Initialize();

            if (Data.IgnoreOnlyNumeric != other.Data.IgnoreOnlyNumeric || Data.IgnoreCase != other.Data.IgnoreCase) { return false; }
            if ((_legacy is object) != (other._legacy is object)) { return false; }

            if (_legacy is object) { return _legacy.IsEquivalentTo(other._legacy); }

            EnsureFrozen();
            other.EnsureFrozen();

            var mine   = _engine.Dictionary.ToBlobs();
            var theirs = other._engine.Dictionary.ToBlobs();
            return mine.count == theirs.count && mine.payload.AsSpan().SequenceEqual(theirs.payload);
        }

        public static ulong HashCombine64(ulong rhs, ulong lhs)
        {
            lhs ^= rhs + 0x9e3779b97f492000 + (lhs << 6) + (lhs >> 2);
            return lhs;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Hash64(ReadOnlySpan<char> key)
        {
            ulong hashedValue = 3074457345618258791ul;
            for (int i = 0; i < key.Length; i++)
            {
                hashedValue += key[i];
                hashedValue *= 3074457345618258799ul;
            }
            return hashedValue;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong IgnoreCaseHash64(ReadOnlySpan<char> key)
        {
            ulong hashedValue = 3074457345618258791ul;
            for (int i = 0; i < key.Length; i++)
            {
                hashedValue += char.ToLowerInvariant(key[i]);
                hashedValue *= 3074457345618258799ul;
            }
            return hashedValue;
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

                Data.Hashes              = null;
                Data.MultiGramHashes     = null;
                Data.TokenizerExceptions = null;
                Data.EntriesPayload      = null;
                Data.EntriesBlockOffsets = null;
                Data.EntriesCount        = 0;
                Data.ExceptionBuckets    = null;
                Data.ExceptionLows       = null;
                Data.ExceptionCount      = 0;
                Data.MinTokenLength      = 0;
                Data.MaxTokenLength      = 0;
            }
        }

        /// <summary>The entries this model recognises, in sorted order.</summary>
        public IEnumerable<string> GetEntries()
        {
            Initialize();
            return _legacy is object ? Enumerable.Empty<string>() : _engine.Entries();
        }

        public CompactHash32Set GetSimpleSpecialCases()
        {
            Initialize();
            EnsureFrozen();
            return _legacy is object ? _legacy.Exceptions : _engine.Exceptions;
        }

        public void AddEntry(string entry)
        {
            Initialize();

            if (_legacy is object) { _legacy.AddEntry(entry, Data, Language); return; }

            _engine.IgnoreCase = Data.IgnoreCase;
            _engine.Add(entry, Data.IgnoreOnlyNumeric);
            Data.MinTokenLength = _engine.MinTokenLength;
            Data.MaxTokenLength = _engine.MaxTokenLength;
        }

        public void AppendList(IEnumerable<string> words)
        {
            foreach (var word in words)
            {
                AddEntry(word);
            }
        }

        private ReaderWriterLockSlim TrainLock = new ReaderWriterLockSlim();

        public void TrainWord2Sense(IEnumerable<IDocument> documents, ParallelOptions parallelOptions, int ngrams = 3, double tooRare = 1E-5, double tooCommon = 0.1, Word2SenseTrainingData trainingData = null)
        {
            Initialize();

            var hashCount          = new ConcurrentDictionary<ulong, int>(trainingData?.HashCount           ?? new Dictionary<ulong, int>());
            var senses             = new ConcurrentDictionary<ulong, ulong[]>(trainingData?.Senses          ?? new Dictionary<ulong, ulong[]>());
            var words              = new ConcurrentDictionary<ulong, string>(trainingData?.Words            ?? new Dictionary<ulong, string>());
            var shapes             = new ConcurrentDictionary<string, ulong>(trainingData?.Shapes           ?? new Dictionary<string, ulong>());
            var shapeExamples      = new ConcurrentDictionary<string, string[]>(trainingData?.ShapeExamples ?? new Dictionary<string, string[]>());

            long totalDocCount     = trainingData?.SeenDocuments ?? 0;
            long totalTokenCount   = trainingData?.SeenTokens ?? 0;

            bool ignoreCase        = Data.IgnoreCase;
            bool ignoreOnlyNumeric = Data.IgnoreOnlyNumeric;
            var stopwords          = new HashSet<ulong>(StopWords.Spacy.For(Language).Select(w => ignoreCase ? IgnoreCaseHash64(w.AsSpan()) : Hash64(w.AsSpan())).ToArray());

            int docCount = 0, tkCount = 0;

            var sw = Stopwatch.StartNew();

            TrainLock.EnterWriteLock();
            try
            {
                Parallel.ForEach(documents, parallelOptions, doc =>
                {
                    try
                    {
                        var stack = new Queue<ulong>(ngrams);

                        if (doc.TokensCount < ngrams) { return; } //Ignore too small documents

                        Interlocked.Add(ref tkCount, doc.TokensCount);

                        foreach (var span in doc)
                        {
                            var tokens = span.GetCapturedTokens().ToArray();

                            for (int i = 0; i < tokens.Length; i++)
                            {
                                var tk = tokens[i];

                                if (!(tk is Tokens))
                                {
                                    var shape = tk.ValueAsSpan.Shape(compact: false);
                                    shapes.AddOrUpdate(shape, 1, (k, v) => v + 1);

                                    shapeExamples.AddOrUpdate(shape, (k) => new[] { tk.Value }, (k, v) =>
                                    {
                                        if (v.Length < 50)
                                        {
                                            v = v.Concat(new[] { tk.Value }).Distinct().ToArray();
                                        }
                                        return v;
                                    });
                                }

                                var hash = ignoreCase ? IgnoreCaseHash64(tk.ValueAsSpan) : Hash64(tk.ValueAsSpan);

                                bool filterPartOfSpeech = !(tk.POS == PartOfSpeech.ADJ || tk.POS == PartOfSpeech.NOUN);

                                bool skipIfHasUpperCase = (!ignoreCase && !tk.ValueAsSpan.IsAllLowerCase());

                                bool skipIfTooSmall = (tk.Length < 3);

                                bool skipIfNotAllLetterOrDigit = !(tk.ValueAsSpan.IsAllLetterOrDigit());

                                bool skipIfStopWordOrEntity = stopwords.Contains(hash) || tk.EntityTypes.Any();

                                //Heuristic for ordinal numbers (i.e. 1st, 2nd, 33rd, etc)
                                bool skipIfMaybeOrdinal = (tk.ValueAsSpan.IndexOfAny(new char[] { '1', '2', '3', '4', '5', '6', '7', '8', '9', '0' }, 0) >= 0 &&
                                                           tk.ValueAsSpan.IndexOfAny(new char[] { 't', 'h', 's', 't', 'r', 'd' }, 0) >= 0 &&
                                                           tk.ValueAsSpan.IndexOfAny(new char[] { 'a', 'b', 'c', 'e', 'f', 'g', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'u', 'v', 'w', 'x', 'y', 'z' }, 0) < 0);

                                bool skipIfOnlyNumeric = ignoreOnlyNumeric ? !tk.ValueAsSpan.IsLetter() : false;

                                //Only filter for POS if language != any, as otherwise we won't have the POS information
                                bool skipThisToken = (filterPartOfSpeech && Language != Language.Any) || skipIfHasUpperCase || skipIfTooSmall || skipIfNotAllLetterOrDigit || skipIfStopWordOrEntity || skipIfMaybeOrdinal || skipIfOnlyNumeric;

                                if (skipThisToken)
                                {
                                    stack.Clear();
                                    continue;
                                }

                                if (!words.ContainsKey(hash)) { words[hash] = ignoreCase ? tk.Value.ToLowerInvariant() : tk.Value; }

                                stack.Enqueue(hash);
                                ulong combined = stack.ElementAt(0);

                                for (int j = 1; j < stack.Count; j++)
                                {
                                    combined = HashCombine64(combined, stack.ElementAt(j));
                                    if (hashCount.ContainsKey(combined))
                                    {
                                        hashCount[combined]++;
                                    }
                                    else
                                    {
                                        senses[combined] = stack.Take(j + 1).ToArray();
                                        hashCount[combined] = 1;
                                    }
                                }

                                if (stack.Count > ngrams) { stack.Dequeue(); }
                            }
                        }

                        int count = Interlocked.Increment(ref docCount);

                        if (count % 1000 == 0)
                        {
                            Logger.LogInformation("Training Word2Sense model - at {DOCCOUNT} documents, {TKCOUNT} tokens - elapsed {ELAPSED} seconds at {KTKS} kTk/s)", docCount, tkCount, sw.Elapsed.TotalSeconds, (tkCount / sw.ElapsedMilliseconds));
                        }
                    }
                    catch (Exception E)
                    {
                        Logger.LogError(E, "Error during training Word2Sense model");
                    }
                });
            }
            catch (OperationCanceledException)
            {
                return;
            }
            finally
            {
                TrainLock.ExitWriteLock();
            }

            Logger.LogInformation("Finish parsing documents for Word2Sense model");

            totalDocCount += docCount;
            totalTokenCount += tkCount;

            int thresholdRare   = Math.Max(2, (int)Math.Floor(tooRare * totalTokenCount));
            int thresholdCommon = (int)Math.Floor(tooCommon * totalTokenCount);

            var toKeep = hashCount.Where(kv => kv.Value >= thresholdRare && kv.Value <= thresholdCommon).OrderByDescending(kv => kv.Value)
                                                .Select(kv => kv.Key).ToArray();

            // A sense is the sequence of words that produced it, so it is stored as the entry it stands for
            // rather than as the combined hash - which is what lets the model be read back and re-exported.
            var sense = new StringBuilder();

            foreach (var key in toKeep)
            {
                if (!senses.TryGetValue(key, out var hashes)) { continue; }

                sense.Clear();
                bool complete = true;

                for (int i = 0; i < hashes.Length; i++)
                {
                    if (!words.TryGetValue(hashes[i], out var word)) { complete = false; break; }
                    if (i > 0) { sense.Append(' '); }
                    sense.Append(word);
                }

                if (complete) { AddEntry(sense.ToString()); }
            }

            if (trainingData is object)
            {
                trainingData.HashCount = new Dictionary<ulong, int>(hashCount);
                trainingData.Senses = new Dictionary<ulong, ulong[]>(senses);
                trainingData.Words = new Dictionary<ulong, string>(words);
                trainingData.SeenDocuments = totalDocCount;
                trainingData.SeenTokens = totalTokenCount;
                trainingData.Shapes = new Dictionary<string, ulong>(shapes);
                trainingData.ShapeExamples = new Dictionary<string, string[]>(shapeExamples);
            }

            Logger.LogInformation("Finish training Word2Sense model");
        }
    }

    [MessagePack.MessagePackObject(keyAsPropertyName: true)]
    public partial class Word2SenseTrainingData
    {
        public Dictionary<ulong, int> HashCount { get; set; } = new Dictionary<ulong, int>();
        public Dictionary<ulong, ulong[]> Senses { get; set; } = new Dictionary<ulong, ulong[]>();
        public Dictionary<ulong, string> Words { get; set; } = new Dictionary<ulong, string>();
        public Dictionary<string, ulong> Shapes { get; set; } = new Dictionary<string, ulong>();
        public Dictionary<string, string[]> ShapeExamples { get; set; } = new Dictionary<string, string[]>();
        public long SeenDocuments { get; set; } = 0;
        public long SeenTokens { get; set; } = 0;
    }
}
