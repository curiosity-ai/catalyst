using Microsoft.Extensions.Logging;
using Mosaik.Core;
using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using UID;

namespace Catalyst.Models
{
    public class LinkedSpotterModel : StorableObjectData
    {
        public Dictionary<ulong, UID128> Hashes { get; set; } = new Dictionary<ulong, UID128>();
        public List<HashSet<ulong>> MultiGramHashes { get; set; } = new List<HashSet<ulong>>();
        public string CaptureTag { get; set; }
        public HashSet<int> TokenizerExceptionsSet { get; set; } = new HashSet<int>();
        public bool IgnoreOnlyNumeric { get; set; }
        public bool IgnoreCase { get; set; }
    }

    public class LinkedSpotter : StorableObjectV2<LinkedSpotter, LinkedSpotterModel>, IEntityRecognizer, IProcess, IHasSimpleSpecialCases, ICanOptimizeMemory
    {
        public string CaptureTag => Data.CaptureTag;

        public bool IgnoreCase { get { return Data.IgnoreCase; } set { Data.IgnoreCase = value; } }

        public const string Separator = "_";

        private ICompactHashMap64   _frozenHashes;
        private ICompactHashSet64[] _frozenMultiGram;
        private bool                _frozen;

        /// <summary>True once the model's hash tables have been compacted into their read-only in-memory form.</summary>
        public bool IsMemoryOptimized => _frozen;

        /// <summary>Estimated bytes held by the compacted hash tables, or 0 when the model is not compacted.</summary>
        public long OptimizedMemoryBytes
        {
            get
            {
                if (!_frozen) { return 0; }
                long mem = _frozenHashes?.EstimatedBytes ?? 0;
                if (_frozenMultiGram is object)
                {
                    foreach (var s in _frozenMultiGram) { mem += s.EstimatedBytes; }
                }
                return mem;
            }
        }

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

        public void TrimExcess()
        {
            if (Data is null) return;

            if (Data.MultiGramHashes is object)
            {

                Data.MultiGramHashes.TrimExcess();

                foreach (var v in Data.MultiGramHashes)
                {
                    v.TrimExcess();
                }
            }
            Data.TokenizerExceptionsSet?.TrimExcess();
            Data.Hashes?.TrimExcess();

            Freeze();
        }

        // Replaces the trained Dictionary/HashSet lookups with compact, read-only equivalents and releases
        // the originals. Keeps TokenizerExceptionsSet intact (consumed later, then dropped by OptimizeMemory).
        private void Freeze()
        {
            if (_frozen || Data is null || Data.Hashes is null) { return; }

            _frozenHashes = CompactHash.BuildMap(Data.Hashes);

            var multi = Data.MultiGramHashes;
            _frozenMultiGram = new ICompactHashSet64[multi?.Count ?? 0];
            for (int i = 0; i < _frozenMultiGram.Length; i++)
            {
                _frozenMultiGram[i] = CompactHash.BuildSet(multi[i]);
            }

            _frozen              = true;
            Data.Hashes          = null;
            Data.MultiGramHashes = null;
        }

        // Rebuilds the mutable Dictionary/HashSet representation from the compact tables so the model can be
        // mutated or re-stored. Only possible for lossless (exact) compaction.
        private void Unfreeze()
        {
            if (!_frozen) { return; }

            if (_frozenHashes is object && !_frozenHashes.CanEnumerateKeys)
            {
                throw new InvalidOperationException("This LinkedSpotter was loaded with fingerprint compression (SpotterCompaction.UseFingerprint32) and cannot be modified or re-stored losslessly. Reload it with fingerprint compression disabled to modify it.");
            }

            var hashes = new Dictionary<ulong, UID128>(_frozenHashes?.Count ?? 0);
            if (_frozenHashes is object)
            {
                foreach (var kv in _frozenHashes.Entries()) { hashes[kv.Key] = kv.Value; }
            }
            Data.Hashes = hashes;

            var multi = new List<HashSet<ulong>>(_frozenMultiGram?.Length ?? 0);
            if (_frozenMultiGram is object)
            {
                foreach (var s in _frozenMultiGram)
                {
                    var hs = new HashSet<ulong>(s.Count);
                    foreach (var k in s.Keys()) { hs.Add(k); }
                    multi.Add(hs);
                }
            }
            Data.MultiGramHashes = multi;

            Data.TokenizerExceptionsSet ??= new HashSet<int>();

            _frozen          = false;
            _frozenHashes    = null;
            _frozenMultiGram = null;
        }

        public override async Task StoreAsync(System.IO.Stream stream)
        {
            bool wasFrozen = _frozen;
            Unfreeze();
            await base.StoreAsync(stream);
            if (wasFrozen) { Freeze(); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool HasMultiGram() => _frozen ? _frozenMultiGram.Length > 0 : Data.MultiGramHashes.Count > 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int MultiGramCount() => _frozen ? _frozenMultiGram.Length : Data.MultiGramHashes.Count;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MultiGramContains(int n, ulong hash) => _frozen ? _frozenMultiGram[n].Contains(hash) : Data.MultiGramHashes[n].Contains(hash);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool HashesTryGetValue(ulong hash, out UID128 uid) => _frozen ? _frozenHashes.TryGetValue(hash, out uid) : Data.Hashes.TryGetValue(hash, out uid);

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

        public void OptimizeMemory()
        {
            Data.TokenizerExceptionsSet?.Clear();
            Data.TokenizerExceptionsSet = null;
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
            _frozen          = false;
            _frozenHashes    = null;
            _frozenMultiGram = null;

            Data.Hashes                 = new Dictionary<ulong, UID128>();
            Data.MultiGramHashes        = new List<HashSet<ulong>>();
            Data.TokenizerExceptionsSet = new HashSet<int>();
        }

        public bool RecognizeEntities(Span ispan, bool stopOnFirstFound = false)
        {
            var pooledTokens = ispan.ToTokenSpanPolled(out var actualLength);
            var tokens = pooledTokens.AsSpan(0, actualLength);

            int N = tokens.Length;
            bool hasMultiGram = HasMultiGram();
            bool foundAny = false;
            for (int i = 0; i < N; i++)
            {
                var tk = tokens[i];
                //if (tk.POS != PartOfSpeechEnum.NOUN && tk.POS != PartOfSpeechEnum.ADJ && tk.POS != PartOfSpeechEnum.PROPN) { continue; }

                var tokenHash = Data.IgnoreCase ? IgnoreCaseHash64(tk.ValueAsSpan) : Hash64(tk.ValueAsSpan);

                if (hasMultiGram && MultiGramContains(0, tokenHash))
                {
                    int window = Math.Min(N - i, MultiGramCount());
                    ulong hash = tokenHash;
                    bool someTokenHasReplacements = tk.Replacement is object;
                    int i_final = i;
                    UID128 uid_final = default;

                    for (int n = 1; n < window; n++)
                    {
                        var next = tokens[n + i];
                        someTokenHasReplacements |= (next.Replacement is object);

                        var nextHash = Data.IgnoreCase ? IgnoreCaseHash64(next.ValueAsSpan) : Hash64(next.ValueAsSpan);
                        if (MultiGramContains(n, nextHash))
                        {
                            //txt += " " + next.Value;
                            //var hashTxt = Hash64(txt);
                            hash = HashCombine64(hash, nextHash);
                            if (HashesTryGetValue(hash, out var uid_multi))
                            {
                                i_final = i + n;
                                uid_final = uid_multi;
                            }
                        }
                        else
                        {
                            break;
                        }
                    }

                    if (i_final > i)
                    {
                        foundAny = true;
                        if (stopOnFirstFound) { return foundAny; } //Used for checking if the document contains any entity
                        tk.AddEntityType(new EntityType(CaptureTag, EntityTag.Begin, uid_final));
                        tokens[i_final].AddEntityType(new EntityType(CaptureTag, EntityTag.End, uid_final));

                        for (int m = i + 1; m < (i_final); m++)
                        {
                            tokens[m].AddEntityType(new EntityType(CaptureTag, EntityTag.Inside, uid_final));
                        }
                    }

                    i = i_final;
                }

                if (HashesTryGetValue(tokenHash, out var uid))
                {
                    foundAny = true;
                    if (stopOnFirstFound) { return foundAny; } //Used for checking if the document contains any entity
                    tk.AddEntityType(new EntityType(CaptureTag, EntityTag.Single, uid));
                }
            }
            
            ArrayPool<Token>.Shared.Return(pooledTokens);

            return foundAny;
        }

        private ReaderWriterLockSlim TrainLock = new ReaderWriterLockSlim();

        public IEnumerable<int> GetSimpleSpecialCases()
        {
            if (Data.TokenizerExceptionsSet is object)
            {
                foreach (var sc in Data.TokenizerExceptionsSet)
                {
                    yield return sc;
                }
            }
        }

        public void AddEntry(string entry, UID128 uid)
        {
            if (string.IsNullOrWhiteSpace(entry)) { return; }

            if (_frozen) { Unfreeze(); }

            if (Data.IgnoreOnlyNumeric && int.TryParse(entry, out _)) { return; } //Ignore pure numerical entries

            //The logic below uses SpanSplitEnumerator and is the allocation-free version of this:
            //  var words = entry.Trim().Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);

            var entrySpan = entry.AsSpan();
            var partsEnumerator = entrySpan.Split(' ' );

            int wordsLength = 0;
            Range currentPart;
            Range validCurrentPart = default;
            while (partsEnumerator.MoveNext())
            {
                currentPart = partsEnumerator.Current;

                if (currentPart.End.Value > currentPart.Start.Value) //Skip empty entries (i.e. 'Hello   World' would be split into: 'Hello', '', '', '', 'World')
                {
                    validCurrentPart = currentPart;
                    wordsLength++;
                }
            }

            if (wordsLength == 1)
            {
                var wordSpan = entrySpan.Slice(validCurrentPart.Start.Value, validCurrentPart.End.Value - validCurrentPart.Start.Value);
                var hash = Data.IgnoreCase ? Spotter.IgnoreCaseHash64(wordSpan) : Spotter.Hash64(wordSpan);
                
                Data.Hashes[hash] = uid;

                if (!wordSpan.IsAllLetterOrDigit())
                {
                    Data.TokenizerExceptionsSet.Add(wordSpan.CaseSensitiveHash32());
                }

                return;
            }

            partsEnumerator = entrySpan.Split(' ');

            ulong combinedHash = 0;
            int n = 0;
            while (partsEnumerator.MoveNext())
            {
                currentPart = partsEnumerator.Current;

                if (currentPart.End.Value > currentPart.Start.Value) //Skip empty entries (i.e. 'Hello   World' would be split into: 'Hello', '', '', '', 'World')
                {
                    var wordSpan = entrySpan.Slice(currentPart.Start.Value, currentPart.End.Value - currentPart.Start.Value);

                    var word_hash = Data.IgnoreCase ? Spotter.IgnoreCaseHash64(wordSpan) : Spotter.Hash64(wordSpan);
                    if (n == 0) { combinedHash = word_hash; } else { combinedHash = Spotter.HashCombine64(combinedHash, word_hash); }

                    if (Data.MultiGramHashes.Count < n + 1)
                    {
                        Data.MultiGramHashes.Add(new HashSet<ulong>());
                    }

                    if (!Data.MultiGramHashes[n].Contains(word_hash))
                    {
                        Data.MultiGramHashes[n].Add(word_hash);
                    }

                    if (!wordSpan.IsAllLetterOrDigit())
                    {
                        Data.TokenizerExceptionsSet.Add(wordSpan.CaseSensitiveHash32());
                    }
                 
                    n++;
                }
            }

            Data.Hashes[combinedHash] = uid;
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