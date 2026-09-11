using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using Mosaik.Core;
using UID;

namespace Catalyst.Models
{
    /// <summary>
    /// The hash-table representation the spotters used before the entry dictionary: a set of 64-bit hashes of
    /// whole entries, plus one set per word position telling a walk whether it may take another token.
    ///
    /// It is kept because it is the only thing a model stored in the old format contains - hashes cannot be
    /// turned back into the strings the dictionary needs - so such a model keeps matching exactly as it did.
    /// Nothing writes this format any more; a model gains the dictionary by being rebuilt from its source.
    /// </summary>
    internal sealed class LegacySpotterTables
    {
        private readonly SpotterModel _data;

        private HashSet<ulong>       _hashes;
        private List<HashSet<ulong>> _multiGram;
        private ICompactHashSet64    _frozenHashes;
        private ICompactHashSet64[]  _frozenMultiGram;
        private readonly CompactHash32Set _exceptions = CompactHash32Set.CreateEmpty();

        public bool IsFrozen { get; private set; }

        public CompactHash32Set Exceptions => _exceptions;

        public long EstimatedBytes
        {
            get
            {
                if (!IsFrozen) { return 0; }
                long memory = _frozenHashes?.EstimatedBytes ?? 0;
                if (_frozenMultiGram is object)
                {
                    foreach (var set in _frozenMultiGram) { memory += set.EstimatedBytes; }
                }
                return memory + _exceptions.EstimatedBytes;
            }
        }

        public LegacySpotterTables(SpotterModel data)
        {
            _data      = data;
            _hashes    = data.Hashes ?? new HashSet<ulong>();
            _multiGram = data.MultiGramHashes ?? new List<HashSet<ulong>>();
        }

        public void Freeze()
        {
            if (IsFrozen) { return; }

            _hashes.TrimExcess();
            foreach (var set in _multiGram) { set.TrimExcess(); }

            _frozenHashes    = CompactHash.BuildSet(_hashes);
            _frozenMultiGram = new ICompactHashSet64[_multiGram.Count];
            for (int i = 0; i < _frozenMultiGram.Length; i++) { _frozenMultiGram[i] = CompactHash.BuildSet(_multiGram[i]); }

            // The spotters only ever recorded "keep this word as it is", so the exception values carry no
            // replacements and the table collapses to a set of hashes.
            if (_data.TokenizerExceptions is object && _data.TokenizerExceptions.Count > 0)
            {
                _exceptions.ReplaceWith(CompactHash32Set.Build(_data.TokenizerExceptions.Keys.ToArray()));
            }

            IsFrozen                  = true;
            _hashes                   = null;
            _multiGram                = null;
            _data.Hashes              = null;
            _data.MultiGramHashes     = null;
            _data.TokenizerExceptions = null;
        }

        public void Unfreeze()
        {
            if (!IsFrozen) { return; }

            if (_frozenHashes is object && !_frozenHashes.CanEnumerateKeys)
            {
                throw new InvalidOperationException("This spotter was loaded with fingerprint compression (SpotterCompaction.UseFingerprint32) and cannot be modified or re-stored losslessly. Reload it with fingerprint compression disabled to modify it.");
            }

            _hashes = new HashSet<ulong>(_frozenHashes?.Count ?? 0);
            if (_frozenHashes is object)
            {
                foreach (var key in _frozenHashes.Keys()) { _hashes.Add(key); }
            }

            _multiGram = new List<HashSet<ulong>>(_frozenMultiGram?.Length ?? 0);
            if (_frozenMultiGram is object)
            {
                foreach (var set in _frozenMultiGram)
                {
                    var rebuilt = new HashSet<ulong>(set.Count);
                    foreach (var key in set.Keys()) { rebuilt.Add(key); }
                    _multiGram.Add(rebuilt);
                }
            }

            var exceptions = new Dictionary<int, TokenizationException>(_exceptions.Count);
            foreach (var hash in _exceptions.Hashes()) { exceptions[hash] = new TokenizationException(null); }

            _data.Hashes              = _hashes;
            _data.MultiGramHashes     = _multiGram;
            _data.TokenizerExceptions = exceptions;

            IsFrozen         = false;
            _frozenHashes    = null;
            _frozenMultiGram = null;
        }

        public bool IsEquivalentTo(LegacySpotterTables other)
        {
            Unfreeze();
            other.Unfreeze();

            return _hashes.SetEquals(other._hashes)
                && _multiGram.Count == other._multiGram.Count
                && _multiGram.Zip(other._multiGram, (a, b) => a.SetEquals(b)).All(equal => equal);
        }

        public void AddEntry(string entry, SpotterModel data, Language language)
        {
            if (string.IsNullOrWhiteSpace(entry)) { return; }
            if (data.IgnoreOnlyNumeric && int.TryParse(entry, out _)) { return; }

            Unfreeze();

            var words = entry.Trim().Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (words.Length == 0) { return; }

            if (words.Length == 1)
            {
                _hashes.Add(data.IgnoreCase ? Spotter.IgnoreCaseHash64(words[0].AsSpan()) : Spotter.Hash64(words[0].AsSpan()));
                ObserveTokenLength(data, words[0].Length);
                if (FastTokenizer.WouldSplit(words[0].AsSpan(), language)) { _data.TokenizerExceptions[words[0].CaseSensitiveHash32()] = new TokenizationException(null); }
                return;
            }

            ulong combinedHash = 0;
            for (int n = 0; n < words.Length; n++)
            {
                var wordHash = data.IgnoreCase ? Spotter.IgnoreCaseHash64(words[n].AsSpan()) : Spotter.Hash64(words[n].AsSpan());
                ObserveTokenLength(data, words[n].Length);
                combinedHash = n == 0 ? wordHash : Spotter.HashCombine64(combinedHash, wordHash);

                while (_multiGram.Count < n + 1) { _multiGram.Add(new HashSet<ulong>()); }
                _multiGram[n].Add(wordHash);

                if (FastTokenizer.WouldSplit(words[n].AsSpan(), language)) { _data.TokenizerExceptions[words[n].CaseSensitiveHash32()] = new TokenizationException(null); }
            }

            _hashes.Add(combinedHash);
        }

        private static void ObserveTokenLength(SpotterModel data, int length)
        {
            if (length <= 0) { return; }
            if (data.MaxTokenLength == 0) { data.MinTokenLength = length; data.MaxTokenLength = length; return; }
            if (length < data.MinTokenLength) { data.MinTokenLength = length; }
            if (length > data.MaxTokenLength) { data.MaxTokenLength = length; }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MultiGramContains(int n, ulong hash) => IsFrozen ? _frozenMultiGram[n].Contains(hash) : _multiGram[n].Contains(hash);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool HashesContains(ulong hash) => IsFrozen ? _frozenHashes.Contains(hash) : _hashes.Contains(hash);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool CouldMatchLength(int length) => _data.MaxTokenLength == 0 || (length >= _data.MinTokenLength && length <= _data.MaxTokenLength);

        public bool Match(Span<Token> tokens, string captureTag, bool stopOnFirstFound)
        {
            int  N            = tokens.Length;
            int  multiGrams   = IsFrozen ? _frozenMultiGram.Length : _multiGram.Count;
            bool hasMultiGram = multiGrams > 0;
            bool foundAny     = false;
            bool ignoreCase   = _data.IgnoreCase;

            for (int i = 0; i < N; i++)
            {
                var token = tokens[i];
                if (!CouldMatchLength(token.Length)) { continue; }

                var tokenHash = ignoreCase ? Spotter.IgnoreCaseHash64(token.ValueAsSpan) : Spotter.Hash64(token.ValueAsSpan);

                if (hasMultiGram && MultiGramContains(0, tokenHash))
                {
                    int   window = Math.Min(N - i, multiGrams);
                    ulong hash   = tokenHash;
                    int   last   = i;

                    for (int n = 1; n < window; n++)
                    {
                        var next = tokens[n + i];
                        if (!CouldMatchLength(next.Length)) { break; }

                        var nextHash = ignoreCase ? Spotter.IgnoreCaseHash64(next.ValueAsSpan) : Spotter.Hash64(next.ValueAsSpan);
                        if (!MultiGramContains(n, nextHash)) { break; }

                        hash = Spotter.HashCombine64(hash, nextHash);
                        if (HashesContains(hash)) { last = i + n; }
                    }

                    if (last > i)
                    {
                        foundAny = true;
                        if (stopOnFirstFound) { return true; }
                        tokens[i].AddEntityType(new EntityType(captureTag, EntityTag.Begin));
                        tokens[last].AddEntityType(new EntityType(captureTag, EntityTag.End));
                        for (int m = i + 1; m < last; m++) { tokens[m].AddEntityType(new EntityType(captureTag, EntityTag.Inside)); }
                    }

                    i = last;
                }

                if (HashesContains(tokenHash))
                {
                    foundAny = true;
                    if (stopOnFirstFound) { return true; }
                    token.AddEntityType(new EntityType(captureTag, EntityTag.Single));
                }
            }

            return foundAny;
        }
    }

    /// <summary>The same, for a <see cref="LinkedSpotter"/> - the entry table maps to a <see cref="UID128"/> instead of being a set.</summary>
    internal sealed class LegacyLinkedSpotterTables
    {
        private readonly LinkedSpotterModel _data;

        private Dictionary<ulong, UID128> _hashes;
        private List<HashSet<ulong>>      _multiGram;
        private ICompactHashMap64         _frozenHashes;
        private ICompactHashSet64[]       _frozenMultiGram;
        private readonly CompactHash32Set _exceptions = CompactHash32Set.CreateEmpty();

        public bool IsFrozen { get; private set; }

        public CompactHash32Set Exceptions => _exceptions;

        public long EstimatedBytes
        {
            get
            {
                if (!IsFrozen) { return 0; }
                long memory = _frozenHashes?.EstimatedBytes ?? 0;
                if (_frozenMultiGram is object)
                {
                    foreach (var set in _frozenMultiGram) { memory += set.EstimatedBytes; }
                }
                return memory + _exceptions.EstimatedBytes;
            }
        }

        public LegacyLinkedSpotterTables(LinkedSpotterModel data)
        {
            _data      = data;
            _hashes    = data.Hashes ?? new Dictionary<ulong, UID128>();
            _multiGram = data.MultiGramHashes ?? new List<HashSet<ulong>>();
        }

        public void Freeze()
        {
            if (IsFrozen) { return; }

            _hashes.TrimExcess();
            foreach (var set in _multiGram) { set.TrimExcess(); }

            _frozenHashes    = CompactHash.BuildMap(_hashes);
            _frozenMultiGram = new ICompactHashSet64[_multiGram.Count];
            for (int i = 0; i < _frozenMultiGram.Length; i++) { _frozenMultiGram[i] = CompactHash.BuildSet(_multiGram[i]); }

            if (_data.TokenizerExceptionsSet is object && _data.TokenizerExceptionsSet.Count > 0)
            {
                _exceptions.ReplaceWith(CompactHash32Set.Build(_data.TokenizerExceptionsSet));
            }

            IsFrozen                     = true;
            _hashes                      = null;
            _multiGram                   = null;
            _data.Hashes                 = null;
            _data.MultiGramHashes        = null;
            _data.TokenizerExceptionsSet = null;
        }

        public void Unfreeze()
        {
            if (!IsFrozen) { return; }

            if (_frozenHashes is object && !_frozenHashes.CanEnumerateKeys)
            {
                throw new InvalidOperationException("This LinkedSpotter was loaded with fingerprint compression (SpotterCompaction.UseFingerprint32) and cannot be modified or re-stored losslessly. Reload it with fingerprint compression disabled to modify it.");
            }

            _hashes = new Dictionary<ulong, UID128>(_frozenHashes?.Count ?? 0);
            if (_frozenHashes is object)
            {
                foreach (var entry in _frozenHashes.Entries()) { _hashes[entry.Key] = entry.Value; }
            }

            _multiGram = new List<HashSet<ulong>>(_frozenMultiGram?.Length ?? 0);
            if (_frozenMultiGram is object)
            {
                foreach (var set in _frozenMultiGram)
                {
                    var rebuilt = new HashSet<ulong>(set.Count);
                    foreach (var key in set.Keys()) { rebuilt.Add(key); }
                    _multiGram.Add(rebuilt);
                }
            }

            _data.Hashes                 = _hashes;
            _data.MultiGramHashes        = _multiGram;
            _data.TokenizerExceptionsSet = new HashSet<int>(_exceptions.Hashes());

            IsFrozen         = false;
            _frozenHashes    = null;
            _frozenMultiGram = null;
        }

        public void AddEntry(string entry, UID128 uid, LinkedSpotterModel data, Language language)
        {
            if (string.IsNullOrWhiteSpace(entry)) { return; }
            if (data.IgnoreOnlyNumeric && int.TryParse(entry, out _)) { return; }

            Unfreeze();

            var words = entry.Trim().Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (words.Length == 0) { return; }

            if (words.Length == 1)
            {
                _hashes[data.IgnoreCase ? Spotter.IgnoreCaseHash64(words[0].AsSpan()) : Spotter.Hash64(words[0].AsSpan())] = uid;
                ObserveTokenLength(data, words[0].Length);
                if (FastTokenizer.WouldSplit(words[0].AsSpan(), language)) { _data.TokenizerExceptionsSet.Add(words[0].CaseSensitiveHash32()); }
                return;
            }

            ulong combinedHash = 0;
            for (int n = 0; n < words.Length; n++)
            {
                var wordHash = data.IgnoreCase ? Spotter.IgnoreCaseHash64(words[n].AsSpan()) : Spotter.Hash64(words[n].AsSpan());
                ObserveTokenLength(data, words[n].Length);
                combinedHash = n == 0 ? wordHash : Spotter.HashCombine64(combinedHash, wordHash);

                while (_multiGram.Count < n + 1) { _multiGram.Add(new HashSet<ulong>()); }
                _multiGram[n].Add(wordHash);

                if (FastTokenizer.WouldSplit(words[n].AsSpan(), language)) { _data.TokenizerExceptionsSet.Add(words[n].CaseSensitiveHash32()); }
            }

            _hashes[combinedHash] = uid;
        }

        private static void ObserveTokenLength(LinkedSpotterModel data, int length)
        {
            if (length <= 0) { return; }
            if (data.MaxTokenLength == 0) { data.MinTokenLength = length; data.MaxTokenLength = length; return; }
            if (length < data.MinTokenLength) { data.MinTokenLength = length; }
            if (length > data.MaxTokenLength) { data.MaxTokenLength = length; }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool MultiGramContains(int n, ulong hash) => IsFrozen ? _frozenMultiGram[n].Contains(hash) : _multiGram[n].Contains(hash);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool HashesTryGetValue(ulong hash, out UID128 uid) => IsFrozen ? _frozenHashes.TryGetValue(hash, out uid) : _hashes.TryGetValue(hash, out uid);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool CouldMatchLength(int length) => _data.MaxTokenLength == 0 || (length >= _data.MinTokenLength && length <= _data.MaxTokenLength);

        public bool Match(Span<Token> tokens, string captureTag, bool stopOnFirstFound)
        {
            int  N            = tokens.Length;
            int  multiGrams   = IsFrozen ? _frozenMultiGram.Length : _multiGram.Count;
            bool hasMultiGram = multiGrams > 0;
            bool foundAny     = false;
            bool ignoreCase   = _data.IgnoreCase;

            for (int i = 0; i < N; i++)
            {
                var token = tokens[i];
                if (!CouldMatchLength(token.Length)) { continue; }

                var tokenHash = ignoreCase ? Spotter.IgnoreCaseHash64(token.ValueAsSpan) : Spotter.Hash64(token.ValueAsSpan);

                if (hasMultiGram && MultiGramContains(0, tokenHash))
                {
                    int    window    = Math.Min(N - i, multiGrams);
                    ulong  hash      = tokenHash;
                    int    last      = i;
                    UID128 lastValue = default;

                    for (int n = 1; n < window; n++)
                    {
                        var next = tokens[n + i];
                        if (!CouldMatchLength(next.Length)) { break; }

                        var nextHash = ignoreCase ? Spotter.IgnoreCaseHash64(next.ValueAsSpan) : Spotter.Hash64(next.ValueAsSpan);
                        if (!MultiGramContains(n, nextHash)) { break; }

                        hash = Spotter.HashCombine64(hash, nextHash);
                        if (HashesTryGetValue(hash, out var uid)) { last = i + n; lastValue = uid; }
                    }

                    if (last > i)
                    {
                        foundAny = true;
                        if (stopOnFirstFound) { return true; }
                        tokens[i].AddEntityType(new EntityType(captureTag, EntityTag.Begin, lastValue));
                        tokens[last].AddEntityType(new EntityType(captureTag, EntityTag.End, lastValue));
                        for (int m = i + 1; m < last; m++) { tokens[m].AddEntityType(new EntityType(captureTag, EntityTag.Inside, lastValue)); }
                    }

                    i = last;
                }

                if (HashesTryGetValue(tokenHash, out var single))
                {
                    foundAny = true;
                    if (stopOnFirstFound) { return true; }
                    token.AddEntityType(new EntityType(captureTag, EntityTag.Single, single));
                }
            }

            return foundAny;
        }
    }
}
