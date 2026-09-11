using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using Mosaik.Core;
using UID;

namespace Catalyst.Models
{
    /// <summary>Receives the matches a <see cref="SpotterEngine"/> walk finds, without the engine knowing how they are tagged.</summary>
    internal interface ISpotterMatchSink
    {
        void OnSingle(ref Token token, int rank);
        void OnMultiGram(Span<Token> tokens, int begin, int end, int rank);
    }

    /// <summary>
    /// The storage and the matching walk shared by <see cref="Spotter"/> and <see cref="LinkedSpotter"/>.
    ///
    /// Entries are held as one sorted, prefix-compressed dictionary of their surface forms rather than as a
    /// hash set of single tokens plus one hash set per word position. A multi-token entry is simply an entry
    /// containing separators, so the walk extends a candidate one token at a time for as long as the
    /// dictionary says some entry continues past what it has - there is no separate multi-gram mechanism to
    /// keep in step, and a lookup yields the entry's rank, which is how the tagged spotter reaches its value
    /// array without storing any keys.
    /// </summary>
    internal sealed class SpotterEngine
    {
        public EntryDictionary  Dictionary { get; private set; } = EntryDictionary.Empty;
        public CompactHash32Set Exceptions { get; } = CompactHash32Set.CreateEmpty();

        private EntryPrefilter    _prefilter = EntryPrefilter.Empty;
        private EntryStoreBuilder _builder;
        private volatile bool     _frozen;

        public bool     IgnoreCase { get; set; }
        public Language Language   { get; set; } = Language.Any;
        public int  MinTokenLength { get; set; }
        public int  MaxTokenLength { get; set; }

        public bool IsFrozen => _frozen;
        public int  Count    => _frozen ? Dictionary.Count : (_builder?.Count ?? 0);

        /// <summary>Bytes held by the read-only structures, or 0 while the engine is still being built.</summary>
        public long EstimatedBytes => _frozen ? Dictionary.EstimatedBytes + Exceptions.EstimatedBytes + _prefilter.EstimatedBytes : 0;

        // ---- building ------------------------------------------------------------------------------

        /// <summary>
        /// Normalizes and stores one entry, registering a tokenizer exception for any of its words the
        /// tokenizer would otherwise split. Returns the insertion index, or -1 when the entry is rejected.
        /// </summary>
        public int Add(string entry, bool ignoreOnlyNumeric)
        {
            if (string.IsNullOrWhiteSpace(entry)) { return -1; }
            if (ignoreOnlyNumeric && int.TryParse(entry, out _)) { return -1; } //Ignore pure numerical entries

            if (_frozen) { Unfreeze(); }
            _builder ??= new EntryStoreBuilder();

            var trimmed = entry.AsSpan().Trim();
            if (trimmed.Length == 0) { return -1; }

            int charCapacity = trimmed.Length;
            int byteCapacity = EntryText.MaxUtf8Bytes(trimmed.Length);
            var chars        = ArrayPool<char>.Shared.Rent(charCapacity);
            var bytes        = ArrayPool<byte>.Shared.Rent(byteCapacity);

            try
            {
                int length = EntryText.Normalize(trimmed, IgnoreCase, chars.AsSpan(0, charCapacity), bytes.AsSpan(0, byteCapacity), out int wordCount);
                if (length <= 0 || wordCount == 0) { return -1; }

                RegisterWords(trimmed);
                return _builder.Add(bytes.AsSpan(0, length));
            }
            finally
            {
                ArrayPool<char>.Shared.Return(chars);
                ArrayPool<byte>.Shared.Return(bytes);
            }
        }

        // Walks the entry's words in their original casing: the tokenizer hashes what it sees in the document,
        // so an exception has to be recorded under the spelling the entry was given in.
        private void RegisterWords(ReadOnlySpan<char> entry)
        {
            int i = 0;
            while (i < entry.Length)
            {
                while (i < entry.Length && entry[i] == ' ') { i++; }
                if (i >= entry.Length) { break; }

                int start = i;
                while (i < entry.Length && entry[i] != ' ') { i++; }

                var word = entry.Slice(start, i - start);
                ObserveTokenLength(word.Length);

                if (FastTokenizer.WouldSplit(word, Language)) { _builder.AddException(word.CaseSensitiveHash32()); }
            }
        }

        private void ObserveTokenLength(int length)
        {
            if (length <= 0) { return; }
            if (MaxTokenLength == 0) // the model was empty until now
            {
                MinTokenLength = length;
                MaxTokenLength = length;
            }
            else
            {
                if (length < MinTokenLength) { MinTokenLength = length; }
                if (length > MaxTokenLength) { MaxTokenLength = length; }
            }
        }

        /// <summary>
        /// Compacts what was added into the read-only structures. Returns the rank -> insertion-index map so
        /// a caller holding per-entry values can lay them out densely by rank, or null when nothing changed.
        /// </summary>
        public int[] Freeze()
        {
            if (_frozen) { return null; }

            if (_builder is null)
            {
                _frozen = true;
                return null;
            }

            var order  = _builder.SortDistinct(out int distinct);
            Dictionary = _builder.BuildDictionary(order, distinct);
            Exceptions.ReplaceWith(_builder.BuildExceptions());
            _prefilter = EntryPrefilter.Build(Dictionary);
            _builder   = null;
            _frozen    = true;

            Array.Resize(ref order, distinct);
            return order;
        }

        /// <summary>
        /// Re-opens the model for editing. Entries come back in rank order, so a caller's value array stays
        /// aligned: insertion index i now holds what rank i held.
        ///
        /// An entry added after the model has been put into a pipeline reaches the tokenizer at the next
        /// compaction - the next recognition call, or an explicit TrimExcess - so a word that needs a
        /// tokenization exception is matched from then on rather than immediately.
        /// </summary>
        public void Unfreeze()
        {
            if (!_frozen)
            {
                _builder ??= new EntryStoreBuilder();
                return;
            }

            var builder = new EntryStoreBuilder(Math.Max(16, Dictionary.Count));
            Dictionary.Visit((rank, entry) => builder.Add(entry));
            builder.AddExceptions(Exceptions.Hashes());

            _builder   = builder;
            Dictionary = EntryDictionary.Empty;
            _prefilter = EntryPrefilter.Empty;
            _frozen    = false;

            // The exception table is deliberately left standing. A tokenizer that already holds it by
            // reference keeps working on the entries the model had while the new ones are being added; the
            // next compaction replaces it with a superset. Emptying it here would silently stop splitting
            // correctly for every document processed in between.
        }

        public void Clear()
        {
            Dictionary     = EntryDictionary.Empty;
            _prefilter     = EntryPrefilter.Empty;
            Exceptions.ReplaceWith(CompactHash32Set.Empty);
            _builder       = new EntryStoreBuilder();
            _frozen        = false;
            MinTokenLength = 0;
            MaxTokenLength = 0;
        }

        /// <summary>Every stored entry, in rank order.</summary>
        public IEnumerable<string> Entries()
        {
            var entries = new List<string>(Count);
            if (_frozen)
            {
                Dictionary.Visit((rank, entry) => entries.Add(System.Text.Encoding.UTF8.GetString(entry)));
            }
            else if (_builder is object)
            {
                for (int i = 0; i < _builder.Count; i++) { entries.Add(System.Text.Encoding.UTF8.GetString(_builder.EntryAt(i))); }
            }
            return entries;
        }

        // ---- matching ------------------------------------------------------------------------------

        // A token whose length falls outside the window of every stored word cannot be part of any entry.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool CouldMatchLength(int length) => MaxTokenLength == 0 || (length >= MinTokenLength && length <= MaxTokenLength);

        public bool Match<TSink>(Span<Token> tokens, bool stopOnFirstFound, ref TSink sink) where TSink : struct, ISpotterMatchSink
        {
            var dictionary = Dictionary;
            if (dictionary.Count == 0) { return false; }

            int  maxBytes     = dictionary.MaxEntryBytes;
            var  keyBuffer    = ArrayPool<byte>.Shared.Rent(maxBytes + 8);
            var  decodeBuffer = ArrayPool<byte>.Shared.Rent(maxBytes + 8);
            var  charBuffer   = ArrayPool<char>.Shared.Rent(maxBytes + 8);
            bool ignoreCase   = IgnoreCase;
            bool foundAny     = false;
            int  N            = tokens.Length;

            try
            {
                for (int i = 0; i < N; i++)
                {
                    var token = tokens[i];
                    if (!CouldMatchLength(token.Length)) { continue; }

                    int length = EntryText.EncodeToken(token.ValueAsSpan, ignoreCase, charBuffer, keyBuffer.AsSpan(0, maxBytes));
                    if (length <= 0) { continue; }

                    if (!_prefilter.MayStart(EntryText.Hash(keyBuffer.AsSpan(0, length)))) { continue; }

                    dictionary.Probe(keyBuffer.AsSpan(0, length), EntryText.SEPARATOR, decodeBuffer, out int singleRank, out bool canExtend);

                    int last = i, lastRank = -1, j = i;

                    while (canExtend && j + 1 < N)
                    {
                        var next = tokens[j + 1];
                        if (!CouldMatchLength(next.Length)) { break; }
                        if (length + 1 >= maxBytes) { break; }

                        int restore = length;
                        keyBuffer[length] = EntryText.SEPARATOR;
                        int extra = EntryText.EncodeToken(next.ValueAsSpan, ignoreCase, charBuffer, keyBuffer.AsSpan(length + 1, maxBytes - length - 1));

                        if (extra <= 0) { length = restore; break; }
                        length += 1 + extra;
                        j++;

                        dictionary.Probe(keyBuffer.AsSpan(0, length), EntryText.SEPARATOR, decodeBuffer, out int rank, out canExtend);
                        if (rank >= 0) { lastRank = rank; last = j; }
                    }

                    if (last > i)
                    {
                        foundAny = true;
                        if (stopOnFirstFound) { return true; }
                        sink.OnMultiGram(tokens, i, last, lastRank);
                    }

                    if (singleRank >= 0)
                    {
                        foundAny = true;
                        if (stopOnFirstFound) { return true; }
                        sink.OnSingle(ref token, singleRank);
                    }

                    i = last;
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(keyBuffer);
                ArrayPool<byte>.Shared.Return(decodeBuffer);
                ArrayPool<char>.Shared.Return(charBuffer);
            }

            return foundAny;
        }

        // ---- serialization -------------------------------------------------------------------------

        public void LoadFrom(byte[] payload, byte[] blockOffsets, int count, int blockSize, int maxEntryBytes,
                             byte[] exceptionBuckets, byte[] exceptionLows, int exceptionCount)
        {
            Dictionary = EntryDictionary.FromBlobs(payload, blockOffsets, count, blockSize, maxEntryBytes);
            Exceptions.ReplaceWith(CompactHash32Set.FromBlobs(exceptionBuckets, exceptionLows, exceptionCount));
            _prefilter = EntryPrefilter.Build(Dictionary);
            _builder   = null;
            _frozen    = true;
        }
    }
}
