using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using Mosaik.Core;
using UID;

namespace Catalyst.Models
{
    /// <summary>Receives the matches a <see cref="SpotterEngine"/> walk finds, without the engine knowing how they are tagged.</summary>
    internal interface ISpotterMatchSink
    {
        void OnSingle(ref Token token, EntrySegment segment, int rank);
        void OnMultiGram(Span<Token> tokens, int begin, int end, EntrySegment segment, int rank);
    }

    /// <summary>Visits one entry of a merged walk over several segments: the segment whose copy of the entry is the newest, and its rank there.</summary>
    internal delegate void MergedEntryVisitor(ReadOnlySpan<byte> entry, EntrySegment segment, int rank);

    /// <summary>
    /// One immutable slice of a spotter's entries: a sorted dictionary, the prefilter over its first words, the
    /// values it links to (by rank), and which of its entries are removals of an entry an older segment holds.
    /// A segment is never changed once built; new data arrives as a new segment and segments are merged.
    /// </summary>
    internal sealed class EntrySegment
    {
        public EntryDictionary Dictionary { get; }
        public EntryPrefilter  Prefilter  { get; }
        public UID128[]        Values     { get; } // by rank; null for a spotter that links to nothing

        private readonly ulong[] _removed;         // by rank; null when the segment removes nothing

        public EntrySegment(EntryDictionary dictionary, UID128[] values, ulong[] removed)
        {
            Dictionary = dictionary;
            Values     = values;
            _removed   = removed;
            Prefilter  = EntryPrefilter.Build(dictionary);
        }

        public int  Count       => Dictionary.Count;
        public bool HasRemovals => _removed is object;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool IsRemoved(int rank) => _removed is object && (_removed[rank >> 6] & (1UL << (rank & 63))) != 0;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public UID128 ValueAt(int rank) => Values is object && (uint)rank < (uint)Values.Length ? Values[rank] : default;

        public long EstimatedBytes => 48L + Dictionary.EstimatedBytes + Prefilter.EstimatedBytes
                                    + (Values   is null ? 0 : 24L + (long)Values.Length   * 16)
                                    + (_removed is null ? 0 : 24L + (long)_removed.Length * sizeof(ulong));
    }

    /// <summary>
    /// The storage and the matching walk shared by <see cref="Spotter"/> and <see cref="LinkedSpotter"/>.
    ///
    /// Entries are held as sorted, prefix-compressed dictionaries of their surface forms rather than as a
    /// hash set of single tokens plus one hash set per word position. A multi-token entry is simply an entry
    /// containing separators, so the walk extends a candidate one token at a time for as long as the
    /// dictionary says some entry continues past what it has, and a lookup yields the entry's rank, which is
    /// how a segment reaches its value array without storing any keys.
    ///
    /// The entries live in a short list of immutable <see cref="EntrySegment"/>s, the way an inverted index
    /// keeps its segments: what is added or removed is buffered, a <see cref="Flush"/> turns the buffer into
    /// a new small segment, and segments of comparable size are merged into one. Adding a few entries to a
    /// model holding millions therefore costs a segment the size of what was added, not a rebuild of what was
    /// there - the large segment is only rewritten once the data added since it was built has grown to a
    /// fraction of it. The newest segment that holds an entry decides it: a later value replaces an earlier
    /// one, and a removal hides it until the merge that reaches the oldest segment drops both.
    ///
    /// Readers take the segment list once per call and never wait on a writer: a flush or a merge publishes
    /// a new list when it is complete.
    /// </summary>
    internal sealed class SpotterEngine
    {
        /// <summary>Two adjacent segments that hold no more than this many entries between them are always merged: rewriting them is cheaper than probing both.</summary>
        internal const int SMALL_SEGMENT_ENTRIES = 1024;

        /// <summary>
        /// A segment is merged into the older one before it unless the older one is more than this many times
        /// its size. Sizes therefore fall geometrically from the oldest segment to the newest, which bounds the
        /// segment count by log4 of the entry count and rewrites any entry about log4 times over its lifetime.
        /// </summary>
        internal const int MERGE_FACTOR = 4;

        private const byte OP_ADD              = 0;
        private const byte OP_REMOVE           = 1;
        private const byte OP_REMOVE_IF_LINKED = 2;

        private const int STATE_INHERIT = 0;
        private const int STATE_LIVE    = 1;
        private const int STATE_REMOVED = 2;

        private readonly bool   _withValues;
        private readonly object _writeLock = new object();

        private volatile EntrySegment[] _segments = Array.Empty<EntrySegment>();

        // What has been added or removed since the last flush, in the order it happened.
        private EntryStoreBuilder _pending;
        private byte[]            _pendingOps;
        private UID128[]          _pendingValues;
        private volatile int      _pendingCount;

        public CompactHash32Set Exceptions { get; } = CompactHash32Set.CreateEmpty();

        public bool     IgnoreCase { get; set; }
        public Language Language   { get; set; } = Language.Any;
        public int  MinTokenLength { get; set; }
        public int  MaxTokenLength { get; set; }

        public SpotterEngine(bool withValues)
        {
            _withValues = withValues;
        }

        /// <summary>True when nothing is waiting for a <see cref="Flush"/>.</summary>
        public bool IsFrozen => _pendingCount == 0;

        public int SegmentCount => _segments.Length;

        /// <summary>Entries held across the segments plus those waiting to be flushed - an upper bound, as a newer segment can repeat or remove an entry of an older one.</summary>
        public int Count
        {
            get
            {
                long count = _pendingCount;
                foreach (var segment in _segments) { count += segment.Count; }
                return (int)Math.Min(int.MaxValue, count);
            }
        }

        /// <summary>Bytes held by the read-only structures, or 0 while entries are waiting to be flushed.</summary>
        public long EstimatedBytes
        {
            get
            {
                if (_pendingCount > 0) { return 0; }

                long bytes = Exceptions.EstimatedBytes;
                foreach (var segment in _segments) { bytes += segment.EstimatedBytes; }
                return bytes;
            }
        }

        // ---- building ------------------------------------------------------------------------------

        /// <summary>
        /// Normalizes and buffers one entry, registering a tokenizer exception for any of its words the
        /// tokenizer would otherwise split. Returns false when the entry is rejected.
        /// </summary>
        public bool Add(string entry, bool ignoreOnlyNumeric, UID128 value = default)
        {
            if (string.IsNullOrWhiteSpace(entry)) { return false; }
            if (ignoreOnlyNumeric && int.TryParse(entry, out _)) { return false; } //Ignore pure numerical entries

            return Append(entry, OP_ADD, value);
        }

        /// <summary>
        /// Buffers the removal of an entry. With <paramref name="onlyIfLinkedTo"/> the entry is only removed
        /// while it still links to <paramref name="linkedTo"/>, so a node letting go of a name does not take it
        /// away from another node that holds the same name.
        /// </summary>
        public bool Remove(string entry, UID128 linkedTo, bool onlyIfLinkedTo)
        {
            if (string.IsNullOrWhiteSpace(entry)) { return false; }

            return Append(entry, onlyIfLinkedTo && _withValues ? OP_REMOVE_IF_LINKED : OP_REMOVE, linkedTo);
        }

        private bool Append(string entry, byte op, UID128 value)
        {
            var trimmed = entry.AsSpan().Trim();
            if (trimmed.Length == 0) { return false; }

            int length = Normalize(trimmed, out var chars, out var bytes);

            try
            {
                if (length <= 0) { return false; }

                lock (_writeLock)
                {
                    _pending ??= new EntryStoreBuilder(64);

                    if (op == OP_ADD) { RegisterWords(trimmed); }

                    int index = _pending.Add(bytes.AsSpan(0, length));

                    if (_pendingOps is null || index >= _pendingOps.Length)
                    {
                        int capacity = Math.Max(64, Math.Max(index + 1, (_pendingOps?.Length ?? 0) * 2));
                        Array.Resize(ref _pendingOps, capacity);
                        if (_withValues) { Array.Resize(ref _pendingValues, capacity); }
                    }

                    _pendingOps[index] = op;
                    if (_withValues) { _pendingValues[index] = value; }

                    _pendingCount = _pending.Count;
                }

                return true;
            }
            finally
            {
                ArrayPool<char>.Shared.Return(chars);
                ArrayPool<byte>.Shared.Return(bytes);
            }
        }

        // Returns the normalized byte length, or -1. The two rented buffers are the caller's to return.
        private int Normalize(ReadOnlySpan<char> trimmed, out char[] chars, out byte[] bytes)
        {
            int charCapacity = trimmed.Length;
            int byteCapacity = EntryText.MaxUtf8Bytes(trimmed.Length);
            chars            = ArrayPool<char>.Shared.Rent(charCapacity);
            bytes            = ArrayPool<byte>.Shared.Rent(byteCapacity);

            int length = EntryText.Normalize(trimmed, IgnoreCase, chars.AsSpan(0, charCapacity), bytes.AsSpan(0, byteCapacity), out int wordCount);
            return wordCount == 0 ? -1 : length;
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

                if (FastTokenizer.WouldSplit(word, Language)) { _pending.AddException(word.CaseSensitiveHash32()); }
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
        /// Turns what was buffered into a new segment and merges segments by the policy above. Everything
        /// buffered is matched from the moment this returns. Returns false when there was nothing to flush.
        /// </summary>
        public bool Flush()
        {
            if (_pendingCount == 0) { return false; }

            lock (_writeLock)
            {
                return FlushLocked();
            }
        }

        /// <summary>
        /// Flushes when something is buffered and no writer holds the model. A reader calls this before it
        /// matches: it never waits on a writer, it matches against what is published until the writer is done.
        /// </summary>
        public void FlushIfIdle()
        {
            if (_pendingCount == 0) { return; }

            if (Monitor.TryEnter(_writeLock))
            {
                try
                {
                    FlushLocked();
                }
                finally
                {
                    Monitor.Exit(_writeLock);
                }
            }
        }

        private bool FlushLocked()
        {
            var pending = _pending;
            if (pending is null || pending.Count == 0) { return false; }

            var ops    = _pendingOps;
            var values = _pendingValues;

            // The tokenizer has to keep a new word whole before the entry holding it can match.
            Exceptions.Add(pending.Exceptions);

            var current = _segments;
            var segment = BuildSegment(pending, ops, values, current);

            if (segment is object)
            {
                _segments = AppendAndMerge(current, segment);
            }

            _pending       = null;
            _pendingOps    = null;
            _pendingValues = null;
            _pendingCount  = 0;
            return true;
        }

        // Replays what happened to each buffered entry in order and keeps the outcome: its last value, a
        // removal of a value an older segment holds, or nothing at all.
        private EntrySegment BuildSegment(EntryStoreBuilder pending, byte[] ops, UID128[] values, EntrySegment[] current)
        {
            int n      = pending.Count;
            var order  = pending.SortGrouped();
            var keep   = new int[n];
            var linked = _withValues ? new UID128[n] : null;
            int kept   = 0;

            List<int> removedRanks = null;
            byte[]    scratch      = null;

            int i = 0;
            while (i < n)
            {
                var entry = pending.EntryAt(order[i]);

                int j = i + 1;
                while (j < n && pending.EntryAt(order[j]).SequenceEqual(entry)) { j++; }

                int    state = STATE_INHERIT;
                UID128 value = default;

                for (int k = i; k < j; k++)
                {
                    int index = order[k];

                    switch (ops[index])
                    {
                        case OP_ADD:    state = STATE_LIVE; value = _withValues ? values[index] : default; break;
                        case OP_REMOVE: state = STATE_REMOVED; break;
                        default:
                        {
                            var expected = values[index];

                            if (state == STATE_LIVE)
                            {
                                if (value == expected) { state = STATE_REMOVED; }
                            }
                            else if (state == STATE_INHERIT)
                            {
                                scratch ??= new byte[MaxEntryBytes(current) + 1];
                                if (TryResolve(current, entry, scratch, out var linkedNow) && linkedNow == expected) { state = STATE_REMOVED; }
                            }
                            break;
                        }
                    }
                }

                if (state == STATE_LIVE)
                {
                    keep[kept] = order[i];
                    if (linked is object) { linked[kept] = value; }
                    kept++;
                }
                else if (state == STATE_REMOVED)
                {
                    // A removal is only worth keeping while an older segment still holds the entry.
                    scratch ??= new byte[MaxEntryBytes(current) + 1];
                    if (TryResolve(current, entry, scratch, out _))
                    {
                        (removedRanks ??= new List<int>()).Add(kept);
                        keep[kept] = order[i];
                        kept++;
                    }
                }

                i = j;
            }

            if (kept == 0) { return null; }

            var dictionary = pending.BuildDictionary(keep, kept);

            if (linked is object && linked.Length != kept) { Array.Resize(ref linked, kept); }

            ulong[] removed = null;
            if (removedRanks is object)
            {
                removed = new ulong[(kept + 63) / 64];
                foreach (var rank in removedRanks) { removed[rank >> 6] |= 1UL << (rank & 63); }
            }

            return new EntrySegment(dictionary, linked, removed);
        }

        private static int MaxEntryBytes(EntrySegment[] segments)
        {
            int max = 0;
            foreach (var segment in segments) { max = Math.Max(max, segment.Dictionary.MaxEntryBytes); }
            return max;
        }

        // The newest segment holding the key decides. False when no segment holds it, or the newest one removes it.
        private static bool TryResolve(EntrySegment[] segments, ReadOnlySpan<byte> key, Span<byte> scratch, out UID128 value)
        {
            for (int s = segments.Length - 1; s >= 0; s--)
            {
                var segment = segments[s];
                int rank    = segment.Dictionary.Find(key, scratch);
                if (rank < 0) { continue; }

                if (segment.IsRemoved(rank)) { break; }

                value = segment.ValueAt(rank);
                return true;
            }

            value = default;
            return false;
        }

        private EntrySegment[] AppendAndMerge(EntrySegment[] current, EntrySegment added)
        {
            var segments = new List<EntrySegment>(current.Length + 1);
            segments.AddRange(current);
            segments.Add(added);

            while (segments.Count > 1)
            {
                int n     = segments.Count;
                var older = segments[n - 2];
                var newer = segments[n - 1];

                bool small      = (long)older.Count + newer.Count <= SMALL_SEGMENT_ENTRIES;
                bool comparable = older.Count <= (long)MERGE_FACTOR * newer.Count;

                if (!small && !comparable) { break; }

                // Only a merge that produces the oldest segment can drop removals: nothing older is left for them to hide.
                var merged = Merge(new[] { older, newer }, dropRemovals: n == 2);

                segments.RemoveRange(n - 2, 2);
                if (merged.Count > 0) { segments.Add(merged); }
            }

            return segments.ToArray();
        }

        /// <summary>Merges every segment and whatever is buffered into one, dropping removals. What a stored model is written from.</summary>
        public void Compact()
        {
            lock (_writeLock)
            {
                FlushLocked();

                var current = _segments;

                if (current.Length > 1 || (current.Length == 1 && current[0].HasRemovals))
                {
                    var merged = Merge(current, dropRemovals: true);
                    _segments  = merged.Count > 0 ? new[] { merged } : Array.Empty<EntrySegment>();
                }
            }
        }

        /// <summary>The single segment of a compacted engine, or null when it holds nothing.</summary>
        public EntrySegment CompactedSegment()
        {
            Compact();
            var segments = _segments;
            return segments.Length == 0 ? null : segments[0];
        }

        private EntrySegment Merge(EntrySegment[] inputs, bool dropRemovals)
        {
            long entries = 0, bytes = 0;
            foreach (var input in inputs)
            {
                entries += input.Count;
                bytes   += input.Dictionary.ToBlobs().payload.Length;
            }

            var     writer  = new EntryDictionary.Writer((int)entries, (int)Math.Min(int.MaxValue - 64, bytes + 64));
            var     values  = _withValues ? new UID128[entries] : null;
            ulong[] removed = null;

            Walk(inputs, (entry, segment, rank) =>
            {
                bool isRemoved = segment.IsRemoved(rank);
                if (isRemoved && dropRemovals) { return; }

                int at = writer.Count;
                if (values is object) { values[at] = segment.ValueAt(rank); }
                if (isRemoved)
                {
                    removed ??= new ulong[(entries + 63) / 64];
                    removed[at >> 6] |= 1UL << (at & 63);
                }
                writer.Add(entry);
            });

            var dictionary = writer.Complete();

            if (values is object && values.Length != dictionary.Count) { Array.Resize(ref values, dictionary.Count); }
            if (removed is object) { Array.Resize(ref removed, (dictionary.Count + 63) / 64); }

            return new EntrySegment(dictionary, values, removed);
        }

        /// <summary>
        /// Walks the entries of <paramref name="inputs"/> (oldest first) in ascending order, once per distinct
        /// entry, handing over the copy in the newest segment holding it - including a removal, which the
        /// visitor decides what to do with. Every segment is streamed; nothing is materialized.
        /// </summary>
        internal static void Walk(EntrySegment[] inputs, MergedEntryVisitor visitor)
        {
            int k         = inputs.Length;
            var positions = new int[k];
            var ranks     = new int[k];
            var lengths   = new int[k];
            var scratch   = new byte[k][];

            for (int c = 0; c < k; c++)
            {
                scratch[c] = new byte[inputs[c].Dictionary.MaxEntryBytes + 1];
                if (inputs[c].Count > 0) { lengths[c] = inputs[c].Dictionary.ReadNext(ref positions[c], scratch[c]); }
            }

            while (true)
            {
                int winner = -1;

                for (int c = 0; c < k; c++)
                {
                    if (ranks[c] >= inputs[c].Count) { continue; }
                    if (winner < 0) { winner = c; continue; }

                    // On a tie the later - newer - segment takes over.
                    if (scratch[c].AsSpan(0, lengths[c]).SequenceCompareTo(scratch[winner].AsSpan(0, lengths[winner])) <= 0) { winner = c; }
                }

                if (winner < 0) { return; }

                var entry = scratch[winner].AsSpan(0, lengths[winner]);
                visitor(entry, inputs[winner], ranks[winner]);

                for (int c = 0; c < k; c++)
                {
                    if (c == winner || ranks[c] >= inputs[c].Count) { continue; }
                    if (scratch[c].AsSpan(0, lengths[c]).SequenceEqual(entry)) { Advance(c); }
                }

                Advance(winner);
            }

            void Advance(int c)
            {
                ranks[c]++;
                if (ranks[c] < inputs[c].Count) { lengths[c] = inputs[c].Dictionary.ReadNext(ref positions[c], scratch[c]); }
            }
        }

        public void Clear()
        {
            lock (_writeLock)
            {
                _segments      = Array.Empty<EntrySegment>();
                _pending       = null;
                _pendingOps    = null;
                _pendingValues = null;
                _pendingCount  = 0;
                Exceptions.ReplaceWith(CompactHash32Set.Empty);
                MinTokenLength = 0;
                MaxTokenLength = 0;
            }
        }

        /// <summary>Every stored entry in sorted order, with what it links to. Flushes first.</summary>
        public List<(string entry, UID128 value)> Entries()
        {
            Flush();

            var entries = new List<(string entry, UID128 value)>(Count);

            Walk(_segments, (entry, segment, rank) =>
            {
                if (!segment.IsRemoved(rank)) { entries.Add((System.Text.Encoding.UTF8.GetString(entry), segment.ValueAt(rank))); }
            });

            return entries;
        }

        /// <summary>What <paramref name="entry"/> links to, as of the last flush.</summary>
        public bool TryGetValue(string entry, out UID128 value)
        {
            value = default;
            if (string.IsNullOrWhiteSpace(entry)) { return false; }

            var segments = _segments;
            if (segments.Length == 0) { return false; }

            int length = Normalize(entry.AsSpan().Trim(), out var chars, out var bytes);

            try
            {
                if (length <= 0) { return false; }
                return TryResolve(segments, bytes.AsSpan(0, length), new byte[MaxEntryBytes(segments) + 1], out value);
            }
            finally
            {
                ArrayPool<char>.Shared.Return(chars);
                ArrayPool<byte>.Shared.Return(bytes);
            }
        }

        // ---- matching ------------------------------------------------------------------------------

        // A token whose length falls outside the window of every stored word cannot be part of any entry.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool CouldMatchLength(int length) => MaxTokenLength == 0 || (length >= MinTokenLength && length <= MaxTokenLength);

        public bool Match<TSink>(Span<Token> tokens, bool stopOnFirstFound, ref TSink sink) where TSink : struct, ISpotterMatchSink
        {
            // The merge policy keeps the count logarithmic in the entry count, far below the 32 the mask holds.
            var segments = _segments;
            int count    = segments.Length;
            if (count == 0) { return false; }

            int maxBytes = MaxEntryBytes(segments);
            if (maxBytes == 0) { return false; }

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

                    ulong hash       = EntryText.Hash(keyBuffer.AsSpan(0, length));
                    uint  candidates = 0;

                    for (int s = 0; s < count; s++)
                    {
                        if (segments[s].Prefilter.MayStart(hash)) { candidates |= 1u << s; }
                    }

                    if (candidates == 0) { continue; }

                    candidates = Probe(segments, candidates, keyBuffer.AsSpan(0, length), decodeBuffer, out var singleSegment, out int singleRank);

                    int          last = i, lastRank = -1, j = i;
                    EntrySegment lastSegment = null;

                    while (candidates != 0 && j + 1 < N)
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

                        candidates = Probe(segments, candidates, keyBuffer.AsSpan(0, length), decodeBuffer, out var segment, out int rank);
                        if (segment is object) { lastSegment = segment; lastRank = rank; last = j; }
                    }

                    if (last > i)
                    {
                        foundAny = true;
                        if (stopOnFirstFound) { return true; }
                        sink.OnMultiGram(tokens, i, last, lastSegment, lastRank);
                    }

                    if (singleSegment is object)
                    {
                        foundAny = true;
                        if (stopOnFirstFound) { return true; }
                        sink.OnSingle(ref token, singleSegment, singleRank);
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

        // Probes the candidate segments newest first. The newest segment holding the key exactly decides whether
        // it matches - a removal there hides any older copy. Returns the segments in which some entry continues
        // past the key, which are the only ones worth probing with the next token appended.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint Probe(EntrySegment[] segments, uint candidates, ReadOnlySpan<byte> key, Span<byte> scratch, out EntrySegment matched, out int matchedRank)
        {
            matched     = null;
            matchedRank = -1;

            bool decided = false;
            uint extend  = 0;

            for (int s = segments.Length - 1; s >= 0; s--)
            {
                uint bit = 1u << s;
                if ((candidates & bit) == 0) { continue; }

                var segment = segments[s];
                segment.Dictionary.Probe(key, EntryText.SEPARATOR, scratch, out int rank, out bool canExtend);

                if (canExtend) { extend |= bit; }

                if (rank >= 0 && !decided)
                {
                    decided = true;
                    if (!segment.IsRemoved(rank))
                    {
                        matched     = segment;
                        matchedRank = rank;
                    }
                }
            }

            return extend;
        }

        // ---- serialization -------------------------------------------------------------------------

        public void LoadFrom(byte[] payload, byte[] blockOffsets, int count, int blockSize, int maxEntryBytes,
                             byte[] exceptionBuckets, byte[] exceptionLows, int exceptionCount, UID128[] values = null)
        {
            lock (_writeLock)
            {
                var dictionary = EntryDictionary.FromBlobs(payload, blockOffsets, count, blockSize, maxEntryBytes);

                Exceptions.ReplaceWith(CompactHash32Set.FromBlobs(exceptionBuckets, exceptionLows, exceptionCount));

                _segments = dictionary.Count > 0
                    ? new[] { new EntrySegment(dictionary, _withValues ? values ?? Array.Empty<UID128>() : null, null) }
                    : Array.Empty<EntrySegment>();

                _pending       = null;
                _pendingOps    = null;
                _pendingValues = null;
                _pendingCount  = 0;
            }
        }
    }
}
