using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Runtime.CompilerServices;

namespace Catalyst.Models
{
    /// <summary>
    /// Read-only set of 32-bit hashes stored as two flat byte arrays, used for the tokenizer-exception tables
    /// the spotters hand to <see cref="FastTokenizer"/>.
    ///
    /// The keys are bucketed by their high 16 bits, so only the low 16 bits are stored: a fixed 65,537-entry
    /// offset table plus two bytes per key, against the ~24 bytes per key a <see cref="HashSet{T}"/> of
    /// <see cref="int"/> occupies once its load factor and per-entry hash/next fields are counted. A lookup
    /// reads one offset pair and binary-searches a handful of contiguous values, which is also fewer cache
    /// misses than the hash set it replaces.
    ///
    /// Both arrays are the little-endian bytes that get serialized, read in place - the stored form and the
    /// in-memory form are the same bytes, so loading a model does not rebuild the table, and the whole
    /// structure is two managed objects whatever the key count.
    /// </summary>
    public sealed class CompactHash32Set
    {
        internal const int BUCKET_COUNT = 1 << 16;
        internal const int BUCKET_BYTES = (BUCKET_COUNT + 1) * sizeof(int);

        // Hashes added to a built set wait in a small sorted array beside it, so a model taking a handful of
        // new entries does not rebuild the 256 KB bucket table each time; past this many they are folded in.
        internal const int OVERFLOW_LIMIT = 4096;

        // The contents sit behind one immutable snapshot so a model can be re-frozen - after new entries are
        // added to it - without the tokenizer, which holds this object by reference, ever seeing a torn or
        // stale table.
        private sealed class Contents
        {
            public byte[] Buckets;  // (BUCKET_COUNT + 1) int32 offsets into Lows
            public byte[] Lows;     // Count uint16 low halves, ascending within each bucket
            public int    Count;
            public uint[] Overflow; // ascending, none of them in the table above
        }

        private volatile Contents _contents;

        /// <summary>Number of distinct hashes in the set.</summary>
        public int Count { get { var c = _contents; return c.Count + c.Overflow.Length; } }

        /// <summary>Bytes held by this structure, counting every array header.</summary>
        public long EstimatedBytes
        {
            get { var c = _contents; return 40L + 32L + 24L + c.Buckets.Length + 24L + c.Lows.Length + 24L + (long)c.Overflow.Length * sizeof(uint); }
        }

        private CompactHash32Set(byte[] buckets, byte[] lows, int count)
        {
            _contents = new Contents { Buckets = buckets, Lows = lows, Count = count, Overflow = Array.Empty<uint>() };
        }

        /// <summary>A new, empty set. Each model owns its own so it can be refilled in place.</summary>
        public static CompactHash32Set CreateEmpty() => new CompactHash32Set(EMPTY_BUCKETS, Array.Empty<byte>(), 0);

        private static readonly byte[] EMPTY_BUCKETS = new byte[BUCKET_BYTES];

        /// <summary>A shared empty set for callers that will never refill it.</summary>
        public static readonly CompactHash32Set Empty = CreateEmpty();

        /// <summary>Replaces the contents in place, so anything holding this object by reference sees the new table.</summary>
        internal void ReplaceWith(CompactHash32Set other)
        {
            _contents = other._contents;
        }

        /// <summary>Builds the set from an unsorted, possibly repeating sequence of hashes.</summary>
        public static CompactHash32Set Build(ICollection<int> hashes)
        {
            if (hashes is null || hashes.Count == 0) { return Empty; }

            var keys = new uint[hashes.Count];
            int n    = 0;
            foreach (var h in hashes) { keys[n++] = unchecked((uint)h); }
            return Build(keys, n);
        }

        /// <summary>Builds the set from the first <paramref name="count"/> entries of <paramref name="keys"/>, which is sorted in place.</summary>
        internal static CompactHash32Set Build(uint[] keys, int count)
        {
            if (count == 0) { return Empty; }

            Array.Sort(keys, 0, count);

            int distinct = 0;
            for (int i = 0; i < count; i++)
            {
                if (i == 0 || keys[i] != keys[i - 1]) { keys[distinct++] = keys[i]; }
            }

            var buckets = new byte[BUCKET_BYTES];
            var lows    = new byte[distinct * sizeof(ushort)];

            int at = 0;
            for (int b = 0; b < BUCKET_COUNT; b++)
            {
                BinaryPrimitives.WriteInt32LittleEndian(buckets.AsSpan(b * sizeof(int)), at);
                while (at < distinct && (keys[at] >> 16) == (uint)b)
                {
                    BinaryPrimitives.WriteUInt16LittleEndian(lows.AsSpan(at * sizeof(ushort)), (ushort)keys[at]);
                    at++;
                }
            }
            BinaryPrimitives.WriteInt32LittleEndian(buckets.AsSpan(BUCKET_COUNT * sizeof(int)), at);

            return new CompactHash32Set(buckets, lows, distinct);
        }

        /// <summary>
        /// Adds hashes to the set in place. A few go into the overflow beside the table; the table is only rebuilt
        /// once the overflow fills, so its cost is paid once per <see cref="OVERFLOW_LIMIT"/> additions.
        /// </summary>
        internal void Add(ReadOnlySpan<uint> hashes)
        {
            if (hashes.Length == 0) { return; }

            var contents = _contents;
            var added    = new List<uint>();

            foreach (var h in hashes)
            {
                if (!Contains(contents, h)) { added.Add(h); }
            }

            if (added.Count == 0) { return; }

            added.Sort();

            int unique = 0;
            for (int i = 0; i < added.Count; i++)
            {
                if (i == 0 || added[i] != added[i - 1]) { added[unique++] = added[i]; }
            }

            var overflow = new uint[contents.Overflow.Length + unique];
            int a = 0, b = 0, o = 0;
            while (a < contents.Overflow.Length || b < unique)
            {
                if (b >= unique || (a < contents.Overflow.Length && contents.Overflow[a] < added[b])) { overflow[o++] = contents.Overflow[a++]; }
                else { overflow[o++] = added[b++]; }
            }

            if (overflow.Length <= OVERFLOW_LIMIT)
            {
                _contents = new Contents { Buckets = contents.Buckets, Lows = contents.Lows, Count = contents.Count, Overflow = overflow };
                return;
            }

            _contents = Fold(contents, overflow);
        }

        // The table rebuilt with the overflow inside it.
        private static Contents Fold(Contents contents, uint[] overflow)
        {
            var keys = new uint[contents.Count + overflow.Length];
            int n    = 0;
            foreach (var h in EnumerateTable(contents)) { keys[n++] = unchecked((uint)h); }
            foreach (var h in overflow)                 { keys[n++] = h; }
            return Build(keys, n)._contents;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Contains(int hash) => Contains(_contents, unchecked((uint)hash));

        private static bool Contains(Contents contents, uint h)
        {
            if (contents.Overflow.Length > 0 && Array.BinarySearch(contents.Overflow, h) >= 0) { return true; }
            if (contents.Count == 0) { return false; }

            int  bucket = (int)(h >> 16);
            int  lo     = BinaryPrimitives.ReadInt32LittleEndian(contents.Buckets.AsSpan(bucket * sizeof(int)));
            int  hi     = BinaryPrimitives.ReadInt32LittleEndian(contents.Buckets.AsSpan((bucket + 1) * sizeof(int))) - 1;

            if (lo > hi) { return false; }

            var    lows = contents.Lows.AsSpan();
            ushort want = (ushort)h;
            while (lo <= hi)
            {
                int    mid = (int)(((uint)lo + (uint)hi) >> 1);
                ushort v   = BinaryPrimitives.ReadUInt16LittleEndian(lows.Slice(mid * sizeof(ushort)));
                if (v == want) { return true; }
                if (v < want) { lo = mid + 1; } else { hi = mid - 1; }
            }
            return false;
        }

        /// <summary>Every hash in the set: the ones not yet folded into the table first, then the table ascending by bucket. Used when a model has to be rebuilt or compared.</summary>
        public IEnumerable<int> Hashes() => Enumerate(_contents);

        private static IEnumerable<int> Enumerate(Contents contents)
        {
            foreach (var h in contents.Overflow)        { yield return unchecked((int)h); }
            foreach (var h in EnumerateTable(contents)) { yield return h; }
        }

        private static IEnumerable<int> EnumerateTable(Contents contents)
        {
            for (int b = 0; b < BUCKET_COUNT && contents.Count > 0; b++)
            {
                int from = BinaryPrimitives.ReadInt32LittleEndian(contents.Buckets.AsSpan(b * sizeof(int)));
                int to   = BinaryPrimitives.ReadInt32LittleEndian(contents.Buckets.AsSpan((b + 1) * sizeof(int)));
                for (int i = from; i < to; i++)
                {
                    ushort low = BinaryPrimitives.ReadUInt16LittleEndian(contents.Lows.AsSpan(i * sizeof(ushort)));
                    yield return unchecked((int)(((uint)b << 16) | low));
                }
            }
        }

        /// <summary>The backing blobs, for the model to store. They are the in-memory form verbatim.</summary>
        internal (byte[] buckets, byte[] lows, int count) ToBlobs()
        {
            var contents = _contents;

            if (contents.Overflow.Length > 0)
            {
                contents  = Fold(contents, contents.Overflow);
                _contents = contents;
            }

            return (contents.Buckets, contents.Lows, contents.Count);
        }

        internal static CompactHash32Set FromBlobs(byte[] buckets, byte[] lows, int count)
        {
            if (buckets is null || lows is null || count <= 0) { return Empty; }
            if (buckets.Length != BUCKET_BYTES || lows.Length != count * sizeof(ushort)) { return Empty; }
            return new CompactHash32Set(buckets, lows, count);
        }
    }
}
