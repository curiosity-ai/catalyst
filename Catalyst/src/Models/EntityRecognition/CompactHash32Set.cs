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

        // The contents sit behind one immutable snapshot so a model can be re-frozen - after new entries are
        // added to it - without the tokenizer, which holds this object by reference, ever seeing a torn or
        // stale table.
        private sealed class Contents
        {
            public byte[] Buckets; // (BUCKET_COUNT + 1) int32 offsets into Lows
            public byte[] Lows;    // Count uint16 low halves, ascending within each bucket
            public int    Count;
        }

        private volatile Contents _contents;

        /// <summary>Number of distinct hashes in the set.</summary>
        public int Count => _contents.Count;

        /// <summary>Bytes held by this structure, counting every array header.</summary>
        public long EstimatedBytes
        {
            get { var c = _contents; return 40L + 24L + 24L + c.Buckets.Length + 24L + c.Lows.Length; }
        }

        private CompactHash32Set(byte[] buckets, byte[] lows, int count)
        {
            _contents = new Contents { Buckets = buckets, Lows = lows, Count = count };
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Contains(int hash)
        {
            var contents = _contents;
            if (contents.Count == 0) { return false; }

            uint h      = unchecked((uint)hash);
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

        /// <summary>Every hash in the set, ascending by bucket. Used when a model has to be rebuilt or compared.</summary>
        public IEnumerable<int> Hashes()
        {
            var contents = _contents;
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
