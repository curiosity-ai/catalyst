using System;
using System.Buffers.Binary;
using System.Runtime.CompilerServices;

namespace Catalyst.Models
{
    /// <summary>Visits one stored entry during a sequential walk of an <see cref="EntryDictionary"/>.</summary>
    internal delegate void EntryVisitor(int rank, ReadOnlySpan<byte> entry);

    /// <summary>
    /// Read-only, sorted, front-coded dictionary of the entries a spotter recognises, stored as UTF-8.
    ///
    /// Entries are sorted and cut into blocks. The first entry of each block is stored whole; the rest store
    /// only the number of leading bytes shared with their predecessor and the bytes that differ. Catalogue
    /// data - part numbers, product codes, names sharing a stem - shares most of its characters with its
    /// sorted neighbour, so this costs a few bytes per entry against the ten a 64-bit hash table spends on
    /// the key alone, and it keeps the entry itself rather than a hash of it.
    ///
    /// Three properties earn it its place over the hash tables it replaces:
    ///  - a lookup returns the entry's *rank*, so a tagged spotter indexes a dense value array by rank and
    ///    stores no keys at all;
    ///  - it answers "does any entry continue past this one", which is what walking a multi-token entry needs
    ///    and what the tokenizer needs to decide whether to keep a word whole;
    ///  - it can hand back what it stores, so a model can say why something did not match, and can be
    ///    re-opened for editing without losing the original strings.
    ///
    /// The payload and the block-offset table are the bytes that get serialized, read in place: the stored
    /// form is the in-memory form, and the structure is three managed objects whatever the entry count.
    /// </summary>
    internal sealed class EntryDictionary
    {
        /// <summary>
        /// Entries per block. Larger blocks share more prefix (less memory) and scan longer (slower lookup).
        /// 32 sits at the knee: past it the payload barely shrinks while the scan keeps growing.
        /// </summary>
        internal const int DEFAULT_BLOCK_SIZE = 32;

        private readonly byte[] _payload;
        private readonly byte[] _blockOffsets; // (Blocks + 1) int32 offsets into _payload
        private readonly int    _blockSize;

        public int Count         { get; }
        public int MaxEntryBytes { get; }
        public int Blocks        => _blockOffsets.Length / sizeof(int) - 1;

        /// <summary>Bytes held by this structure, counting all three object headers.</summary>
        public long EstimatedBytes => 48L + 24L + _payload.Length + 24L + _blockOffsets.Length;

        public static readonly EntryDictionary Empty = new EntryDictionary(Array.Empty<byte>(), new byte[sizeof(int)], DEFAULT_BLOCK_SIZE, 0, 0);

        private EntryDictionary(byte[] payload, byte[] blockOffsets, int blockSize, int count, int maxEntryBytes)
        {
            _payload       = payload;
            _blockOffsets  = blockOffsets;
            _blockSize     = blockSize;
            Count          = count;
            MaxEntryBytes  = maxEntryBytes;
        }

        // ---- building ------------------------------------------------------------------------------

        /// <summary>
        /// Builds the dictionary from <paramref name="count"/> entries of <paramref name="blob"/>, addressed
        /// by <paramref name="order"/> (ascending byte order, already de-duplicated).
        /// </summary>
        public static EntryDictionary Build(byte[] blob, int[] offsets, int[] order, int count, int blockSize = DEFAULT_BLOCK_SIZE)
        {
            if (count == 0) { return Empty; }

            int blocks       = (count + blockSize - 1) / blockSize;
            var blockOffsets = new byte[(blocks + 1) * sizeof(int)];

            long capacity = 0;
            int  maxEntry = 0;
            for (int i = 0; i < count; i++)
            {
                int len   = offsets[order[i] + 1] - offsets[order[i]];
                capacity += len + 5;
                if (len > maxEntry) { maxEntry = len; }
            }

            var payload = new byte[capacity];
            int p       = 0;

            for (int b = 0; b < blocks; b++)
            {
                BinaryPrimitives.WriteInt32LittleEndian(blockOffsets.AsSpan(b * sizeof(int)), p);

                int from = b * blockSize;
                int to   = Math.Min(from + blockSize, count);

                var previous = Entry(blob, offsets, order[from]);
                p = WriteEntry(payload, p, 0, previous);

                for (int i = from + 1; i < to; i++)
                {
                    var current = Entry(blob, offsets, order[i]);
                    int shared  = SharedPrefix(previous, current);
                    p           = WriteEntry(payload, p, shared, current.Slice(shared));
                    previous    = current;
                }
            }
            BinaryPrimitives.WriteInt32LittleEndian(blockOffsets.AsSpan(blocks * sizeof(int)), p);

            Array.Resize(ref payload, p);
            return new EntryDictionary(payload, blockOffsets, blockSize, count, maxEntry);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ReadOnlySpan<byte> Entry(byte[] blob, int[] offsets, int index) => blob.AsSpan(offsets[index], offsets[index + 1] - offsets[index]);

        private static int SharedPrefix(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
        {
            int n = Math.Min(a.Length, b.Length), i = 0;
            while (i < n && a[i] == b[i]) { i++; }
            return i;
        }

        // One control byte carries both lengths as nibbles; either escapes to a varint when it does not fit.
        private static int WriteEntry(byte[] buffer, int p, int shared, ReadOnlySpan<byte> suffix)
        {
            int suffixLength = suffix.Length;
            buffer[p++] = (byte)((Math.Min(shared, 15) << 4) | Math.Min(suffixLength, 15));
            if (shared >= 15)       { p = WriteVarint(buffer, p, shared - 15); }
            if (suffixLength >= 15) { p = WriteVarint(buffer, p, suffixLength - 15); }
            suffix.CopyTo(buffer.AsSpan(p));
            return p + suffixLength;
        }

        private static int WriteVarint(byte[] buffer, int p, int value)
        {
            while (value >= 0x80) { buffer[p++] = (byte)(value | 0x80); value >>= 7; }
            buffer[p++] = (byte)value;
            return p;
        }

        // ---- reading -------------------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int BlockStart(int block) => BinaryPrimitives.ReadInt32LittleEndian(_blockOffsets.AsSpan(block * sizeof(int)));

        // Decodes the entry at byte position p on top of whatever prefix scratch already holds, advancing p.
        private int ReadEntry(ref int p, Span<byte> scratch)
        {
            byte control      = _payload[p++];
            int  shared       = control >> 4;
            int  suffixLength = control & 0x0F;
            if (shared == 15)       { shared       = 15 + ReadVarint(ref p); }
            if (suffixLength == 15) { suffixLength = 15 + ReadVarint(ref p); }
            _payload.AsSpan(p, suffixLength).CopyTo(scratch.Slice(shared));
            p += suffixLength;
            return shared + suffixLength;
        }

        private int ReadVarint(ref int p)
        {
            int value = 0, shift = 0;
            while (true)
            {
                byte b = _payload[p++];
                value |= (b & 0x7F) << shift;
                if ((b & 0x80) == 0) { return value; }
                shift += 7;
            }
        }

        /// <summary>
        /// Looks <paramref name="key"/> up and, in the same pass, reports whether any stored entry continues
        /// past it with <paramref name="separator"/> - which is how a multi-token walk knows to take another
        /// token, and how the tokenizer knows a word is the start of something longer.
        /// </summary>
        /// <param name="scratch">Decode buffer, at least <see cref="MaxEntryBytes"/> bytes.</param>
        /// <param name="rank">The entry's rank when stored exactly, otherwise -1.</param>
        /// <param name="canExtend">True when some stored entry equals <paramref name="key"/> + <paramref name="separator"/> + more.</param>
        public void Probe(ReadOnlySpan<byte> key, byte separator, Span<byte> scratch, out int rank, out bool canExtend)
        {
            rank      = -1;
            canExtend = false;
            if (Count == 0 || key.Length == 0 || key.Length > MaxEntryBytes) { return; }

            int index = LowerBound(key, scratch, out int position, out int length);

            // Entries sharing the prefix are contiguous and ordered by the byte that follows it, so the scan
            // stops at the first entry whose next byte is past the separator.
            while (index < Count)
            {
                if (length < key.Length || !scratch.Slice(0, key.Length).SequenceEqual(key)) { return; }

                if (length == key.Length)
                {
                    rank = index;
                }
                else
                {
                    byte next = scratch[key.Length];
                    if (next == separator) { canExtend = true; return; }
                    if (next > separator) { return; }
                }

                index++;
                if (index >= Count) { return; }
                length = ReadEntry(ref position, scratch);
            }
        }

        /// <summary>Rank of <paramref name="key"/>, or -1 when it is not stored.</summary>
        public int Find(ReadOnlySpan<byte> key, Span<byte> scratch)
        {
            if (Count == 0 || key.Length == 0 || key.Length > MaxEntryBytes) { return -1; }

            int index  = LowerBound(key, scratch, out _, out int length);
            if (index >= Count || length != key.Length) { return -1; }
            return scratch.Slice(0, length).SequenceEqual(key) ? index : -1;
        }

        // Rank of the first entry >= key, with scratch/position/length left on that entry. Returns Count when
        // the key sorts past everything stored.
        private int LowerBound(ReadOnlySpan<byte> key, Span<byte> scratch, out int position, out int length)
        {
            int blocks = Blocks;
            int lo = 0, hi = blocks - 1, block = -1;

            while (lo <= hi)
            {
                int mid = (int)(((uint)lo + (uint)hi) >> 1);
                int p   = BlockStart(mid);
                int len = ReadEntry(ref p, scratch);
                int cmp = scratch.Slice(0, len).SequenceCompareTo(key);

                if (cmp == 0)
                {
                    position = p;
                    length   = len;
                    return mid * _blockSize;
                }
                if (cmp < 0) { block = mid; lo = mid + 1; } else { hi = mid - 1; }
            }

            if (block < 0)
            {
                // The key sorts before everything; the lower bound is the very first entry.
                position = BlockStart(0);
                length   = ReadEntry(ref position, scratch);
                return 0;
            }

            int start = block * _blockSize;
            int last  = Math.Min(_blockSize, Count - start);
            position  = BlockStart(block);

            for (int i = 0; i < last; i++)
            {
                length = ReadEntry(ref position, scratch);
                if (scratch.Slice(0, length).SequenceCompareTo(key) >= 0) { return start + i; }
            }

            // Everything in this block sorts before the key, so the answer is the next block's first entry.
            int next = start + last;
            if (next >= Count) { length = 0; return Count; }
            position = BlockStart(block + 1);
            length   = ReadEntry(ref position, scratch);
            return next;
        }

        /// <summary>Walks every entry in rank order. The span handed to <paramref name="visitor"/> is only valid during the call.</summary>
        public void Visit(EntryVisitor visitor)
        {
            if (Count == 0) { return; }

            Span<byte> scratch = new byte[MaxEntryBytes + 1];
            int position = 0;
            for (int i = 0; i < Count; i++)
            {
                int length = ReadEntry(ref position, scratch);
                visitor(i, scratch.Slice(0, length));
            }
        }

        // ---- serialization -------------------------------------------------------------------------

        internal (byte[] payload, byte[] blockOffsets, int count, int blockSize, int maxEntryBytes) ToBlobs()
            => (_payload, _blockOffsets, Count, _blockSize, MaxEntryBytes);

        internal static EntryDictionary FromBlobs(byte[] payload, byte[] blockOffsets, int count, int blockSize, int maxEntryBytes)
        {
            if (payload is null || blockOffsets is null || count <= 0 || blockSize <= 0) { return Empty; }
            if (blockOffsets.Length != ((count + blockSize - 1) / blockSize + 1) * sizeof(int)) { return Empty; }
            return new EntryDictionary(payload, blockOffsets, blockSize, count, maxEntryBytes);
        }
    }
}
