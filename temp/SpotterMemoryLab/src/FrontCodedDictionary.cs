using System;

namespace SpotterMemoryLab
{
    /// <summary>
    /// Front-coded (prefix-compressed) sorted string dictionary. The entries are sorted; every block of
    /// <c>BlockSize</c> stores its first entry in full and the rest as (shared-prefix length, suffix).
    ///
    /// Two arrays total: one payload byte array and one block-offset array. Lookup is a binary search over
    /// block headers followed by a scan inside one block, and it returns the entry's rank - which is what
    /// makes the tagged (LinkedSpotter) variant cheap: the UID array is indexed by rank and therefore dense,
    /// with no hash-table slack.
    ///
    /// Unlike every hash-based design it is exact (no hash collisions at all), it can enumerate and rebuild
    /// the original strings, and it answers "is this a stored prefix" - which is what the tokenizer needs,
    /// so it can replace the tokenizer-exception table instead of adding to it.
    /// </summary>
    internal sealed class FrontCodedDictionary
    {
        private readonly byte[] _payload;
        private readonly int[]  _blockOffset;
        private readonly int    _blockSize;
        private readonly int    _count;

        public int  Count          => _count;
        public int  Blocks         => _blockOffset.Length - 1;
        public long PayloadBytes   => _payload.Length;
        public int  Objects        => 3;
        public long EstimatedBytes => Sz.Arr(_payload.Length, 1) + Sz.Arr(_blockOffset.Length, 4) + Sz.Obj(2 * Sz.REF + 4 + 4);

        public FrontCodedDictionary(StringBlob blob, int[] sortedOrder, int blockSize)
        {
            _blockSize = blockSize;
            _count     = sortedOrder.Length;

            int blocks    = (_count + blockSize - 1) / blockSize;
            _blockOffset  = new int[blocks + 1];
            var payload   = new byte[blob.TotalChars + _count * 2 + 64];
            int p         = 0;

            for (int b = 0; b < blocks; b++)
            {
                _blockOffset[b] = p;
                int from = b * blockSize;
                int to   = Math.Min(from + blockSize, _count);

                var header = blob[sortedOrder[from]];
                p = WriteEntry(payload, p, 0, header);

                var prev = header;
                for (int i = from + 1; i < to; i++)
                {
                    var cur  = blob[sortedOrder[i]];
                    int plen = SharedPrefix(prev, cur);
                    p        = WriteEntry(payload, p, plen, cur.Slice(plen));
                    prev     = cur;
                }
            }
            _blockOffset[blocks] = p;

            Array.Resize(ref payload, p);
            _payload = payload;
        }

        private static int SharedPrefix(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
        {
            int n = Math.Min(a.Length, b.Length);
            int i = 0;
            while (i < n && a[i] == b[i]) { i++; }
            return i;
        }

        // One control byte carries both lengths as nibbles; either escapes to a varint when it does not fit.
        private static int WriteEntry(byte[] buf, int p, int plen, ReadOnlySpan<byte> suffix)
        {
            int slen = suffix.Length;
            buf[p++] = (byte)((Math.Min(plen, 15) << 4) | Math.Min(slen, 15));
            if (plen >= 15) { p = WriteVarint(buf, p, plen - 15); }
            if (slen >= 15) { p = WriteVarint(buf, p, slen - 15); }
            suffix.CopyTo(buf.AsSpan(p));
            return p + slen;
        }

        private static int WriteVarint(byte[] buf, int p, int v)
        {
            while (v >= 0x80) { buf[p++] = (byte)(v | 0x80); v >>= 7; }
            buf[p++] = (byte)v;
            return p;
        }

        private static int ReadVarint(byte[] buf, ref int p)
        {
            int v = 0, shift = 0;
            while (true)
            {
                byte b = buf[p++];
                v |= (b & 0x7F) << shift;
                if ((b & 0x80) == 0) { return v; }
                shift += 7;
            }
        }

        // Decodes one entry into scratch on top of whatever prefix is already there; returns the new length.
        private int ReadEntry(ref int p, byte[] scratch)
        {
            byte ctrl = _payload[p++];
            int plen  = ctrl >> 4;
            int slen  = ctrl & 0x0F;
            if (plen == 15) { plen = 15 + ReadVarint(_payload, ref p); }
            if (slen == 15) { slen = 15 + ReadVarint(_payload, ref p); }
            Buffer.BlockCopy(_payload, p, scratch, plen, slen);
            p += slen;
            return plen + slen;
        }

        /// <summary>Rank of <paramref name="key"/> in the dictionary, or -1 when absent.</summary>
        public int Find(ReadOnlySpan<byte> key, byte[] scratch)
        {
            int lo = 0, hi = Blocks - 1, block = -1;
            while (lo <= hi)
            {
                int mid = (lo + hi) >> 1;
                int p   = _blockOffset[mid];
                int len = ReadEntry(ref p, scratch);
                int cmp = scratch.AsSpan(0, len).SequenceCompareTo(key);
                if (cmp == 0) { return mid * _blockSize; }
                if (cmp < 0) { block = mid; lo = mid + 1; } else { hi = mid - 1; }
            }
            if (block < 0) { return -1; }

            int q    = _blockOffset[block];
            int last = Math.Min(_blockSize, _count - block * _blockSize);
            int curLen = ReadEntry(ref q, scratch);
            for (int i = 1; i < last; i++)
            {
                curLen = ReadEntry(ref q, scratch);
                int cmp = scratch.AsSpan(0, curLen).SequenceCompareTo(key);
                if (cmp == 0) { return block * _blockSize + i; }
                if (cmp > 0) { return -1; }
            }
            return -1;
        }
    }
}
