using System;
using System.Collections.Generic;
using System.Text;

namespace SpotterMemoryLab
{
    /// <summary>
    /// The dataset itself: all part numbers concatenated into one ASCII byte array plus an offset table.
    /// Holding 10M values as individual strings would cost ~700 MB of dataset before any model exists, and
    /// 10M objects, which would drown out the numbers we are trying to measure.
    /// </summary>
    internal sealed class StringBlob
    {
        public byte[] Data;
        public int[]  Offsets; // Count + 1 entries
        public int    Count;

        private int _dataLength;

        public StringBlob(int expectedCount, long expectedBytes)
        {
            Data    = new byte[Math.Max(1024, expectedBytes)];
            Offsets = new int[expectedCount + 1];
            Offsets[0] = 0;
        }

        public ReadOnlySpan<byte> this[int i] => Data.AsSpan(Offsets[i], Offsets[i + 1] - Offsets[i]);

        public int LengthOf(int i) => Offsets[i + 1] - Offsets[i];

        public string StringAt(int i) => Encoding.ASCII.GetString(this[i]);

        public void Add(ReadOnlySpan<byte> value)
        {
            if (_dataLength + value.Length > Data.Length)
            {
                Array.Resize(ref Data, Math.Max(Data.Length * 2, _dataLength + value.Length));
            }
            if (Count + 1 >= Offsets.Length)
            {
                Array.Resize(ref Offsets, Offsets.Length * 2);
            }
            value.CopyTo(Data.AsSpan(_dataLength));
            _dataLength      += value.Length;
            Offsets[++Count]  = _dataLength;
        }

        public void Trim()
        {
            Array.Resize(ref Data, _dataLength);
            Array.Resize(ref Offsets, Count + 1);
        }

        public long TotalChars => _dataLength;

        /// <summary>Indices of all entries in lexicographic byte order.</summary>
        public int[] SortedOrder()
        {
            var keys = new ulong[Count];
            var idx  = new int[Count];
            for (int i = 0; i < Count; i++)
            {
                idx[i]  = i;
                keys[i] = PackPrefix(this[i]);
            }

            Array.Sort(keys, idx);

            // Resolve runs that share the packed prefix (entries longer than 8 bytes with an equal head).
            int start = 0;
            while (start < Count)
            {
                int end = start + 1;
                while (end < Count && keys[end] == keys[start]) { end++; }
                if (end - start > 1)
                {
                    var run = new int[end - start];
                    Array.Copy(idx, start, run, 0, run.Length);
                    Array.Sort(run, CompareEntries);
                    Array.Copy(run, 0, idx, start, run.Length);
                }
                start = end;
            }

            return idx;
        }

        private int CompareEntries(int a, int b) => this[a].SequenceCompareTo(this[b]);

        // First 8 bytes, big-endian, zero-padded: shorter strings sort before longer ones with the same head.
        private ulong PackPrefix(ReadOnlySpan<byte> s)
        {
            ulong k = 0;
            for (int i = 0; i < 8; i++)
            {
                k = (k << 8) | (i < s.Length ? s[i] : 0u);
            }
            return k;
        }

        /// <summary>Distinct bytes used by the dataset, ascending.</summary>
        public byte[] Alphabet()
        {
            var seen = new bool[256];
            for (int i = 0; i < _dataLength; i++) { seen[Data[i]] = true; }
            var list = new List<byte>();
            for (int c = 0; c < 256; c++) { if (seen[c]) { list.Add((byte)c); } }
            return list.ToArray();
        }
    }
}
