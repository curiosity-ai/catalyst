using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace SpotterMemoryLab
{
    /// <summary>
    /// Elias-Fano encoded monotone sequence of unsigned integers, used as a membership structure over the
    /// spotter's hashes. Stores n values from a universe U in about n * (log2(U/n) + 2) bits - so a sorted
    /// set of 64-bit hashes costs ~5.5 bytes per key instead of the 10 bytes the perfect-hash slot array
    /// costs, and it is exact with respect to the hashes (nothing is truncated).
    ///
    /// Five arrays total, whatever the key count.
    /// </summary>
    internal sealed class EliasFano
    {
        private readonly ulong[] _low;      // packed low bits, _lowBits per value
        private readonly ulong[] _high;     // for value i, bit (hi_i + i) is set
        private readonly int[]   _z0Word;   // word holding every SAMPLE-th zero of _high
        private readonly int[]   _z0Before; // zeros before that word
        private readonly int     _lowBits;
        private readonly ulong   _lowMask;
        private readonly int     _count;

        private const int SAMPLE = 64;

        public int  Count          => _count;
        public int  LowBitsPerKey  => _lowBits;
        public int  Objects        => 5;
        public long EstimatedBytes => Sz.Arr(_low.Length, 8) + Sz.Arr(_high.Length, 8)
                                    + Sz.Arr(_z0Word.Length, 4) + Sz.Arr(_z0Before.Length, 4)
                                    + Sz.Obj(4 * Sz.REF + 8 + 4 + 4);

        /// <param name="sorted">Ascending, distinct values.</param>
        /// <param name="universeBits">Number of significant bits in the values (64 for full hashes).</param>
        public EliasFano(ulong[] sorted, int count, int universeBits)
        {
            _count = count;

            int lowBits = 0;
            if (count > 0)
            {
                double ratio = Math.Pow(2, universeBits) / count;
                lowBits      = Math.Max(0, Math.Min(universeBits, (int)Math.Floor(Math.Log2(ratio))));
            }
            _lowBits = lowBits;
            _lowMask = lowBits >= 64 ? ulong.MaxValue : (1UL << lowBits) - 1;

            long lowTotalBits = (long)count * lowBits;
            _low              = new ulong[lowTotalBits / 64 + 2];

            long maxHi     = count == 0 ? 0 : (long)(sorted[count - 1] >> lowBits);
            long highTotal = maxHi + count + 1;
            _high          = new ulong[highTotal / 64 + 2];

            for (int i = 0; i < count; i++)
            {
                ulong v = sorted[i];
                WriteLow(i, v & _lowMask);
                long bit = (long)(v >> lowBits) + i;
                _high[bit >> 6] |= 1UL << (int)(bit & 63);
            }

            long totalZeros = (long)_high.Length * 64 - count;
            int  samples    = (int)(totalZeros / SAMPLE) + 2;
            _z0Word         = new int[samples];
            _z0Before       = new int[samples];

            int  j     = 0;
            long zeros = 0;
            for (int w = 0; w < _high.Length; w++)
            {
                int z = 64 - BitOperations.PopCount(_high[w]);
                while (j < samples && (long)j * SAMPLE < zeros + z)
                {
                    _z0Word[j]   = w;
                    _z0Before[j] = (int)zeros;
                    j++;
                }
                zeros += z;
            }
            while (j < samples) { _z0Word[j] = _high.Length - 1; _z0Before[j] = (int)zeros; j++; }
        }

        private void WriteLow(int i, ulong value)
        {
            if (_lowBits == 0) { return; }
            long bit  = (long)i * _lowBits;
            int  word = (int)(bit >> 6);
            int  off  = (int)(bit & 63);
            _low[word] |= value << off;
            if (off + _lowBits > 64) { _low[word + 1] |= value >> (64 - off); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private ulong ReadLow(int i)
        {
            if (_lowBits == 0) { return 0; }
            long bit  = (long)i * _lowBits;
            int  word = (int)(bit >> 6);
            int  off  = (int)(bit & 63);
            ulong v   = _low[word] >> off;
            if (off + _lowBits > 64) { v |= _low[word + 1] << (64 - off); }
            return v & _lowMask;
        }

        public bool Contains(ulong value)
        {
            if (_count == 0) { return false; }

            ulong hi = value >> _lowBits;
            ulong lo = value & _lowMask;

            int i;
            if (hi == 0)
            {
                i = 0;
            }
            else
            {
                long p = Select0((long)hi - 1);
                if (p < 0) { return false; }
                i = (int)(p + 1 - (long)hi);
                if (i < 0 || i >= _count) { return false; }
            }

            long bit = (long)hi + i;
            while (i < _count && ((_high[bit >> 6] >> (int)(bit & 63)) & 1UL) != 0)
            {
                if (ReadLow(i) == lo) { return true; }
                i++; bit++;
            }
            return false;
        }

        // Position of the k-th zero bit (0-indexed) in _high.
        private long Select0(long k)
        {
            int j = (int)(k / SAMPLE);
            if (j >= _z0Word.Length) { return -1; }

            int  w     = _z0Word[j];
            long zeros = _z0Before[j];

            while (w < _high.Length)
            {
                int z = 64 - BitOperations.PopCount(_high[w]);
                if (zeros + z > k)
                {
                    ulong word = ~_high[w];
                    int rem    = (int)(k - zeros);
                    while (true)
                    {
                        int b = BitOperations.TrailingZeroCount(word);
                        if (rem == 0) { return (long)w * 64 + b; }
                        rem--;
                        word &= word - 1;
                    }
                }
                zeros += z;
                w++;
            }
            return -1;
        }
    }
}
