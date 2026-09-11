using System;
using System.Runtime.CompilerServices;

namespace SpotterMemoryLab
{
    /// <summary>
    /// Blocked Bloom filter: every key touches k bits inside one 512-bit block, so a probe costs a single
    /// cache line. Used as a *prefilter* in front of an exact structure - it never says "absent" about a
    /// stored key, so the combination stays exact while ordinary text is rejected at hash-table speed and
    /// only the ~1% that survives pays for the dictionary or automaton walk.
    /// </summary>
    internal sealed class BlockedBloom
    {
        private readonly ulong[] _bits;
        private readonly int     _blocks;
        private readonly int     _k;

        public int  Objects        => 2;
        public long EstimatedBytes => Sz.Arr(_bits.Length, 8) + Sz.Obj(Sz.REF + 4 + 4);
        public int  BitsPerKey     { get; }

        public BlockedBloom(ulong[] keys, int count, int bitsPerKey)
        {
            BitsPerKey = bitsPerKey;
            _k         = Math.Max(1, Math.Min(7, (int)Math.Round(bitsPerKey * 0.693)));
            _blocks    = Math.Max(1, (int)(((long)count * bitsPerKey + 511) / 512));
            _bits      = new ulong[(long)_blocks * 8];

            for (int i = 0; i < count; i++) { Add(keys[i]); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void Add(ulong key)
        {
            ulong h    = RefHashSet.Mix(key);
            long  bas  = (long)((uint)(h >> 32) % (uint)_blocks) * 8;
            ulong bits = RefHashSet.Mix(h ^ 0x9E3779B97F4A7C15UL);
            for (int i = 0; i < _k; i++)
            {
                int b = (int)(bits & 511);
                bits >>= 9;
                _bits[bas + (b >> 6)] |= 1UL << (b & 63);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool MayContain(ulong key)
        {
            ulong h    = RefHashSet.Mix(key);
            long  bas  = (long)((uint)(h >> 32) % (uint)_blocks) * 8;
            ulong bits = RefHashSet.Mix(h ^ 0x9E3779B97F4A7C15UL);
            for (int i = 0; i < _k; i++)
            {
                int b = (int)(bits & 511);
                bits >>= 9;
                if ((_bits[bas + (b >> 6)] & (1UL << (b & 63))) == 0) { return false; }
            }
            return true;
        }
    }
}
