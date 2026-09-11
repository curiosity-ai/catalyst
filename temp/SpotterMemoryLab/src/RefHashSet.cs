using System;
using System.Runtime.CompilerServices;

namespace SpotterMemoryLab
{
    /// <summary>
    /// Open-addressed set of 64-bit keys, equivalent to Catalyst's ExactHashSet64. Present only as a speed
    /// reference: it is the one-probe behaviour every hash-based design in the lab shares.
    /// </summary>
    internal sealed class RefHashSet
    {
        private readonly ulong[] _slots;
        private readonly ulong   _mask;
        private readonly int     _count;

        public int  Objects        => 2;
        public long EstimatedBytes => Sz.Arr(_slots.Length, 8) + Sz.Obj(Sz.REF + 8 + 4);

        public RefHashSet(ulong[] keys, int count)
        {
            _count = count;
            long needed = (long)(count / 0.75) + 1;
            int  m      = 1;
            while (m < needed) { m <<= 1; }
            _slots = new ulong[m];
            _mask  = (ulong)(m - 1);

            for (int i = 0; i < count; i++)
            {
                ulong k   = keys[i];
                ulong idx = Mix(k) & _mask;
                while (_slots[idx] != 0)
                {
                    if (_slots[idx] == k) { goto next; }
                    idx = (idx + 1) & _mask;
                }
                _slots[idx] = k;
                next: ;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mix(ulong x)
        {
            x ^= x >> 33; x *= 0xff51afd7ed558ccdUL;
            x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53UL;
            x ^= x >> 33;
            return x;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Contains(ulong key)
        {
            ulong idx = Mix(key) & _mask;
            ulong v;
            while ((v = _slots[idx]) != 0)
            {
                if (v == key) { return true; }
                idx = (idx + 1) & _mask;
            }
            return false;
        }
    }
}
