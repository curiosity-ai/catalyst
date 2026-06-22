using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UID;

namespace Catalyst.Models
{
    // CHD-style (hash, displace) perfect hash for a known, distinct, non-zero key set, used to back the
    // compact spotter tables at a load factor close to 1.0 (no power-of-two waste, deterministic footprint).
    //
    // Design notes:
    //  - Sequential, single pass, no multithreading.
    //  - Allocation-light: every transient build buffer comes from ArrayPool and is returned in finally; the
    //    only persistent allocation is the displacement array.
    //  - Never throws and always completes: the per-bucket displacement search is bounded (DMax) and any
    //    bucket that does not place is sent to a small overflow table the caller keeps - so we never grind
    //    the hard tail or reseed the whole table. Alpha (< 1) leaves a few percent of slots empty, which
    //    keeps overflow tiny (~0.02%) and the build fast and ~linear.
    //  - Slot indices are reduced mod m BEFORE the displacement multiply so the arithmetic cannot overflow
    //    2^64 (m is not a power of two, so a wrap would not commute with mod m).
    internal sealed class Mph
    {
        public const double Alpha     = 0.80; // slot-array load-factor target (lower => faster build, tiny overflow, a little more memory)
        public const int    Lambda    = 4;    // average keys per bucket
        public const int    DMax       = 256; // bounded displacement search before a bucket overflows
        private const int   MaxBucket  = 4096;

        private readonly int   _m;
        private readonly int   _r;
        private readonly ulong _s1;
        private readonly ulong _s2;
        private readonly int[] _disp;

        public int  M                 => _m;
        public long DisplacementBytes => 24 + (long)_disp.Length * sizeof(int);

        private Mph(int m, int r, ulong s1, ulong s2, int[] disp) { _m = m; _r = r; _s1 = s1; _s2 = s2; _disp = disp; }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Slot(ulong key)
        {
            ulong mm = (ulong)_m;
            ulong h1 = CompactHash.Mix(key ^ _s1);
            int   d  = _disp[(int)(h1 % (ulong)_r)];
            ulong am = h1 % mm;
            ulong bm = (CompactHash.Mix(key ^ _s2) | 1UL) % mm;
            if (bm == 0) { bm = 1; }
            return (int)((am + (ulong)d * bm) % mm);
        }

        // Builds the function. overflowIdx[0..overflowCount) are indices (into keys) of keys not placed perfectly.
        public static Mph Build(ulong[] keys, int n, out int[] overflowIdx, out int overflowCount)
        {
            int m = Math.Max(1, (int)Math.Ceiling(n / Alpha));
            int r = Math.Max(1, n / Lambda);

            var pi  = ArrayPool<int>.Shared;
            var pul = ArrayPool<ulong>.Shared;

            int     words   = (m >> 6) + 1;
            int[]   am       = pi.Rent(n == 0 ? 1 : n);
            int[]   bm       = pi.Rent(n == 0 ? 1 : n);
            int[]   bucketOf = pi.Rent(n == 0 ? 1 : n);
            int[]   count   = pi.Rent(r);
            int[]   start   = pi.Rent(r + 1);
            int[]   items   = pi.Rent(n == 0 ? 1 : n);
            int[]   cursor  = pi.Rent(r);
            int[]   order   = pi.Rent(r);
            int[]   disp    = new int[r];
            ulong[] occ     = pul.Rent(words);
            int[]   tent    = pi.Rent(MaxBucket);
            int[]   cur     = pi.Rent(MaxBucket);
            int[]   sizeHist = pi.Rent(MaxBucket + 2);

            var overflow = new List<int>();

            try
            {
                ulong seed = 0x243F6A8885A308D3UL;
                ulong s1 = NextSeed(ref seed);
                ulong s2 = NextSeed(ref seed);

                Array.Clear(count, 0, r);
                int maxBucket = 0;
                for (int i = 0; i < n; i++)
                {
                    ulong h1 = CompactHash.Mix(keys[i] ^ s1);
                    ulong b  = CompactHash.Mix(keys[i] ^ s2) | 1UL;
                    am[i]       = (int)(h1 % (ulong)m);
                    int bmod    = (int)(b % (ulong)m);
                    bm[i]       = bmod == 0 ? 1 : bmod;
                    int bk      = (int)(h1 % (ulong)r);
                    bucketOf[i] = bk;
                    int c = ++count[bk];
                    if (c > maxBucket) { maxBucket = c; }
                }

                start[0] = 0;
                for (int b = 0; b < r; b++) { start[b + 1] = start[b] + count[b]; }
                Array.Copy(start, cursor, r);
                for (int i = 0; i < n; i++) { int bk = bucketOf[i]; items[cursor[bk]++] = i; }

                // counting sort of buckets by DESCENDING size (no comparator allocation)
                int hist = Math.Min(maxBucket, MaxBucket);
                Array.Clear(sizeHist, 0, hist + 2);
                for (int b = 0; b < r; b++) { sizeHist[Math.Min(count[b], hist + 1)]++; }
                int pos = 0;
                for (int s = hist + 1; s >= 0; s--) { int c = sizeHist[s]; sizeHist[s] = pos; pos += c; }
                for (int b = 0; b < r; b++) { order[sizeHist[Math.Min(count[b], hist + 1)]++] = b; }

                Array.Clear(occ, 0, words);

                for (int oi = 0; oi < r; oi++)
                {
                    int b = order[oi];
                    int s = count[b];
                    if (s == 0) { disp[b] = 0; continue; }
                    int bs = start[b];

                    if (s > MaxBucket) { for (int j = 0; j < s; j++) overflow.Add(items[bs + j]); disp[b] = 0; continue; }

                    for (int j = 0; j < s; j++) { cur[j] = am[items[bs + j]]; }

                    bool placed = false;
                    for (int d = 0; d < DMax; d++)
                    {
                        int t = 0; bool good = true;
                        for (int j = 0; j < s; j++)
                        {
                            int slot = cur[j];
                            ref ulong w = ref occ[slot >> 6];
                            ulong bit = 1UL << (slot & 63);
                            if ((w & bit) != 0) { good = false; break; }
                            w |= bit; // tentative
                            tent[t++] = slot;
                        }
                        if (good) { disp[b] = d; placed = true; break; }
                        for (int k = 0; k < t; k++) { int sl = tent[k]; occ[sl >> 6] &= ~(1UL << (sl & 63)); }
                        for (int j = 0; j < s; j++) { int v = cur[j] + bm[items[bs + j]]; cur[j] = v >= m ? v - m : v; }
                    }
                    if (!placed) { for (int j = 0; j < s; j++) overflow.Add(items[bs + j]); disp[b] = 0; }
                }

                overflowCount = overflow.Count;
                overflowIdx   = overflowCount == 0 ? Array.Empty<int>() : overflow.ToArray();
                return new Mph(m, r, s1, s2, disp);
            }
            finally
            {
                pi.Return(am); pi.Return(bm); pi.Return(bucketOf); pi.Return(count); pi.Return(start);
                pi.Return(items); pi.Return(cursor); pi.Return(order); pi.Return(tent); pi.Return(cur);
                pi.Return(sizeHist); pul.Return(occ);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong NextSeed(ref ulong x)
        {
            x += 0x9E3779B97F4A7C15UL;
            ulong z = x;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static uint Fingerprint(ulong key)
        {
            uint f = (uint)CompactHash.Mix(key);
            return f == 0 ? 1u : f;
        }
    }

    /// <summary>
    /// Perfect-hash membership set over 64-bit keys (load factor ~1.0). One <see cref="ulong"/> slot per key
    /// plus a small displacement array; a tiny overflow set holds the &lt;~0.02% of keys not placed perfectly.
    /// </summary>
    internal sealed class MphHashSet64 : ICompactHashSet64
    {
        private readonly Mph            _mph;
        private readonly ulong[]        _keys;
        private readonly HashSet<ulong> _overflow;
        private readonly bool           _hasZero;
        private readonly int            _count;
        private readonly ExactHashSet64 _fallback;

        public MphHashSet64(ICollection<ulong> source)
        {
            _count = source.Count;
            var pul = ArrayPool<ulong>.Shared;
            ulong[] scratch = pul.Rent(_count == 0 ? 1 : _count);
            try
            {
                int n = 0;
                foreach (var k in source) { if (k == 0) _hasZero = true; else scratch[n++] = k; }

                var mph = Mph.Build(scratch, n, out var ofIdx, out var ofCount);
                if (mph is null) { _fallback = new ExactHashSet64(source); return; }
                _mph  = mph;
                _keys = new ulong[_mph.M];

                var ofSet = ofCount > 0 ? new HashSet<int>(ofCount) : null;
                for (int i = 0; i < ofCount; i++) { ofSet.Add(ofIdx[i]); }
                for (int i = 0; i < n; i++)
                {
                    if (ofSet is object && ofSet.Contains(i)) { continue; }
                    _keys[_mph.Slot(scratch[i])] = scratch[i];
                }
                if (ofCount > 0)
                {
                    _overflow = new HashSet<ulong>(ofCount);
                    for (int i = 0; i < ofCount; i++) { _overflow.Add(scratch[ofIdx[i]]); }
                }
            }
            finally { pul.Return(scratch); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Contains(ulong key)
        {
            if (_mph is null) { return _fallback.Contains(key); }
            if (key == 0) { return _hasZero; }
            if (_keys[_mph.Slot(key)] == key) { return true; }
            return _overflow is object && _overflow.Contains(key);
        }

        public int  Count            => _mph is null ? _fallback.Count : _count;
        public bool CanEnumerateKeys => true;
        public long EstimatedBytes   => _mph is null ? _fallback.EstimatedBytes
            : 24 + (long)_keys.Length * sizeof(ulong) + _mph.DisplacementBytes + (_overflow is null ? 0 : 32L * _overflow.Count + 64);

        public IEnumerable<ulong> Keys()
        {
            if (_mph is null) { foreach (var k in _fallback.Keys()) yield return k; yield break; }
            if (_hasZero) { yield return 0; }
            foreach (var v in _keys) { if (v != 0) yield return v; }
            if (_overflow is object) { foreach (var v in _overflow) yield return v; }
        }
    }

    /// <summary>
    /// Perfect-hash map from a 64-bit key to a <see cref="UID128"/> (load factor ~1.0).
    /// </summary>
    internal sealed class MphHashMap64 : ICompactHashMap64
    {
        private readonly Mph                          _mph;
        private readonly ulong[]                       _keys;
        private readonly UID128[]                      _values;
        private readonly Dictionary<ulong, UID128>     _overflow;
        private readonly bool                          _hasZero;
        private readonly UID128                         _zeroValue;
        private readonly int                           _count;
        private readonly ExactHashMap64                _fallback;

        public MphHashMap64(IDictionary<ulong, UID128> source)
        {
            _count = source.Count;
            var pul = ArrayPool<ulong>.Shared;
            ulong[] scratch = pul.Rent(_count == 0 ? 1 : _count);
            try
            {
                int n = 0;
                foreach (var kv in source) { if (kv.Key == 0) { _hasZero = true; _zeroValue = kv.Value; } else scratch[n++] = kv.Key; }

                var mph = Mph.Build(scratch, n, out var ofIdx, out var ofCount);
                if (mph is null) { _fallback = new ExactHashMap64(source); return; }
                _mph    = mph;
                _keys   = new ulong[_mph.M];
                _values = new UID128[_mph.M];

                var ofSet = ofCount > 0 ? new HashSet<int>(ofCount) : null;
                for (int i = 0; i < ofCount; i++) { ofSet.Add(ofIdx[i]); }
                for (int i = 0; i < n; i++)
                {
                    if (ofSet is object && ofSet.Contains(i)) { continue; }
                    int s = _mph.Slot(scratch[i]); _keys[s] = scratch[i]; _values[s] = source[scratch[i]];
                }
                if (ofCount > 0)
                {
                    _overflow = new Dictionary<ulong, UID128>(ofCount);
                    for (int i = 0; i < ofCount; i++) { ulong k = scratch[ofIdx[i]]; _overflow[k] = source[k]; }
                }
            }
            finally { pul.Return(scratch); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryGetValue(ulong key, out UID128 value)
        {
            if (_mph is null) { return _fallback.TryGetValue(key, out value); }
            if (key == 0) { value = _zeroValue; return _hasZero; }
            int s = _mph.Slot(key);
            if (_keys[s] == key) { value = _values[s]; return true; }
            if (_overflow is object) { return _overflow.TryGetValue(key, out value); }
            value = default;
            return false;
        }

        public int  Count            => _mph is null ? _fallback.Count : _count;
        public bool CanEnumerateKeys => true;
        public long EstimatedBytes   => _mph is null ? _fallback.EstimatedBytes
            : 32 + (long)_keys.Length * (sizeof(ulong) + 16) + _mph.DisplacementBytes + (_overflow is null ? 0 : 48L * _overflow.Count + 64);

        public IEnumerable<KeyValuePair<ulong, UID128>> Entries()
        {
            if (_mph is null) { foreach (var e in _fallback.Entries()) yield return e; yield break; }
            if (_hasZero) { yield return new KeyValuePair<ulong, UID128>(0, _zeroValue); }
            for (int i = 0; i < _keys.Length; i++) { if (_keys[i] != 0) yield return new KeyValuePair<ulong, UID128>(_keys[i], _values[i]); }
            if (_overflow is object) { foreach (var e in _overflow) yield return e; }
        }
    }

    /// <summary>
    /// Lossy perfect-hash set storing a 32-bit fingerprint per key (load factor ~1.0). ~2^-32 false-positive
    /// rate; never false negatives. The small overflow set keeps full keys, so overflow members are exact.
    /// </summary>
    internal sealed class MphFingerprintSet64 : ICompactHashSet64
    {
        private readonly Mph                  _mph;
        private readonly uint[]               _fp;
        private readonly HashSet<ulong>       _overflow;
        private readonly bool                 _hasZero;
        private readonly int                  _count;
        private readonly FingerprintHashSet64 _fallback;

        public MphFingerprintSet64(ICollection<ulong> source)
        {
            _count = source.Count;
            var pul = ArrayPool<ulong>.Shared;
            ulong[] scratch = pul.Rent(_count == 0 ? 1 : _count);
            try
            {
                int n = 0;
                foreach (var k in source) { if (k == 0) _hasZero = true; else scratch[n++] = k; }

                var mph = Mph.Build(scratch, n, out var ofIdx, out var ofCount);
                if (mph is null) { _fallback = new FingerprintHashSet64(source); return; }
                _mph = mph;
                _fp  = new uint[_mph.M];

                var ofSet = ofCount > 0 ? new HashSet<int>(ofCount) : null;
                for (int i = 0; i < ofCount; i++) { ofSet.Add(ofIdx[i]); }
                for (int i = 0; i < n; i++)
                {
                    if (ofSet is object && ofSet.Contains(i)) { continue; }
                    _fp[_mph.Slot(scratch[i])] = Mph.Fingerprint(scratch[i]);
                }
                if (ofCount > 0)
                {
                    _overflow = new HashSet<ulong>(ofCount);
                    for (int i = 0; i < ofCount; i++) { _overflow.Add(scratch[ofIdx[i]]); }
                }
            }
            finally { pul.Return(scratch); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Contains(ulong key)
        {
            if (_mph is null) { return _fallback.Contains(key); }
            if (key == 0) { return _hasZero; }
            if (_fp[_mph.Slot(key)] == Mph.Fingerprint(key)) { return true; }
            return _overflow is object && _overflow.Contains(key);
        }

        public int  Count            => _mph is null ? _fallback.Count : _count;
        public bool CanEnumerateKeys => false;
        public long EstimatedBytes   => _mph is null ? _fallback.EstimatedBytes
            : 24 + (long)_fp.Length * sizeof(uint) + _mph.DisplacementBytes + (_overflow is null ? 0 : 32L * _overflow.Count + 64);

        public IEnumerable<ulong> Keys() => throw new NotSupportedException("A fingerprint-compressed spotter cannot enumerate its original keys.");
    }

    /// <summary>
    /// Lossy perfect-hash map storing a 32-bit fingerprint + value per key (load factor ~1.0). Overflow keys
    /// are kept exactly (full key) in a tiny side dictionary.
    /// </summary>
    internal sealed class MphFingerprintMap64 : ICompactHashMap64
    {
        private readonly Mph                          _mph;
        private readonly uint[]                        _fp;
        private readonly UID128[]                      _values;
        private readonly Dictionary<ulong, UID128>     _overflow;
        private readonly bool                          _hasZero;
        private readonly UID128                         _zeroValue;
        private readonly int                           _count;
        private readonly FingerprintHashMap64          _fallback;

        public MphFingerprintMap64(IDictionary<ulong, UID128> source)
        {
            _count = source.Count;
            var pul = ArrayPool<ulong>.Shared;
            ulong[] scratch = pul.Rent(_count == 0 ? 1 : _count);
            try
            {
                int n = 0;
                foreach (var kv in source) { if (kv.Key == 0) { _hasZero = true; _zeroValue = kv.Value; } else scratch[n++] = kv.Key; }

                var mph = Mph.Build(scratch, n, out var ofIdx, out var ofCount);
                if (mph is null) { _fallback = new FingerprintHashMap64(source); return; }
                _mph    = mph;
                _fp     = new uint[_mph.M];
                _values = new UID128[_mph.M];

                var ofSet = ofCount > 0 ? new HashSet<int>(ofCount) : null;
                for (int i = 0; i < ofCount; i++) { ofSet.Add(ofIdx[i]); }
                for (int i = 0; i < n; i++)
                {
                    if (ofSet is object && ofSet.Contains(i)) { continue; }
                    int s = _mph.Slot(scratch[i]); _fp[s] = Mph.Fingerprint(scratch[i]); _values[s] = source[scratch[i]];
                }
                if (ofCount > 0)
                {
                    _overflow = new Dictionary<ulong, UID128>(ofCount);
                    for (int i = 0; i < ofCount; i++) { ulong k = scratch[ofIdx[i]]; _overflow[k] = source[k]; }
                }
            }
            finally { pul.Return(scratch); }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryGetValue(ulong key, out UID128 value)
        {
            if (_mph is null) { return _fallback.TryGetValue(key, out value); }
            if (key == 0) { value = _zeroValue; return _hasZero; }
            int s = _mph.Slot(key);
            if (_fp[s] == Mph.Fingerprint(key)) { value = _values[s]; return true; }
            if (_overflow is object) { return _overflow.TryGetValue(key, out value); }
            value = default;
            return false;
        }

        public int  Count            => _mph is null ? _fallback.Count : _count;
        public bool CanEnumerateKeys => false;
        public long EstimatedBytes   => _mph is null ? _fallback.EstimatedBytes
            : 32 + (long)_fp.Length * (sizeof(uint) + 16) + _mph.DisplacementBytes + (_overflow is null ? 0 : 48L * _overflow.Count + 64);

        public IEnumerable<KeyValuePair<ulong, UID128>> Entries() => throw new NotSupportedException("A fingerprint-compressed spotter cannot enumerate its original keys.");
    }
}
