using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using UID;

namespace Catalyst.Models
{
    /// <summary>
    /// Global options controlling how spotter models compact their in-memory hash tables once loaded.
    /// </summary>
    public static class SpotterCompaction
    {
        /// <summary>
        /// When <c>true</c>, spotters loaded via <c>FromStoreAsync</c> replace their full 64-bit hash keys
        /// with 32-bit fingerprints, roughly halving the memory of the membership tables at the cost of a
        /// ~2.3e-10 per-lookup false-positive rate. Off by default to preserve the exact (lossless) behavior.
        /// This is opt-in: turning it on trades a negligible amount of recognition precision for memory.
        /// A model frozen in fingerprint mode is read-only - it cannot be mutated or re-stored losslessly.
        /// </summary>
        public static bool UseFingerprint32 { get; set; } = false;
    }

    /// <summary>
    /// Read-only, append-free membership table over 64-bit keys. Built once from a finished model and used
    /// in place of the <see cref="HashSet{T}"/> the model was trained with, to reduce memory footprint.
    /// </summary>
    internal interface ICompactHashSet64
    {
        bool Contains(ulong key);
        int Count { get; }
        bool CanEnumerateKeys { get; }
        IEnumerable<ulong> Keys();
        long EstimatedBytes { get; }
    }

    /// <summary>
    /// Read-only, append-free map from a 64-bit key to a <see cref="UID128"/> value.
    /// </summary>
    internal interface ICompactHashMap64
    {
        bool TryGetValue(ulong key, out UID128 value);
        int Count { get; }
        bool CanEnumerateKeys { get; }
        IEnumerable<KeyValuePair<ulong, UID128>> Entries();
        long EstimatedBytes { get; }
    }

    internal static class CompactHash
    {
        // 64-bit finalizer (fmix64 from MurmurHash3). The spotter token hashes are already well distributed,
        // but mixing again decouples the slot index from any structure in the original hash and gives linear
        // probing good behavior on both the low bits (slot) and the high bits (fingerprint).
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static ulong Mix(ulong x)
        {
            x ^= x >> 33;
            x *= 0xff51afd7ed558ccdUL;
            x ^= x >> 33;
            x *= 0xc4ceb9fe1a85ec53UL;
            x ^= x >> 33;
            return x;
        }

        // Power-of-two table size keeping the load factor at or below ~0.75, with at least one always-empty
        // slot so linear-probe lookups for absent keys are guaranteed to terminate.
        public static int CapacityFor(int count)
        {
            long needed = (long)(count / 0.75) + 1;
            int m = 1;
            while (m < needed) { m <<= 1; }
            return m;
        }

        // Primary builders: perfect-hash backed (load factor ~1.0, deterministic footprint). The open-addressed
        // ExactHashSet64/ExactHashMap64 (and fingerprint variants) remain as the never-fail fallback used
        // internally by the MPH structures if perfect-hash construction ever fails to produce a function.
        public static ICompactHashSet64 BuildSet(ICollection<ulong> keys)
        {
            return SpotterCompaction.UseFingerprint32
                ? new MphFingerprintSet64(keys)
                : (ICompactHashSet64)new MphHashSet64(keys);
        }

        public static ICompactHashMap64 BuildMap(IDictionary<ulong, UID128> map)
        {
            return SpotterCompaction.UseFingerprint32
                ? new MphFingerprintMap64(map)
                : (ICompactHashMap64)new MphHashMap64(map);
        }
    }

    /// <summary>
    /// Lossless open-addressed set of 64-bit keys (8 bytes/slot). Replaces a <see cref="HashSet{T}"/> of
    /// <see cref="ulong"/> at roughly a third of the memory while keeping O(1) lookups and exact semantics.
    /// </summary>
    internal sealed class ExactHashSet64 : ICompactHashSet64
    {
        private readonly ulong[] _slots; // 0 marks an empty slot; the key 0 is tracked separately
        private readonly ulong   _mask;
        private readonly bool    _hasZero;
        private readonly int     _count;

        public ExactHashSet64(ICollection<ulong> keys)
        {
            _count = keys.Count;
            int m  = CompactHash.CapacityFor(_count);
            _slots = new ulong[m];
            _mask  = (ulong)(m - 1);

            bool hasZero = false;
            foreach (var k in keys)
            {
                if (k == 0) { hasZero = true; continue; }
                Insert(k);
            }
            _hasZero = hasZero;
        }

        private void Insert(ulong key)
        {
            ulong idx = CompactHash.Mix(key) & _mask;
            while (_slots[idx] != 0)
            {
                if (_slots[idx] == key) { return; }
                idx = (idx + 1) & _mask;
            }
            _slots[idx] = key;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Contains(ulong key)
        {
            if (key == 0) { return _hasZero; }

            ulong idx = CompactHash.Mix(key) & _mask;
            ulong v;
            while ((v = _slots[idx]) != 0)
            {
                if (v == key) { return true; }
                idx = (idx + 1) & _mask;
            }
            return false;
        }

        public int  Count            => _count;
        public bool CanEnumerateKeys => true;
        public long EstimatedBytes   => 24 + (long)_slots.Length * sizeof(ulong);

        public IEnumerable<ulong> Keys()
        {
            if (_hasZero) { yield return 0; }
            foreach (var v in _slots)
            {
                if (v != 0) { yield return v; }
            }
        }
    }

    /// <summary>
    /// Lossy open-addressed set storing a 32-bit fingerprint per key (4 bytes/slot). Halves the memory of
    /// <see cref="ExactHashSet64"/> at the cost of a ~2^-32 per-lookup false-positive rate. Opt-in.
    /// </summary>
    internal sealed class FingerprintHashSet64 : ICompactHashSet64
    {
        private readonly uint[] _slots; // 0 marks an empty slot; fingerprints are forced non-zero
        private readonly ulong  _mask;
        private readonly bool   _hasZero;
        private readonly int    _count;

        public FingerprintHashSet64(ICollection<ulong> keys)
        {
            _count = keys.Count;
            int m  = CompactHash.CapacityFor(_count);
            _slots = new uint[m];
            _mask  = (ulong)(m - 1);

            bool hasZero = false;
            foreach (var k in keys)
            {
                if (k == 0) { hasZero = true; continue; }
                Insert(k);
            }
            _hasZero = hasZero;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint Fingerprint(ulong h)
        {
            uint fp = (uint)h;
            return fp == 0 ? 1u : fp;
        }

        private void Insert(ulong key)
        {
            ulong h   = CompactHash.Mix(key);
            uint  fp  = Fingerprint(h);
            ulong idx = (h >> 32) & _mask;
            while (_slots[idx] != 0)
            {
                if (_slots[idx] == fp) { return; } // fingerprint collision: treated as already present
                idx = (idx + 1) & _mask;
            }
            _slots[idx] = fp;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool Contains(ulong key)
        {
            if (key == 0) { return _hasZero; }

            ulong h   = CompactHash.Mix(key);
            uint  fp  = Fingerprint(h);
            ulong idx = (h >> 32) & _mask;
            uint  v;
            while ((v = _slots[idx]) != 0)
            {
                if (v == fp) { return true; }
                idx = (idx + 1) & _mask;
            }
            return false;
        }

        public int  Count            => _count;
        public bool CanEnumerateKeys => false;
        public long EstimatedBytes   => 24 + (long)_slots.Length * sizeof(uint);

        public IEnumerable<ulong> Keys() => throw new NotSupportedException("A fingerprint-compressed spotter cannot enumerate its original keys.");
    }

    /// <summary>
    /// Lossless open-addressed map from a 64-bit key to a <see cref="UID128"/> value.
    /// </summary>
    internal sealed class ExactHashMap64 : ICompactHashMap64
    {
        private readonly ulong[]  _keys;   // 0 marks an empty slot; the key 0 is tracked separately
        private readonly UID128[] _values;
        private readonly ulong    _mask;
        private readonly bool     _hasZero;
        private readonly UID128   _zeroValue;
        private readonly int      _count;

        public ExactHashMap64(IDictionary<ulong, UID128> map)
        {
            _count  = map.Count;
            int m   = CompactHash.CapacityFor(_count);
            _keys   = new ulong[m];
            _values = new UID128[m];
            _mask   = (ulong)(m - 1);

            bool   hasZero   = false;
            UID128 zeroValue = default;
            foreach (var kv in map)
            {
                if (kv.Key == 0) { hasZero = true; zeroValue = kv.Value; continue; }
                Insert(kv.Key, kv.Value);
            }
            _hasZero   = hasZero;
            _zeroValue = zeroValue;
        }

        private void Insert(ulong key, UID128 value)
        {
            ulong idx = CompactHash.Mix(key) & _mask;
            while (_keys[idx] != 0)
            {
                if (_keys[idx] == key) { _values[idx] = value; return; }
                idx = (idx + 1) & _mask;
            }
            _keys[idx]   = key;
            _values[idx] = value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryGetValue(ulong key, out UID128 value)
        {
            if (key == 0)
            {
                value = _zeroValue;
                return _hasZero;
            }

            ulong idx = CompactHash.Mix(key) & _mask;
            ulong k;
            while ((k = _keys[idx]) != 0)
            {
                if (k == key) { value = _values[idx]; return true; }
                idx = (idx + 1) & _mask;
            }
            value = default;
            return false;
        }

        public int  Count            => _count;
        public bool CanEnumerateKeys => true;
        public long EstimatedBytes   => 32 + (long)_keys.Length * (sizeof(ulong) + 16);

        public IEnumerable<KeyValuePair<ulong, UID128>> Entries()
        {
            if (_hasZero) { yield return new KeyValuePair<ulong, UID128>(0, _zeroValue); }
            for (int i = 0; i < _keys.Length; i++)
            {
                if (_keys[i] != 0) { yield return new KeyValuePair<ulong, UID128>(_keys[i], _values[i]); }
            }
        }
    }

    /// <summary>
    /// Lossy open-addressed map storing a 32-bit fingerprint per key plus the <see cref="UID128"/> value.
    /// A ~2^-32 fraction of distinct keys may collide on their fingerprint and resolve to the wrong value
    /// (or be reported present when absent). Opt-in, read-only.
    /// </summary>
    internal sealed class FingerprintHashMap64 : ICompactHashMap64
    {
        private readonly uint[]   _fingerprints; // 0 marks an empty slot; fingerprints are forced non-zero
        private readonly UID128[] _values;
        private readonly ulong    _mask;
        private readonly bool     _hasZero;
        private readonly UID128   _zeroValue;
        private readonly int      _count;

        public FingerprintHashMap64(IDictionary<ulong, UID128> map)
        {
            _count        = map.Count;
            int m         = CompactHash.CapacityFor(_count);
            _fingerprints = new uint[m];
            _values       = new UID128[m];
            _mask         = (ulong)(m - 1);

            bool   hasZero   = false;
            UID128 zeroValue = default;
            foreach (var kv in map)
            {
                if (kv.Key == 0) { hasZero = true; zeroValue = kv.Value; continue; }
                Insert(kv.Key, kv.Value);
            }
            _hasZero   = hasZero;
            _zeroValue = zeroValue;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static uint Fingerprint(ulong h)
        {
            uint fp = (uint)h;
            return fp == 0 ? 1u : fp;
        }

        private void Insert(ulong key, UID128 value)
        {
            ulong h   = CompactHash.Mix(key);
            uint  fp  = Fingerprint(h);
            ulong idx = (h >> 32) & _mask;
            while (_fingerprints[idx] != 0)
            {
                if (_fingerprints[idx] == fp) { _values[idx] = value; return; }
                idx = (idx + 1) & _mask;
            }
            _fingerprints[idx] = fp;
            _values[idx]       = value;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public bool TryGetValue(ulong key, out UID128 value)
        {
            if (key == 0)
            {
                value = _zeroValue;
                return _hasZero;
            }

            ulong h   = CompactHash.Mix(key);
            uint  fp  = Fingerprint(h);
            ulong idx = (h >> 32) & _mask;
            uint  v;
            while ((v = _fingerprints[idx]) != 0)
            {
                if (v == fp) { value = _values[idx]; return true; }
                idx = (idx + 1) & _mask;
            }
            value = default;
            return false;
        }

        public int  Count            => _count;
        public bool CanEnumerateKeys => false;
        public long EstimatedBytes   => 32 + (long)_fingerprints.Length * (sizeof(uint) + 16);

        public IEnumerable<KeyValuePair<ulong, UID128>> Entries() => throw new NotSupportedException("A fingerprint-compressed spotter cannot enumerate its original keys.");
    }
}
