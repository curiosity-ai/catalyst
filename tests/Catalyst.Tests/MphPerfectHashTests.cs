using System;
using System.Collections.Generic;
using System.Linq;
using Catalyst.Models;
using UID;
using Xunit;

namespace Catalyst.Tests
{
    public class MphPerfectHashTests
    {
        private static ulong[] RandomKeys(int count, int seed, bool includeZero = false)
        {
            var rng = new Random(seed);
            var set = new HashSet<ulong>();
            if (includeZero) { set.Add(0); }
            while (set.Count < count) { set.Add((((ulong)(uint)rng.Next()) << 32) | (uint)rng.Next()); }
            return set.ToArray();
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        [InlineData(9)]
        [InlineData(1000)]
        [InlineData(100_000)] // large enough to exercise the overflow path
        public void MphSet_MatchesHashSet(int count)
        {
            var keys   = RandomKeys(count, 11 + count, includeZero: count > 0);
            var source = new HashSet<ulong>(keys);
            var set    = new MphHashSet64(source);

            Assert.Equal(source.Count, set.Count);

            // no false negatives - every member must be found
            foreach (var k in keys) { Assert.True(set.Contains(k)); }

            // parity with HashSet over many random probes (exact: no false positives either)
            var rng = new Random(2024);
            for (int i = 0; i < 200_000; i++)
            {
                ulong probe = (((ulong)(uint)rng.Next()) << 32) | (uint)rng.Next();
                Assert.Equal(source.Contains(probe), set.Contains(probe));
            }

            // enumeration round-trips to the original set (incl. any overflow keys)
            Assert.True(set.CanEnumerateKeys);
            Assert.True(source.SetEquals(set.Keys()));
        }

        [Fact]
        public void MphSet_HandlesZeroKey()
        {
            var withZero = new MphHashSet64(new HashSet<ulong> { 0, 5, 99, 12345 });
            Assert.True(withZero.Contains(0));
            Assert.True(withZero.Contains(5));
            Assert.False(withZero.Contains(7));

            var withoutZero = new MphHashSet64(new HashSet<ulong> { 5, 99 });
            Assert.False(withoutZero.Contains(0));
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        [InlineData(1000)]
        [InlineData(100_000)]
        public void MphMap_MatchesDictionary(int count)
        {
            var keys = RandomKeys(count, 31 + count, includeZero: count > 0);
            var source = new Dictionary<ulong, UID128>();
            foreach (var k in keys) { source[k] = UID128.New(); }

            var map = new MphHashMap64(source);
            Assert.Equal(source.Count, map.Count);

            foreach (var kv in source)
            {
                Assert.True(map.TryGetValue(kv.Key, out var v));
                Assert.Equal(kv.Value, v);
            }

            var rng = new Random(7);
            for (int i = 0; i < 200_000; i++)
            {
                ulong probe = (((ulong)(uint)rng.Next()) << 32) | (uint)rng.Next();
                Assert.Equal(source.ContainsKey(probe), map.TryGetValue(probe, out _));
            }

            Assert.True(map.CanEnumerateKeys);
            var roundTrip = map.Entries().ToDictionary(e => e.Key, e => e.Value);
            Assert.Equal(source.Count, roundTrip.Count);
            foreach (var kv in source) { Assert.Equal(kv.Value, roundTrip[kv.Key]); }
        }

        [Fact]
        public void MphMap_HandlesZeroKey()
        {
            var zeroValue = UID128.New();
            var map = new MphHashMap64(new Dictionary<ulong, UID128> { { 0, zeroValue }, { 42, UID128.New() }, { 9999, UID128.New() } });
            Assert.True(map.TryGetValue(0, out var v));
            Assert.Equal(zeroValue, v);
            Assert.False(map.TryGetValue(7, out _));
        }

        [Theory]
        [InlineData(1000)]
        [InlineData(100_000)]
        public void MphFingerprintSet_HasNoFalseNegatives(int count)
        {
            var keys = RandomKeys(count, 53 + count, includeZero: true);
            var set  = new MphFingerprintSet64(new HashSet<ulong>(keys));

            foreach (var k in keys) { Assert.True(set.Contains(k)); }

            Assert.False(set.CanEnumerateKeys);
            Assert.Throws<NotSupportedException>(() => set.Keys().ToArray());
        }

        [Theory]
        [InlineData(1000)]
        [InlineData(100_000)]
        public void MphFingerprintMap_ResolvesKnownKeys(int count)
        {
            var keys = RandomKeys(count, 71 + count, includeZero: true);
            var source = new Dictionary<ulong, UID128>();
            foreach (var k in keys) { source[k] = UID128.New(); }

            var map = new MphFingerprintMap64(source);
            foreach (var kv in source)
            {
                Assert.True(map.TryGetValue(kv.Key, out var v));
                Assert.Equal(kv.Value, v);
            }
            Assert.False(map.CanEnumerateKeys);
        }

        [Theory]
        [InlineData(1000)]
        [InlineData(100_000)]
        public void MphFingerprint16Set_NoFalseNegatives_AndLowFalsePositiveRate(int count)
        {
            var keys    = RandomKeys(count, 91 + count, includeZero: true);
            var present = new HashSet<ulong>(keys);
            var set     = new MphFingerprint16Set64(present);

            Assert.Equal(present.Count, set.Count);
            Assert.False(set.CanEnumerateKeys);

            // no false negatives - every member must be found
            foreach (var k in keys) { Assert.True(set.Contains(k)); }

            // false positives are allowed but should be ~2^-16; with 500k probes the expectation is ~7.6
            var rng = new Random(123);
            int fp = 0; const int trials = 500_000;
            for (int i = 0; i < trials; i++)
            {
                ulong probe = (((ulong)(uint)rng.Next()) << 32) | (uint)rng.Next();
                if (!present.Contains(probe) && set.Contains(probe)) { fp++; }
            }
            Assert.True(fp < 100, $"false-positive rate unexpectedly high: {fp}/{trials}");
        }

        [Fact]
        public void Factory_BuildsPerfectHashStructures()
        {
            var keys = new HashSet<ulong> { 1, 2, 3, 4, 5 };
            var map  = new Dictionary<ulong, UID128> { { 1, UID128.New() } };

            var previous = SpotterCompaction.UseFingerprint32;
            try
            {
                SpotterCompaction.UseFingerprint32 = false;
                Assert.IsType<MphHashSet64>(CompactHash.BuildSet(keys));
                Assert.IsType<MphHashMap64>(CompactHash.BuildMap(map));

                SpotterCompaction.UseFingerprint32 = true;
                Assert.IsType<MphFingerprintSet64>(CompactHash.BuildSet(keys));
                Assert.IsType<MphFingerprintMap64>(CompactHash.BuildMap(map));
            }
            finally
            {
                SpotterCompaction.UseFingerprint32 = previous;
            }
        }
    }
}
