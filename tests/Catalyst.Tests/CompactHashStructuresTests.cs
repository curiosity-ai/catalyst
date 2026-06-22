using System;
using System.Collections.Generic;
using System.Linq;
using Catalyst.Models;
using UID;
using Xunit;

namespace Catalyst.Tests
{
    public class CompactHashStructuresTests
    {
        private static ulong[] RandomKeys(int count, int seed, bool includeZero = false)
        {
            var rng  = new Random(seed);
            var set  = new HashSet<ulong>();
            if (includeZero) { set.Add(0); }
            while (set.Count < count)
            {
                ulong hi = (ulong)(uint)rng.Next();
                ulong lo = (ulong)(uint)rng.Next();
                set.Add((hi << 32) | lo);
            }
            return set.ToArray();
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        [InlineData(7)]
        [InlineData(1000)]
        [InlineData(50000)]
        public void ExactSet_MatchesHashSet(int count)
        {
            var keys   = RandomKeys(count, seed: count + 1, includeZero: count > 0);
            var source = new HashSet<ulong>(keys);
            var set    = new ExactHashSet64(source);

            Assert.Equal(source.Count, set.Count);

            foreach (var k in keys)
            {
                Assert.True(set.Contains(k));
            }

            // Probe a large set of values that are (almost surely) not members and require parity with HashSet.
            var rng = new Random(12345);
            for (int i = 0; i < 100_000; i++)
            {
                ulong probe = ((ulong)(uint)rng.Next() << 32) | (uint)rng.Next();
                Assert.Equal(source.Contains(probe), set.Contains(probe));
            }

            // Enumeration round-trips to the original set.
            Assert.True(source.SetEquals(set.Keys()));
        }

        [Fact]
        public void ExactSet_HandlesZeroKey()
        {
            var withZero = new ExactHashSet64(new HashSet<ulong> { 0, 5, 99 });
            Assert.True(withZero.Contains(0));
            Assert.True(withZero.Contains(5));
            Assert.False(withZero.Contains(7));

            var withoutZero = new ExactHashSet64(new HashSet<ulong> { 5, 99 });
            Assert.False(withoutZero.Contains(0));
        }

        [Theory]
        [InlineData(1000)]
        [InlineData(50000)]
        public void FingerprintSet_HasNoFalseNegatives(int count)
        {
            var keys = RandomKeys(count, seed: count + 7, includeZero: true);
            var set  = new FingerprintHashSet64(new HashSet<ulong>(keys));

            // The fingerprint variant may have rare false positives, but must never have false negatives.
            foreach (var k in keys)
            {
                Assert.True(set.Contains(k));
            }

            Assert.False(set.CanEnumerateKeys);
            Assert.Throws<NotSupportedException>(() => set.Keys().ToArray());
        }

        [Theory]
        [InlineData(0)]
        [InlineData(1)]
        [InlineData(1000)]
        [InlineData(50000)]
        public void ExactMap_MatchesDictionary(int count)
        {
            var keys = RandomKeys(count, seed: count + 3, includeZero: count > 0);
            var source = new Dictionary<ulong, UID128>();
            foreach (var k in keys) { source[k] = UID128.New(); }

            var map = new ExactHashMap64(source);
            Assert.Equal(source.Count, map.Count);

            foreach (var kv in source)
            {
                Assert.True(map.TryGetValue(kv.Key, out var v));
                Assert.Equal(kv.Value, v);
            }

            var rng = new Random(999);
            for (int i = 0; i < 100_000; i++)
            {
                ulong probe = ((ulong)(uint)rng.Next() << 32) | (uint)rng.Next();
                Assert.Equal(source.ContainsKey(probe), map.TryGetValue(probe, out _));
            }

            Assert.Equal(source.Count, map.Entries().Count());
            foreach (var kv in map.Entries())
            {
                Assert.Equal(source[kv.Key], kv.Value);
            }
        }

        [Fact]
        public void ExactMap_HandlesZeroKey()
        {
            var zeroValue = UID128.New();
            var map = new ExactHashMap64(new Dictionary<ulong, UID128> { { 0, zeroValue }, { 42, UID128.New() } });

            Assert.True(map.TryGetValue(0, out var v));
            Assert.Equal(zeroValue, v);
            Assert.False(map.TryGetValue(7, out _));
        }

        [Theory]
        [InlineData(1000)]
        [InlineData(50000)]
        public void FingerprintMap_ResolvesKnownKeys(int count)
        {
            var keys = RandomKeys(count, seed: count + 11, includeZero: true);
            var source = new Dictionary<ulong, UID128>();
            foreach (var k in keys) { source[k] = UID128.New(); }

            var map = new FingerprintHashMap64(source);

            // With 32-bit fingerprints, the expected number of intra-set collisions across 50k random keys is
            // ~50k^2 / 2^33 < 0.0003, so every known key resolves to its exact value in practice.
            foreach (var kv in source)
            {
                Assert.True(map.TryGetValue(kv.Key, out var v));
                Assert.Equal(kv.Value, v);
            }
        }

        // Note: the open-addressed structures above are now the fallback; the CompactHash factory builds the
        // perfect-hash variants by default - that is covered by MphPerfectHashTests.Factory_BuildsPerfectHashStructures.
    }
}
