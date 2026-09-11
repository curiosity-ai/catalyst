using System;

namespace SpotterMemoryLab
{
    /// <summary>
    /// Analytic size model of the structures the spotters use today, derived from the layouts in
    /// CompactHashStructures.cs / MphPerfectHash.cs. Program.cs cross-checks these against a real
    /// Spotter/LinkedSpotter built over the same dataset.
    /// </summary>
    internal static class BaselineModel
    {
        public const double MPH_ALPHA  = 0.80; // Mph.Alpha
        public const int    MPH_LAMBDA = 4;    // Mph.Lambda

        public static long MphSlots(long n)         => Math.Max(1, (long)Math.Ceiling(n / MPH_ALPHA));
        public static long MphDisplacements(long n) => Math.Max(1, n / MPH_LAMBDA);

        public static long DisplacementBytes(long n) => Sz.Arr(MphDisplacements(n), 4);

        /// <summary>MphHashSet64: one ulong slot per perfect-hash slot.</summary>
        public static long SetLossless(long n, out int objects)
        {
            objects = 4; // set + key array + Mph + displacement array
            return Sz.Arr(MphSlots(n), 8) + DisplacementBytes(n) + Sz.Obj(4 * Sz.REF + 8) + Sz.Obj(3 * Sz.REF + 24);
        }

        /// <summary>MphFingerprintSet64: one uint fingerprint per slot, ~2^-32 false positives.</summary>
        public static long SetFingerprint(long n, out int objects)
        {
            objects = 4;
            return Sz.Arr(MphSlots(n), 4) + DisplacementBytes(n) + Sz.Obj(4 * Sz.REF + 8) + Sz.Obj(3 * Sz.REF + 24);
        }

        /// <summary>MphHashMap64: a ulong key and a 16-byte UID128 per slot.</summary>
        public static long MapLossless(long n, out int objects)
        {
            objects = 5;
            return Sz.Arr(MphSlots(n), 8) + Sz.Arr(MphSlots(n), 16) + DisplacementBytes(n) + Sz.Obj(5 * Sz.REF + 24) + Sz.Obj(3 * Sz.REF + 24);
        }

        /// <summary>MphFingerprintMap64: a uint fingerprint and a 16-byte UID128 per slot.</summary>
        public static long MapFingerprint(long n, out int objects)
        {
            objects = 5;
            return Sz.Arr(MphSlots(n), 4) + Sz.Arr(MphSlots(n), 16) + DisplacementBytes(n) + Sz.Obj(5 * Sz.REF + 24) + Sz.Obj(3 * Sz.REF + 24);
        }

        /// <summary>
        /// Proposed: perfect hash over 32-bit fingerprints plus an occupancy bitmap with a rank index, so
        /// the UID array is indexed by dense rank instead of by slot. Removes the 20% slack the perfect
        /// hash leaves in the (large) value array for the cost of ~1.2 bits per slot.
        /// </summary>
        public static long MapFingerprintRanked(long n, out int objects)
        {
            objects = 7;
            long m = MphSlots(n);
            return Sz.Arr(m, 4)                    // fingerprints
                 + Sz.Bits(m)                      // occupancy
                 + Sz.Arr(m / 512 + 1, 4)           // rank samples, one per 512 bits
                 + Sz.Arr(n, 16)                    // dense UID128 values
                 + DisplacementBytes(n)
                 + Sz.Obj(6 * Sz.REF + 24) + Sz.Obj(3 * Sz.REF + 24);
        }

        /// <summary>Dictionary&lt;int, TokenizationException&gt; as the Spotter keeps it today.</summary>
        public static long SpotterExceptionTable(long count, long capacity, out int objects)
        {
            int entry = Sz.DictionaryEntry(keyBytes: 4, keyAlign: 4, valueBytes: 8, valueAlign: 8); // struct { string[] }
            return Sz.HashTable(capacity, entry, out objects);
        }

        /// <summary>HashSet&lt;int&gt; as the LinkedSpotter keeps it today.</summary>
        public static long LinkedExceptionTable(long count, long capacity, out int objects)
        {
            int entry = Sz.HashSetEntry(valueBytes: 4, valueAlign: 4);
            return Sz.HashTable(capacity, entry, out objects);
        }

        /// <summary>Exact replacement for either exception table: the 32-bit hashes, sorted, as one array.</summary>
        public static long SortedExceptionArray(long count, out int objects)
        {
            objects = 2; // the array plus a small bucket index
            return Sz.Arr(count, 4) + Sz.Arr(1 << 16, 4);
        }

        /// <summary>Blocked Bloom filter over the exception hashes at the given bits per key.</summary>
        public static long BloomExceptionFilter(long count, int bitsPerKey, out int objects, out double falsePositiveRate)
        {
            objects = 2;
            int k = Math.Max(1, (int)Math.Round(bitsPerKey * 0.693));
            falsePositiveRate = Math.Pow(1 - Math.Exp(-k / (double)bitsPerKey), k);
            return Sz.Bits(count * bitsPerKey);
        }
    }
}
