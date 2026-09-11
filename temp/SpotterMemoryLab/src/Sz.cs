using System;
using System.Collections.Generic;

namespace SpotterMemoryLab
{
    /// <summary>
    /// Analytic size model for CLR data structures on 64-bit. Every reported number in this lab comes from
    /// this model (counted array elements / object fields), never from process working set or GC counters.
    /// </summary>
    internal static class Sz
    {
        public const int OBJECT_HEADER = 16; // sync block index + method table pointer
        public const int ARRAY_HEADER  = 24; // object header + 8-byte length, payload starts 8-byte aligned
        public const int REF           = 8;

        public static long Align8(long bytes) => (bytes + 7) & ~7L;

        /// <summary>Bytes held by a single managed array of <paramref name="count"/> elements.</summary>
        public static long Arr(long count, int elementBytes) => Align8(ARRAY_HEADER + count * elementBytes);

        /// <summary>Bytes held by a bitmap of <paramref name="bits"/> bits stored as ulong[].</summary>
        public static long Bits(long bits) => Arr((bits + 63) / 64, 8);

        /// <summary>Bytes held by one instance of a reference type with the given total instance-field size.</summary>
        public static long Obj(int instanceFieldBytes) => Align8(OBJECT_HEADER + instanceFieldBytes);

        // ---- BCL collections -------------------------------------------------------------------------

        // Dictionary<K,V>: _buckets int[], _entries Entry[], plus 8 scalar/ref fields.
        public const int DICTIONARY_SELF = 16 + 8 + 8 + 8 + 8 + 4 + 4 + 4 + 4; // ~64 rounded below
        // HashSet<T>: _buckets int[], _entries Entry[], plus the same bookkeeping.
        public const int HASHSET_SELF = DICTIONARY_SELF;

        /// <summary>
        /// Bytes held by a Dictionary/HashSet with <paramref name="capacity"/> slots and the given entry
        /// struct size. capacity is the prime the BCL actually allocated (see <see cref="GrownCapacity"/>).
        /// </summary>
        public static long HashTable(long capacity, int entryBytes, out int objects)
        {
            objects = 3; // the collection itself + buckets + entries
            return Align8(OBJECT_HEADER + DICTIONARY_SELF) + Arr(capacity, 4) + Arr(capacity, entryBytes);
        }

        /// <summary>
        /// Entry struct size for Dictionary&lt;K,V&gt;: uint hashCode, int next, K key, V value, laid out
        /// with the natural alignment the CLR gives the struct.
        /// </summary>
        public static int DictionaryEntry(int keyBytes, int keyAlign, int valueBytes, int valueAlign)
        {
            int align  = Math.Max(4, Math.Max(keyAlign, valueAlign));
            long off   = 8; // hashCode + next
            off        = RoundUp(off, keyAlign) + keyBytes;
            off        = RoundUp(off, valueAlign) + valueBytes;
            return (int)RoundUp(off, align);
        }

        /// <summary>Entry struct size for HashSet&lt;T&gt;: int hashCode, int next, T value.</summary>
        public static int HashSetEntry(int valueBytes, int valueAlign)
        {
            int align = Math.Max(4, valueAlign);
            long off  = 8;
            off       = RoundUp(off, valueAlign) + valueBytes;
            return (int)RoundUp(off, align);
        }

        private static long RoundUp(long v, int a) => (v + a - 1) / a * a;

        /// <summary>
        /// The capacity a Dictionary/HashSet ends up with after <paramref name="count"/> sequential Add calls
        /// starting from an empty, capacity-less collection. Mirrors HashHelpers' prime-doubling chain.
        /// </summary>
        public static long GrownCapacity(long count)
        {
            if (count == 0) { return 0; }
            long size = GetPrime(3);
            long used = 0;
            while (true)
            {
                if (used + (count - used) <= size) { return size; }
                used = size;
                size = GetPrime(size * 2);
            }
        }

        /// <summary>The capacity for a collection constructed with an explicit capacity hint.</summary>
        public static long ExactCapacity(long count) => count == 0 ? 0 : GetPrime(count);

        private static readonly int[] PRIMES =
        {
            3, 7, 11, 17, 23, 29, 37, 47, 59, 71, 89, 107, 131, 163, 197, 239, 293, 353, 431, 521, 631, 761,
            919, 1103, 1327, 1597, 1931, 2333, 2801, 3371, 4049, 4861, 5839, 7013, 8419, 10103, 12143, 14591,
            17519, 21023, 25229, 30293, 36353, 43627, 52361, 62851, 75431, 90523, 108631, 130363, 156437,
            187751, 225307, 270371, 324449, 389357, 467237, 560689, 672827, 807403, 968897, 1162687, 1395263,
            1674319, 2009191, 2411033, 2893249, 3471899, 4166287, 4999559, 5999471, 7199369
        };

        private static long GetPrime(long min)
        {
            foreach (var p in PRIMES)
            {
                if (p >= min) { return p; }
            }
            for (long i = min | 1; i < int.MaxValue; i += 2)
            {
                if (IsPrime(i) && (i - 1) % 101 != 0) { return i; }
            }
            return min;
        }

        private static bool IsPrime(long c)
        {
            if ((c & 1) == 0) { return c == 2; }
            long limit = (long)Math.Sqrt(c);
            for (long d = 3; d <= limit; d += 2)
            {
                if (c % d == 0) { return false; }
            }
            return true;
        }

        // ---- formatting -----------------------------------------------------------------------------

        public static string MB(long bytes) => (bytes / (1024.0 * 1024.0)).ToString("n1") + " MB";

        public static string PerKey(long bytes, long keys) => keys == 0 ? "-" : (bytes / (double)keys).ToString("n2");
    }

    /// <summary>One measured design: its estimated bytes, its live managed-object count, and how it behaves.</summary>
    internal sealed class Measurement
    {
        public string Name;
        public string Group;
        public long   Bytes;
        public long   Objects;
        public string Exactness = "exact";
        public double FalsePositiveRate = -1;
        public double BuildSeconds;
        public double LookupNsPerQuery = -1;
        public string Notes = "";
        public List<(string part, long bytes)> Breakdown = new();

        public Measurement Add(string part, long bytes)
        {
            Breakdown.Add((part, bytes));
            Bytes += bytes;
            return this;
        }
    }
}
