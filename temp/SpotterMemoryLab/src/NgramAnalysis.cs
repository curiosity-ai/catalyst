using System;
using System.Collections.Generic;

namespace SpotterMemoryLab
{
    /// <summary>
    /// The two things an n-gram decomposition can actually be used for, measured separately because they
    /// have opposite outcomes:
    ///
    ///  - as an INDEX (gram -> posting list of entry ids) it is a candidate generator for fuzzy matching,
    ///    and it costs more than the whole exact model it would replace, because every entry appears in
    ///    (length - k + 1) posting lists and an exact answer still needs the keys to verify against;
    ///  - as a COMPRESSOR (replace frequent grams with a 2-byte code) it does shrink the stored surface
    ///    forms, which is the part of the idea worth keeping - though a front-coded or automaton
    ///    representation of the same strings exploits the same redundancy and needs no separate index.
    /// </summary>
    internal static class NgramAnalysis
    {
        public sealed class IndexEstimate
        {
            public int  K;
            public int  DistinctGrams;
            public long TotalPostings;
            public long BytesFixed32;
            public long BytesDeltaVarint;
            public long Objects;
            public double MedianMinDocFrequency;
        }

        public static IndexEstimate EstimateInvertedIndex(StringBlob blob, int k, int sampleQueries = 20000)
        {
            var map  = new byte[256];
            var used = blob.Alphabet();
            for (int i = 0; i < used.Length; i++) { map[used[i]] = (byte)(i + 1); }
            int K = used.Length + 1;

            int space = 1;
            for (int i = 0; i < k; i++) { space *= K; }
            var df = new int[space];

            long totalPostings = 0;
            for (int i = 0; i < blob.Count; i++)
            {
                var s = blob[i];
                for (int p = 0; p + k <= s.Length; p++)
                {
                    int g = 0;
                    for (int j = 0; j < k; j++) { g = g * K + map[s[p + j]]; }
                    df[g]++;
                    totalPostings++;
                }
            }

            long n = blob.Count;
            long deltaBytes = 0;
            int  distinct   = 0;
            foreach (var c in df)
            {
                if (c == 0) { continue; }
                distinct++;
                double avgGap = Math.Max(1.0, n / (double)c);
                deltaBytes   += (long)c * VarintBytes((long)avgGap);
            }

            // Best-case candidate count for a query: the rarest gram it contains.
            var mins = new List<int>(sampleQueries);
            int step = Math.Max(1, blob.Count / sampleQueries);
            for (int i = 0; i < blob.Count; i += step)
            {
                var s   = blob[i];
                int min = int.MaxValue;
                for (int p = 0; p + k <= s.Length; p++)
                {
                    int g = 0;
                    for (int j = 0; j < k; j++) { g = g * K + map[s[p + j]]; }
                    if (df[g] < min) { min = df[g]; }
                }
                if (min != int.MaxValue) { mins.Add(min); }
            }
            mins.Sort();

            long dictBytes   = Sz.HashTable(Sz.ExactCapacity(distinct), Sz.DictionaryEntry(8, 8, 4, 4), out int dictObjects);
            long offsetBytes = Sz.Arr(distinct + 1, 8);

            return new IndexEstimate
            {
                K                     = k,
                DistinctGrams         = distinct,
                TotalPostings         = totalPostings,
                BytesFixed32          = dictBytes + offsetBytes + Sz.Arr(totalPostings, 4),
                BytesDeltaVarint      = dictBytes + offsetBytes + Sz.Arr(deltaBytes, 1),
                Objects               = dictObjects + 2,
                MedianMinDocFrequency = mins.Count == 0 ? 0 : mins[mins.Count / 2],
            };
        }

        private static int VarintBytes(long v)
        {
            int n = 1;
            while (v >= 0x80) { v >>= 7; n++; }
            return n;
        }

        public sealed class CodingEstimate
        {
            public int  DictionarySize;
            public long CodedBytes;
            public long DictionaryBytes;
            public double BytesPerEntry;
            public double BytesPerChar;
        }

        /// <summary>
        /// Greedy longest-match coding against a dictionary of the most valuable 3- and 4-grams. A literal
        /// costs one byte (the mapped symbol, always &lt; 128); a dictionary hit costs two (a 15-bit code
        /// with the high bit set).
        /// </summary>
        public static CodingEstimate EstimateGramCoding(StringBlob blob, int dictionarySize = 32768)
        {
            var map = new byte[256];
            var used = blob.Alphabet();
            for (int i = 0; i < used.Length; i++) { map[used[i]] = (byte)(i + 1); }
            int K = used.Length + 1;

            var freq3 = new int[K * K * K];
            var freq4 = new int[K * K * K * K];

            for (int i = 0; i < blob.Count; i++)
            {
                var s = blob[i];
                for (int p = 0; p + 3 <= s.Length; p++)
                {
                    int g = (map[s[p]] * K + map[s[p + 1]]) * K + map[s[p + 2]];
                    freq3[g]++;
                    if (p + 4 <= s.Length) { freq4[g * K + map[s[p + 3]]]++; }
                }
            }

            // Value of adding a gram: bytes saved if every occurrence is coded (3-gram saves 1, 4-gram 2).
            var heap = new List<(long value, int len, int gram)>();
            for (int g = 0; g < freq3.Length; g++) { if (freq3[g] > 0) { heap.Add(((long)freq3[g] * 1, 3, g)); } }
            for (int g = 0; g < freq4.Length; g++) { if (freq4[g] > 0) { heap.Add(((long)freq4[g] * 2, 4, g)); } }
            heap.Sort((a, b) => b.value.CompareTo(a.value));

            var in3 = new bool[freq3.Length];
            var in4 = new bool[freq4.Length];
            int take = Math.Min(dictionarySize, heap.Count);
            long dictChars = 0;
            for (int i = 0; i < take; i++)
            {
                var (_, len, gram) = heap[i];
                if (len == 3) { in3[gram] = true; } else { in4[gram] = true; }
                dictChars += len;
            }

            long coded = 0;
            for (int i = 0; i < blob.Count; i++)
            {
                var s = blob[i];
                int p = 0;
                while (p < s.Length)
                {
                    if (p + 4 <= s.Length)
                    {
                        int g4 = ((map[s[p]] * K + map[s[p + 1]]) * K + map[s[p + 2]]) * K + map[s[p + 3]];
                        if (in4[g4]) { coded += 2; p += 4; continue; }
                    }
                    if (p + 3 <= s.Length)
                    {
                        int g3 = (map[s[p]] * K + map[s[p + 1]]) * K + map[s[p + 2]];
                        if (in3[g3]) { coded += 2; p += 3; continue; }
                    }
                    coded += 1; p += 1;
                }
            }

            return new CodingEstimate
            {
                DictionarySize  = take,
                CodedBytes      = coded,
                DictionaryBytes = Sz.Arr(dictChars, 1) + Sz.Arr(take + 1, 4),
                BytesPerEntry   = coded / (double)blob.Count,
                BytesPerChar    = coded / (double)blob.TotalChars,
            };
        }
    }
}
