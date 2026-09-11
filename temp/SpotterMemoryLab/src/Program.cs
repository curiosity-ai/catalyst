using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using Catalyst.Models;
using Mosaik.Core;
using UID;

namespace SpotterMemoryLab
{
    internal static class Program
    {
        private static int    N       = 10_000_000;
        private static ulong  SEED    = 20260911;
        private static bool   RUNREAL = true;
        private static bool   FLAT    = false;

        private static readonly List<Measurement> Results = new();

        private static void Main(string[] args)
        {
            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--n")        { N = int.Parse(args[++i]); }
                else if (args[i] == "--seed"){ SEED = ulong.Parse(args[++i]); }
                else if (args[i] == "--no-real") { RUNREAL = false; }
                else if (args[i] == "--flat") { FLAT = true; }
            }

            var sw = Stopwatch.StartNew();
            Console.WriteLine($"# Spotter memory lab - target {N:n0} aviation part numbers, seed {SEED}");
            Console.WriteLine();

            var gen    = new PartNumberGenerator(SEED) { LowStructure = FLAT };
            var blob   = gen.Generate(N);
            var sorted = Dedupe(blob, blob.SortedOrder(), out int duplicates);
            int n      = sorted.Length;

            DescribeDataset(blob, sorted, duplicates, sw);

            var hashes   = BuildHashes(blob, sorted, out int distinctHashes, out long exceptionCount, out int[] exceptionHashes, out int distinctExceptionHashes);
            var nonMembers = BuildNonMembers(blob, sorted, 1_000_000, out byte[] nonBlobData, out int[] nonOffsets);

            Console.WriteLine();
            Console.WriteLine($"Distinct 64-bit entry hashes : {distinctHashes:n0}  (collisions: {n - distinctHashes:n0})");
            Console.WriteLine($"Entries needing a tokenizer exception (not all letter-or-digit): {exceptionCount:n0} ({100.0 * exceptionCount / n:n1}%)");
            Console.WriteLine($"Distinct 32-bit exception hashes: {distinctExceptionHashes:n0}");
            Console.WriteLine();

            if (RUNREAL) { MeasureRealSpotters(blob, sorted, exceptionCount); }

            MeasureBaselineModels(n, distinctExceptionHashes);
            MeasureHashBasedCandidates(hashes, distinctHashes, n, distinctExceptionHashes, nonMembers);
            MeasureStringCandidates(blob, sorted, n, nonBlobData, nonOffsets, distinctExceptionHashes, hashes, distinctHashes, nonMembers);
            MeasureNgrams(blob, n);

            Report();
            Console.WriteLine($"# total lab time {sw.Elapsed.TotalSeconds:n0}s");
        }

        // ----------------------------------------------------------------------------------------------

        private static int[] Dedupe(StringBlob blob, int[] sorted, out int duplicates)
        {
            var outv = new int[sorted.Length];
            int k    = 0;
            duplicates = 0;
            for (int i = 0; i < sorted.Length; i++)
            {
                if (i > 0 && blob[sorted[i]].SequenceEqual(blob[sorted[i - 1]])) { duplicates++; continue; }
                outv[k++] = sorted[i];
            }
            Array.Resize(ref outv, k);
            return outv;
        }

        private static void DescribeDataset(StringBlob blob, int[] sorted, int duplicates, Stopwatch sw)
        {
            long totalChars = 0, sharedPrefix = 0;
            int  minLen = int.MaxValue, maxLen = 0, withSpace = 0, notAlnum = 0;
            var  lengthHistogram = new int[64];

            for (int i = 0; i < sorted.Length; i++)
            {
                var s = blob[sorted[i]];
                totalChars += s.Length;
                minLen = Math.Min(minLen, s.Length);
                maxLen = Math.Max(maxLen, s.Length);
                lengthHistogram[Math.Min(63, s.Length)]++;

                bool space = false, alnum = true;
                foreach (var c in s)
                {
                    if (c == ' ') { space = true; }
                    if (!((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) { alnum = false; }
                }
                if (space) { withSpace++; }
                if (!alnum) { notAlnum++; }

                if (i > 0)
                {
                    var p = blob[sorted[i - 1]];
                    int m = Math.Min(p.Length, s.Length), j = 0;
                    while (j < m && p[j] == s[j]) { j++; }
                    sharedPrefix += j;
                }
            }

            int n = sorted.Length;
            Console.WriteLine("## Dataset");
            Console.WriteLine($"entries                       : {n:n0} distinct ({duplicates:n0} generated duplicates removed)");
            Console.WriteLine($"total characters              : {totalChars:n0}");
            Console.WriteLine($"length  min / avg / max       : {minLen} / {totalChars / (double)n:n2} / {maxLen}");
            Console.WriteLine($"alphabet                      : {blob.Alphabet().Length} distinct bytes -> \"{Encoding.ASCII.GetString(blob.Alphabet())}\"");
            Console.WriteLine($"average shared prefix (sorted): {sharedPrefix / (double)(n - 1):n2} chars  ({100.0 * sharedPrefix / totalChars:n1}% of all characters)");
            Console.WriteLine($"multi-token (contains space)  : {withSpace:n0} ({100.0 * withSpace / n:n2}%)");
            Console.WriteLine($"not all letter-or-digit       : {notAlnum:n0} ({100.0 * notAlnum / n:n1}%)");
            Console.Write("examples                      : ");
            for (int i = 0; i < 8; i++) { Console.Write(blob.StringAt(sorted[(int)((long)n * i / 8) + 12345]) + "  "); }
            Console.WriteLine();
            Console.WriteLine($"[{sw.Elapsed.TotalSeconds:n0}s]");
        }

        private static ulong[] BuildHashes(StringBlob blob, int[] sorted, out int distinct, out long exceptionCount, out int[] exceptionHashes, out int distinctExceptionHashes)
        {
            int n       = sorted.Length;
            var hashes  = new ulong[n];
            var scratch = new char[128];
            var excs    = new List<int>(n / 2);
            exceptionCount = 0;

            for (int i = 0; i < n; i++)
            {
                var s = blob[sorted[i]];
                for (int j = 0; j < s.Length; j++) { scratch[j] = (char)s[j]; }
                var span  = new ReadOnlySpan<char>(scratch, 0, s.Length);
                hashes[i] = RefHashSet.Mix(Spotter.Hash64(span));

                bool alnum = true;
                foreach (var c in s)
                {
                    if (!((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) { alnum = false; break; }
                }
                if (!alnum)
                {
                    exceptionCount++;
                    // The word-level exceptions the spotters register: one per whitespace-separated word.
                    int start = 0;
                    for (int j = 0; j <= s.Length; j++)
                    {
                        if (j == s.Length || s[j] == ' ')
                        {
                            if (j > start)
                            {
                                var w = new ReadOnlySpan<char>(scratch, start, j - start);
                                excs.Add(CaseSensitiveHash32(w));
                            }
                            start = j + 1;
                        }
                    }
                }
            }

            exceptionHashes = excs.ToArray();
            Array.Sort(exceptionHashes);
            distinctExceptionHashes = CountDistinct(exceptionHashes);

            var copy = (ulong[])hashes.Clone();
            Array.Sort(copy);
            distinct = CountDistinctU(copy);
            return copy; // sorted, used by the sorted-structure candidates
        }

        private static int CaseSensitiveHash32(ReadOnlySpan<char> key)
        {
            // Mirrors Mosaik.Core's CaseSensitiveHash32 well enough for a size/collision estimate.
            unchecked
            {
                uint h = 2166136261;
                for (int i = 0; i < key.Length; i++) { h = (h ^ key[i]) * 16777619; }
                return (int)h;
            }
        }

        private static int CountDistinct(int[] sortedArr)
        {
            int d = 0;
            for (int i = 0; i < sortedArr.Length; i++) { if (i == 0 || sortedArr[i] != sortedArr[i - 1]) { d++; } }
            return d;
        }

        private static int CountDistinctU(ulong[] sortedArr)
        {
            int d = 0;
            for (int i = 0; i < sortedArr.Length; i++) { if (i == 0 || sortedArr[i] != sortedArr[i - 1]) { d++; } }
            return d;
        }

        // Near-miss non-members: real part numbers with one character changed, plus fresh random ones.
        private static ulong[] BuildNonMembers(StringBlob blob, int[] sorted, int count, out byte[] data, out int[] offsets)
        {
            var rng   = new PartNumberGenerator(SEED ^ 0xDEADBEEF) { LowStructure = FLAT };
            var other = rng.Generate(count / 2, multiTokenShare: 0.015);

            var members = new HashSet<ulong>();
            var scratch = new char[128];
            for (int i = 0; i < sorted.Length; i++)
            {
                var s = blob[sorted[i]];
                for (int j = 0; j < s.Length; j++) { scratch[j] = (char)s[j]; }
                members.Add(Spotter.Hash64(new ReadOnlySpan<char>(scratch, 0, s.Length)));
            }

            var outHashes = new List<ulong>(count);
            var bytes     = new List<byte>(count * 14);
            var offs      = new List<int> { 0 };

            void TryAdd(ReadOnlySpan<byte> s)
            {
                for (int j = 0; j < s.Length; j++) { scratch[j] = (char)s[j]; }
                var h = Spotter.Hash64(new ReadOnlySpan<char>(scratch, 0, s.Length));
                if (members.Contains(h)) { return; }
                outHashes.Add(RefHashSet.Mix(h));
                for (int j = 0; j < s.Length; j++) { bytes.Add(s[j]); }
                offs.Add(bytes.Count);
            }

            for (int i = 0; i < other.Count && outHashes.Count < count / 2; i++) { TryAdd(other[i]); }

            var mutated = new byte[64];
            for (int i = 0; outHashes.Count < count && i < sorted.Length; i += Math.Max(1, sorted.Length / (count / 2 + 1)))
            {
                var s = blob[sorted[i]];
                s.CopyTo(mutated);
                int at = (i * 7 + 3) % s.Length;
                mutated[at] = (byte)('0' + ((mutated[at] + i) % 10));
                TryAdd(new ReadOnlySpan<byte>(mutated, 0, s.Length));
            }

            data    = bytes.ToArray();
            offsets = offs.ToArray();
            return outHashes.ToArray();
        }

        // ----------------------------------------------------------------------------------------------

        private static void MeasureRealSpotters(StringBlob blob, int[] sorted, long exceptionCount)
        {
            Console.WriteLine("## Real Catalyst models, as implemented (built over the same dataset)");

            var sw = Stopwatch.StartNew();
            {
                var spotter = new Spotter(Language.Any, 0, "lab", "PN");
                for (int i = 0; i < sorted.Length; i++) { spotter.AddEntry(blob.StringAt(sorted[i])); }
                double addSeconds = sw.Elapsed.TotalSeconds;

                sw.Restart();
                spotter.TrimExcess();
                double freezeSeconds = sw.Elapsed.TotalSeconds;

                long tables    = spotter.OptimizedMemoryBytes;
                int  exceptions = spotter.GetSimpleSpecialCases().Count;

                var m = new Measurement { Name = "N0  Spotter, implemented (entry dictionary + prefilter)", Group = "Spotter", Objects = 10 }; // engine + dictionary(3) + exception set(4) + prefilter(2)
                m.Add("entry dictionary + exception table + prefilter (measured)", tables);
                m.Notes        = $"entries={sorted.Length:n0}, tokenizer exceptions={exceptions:n0}; add {addSeconds:n0}s, compact {freezeSeconds:n0}s";
                m.BuildSeconds = addSeconds + freezeSeconds;
                Results.Add(m);

                Console.WriteLine($"Spotter        : {Sz.MB(tables)} total, {exceptions:n0} tokenizer exceptions, add {addSeconds:n0}s + compact {freezeSeconds:n0}s");
                spotter.ClearModel();
            }
            GC.Collect(2, GCCollectionMode.Aggressive, true, true);

            sw.Restart();
            {
                var linked = new LinkedSpotter(Language.Any, 0, "lab", "PN");
                for (int i = 0; i < sorted.Length; i++) { linked.AddEntry(blob.StringAt(sorted[i]), UID128.New()); }
                double addSeconds = sw.Elapsed.TotalSeconds;

                sw.Restart();
                linked.TrimExcess();
                double freezeSeconds = sw.Elapsed.TotalSeconds;

                long tables     = linked.OptimizedMemoryBytes;
                int  exceptions = linked.GetSimpleSpecialCases().Count;

                var m = new Measurement { Name = "N1  LinkedSpotter, implemented (dictionary + dense UIDs + prefilter)", Group = "LinkedSpotter", Objects = 11 }; // the same, plus the dense UID array
                m.Add("entry dictionary + UID array + exception table + prefilter (measured)", tables);
                m.Notes        = $"entries={sorted.Length:n0}, tokenizer exceptions={exceptions:n0}; add {addSeconds:n0}s, compact {freezeSeconds:n0}s";
                m.BuildSeconds = addSeconds + freezeSeconds;
                Results.Add(m);

                Console.WriteLine($"LinkedSpotter  : {Sz.MB(tables)} total, {exceptions:n0} tokenizer exceptions, add {addSeconds:n0}s + compact {freezeSeconds:n0}s");
                linked.ClearModel();
            }
            GC.Collect(2, GCCollectionMode.Aggressive, true, true);
            Console.WriteLine();
        }

        private static void MeasureBaselineModels(int n, int exceptionKeys)
        {
            long excCap   = Sz.GrownCapacity(exceptionKeys);
            long excSpot  = BaselineModel.SpotterExceptionTable(exceptionKeys, excCap, out int excObjSpot);
            long excLink  = BaselineModel.LinkedExceptionTable(exceptionKeys, excCap, out int excObjLink);

            {
                long t = BaselineModel.SetLossless(n, out int o);
                var m = new Measurement { Name = "S0' Spotter today (modelled)", Group = "Spotter", Objects = o + excObjSpot * 2 };
                m.Add("perfect-hash key table", t);
                m.Add("TokenizerExceptions dictionary", excSpot);
                m.Add("copy inside the pipeline tokenizer", excSpot);
                Results.Add(m);
            }
            {
                long t = BaselineModel.SetFingerprint(n, out int o);
                var m = new Measurement { Name = "S1  Spotter, UseFingerprint32 (today, opt-in)", Group = "Spotter", Objects = o + excObjSpot * 2, Exactness = "lossy", FalsePositiveRate = Math.Pow(2, -32) };
                m.Add("perfect-hash fingerprint table", t);
                m.Add("TokenizerExceptions dictionary", excSpot);
                m.Add("copy inside the pipeline tokenizer", excSpot);
                Results.Add(m);
            }
            {
                long t = BaselineModel.MapLossless(n, out int o);
                var m = new Measurement { Name = "L0' LinkedSpotter today (modelled)", Group = "LinkedSpotter", Objects = o + excObjLink * 2 };
                m.Add("perfect-hash key+UID table", t);
                m.Add("TokenizerExceptionsSet", excLink);
                m.Add("copy inside the pipeline tokenizer", excLink);
                Results.Add(m);
            }
            {
                long t = BaselineModel.MapFingerprint(n, out int o);
                var m = new Measurement { Name = "L1  LinkedSpotter, UseFingerprint32 (today, opt-in)", Group = "LinkedSpotter", Objects = o + excObjLink * 2, Exactness = "lossy", FalsePositiveRate = Math.Pow(2, -32) };
                m.Add("perfect-hash fingerprint+UID table", t);
                m.Add("TokenizerExceptionsSet", excLink);
                m.Add("copy inside the pipeline tokenizer", excLink);
                Results.Add(m);
            }
            {
                long t = BaselineModel.MapFingerprintRanked(n, out int o);
                long e = BaselineModel.SortedExceptionArray(exceptionKeys, out int eo);
                var m = new Measurement { Name = "L2  MPH fingerprints + rank + dense UID array", Group = "LinkedSpotter", Objects = o + eo, Exactness = "lossy", FalsePositiveRate = Math.Pow(2, -32) };
                m.Add("fingerprints + occupancy + rank + dense UIDs", t);
                m.Add("exceptions as one sorted uint[] shared with the tokenizer", e);
                m.Notes = "removes the 20% perfect-hash slack from the 16-byte value array";
                Results.Add(m);
            }

            // Exception-table variants on their own, so the component is visible.
            {
                var m = new Measurement { Name = "E0  exceptions today: Dictionary<int,TokenizationException> + tokenizer copy", Group = "Exceptions", Objects = excObjSpot * 2 };
                m.Add("model", excSpot); m.Add("tokenizer copy", excSpot);
                Results.Add(m);
            }
            {
                var m = new Measurement { Name = "E0b exceptions today: HashSet<int> + tokenizer copy (LinkedSpotter)", Group = "Exceptions", Objects = excObjLink * 2 };
                m.Add("model", excLink); m.Add("tokenizer copy", excLink);
                Results.Add(m);
            }
            {
                long e = BaselineModel.SortedExceptionArray(exceptionKeys, out int eo);
                var m = new Measurement { Name = "E1  exceptions as a sorted uint[] shared with the tokenizer", Group = "Exceptions", Objects = eo };
                m.Add("sorted hashes + 64K bucket index", e);
                Results.Add(m);
            }
            foreach (var bits in new[] { 8, 10, 12, 16 })
            {
                long e = BaselineModel.BloomExceptionFilter(exceptionKeys, bits, out int eo, out double fp);
                var m = new Measurement { Name = $"E2  exceptions as a Bloom filter, {bits} bits/key", Group = "Exceptions", Objects = eo, Exactness = "lossy", FalsePositiveRate = fp };
                m.Add("filter", e);
                m.Notes = "a false positive keeps a hyphenated word un-split; it never loses a match";
                Results.Add(m);
            }
            {
                var m = new Measurement { Name = "E3  exceptions subsumed by a string dictionary (no separate table)", Group = "Exceptions", Objects = 0 };
                m.Add("nothing", 0);
                m.Notes = "only available to the front-coded / automaton designs, which can answer the prefix query directly";
                Results.Add(m);
            }
        }

        // ----------------------------------------------------------------------------------------------

        private static void MeasureHashBasedCandidates(ulong[] sortedHashes, int distinct, int n, int exceptionKeys, ulong[] nonMembers)
        {
            long excSorted = BaselineModel.SortedExceptionArray(exceptionKeys, out int excObjects);

            var uniq = new ulong[distinct];
            int k = 0;
            for (int i = 0; i < sortedHashes.Length; i++) { if (i == 0 || sortedHashes[i] != sortedHashes[i - 1]) { uniq[k++] = sortedHashes[i]; } }

            {
                var sw = Stopwatch.StartNew();
                var set = new RefHashSet(uniq, distinct);
                double build = sw.Elapsed.TotalSeconds;
                var (ns, fp) = Time(uniq, nonMembers, set.Contains);
                var m = new Measurement { Name = "S2  open-addressed ulong[] set (speed reference)", Group = "Spotter", Objects = set.Objects + excObjects, Exactness = "lossy (hash only)", FalsePositiveRate = fp, BuildSeconds = build, LookupNsPerQuery = ns };
                m.Add("slot array", set.EstimatedBytes);
                m.Add("exceptions as one sorted uint[]", excSorted);
                Results.Add(m);
            }
            {
                var sw = Stopwatch.StartNew();
                var ef = new EliasFano(uniq, distinct, 64);
                double build = sw.Elapsed.TotalSeconds;
                var (ns, fp) = Time(uniq, nonMembers, ef.Contains);
                var m = new Measurement { Name = "S3  Elias-Fano over the sorted 64-bit hashes", Group = "Spotter", Objects = ef.Objects + excObjects, Exactness = "lossy (hash only)", FalsePositiveRate = fp, BuildSeconds = build, LookupNsPerQuery = ns };
                m.Add("Elias-Fano", ef.EstimatedBytes);
                m.Add("exceptions as one sorted uint[]", excSorted);
                m.Notes = $"{ef.LowBitsPerKey} low bits/key; exact with respect to the 64-bit hash";
                Results.Add(m);
            }
            {
                // Same structure over 40-bit truncated hashes: cheaper, with a real false-positive rate.
                var t = new ulong[distinct];
                for (int i = 0; i < distinct; i++) { t[i] = uniq[i] >> 24; }
                Array.Sort(t);
                int td = 0;
                for (int i = 0; i < t.Length; i++) { if (i == 0 || t[i] != t[i - 1]) { t[td++] = t[i]; } }
                var ef = new EliasFano(t, td, 40);
                var m = new Measurement { Name = "S4  Elias-Fano over 40-bit truncated hashes", Group = "Spotter", Objects = ef.Objects + excObjects, Exactness = "lossy", FalsePositiveRate = n / Math.Pow(2, 40) };
                m.Add("Elias-Fano", ef.EstimatedBytes);
                m.Add("exceptions as one sorted uint[]", excSorted);
                m.Notes = $"{ef.LowBitsPerKey} low bits/key";
                Results.Add(m);
            }
            {
                // Analytic: a binary fuse / xor filter needs ~1.08x the information-theoretic bound and no
                // displacement array, so it is the cheapest possible approximate membership at a given rate.
                foreach (var bits in new[] { 16, 32 })
                {
                    long f = Sz.Arr((long)(n * 1.125), bits / 8);
                    var m = new Measurement { Name = $"S5  binary-fuse filter, {bits}-bit fingerprints (analytic)", Group = "Spotter", Objects = 2 + excObjects, Exactness = "lossy", FalsePositiveRate = Math.Pow(2, -bits) };
                    m.Add("filter", f);
                    m.Add("exceptions as one sorted uint[]", excSorted);
                    Results.Add(m);
                }
            }
        }

        private static void MeasureStringCandidates(StringBlob blob, int[] sorted, int n, byte[] nonData, int[] nonOffsets, int exceptionKeys, ulong[] sortedHashes, int distinctHashes, ulong[] nonMemberHashes)
        {
            var scratch = new byte[256];
            FrontCodedDictionary fcd16 = null, fcd64 = null;

            foreach (var blockSize in new[] { 8, 16, 32, 64 })
            {
                var sw  = Stopwatch.StartNew();
                var fcd = new FrontCodedDictionary(blob, sorted, blockSize);
                if (blockSize == 16) { fcd16 = fcd; }
                if (blockSize == 64) { fcd64 = fcd; }
                double build = sw.Elapsed.TotalSeconds;

                int bad = 0;
                for (int i = 0; i < n; i += Math.Max(1, n / 500_000))
                {
                    if (fcd.Find(blob[sorted[i]], scratch) != i) { bad++; }
                }

                int fp = 0, tested = 0;
                sw.Restart();
                for (int i = 0; i + 1 < nonOffsets.Length; i++)
                {
                    var s = new ReadOnlySpan<byte>(nonData, nonOffsets[i], nonOffsets[i + 1] - nonOffsets[i]);
                    if (fcd.Find(s, scratch) >= 0) { fp++; }
                    tested++;
                }
                double ns = sw.Elapsed.TotalMilliseconds * 1e6 / Math.Max(1, tested);

                var m = new Measurement
                {
                    Name             = $"S6/L3  front-coded sorted dictionary, block {blockSize}",
                    Group            = blockSize == 16 ? "Both" : "Front-coding sweep",
                    Objects          = fcd.Objects,
                    Exactness        = bad == 0 ? "exact" : $"BROKEN ({bad} mismatches)",
                    FalsePositiveRate = fp / (double)Math.Max(1, tested),
                    BuildSeconds     = build,
                    LookupNsPerQuery = ns,
                };
                m.Add("payload + block offsets", fcd.EstimatedBytes);
                m.Notes = $"{fcd.PayloadBytes / (double)n:n2} payload bytes/entry; rank-addressable, so it also serves the tokenizer exceptions and a dense UID array";
                Results.Add(m);

                if (blockSize == 16)
                {
                    var lm = new Measurement { Name = "L3  front-coded dictionary + dense UID128[] by rank", Group = "LinkedSpotter", Objects = fcd.Objects + 1, Exactness = "exact", LookupNsPerQuery = ns };
                    lm.Add("front-coded strings", fcd.EstimatedBytes);
                    lm.Add("dense UID128 values", Sz.Arr(n, 16));
                    lm.Notes = "no exception table needed";
                    Results.Add(lm);

                    var sm = new Measurement { Name = "S6  front-coded dictionary (membership only)", Group = "Spotter", Objects = fcd.Objects, Exactness = "exact", LookupNsPerQuery = ns };
                    sm.Add("front-coded strings", fcd.EstimatedBytes);
                    sm.Notes = "no exception table needed";
                    Results.Add(sm);
                }
            }

            {
                var sw = Stopwatch.StartNew();
                DAFSA = new Dafsa(blob, sorted);
                var dafsa = DAFSA;
                double build = sw.Elapsed.TotalSeconds;

                int bad = 0;
                for (int i = 0; i < n; i += Math.Max(1, n / 500_000))
                {
                    if (dafsa.Rank(blob[sorted[i]]) != i) { bad++; }
                }

                int fp = 0, tested = 0;
                sw.Restart();
                for (int i = 0; i + 1 < nonOffsets.Length; i++)
                {
                    var s = new ReadOnlySpan<byte>(nonData, nonOffsets[i], nonOffsets[i + 1] - nonOffsets[i]);
                    if (dafsa.Rank(s) >= 0) { fp++; }
                    tested++;
                }
                double ns = sw.Elapsed.TotalMilliseconds * 1e6 / Math.Max(1, tested);

                Console.WriteLine("## Automaton");
                Console.WriteLine($"trie states (no minimisation): {dafsa.TrieStates:n0}");
                Console.WriteLine($"DAFSA states                 : {dafsa.States:n0}");
                Console.WriteLine($"DAFSA transitions            : {dafsa.Transitions:n0}  ({dafsa.Transitions / (double)n:n2} per entry)");
                Console.WriteLine($"alphabet                     : {dafsa.Alphabet} symbols, packs into 6 bits: {dafsa.Alphabet <= 63}");
                Console.WriteLine($"32-bit packed encoding usable: {dafsa.CanPack32}");
                Console.WriteLine($"built in                     : {build:n0}s");
                Console.WriteLine();

                {
                    var m = new Measurement { Name = "S7  DAFSA, plain arrays (membership)", Group = "Spotter", Objects = 4, Exactness = bad == 0 ? "exact" : $"BROKEN ({bad})", FalsePositiveRate = fp / (double)Math.Max(1, tested), BuildSeconds = build, LookupNsPerQuery = ns };
                    m.Add("state offsets + symbols + targets + final bits", dafsa.BytesPlainMembership);
                    m.Notes = "no exception table needed - the automaton answers the tokenizer's prefix question directly";
                    Results.Add(m);
                }
                if (dafsa.CanPack32)
                {
                    var m = new Measurement { Name = "S8  DAFSA, 32-bit packed transitions (membership)", Group = "Spotter", Objects = 1, Exactness = "exact", LookupNsPerQuery = ns };
                    m.Add("one uint per transition", dafsa.BytesPacked32);
                    m.Notes = "symbol 6 bits + target 24 bits + last + final; valid while transitions < 16.7M";
                    Results.Add(m);
                }
                {
                    var m = new Measurement { Name = "L4  DAFSA + rank counters + dense UID128[]", Group = "LinkedSpotter", Objects = (dafsa.CanPack32 ? 2 : 5) + 1, Exactness = "exact", LookupNsPerQuery = ns };
                    m.Add("automaton", dafsa.CanPack32 ? dafsa.BytesPacked32Ranked : dafsa.BytesPlainRanked);
                    m.Add("dense UID128 values", Sz.Arr(n, 16));
                    m.Notes = "the automaton is a minimal perfect hash, so the UID array is dense and needs no keys";
                    Results.Add(m);
                }
                {
                    var m = new Measurement { Name = "S9  plain trie (no suffix minimisation), for contrast", Group = "Spotter", Objects = 4, Exactness = "exact" };
                    m.Add("state offsets + symbols + targets + final bits",
                          Sz.Arr(dafsa.TrieStates + 1L, 4) + Sz.Arr(dafsa.TrieStates, 1) + Sz.Arr(dafsa.TrieStates, 4) + Sz.Bits(dafsa.TrieStates));
                    m.Notes = $"{dafsa.TrieStates:n0} states vs {dafsa.States:n0} minimised";
                    Results.Add(m);
                }
            }

            MeasureHybrids(blob, sorted, n, nonData, nonOffsets, sortedHashes, distinctHashes, nonMemberHashes, fcd16, fcd64);
        }

        private static Dafsa DAFSA;

        /// <summary>
        /// The exact designs walk a dictionary or an automaton, which is several cache misses. In a real
        /// document almost every token is a miss, so a blocked Bloom filter in front of them turns the
        /// common case back into a single cache line while the combination stays exact.
        /// </summary>
        private static void MeasureHybrids(StringBlob blob, int[] sorted, int n, byte[] nonData, int[] nonOffsets,
                                           ulong[] sortedHashes, int distinctHashes, ulong[] nonMemberHashes,
                                           FrontCodedDictionary fcd16, FrontCodedDictionary fcd64)
        {
            var uniq = new ulong[distinctHashes];
            int u = 0;
            for (int i = 0; i < sortedHashes.Length; i++) { if (i == 0 || sortedHashes[i] != sortedHashes[i - 1]) { uniq[u++] = sortedHashes[i]; } }

            // Realistic token stream: mostly tokens that are not part numbers at all.
            const double MEMBER_SHARE = 0.02;
            int queries    = 2_000_000;
            var qData      = new List<byte>(queries * 12);
            var qOffsets   = new List<int> { 0 };
            var qHashes    = new ulong[queries];
            int nonCount   = nonOffsets.Length - 1;
            var scratchC   = new char[128];

            for (int i = 0; i < queries; i++)
            {
                ReadOnlySpan<byte> s;
                if ((i * 50) % 1000 < MEMBER_SHARE * 1000) { s = blob[sorted[(int)((long)i * 7919 % n)]]; }
                else { s = new ReadOnlySpan<byte>(nonData, nonOffsets[i % nonCount], nonOffsets[i % nonCount + 1] - nonOffsets[i % nonCount]); }

                for (int j = 0; j < s.Length; j++) { qData.Add(s[j]); scratchC[j] = (char)s[j]; }
                qOffsets.Add(qData.Count);
                qHashes[i] = RefHashSet.Mix(Catalyst.Models.Spotter.Hash64(new ReadOnlySpan<char>(scratchC, 0, s.Length)));
            }
            var qBytes = qData.ToArray();
            var qOffs  = qOffsets.ToArray();

            Console.WriteLine("## Mixed token stream (2% of tokens are catalogue entries, 98% are not)");

            var reference = new RefHashSet(uniq, distinctHashes);
            double refNs  = TimeStream(queries, i => reference.Contains(qHashes[i]), out int refHits);
            Console.WriteLine($"{"open-addressed hash set (today's shape)",-58} {refNs,7:n1} ns/token");

            var scratch = new byte[256];
            double fcdNs   = TimeStream(queries, i => fcd64.Find(new ReadOnlySpan<byte>(qBytes, qOffs[i], qOffs[i + 1] - qOffs[i]), scratch) >= 0, out _);
            double dafsaNs = TimeStream(queries, i => DAFSA.Rank(new ReadOnlySpan<byte>(qBytes, qOffs[i], qOffs[i + 1] - qOffs[i])) >= 0, out _);
            Console.WriteLine($"{"front-coded dictionary, block 64, alone",-58} {fcdNs,7:n1} ns/token");
            Console.WriteLine($"{"DAFSA alone",-58} {dafsaNs,7:n1} ns/token");

            foreach (var bits in new[] { 8, 10, 12 })
            {
                var bloom = new BlockedBloom(uniq, distinctHashes, bits);

                int survived = 0;
                for (int i = 0; i < nonMemberHashes.Length; i++) { if (bloom.MayContain(nonMemberHashes[i])) { survived++; } }
                double fp = survived / (double)nonMemberHashes.Length;

                double bloomOnly = TimeStream(queries, i => bloom.MayContain(qHashes[i]), out _);
                double hybridFcd = TimeStream(queries, i => bloom.MayContain(qHashes[i]) && fcd64.Find(new ReadOnlySpan<byte>(qBytes, qOffs[i], qOffs[i + 1] - qOffs[i]), scratch) >= 0, out int h1);
                double hybridDaf = TimeStream(queries, i => bloom.MayContain(qHashes[i]) && DAFSA.Rank(new ReadOnlySpan<byte>(qBytes, qOffs[i], qOffs[i + 1] - qOffs[i])) >= 0, out int h2);

                Console.WriteLine($"  bloom {bits} bits/key: measured fp {fp:P3}, probe {bloomOnly,5:n1} ns, + front-coded {hybridFcd,5:n1} ns, + DAFSA {hybridDaf,5:n1} ns  ({bloom.EstimatedBytes / 1048576.0:n1} MB)");

                if (bits == 10)
                {
                    {
                        var m = new Measurement { Name = "R1  packed DAFSA + 10-bit blocked Bloom prefilter", Group = "Recommended", Objects = 1 + bloom.Objects, Exactness = "exact", LookupNsPerQuery = hybridDaf };
                        m.Add("automaton (32-bit packed transitions)", DAFSA.BytesPacked32);
                        m.Add("prefilter", bloom.EstimatedBytes);
                        m.Notes = "replaces Hashes, MultiGramHashes and the whole tokenizer-exception table";
                        Results.Add(m);
                    }
                    {
                        var m = new Measurement { Name = "R2  front-coded dictionary (block 64) + 10-bit prefilter", Group = "Recommended", Objects = fcd64.Objects + bloom.Objects, Exactness = "exact", LookupNsPerQuery = hybridFcd };
                        m.Add("front-coded strings", fcd64.EstimatedBytes);
                        m.Add("prefilter", bloom.EstimatedBytes);
                        m.Notes = "same, and it can also enumerate / rebuild the stored strings";
                        Results.Add(m);
                    }
                    {
                        var m = new Measurement { Name = "R3  LinkedSpotter: DAFSA rank + dense UID128[] + prefilter", Group = "Recommended", Objects = 2 + 1 + bloom.Objects, Exactness = "exact", LookupNsPerQuery = hybridDaf };
                        m.Add("automaton with rank counters", DAFSA.BytesPacked32Ranked);
                        m.Add("dense UID128 values", Sz.Arr(n, 16));
                        m.Add("prefilter", bloom.EstimatedBytes);
                        Results.Add(m);
                    }
                    {
                        var m = new Measurement { Name = "R4  LinkedSpotter: front-coded rank + dense UID128[] + prefilter", Group = "Recommended", Objects = fcd64.Objects + 1 + bloom.Objects, Exactness = "exact", LookupNsPerQuery = hybridFcd };
                        m.Add("front-coded strings", fcd64.EstimatedBytes);
                        m.Add("dense UID128 values", Sz.Arr(n, 16));
                        m.Add("prefilter", bloom.EstimatedBytes);
                        Results.Add(m);
                    }
                }
            }
            Console.WriteLine();
        }

        private static double TimeStream(int queries, Func<int, bool> probe, out int hits)
        {
            hits = 0;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < queries; i++) { if (probe(i)) { hits++; } }
            return sw.Elapsed.TotalMilliseconds * 1e6 / queries;
        }

        private static void MeasureNgrams(StringBlob blob, int n)
        {
            Console.WriteLine("## n-grams");
            foreach (var k in new[] { 3, 4 })
            {
                var e = NgramAnalysis.EstimateInvertedIndex(blob, k);
                Console.WriteLine($"k={k}: {e.DistinctGrams:n0} distinct grams, {e.TotalPostings:n0} postings ({e.TotalPostings / (double)n:n1} per entry), median rarest-gram posting list {e.MedianMinDocFrequency:n0}");

                var m = new Measurement { Name = $"S10 {k}-gram inverted index (candidate generator only)", Group = "n-gram", Objects = e.Objects, Exactness = "candidates only - still needs the exact keys to verify" };
                m.Add("gram dictionary + offsets + postings (delta+varint)", e.BytesDeltaVarint);
                m.Notes = $"{e.TotalPostings / (double)n:n1} postings/entry; a query still scans ~{e.MedianMinDocFrequency:n0} candidates";
                Results.Add(m);

                var m2 = new Measurement { Name = $"S10b {k}-gram inverted index, fixed 32-bit postings", Group = "n-gram", Objects = e.Objects, Exactness = "candidates only" };
                m2.Add("gram dictionary + offsets + postings (uint32)", e.BytesFixed32);
                Results.Add(m2);
            }

            var c = NgramAnalysis.EstimateGramCoding(blob);
            Console.WriteLine($"gram coding: dictionary {c.DictionarySize:n0} grams -> {c.BytesPerEntry:n2} bytes/entry ({c.BytesPerChar:n2} bytes/char)");
            Console.WriteLine();

            {
                long m2 = BaselineModel.MphSlots(n);
                var m = new Measurement { Name = "S11 n-gram coded strings + perfect-hash index", Group = "n-gram", Objects = 6, Exactness = "exact" };
                m.Add("coded payload", Sz.Arr(c.CodedBytes, 1));
                m.Add("gram dictionary", c.DictionaryBytes);
                m.Add("entry offsets (uint32)", Sz.Arr(n + 1L, 4));
                m.Add("perfect-hash slot -> entry id", Sz.Arr(m2, 4) + BaselineModel.DisplacementBytes(n));
                m.Notes = "the compression works; the separate index is what makes it lose to front-coding";
                Results.Add(m);
            }
        }

        // ----------------------------------------------------------------------------------------------

        private static (double nsPerQuery, double falsePositiveRate) Time(ulong[] members, ulong[] nonMembers, Func<ulong, bool> probe)
        {
            int found = 0;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < members.Length; i++) { if (probe(members[i])) { found++; } }
            double memberNs = sw.Elapsed.TotalMilliseconds * 1e6 / members.Length;

            int fp = 0;
            sw.Restart();
            for (int i = 0; i < nonMembers.Length; i++) { if (probe(nonMembers[i])) { fp++; } }
            double nonNs = sw.Elapsed.TotalMilliseconds * 1e6 / Math.Max(1, nonMembers.Length);

            if (found != members.Length) { Console.WriteLine($"  !! design missed {members.Length - found:n0} members"); }
            return ((memberNs + nonNs) / 2, fp / (double)Math.Max(1, nonMembers.Length));
        }

        private static void Report()
        {
            Console.WriteLine();
            Console.WriteLine("## Results");
            foreach (var group in new[] { "Recommended", "Spotter", "LinkedSpotter", "Both", "n-gram", "Exceptions", "Front-coding sweep" })
            {
                Console.WriteLine();
                Console.WriteLine($"### {group}");
                Console.WriteLine($"{"design",-56} {"total",12} {"B/entry",9} {"objects",9} {"ns/query",9}  {"exactness",-28} notes");
                foreach (var m in Results)
                {
                    if (m.Group != group) { continue; }
                    string fp  = m.FalsePositiveRate >= 0 ? $"{m.Exactness} fp={m.FalsePositiveRate:0.###e+0}" : m.Exactness;
                    string lat = m.LookupNsPerQuery >= 0 ? m.LookupNsPerQuery.ToString("n0") : "-";
                    Console.WriteLine($"{m.Name,-56} {Sz.MB(m.Bytes),12} {Sz.PerKey(m.Bytes, N),9} {m.Objects,9} {lat,9}  {fp,-28} {m.Notes}");
                    foreach (var (part, bytes) in m.Breakdown)
                    {
                        if (m.Breakdown.Count > 1) { Console.WriteLine($"    - {part,-50} {Sz.MB(bytes),12}"); }
                    }
                }
            }
        }
    }
}
