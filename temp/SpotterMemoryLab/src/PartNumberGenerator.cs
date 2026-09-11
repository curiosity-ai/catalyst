using System;
using System.Collections.Generic;
using System.Text;

namespace SpotterMemoryLab
{
    /// <summary>
    /// Generates a synthetic but structurally realistic aviation part-number catalogue.
    ///
    /// The shape that matters for every design measured here is that real catalogues are *families*: a base
    /// standard or drawing number with a set of dash numbers hanging off it (NAS1149F03 -> -32P, -63P, ...),
    /// a long tail of one-off OEM numbers, and a small shared vocabulary of suffixes. Each family draws its
    /// suffixes as a random subset of a shared pool, so suffix languages differ between families - that is
    /// deliberately unfavourable to the automaton designs rather than favourable.
    /// </summary>
    internal sealed class PartNumberGenerator
    {
        private ulong _state;

        public PartNumberGenerator(ulong seed) { _state = seed == 0 ? 0x9E3779B97F4A7C15UL : seed; }

        private ulong Next()
        {
            _state ^= _state << 13;
            _state ^= _state >> 7;
            _state ^= _state << 17;
            return _state;
        }

        private int Next(int maxExclusive) => (int)(Next() % (ulong)maxExclusive);
        private int Next(int min, int maxExclusive) => min + Next(maxExclusive - min);
        private double NextDouble() => (Next() >> 11) * (1.0 / (1UL << 53));

        private const string LETTERS = "ABCDEFGHJKLMNPRSTUVWXYZ";           // aerospace codes skip I, O, Q
        private const string DIGITS  = "0123456789";
        private static readonly string[] STD_PREFIX   = { "AN", "MS", "NAS", "NASM", "NA", "NSA", "ASNA", "LN", "EN", "DIN", "ISO", "AS", "MIL" };
        private static readonly string[] BAC_FAMILY   = { "B", "R", "N", "C", "P", "D", "S", "J" };
        private static readonly string[] OEM_PREFIX   = { "145", "120", "190", "170", "69", "10", "65", "251", "273", "312", "413", "776", "3214", "5001", "7712" };
        private static readonly string[] MATERIALS    = { "7075", "2024", "6061", "5052", "4130", "17-4PH", "15-5PH", "321", "347", "718", "625", "6AL-4V" };
        private static readonly string[] TEMPERS      = { "T6", "T3", "T351", "T73", "T7351", "O", "H14", "H32", "A", "STA", "AMS" };
        private static readonly string[] BRAND_PREFIX = { "HL", "HST", "CR", "MBF", "SL", "NAS", "TL", "LGP", "RV" };

        public enum Kind { StdHardware, BoeingAlpha, BoeingNumeric, AirbusNumeric, MilConnector, FastenerBrand, OemSequential, Bearing, MaterialSpec, VendorFreeform }

        private static readonly (Kind kind, double weight)[] MIX =
        {
            (Kind.StdHardware,    0.30),
            (Kind.BoeingAlpha,    0.08),
            (Kind.BoeingNumeric,  0.10),
            (Kind.AirbusNumeric,  0.07),
            (Kind.MilConnector,   0.06),
            (Kind.FastenerBrand,  0.09),
            (Kind.OemSequential,  0.20),
            (Kind.Bearing,        0.04),
            (Kind.MaterialSpec,   0.03),
            (Kind.VendorFreeform, 0.03),
        };

        // Shared suffix pools. A family picks one pool and a random subset of it, which is what real
        // stocking lists look like (a family is stocked in some of the available dash numbers, not all).
        private readonly List<string[]> _pools = new();

        private void BuildPools()
        {
            var grip = new List<string>();
            for (int i = 2; i <= 80; i++) { grip.Add("-" + i); }
            _pools.Add(grip.ToArray());

            var dash2 = new List<string>();
            for (int i = 1; i <= 99; i++) { dash2.Add("-" + i.ToString("00")); }
            _pools.Add(dash2.ToArray());

            var dash3 = new List<string>();
            for (int i = 1; i <= 999; i += 1) { dash3.Add("-" + i.ToString("000")); }
            _pools.Add(dash3.ToArray());

            var dash4 = new List<string>();
            for (int i = 1; i <= 2000; i++) { dash4.Add("-" + i); }
            _pools.Add(dash4.ToArray());

            var letters = new List<string>();
            foreach (var c in LETTERS) { letters.Add(c.ToString()); }
            foreach (var c in LETTERS) { for (int i = 1; i <= 32; i++) { letters.Add(c.ToString() + i); } }
            _pools.Add(letters.ToArray());

            var diaGrip = new List<string>();
            for (int d = 3; d <= 20; d++) { for (int g = 2; g <= 40; g++) { diaGrip.Add("-" + d + "-" + g); } }
            _pools.Add(diaGrip.ToArray());

            var revs = new List<string>();
            for (int i = 1; i <= 40; i++) { revs.Add("-" + i); }
            for (int i = 0; i < LETTERS.Length; i++) { revs.Add("/" + LETTERS[i]); }
            _pools.Add(revs.ToArray());
        }

        /// <summary>
        /// When true, every family is tiny - the pessimistic dataset that removes almost all of the
        /// prefix/suffix structure a real catalogue has, used to bracket the structure-dependent designs.
        /// </summary>
        public bool LowStructure { get; set; }

        // Family-size distribution, per kind: standards hardware really is stocked in hundreds of dash
        // numbers off one base number, while an OEM drawing number usually has a handful of revisions and
        // a vendor's own number has none.
        private int SampleFamilySize(Kind kind)
        {
            if (LowStructure) { return NextDouble() < 0.8 ? 1 : Next(2, 4); }

            double r = NextDouble();
            switch (kind)
            {
                case Kind.StdHardware:
                case Kind.FastenerBrand:
                case Kind.MilConnector:
                case Kind.AirbusNumeric:
                    if (r < 0.30) { return 1; }
                    if (r < 0.55) { return Next(2, 6); }
                    if (r < 0.80) { return Next(6, 31); }
                    if (r < 0.94) { return Next(31, 151); }
                    if (r < 0.99) { return Next(151, 701); }
                    return Next(701, 3001);

                case Kind.BoeingAlpha:
                case Kind.BoeingNumeric:
                case Kind.Bearing:
                case Kind.MaterialSpec:
                    if (r < 0.45) { return 1; }
                    if (r < 0.75) { return Next(2, 6); }
                    if (r < 0.93) { return Next(6, 26); }
                    if (r < 0.99) { return Next(26, 121); }
                    return Next(121, 501);

                case Kind.OemSequential:
                    if (r < 0.62) { return 1; }
                    if (r < 0.90) { return Next(2, 5); }
                    if (r < 0.99) { return Next(5, 13); }
                    return Next(13, 41);

                default:
                    return 1;
            }
        }

        private string Pick(string[] a) => a[Next(a.Length)];
        private char PickC(string s) => s[Next(s.Length)];

        private string Digits(int n)
        {
            var sb = new StringBuilder(n);
            for (int i = 0; i < n; i++) { sb.Append(PickC(DIGITS)); }
            return sb.ToString();
        }

        private string Alpha(int n)
        {
            var sb = new StringBuilder(n);
            for (int i = 0; i < n; i++) { sb.Append(PickC(LETTERS)); }
            return sb.ToString();
        }

        private string MakeStem(Kind kind)
        {
            switch (kind)
            {
                case Kind.StdHardware:
                {
                    var sb = new StringBuilder();
                    sb.Append(Pick(STD_PREFIX));
                    sb.Append(Digits(Next(3, 6)));
                    if (NextDouble() < 0.45) { sb.Append(Alpha(Next(1, 3))); }
                    if (NextDouble() < 0.30) { sb.Append(Digits(Next(2, 5))); }
                    return sb.ToString();
                }
                case Kind.BoeingAlpha:
                {
                    var sb = new StringBuilder("BAC");
                    sb.Append(Pick(BAC_FAMILY));
                    sb.Append(Digits(2));
                    sb.Append(Alpha(Next(1, 4)));
                    if (NextDouble() < 0.6) { sb.Append(Digits(1)); }
                    return sb.ToString();
                }
                case Kind.BoeingNumeric:
                {
                    var sb = new StringBuilder();
                    sb.Append(Pick(OEM_PREFIX));
                    if (NextDouble() < 0.15) { sb.Append(Alpha(1)); }
                    sb.Append('-');
                    sb.Append(Digits(5));
                    return sb.ToString();
                }
                case Kind.AirbusNumeric:
                {
                    var sb = new StringBuilder();
                    sb.Append(NextDouble() < 0.5 ? Pick(new[] { "ASNA", "NSA", "ABS", "ABM" }) : Pick(new[] { "LN", "EN", "NSA" }));
                    sb.Append(Digits(Next(4, 6)));
                    return sb.ToString();
                }
                case Kind.MilConnector:
                {
                    if (NextDouble() < 0.5)
                    {
                        return "M" + Digits(5) + "/" + Digits(Next(1, 3));
                    }
                    return "D38999/" + Digits(2) + Alpha(Next(1, 3));
                }
                case Kind.FastenerBrand:
                {
                    var sb = new StringBuilder();
                    sb.Append(Pick(BRAND_PREFIX));
                    sb.Append(Digits(Next(2, 5)));
                    if (NextDouble() < 0.7) { sb.Append(Alpha(Next(1, 3))); }
                    return sb.ToString();
                }
                case Kind.OemSequential:
                {
                    double r = NextDouble();
                    if (r < 0.4) { return Digits(Next(6, 9)); }
                    if (r < 0.7) { return Pick(OEM_PREFIX) + "-" + Digits(Next(4, 6)); }
                    return Alpha(2) + Digits(Next(4, 7));
                }
                case Kind.Bearing:
                {
                    var sb = new StringBuilder();
                    sb.Append(Pick(new[] { "KP", "SR", "MS141", "S16", "GE", "MB", "SB", "RBC", "KSP", "COM" }));
                    sb.Append(Digits(Next(2, 5)));
                    if (NextDouble() < 0.5) { sb.Append(Alpha(1)); }
                    return sb.ToString();
                }
                case Kind.MaterialSpec:
                {
                    double r = NextDouble();
                    if (r < 0.4) { return Pick(MATERIALS) + "-" + Pick(TEMPERS); }
                    if (r < 0.7) { return "AMS" + Digits(4); }
                    return "BMS" + Digits(Next(1, 3)) + "-" + Digits(Next(2, 4));
                }
                default:
                {
                    var sb = new StringBuilder();
                    int n = Next(6, 11);
                    for (int i = 0; i < n; i++) { sb.Append(NextDouble() < 0.5 ? PickC(LETTERS) : PickC(DIGITS)); }
                    return sb.ToString();
                }
            }
        }

        /// <summary>Generates <paramref name="target"/> distinct part numbers into a blob.</summary>
        public StringBlob Generate(int target, double multiTokenShare = 0.015)
        {
            BuildPools();

            var blob      = new StringBlob(target + 1024, (long)target * 14);
            var stemSeen  = new HashSet<string>(1 << 20);
            var buffer    = new byte[64];
            var chosen    = new List<int>();

            int emitted = 0;
            foreach (var (kind, weight) in MIX)
            {
                int kindTarget = (int)(target * weight);
                int kindDone   = 0;
                int guard      = 0;

                while (kindDone < kindTarget && guard < kindTarget * 20 + 1000)
                {
                    guard++;
                    var stem = MakeStem(kind);
                    if (!stemSeen.Add(stem)) { continue; }

                    int  size = Math.Min(SampleFamilySize(kind), kindTarget - kindDone);
                    var  pool = _pools[Next(_pools.Count)];

                    if (size == 1)
                    {
                        Write(blob, buffer, stem, multiTokenShare);
                        kindDone++; emitted++;
                        continue;
                    }

                    // Sample `size` distinct suffixes from the pool without replacement.
                    chosen.Clear();
                    if (size >= pool.Length)
                    {
                        for (int i = 0; i < pool.Length; i++) { chosen.Add(i); }
                    }
                    else
                    {
                        var taken = new HashSet<int>(size);
                        while (taken.Count < size)
                        {
                            int c = Next(pool.Length);
                            if (taken.Add(c)) { chosen.Add(c); }
                        }
                    }

                    foreach (var c in chosen)
                    {
                        Write(blob, buffer, stem + pool[c], multiTokenShare);
                        kindDone++; emitted++;
                    }
                }
            }

            // Top up with free-form values if any kind ran out of room.
            while (emitted < target)
            {
                var s = MakeStem(Kind.VendorFreeform);
                if (!stemSeen.Add(s)) { continue; }
                Write(blob, buffer, s, 0);
                emitted++;
            }

            blob.Trim();
            return blob;
        }

        // A small share of catalogue values really do carry a space (a standard written out, a size callout).
        // Keeping some of them exercises the multi-gram path of every design instead of only the single-token one.
        private void Write(StringBlob blob, byte[] buffer, string value, double multiTokenShare)
        {
            if (multiTokenShare > 0 && NextDouble() < multiTokenShare && value.Length > 6)
            {
                int at = Next(3, value.Length - 2);
                value  = value.Substring(0, at) + " " + value.Substring(at);
            }

            int n = Math.Min(value.Length, buffer.Length);
            for (int i = 0; i < n; i++)
            {
                char c    = value[i];
                buffer[i] = (byte)(c < 128 ? c : '?');
            }
            blob.Add(buffer.AsSpan(0, n));
        }
    }
}
