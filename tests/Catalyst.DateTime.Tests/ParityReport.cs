using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Catalyst.DateTimeRecognition;
using Mosaik.Core;

namespace Catalyst.Tests.DateTimeRecognition
{
    public sealed class ParityStats
    {
        public int SpanMatches     { get; set; }
        public int ValueMatches    { get; set; }
        /// <summary>Same reading, allowing a span that differs from the expected one only by glue.</summary>
        public int ReadingMatches  { get; set; }
        public int Expected        { get; set; }
        public int Produced        { get; set; }

        public double SpanRate    => Expected == 0 ? 1 : (double)SpanMatches    / Expected;
        public double ValueRate   => Expected == 0 ? 1 : (double)ValueMatches   / Expected;
        public double ReadingRate => Expected == 0 ? 1 : (double)ReadingMatches / Expected;
    }

    public sealed class ParityResult
    {
        public ParityStats                      Overall  { get; } = new ParityStats();
        public Dictionary<string, ParityStats>  ByType   { get; } = new Dictionary<string, ParityStats>();
        public List<string>                     Failures { get; } = new List<string>();
        public int                              Cases    { get; set; }
        public int                              PerfectCases { get; set; }
        /// <summary>Readings the engine got right, bounded one function word differently.</summary>
        public int                              GlueOnly { get; set; }

        public ParityStats For(string type)
        {
            if (!ByType.TryGetValue(type, out var s))
            {
                s = new ParityStats();
                ByType[type] = s;
            }

            return s;
        }
    }

    /// <summary>
    /// Runs one of the two engines over a language's spec suite and scores it: how often it finds the same
    /// span and type, and how often the resolution matches field for field.
    /// </summary>
    public static class ParityReport
    {
        public delegate List<Hit> Engine(string language, string input, DateTime reference);

        public static ParityResult Run(string language, Engine engine, int maxFailuresRecorded = 3000)
        {
            var result = new ParityResult();
            var cases  = SpecLoader.Load(language);
            var glue   = GlueOf(language);

            foreach (var c in cases)
            {
                result.Cases++;

                var expected = Engines.FromSpec(c);
                List<Hit> actual;

                try
                {
                    actual = engine(language, c.Input, c.ReferenceDateTime);
                }
                catch (Exception e)
                {
                    actual = new List<Hit>();
                    if (result.Failures.Count < maxFailuresRecorded) result.Failures.Add($"THREW  {c.Input}\n         {e.GetType().Name}: {e.Message}");
                }

                result.Overall.Produced += actual.Count;
                bool perfect = expected.Count == actual.Count;

                foreach (var want in expected)
                {
                    var stats = result.For(want.Type);
                    stats.Expected++;
                    result.Overall.Expected++;

                    var got = actual.FirstOrDefault(a => a.Start == want.Start && a.End == want.End && a.Type == want.Type)
                           ?? actual.FirstOrDefault(a => a.Start == want.Start && a.Type == want.Type)
                           ?? actual.FirstOrDefault(a => Overlaps(a, want) && a.Type == want.Type);

                    if (got is null)
                    {
                        perfect = false;
                        if (result.Failures.Count < maxFailuresRecorded)
                        {
                            var near = actual.FirstOrDefault(a => Overlaps(a, want));
                            result.Failures.Add($"MISS   \"{c.Input}\"\n         want {want.Describe()}\n         got  {(near is null ? "(nothing)" : near.Describe())}");
                        }
                        continue;
                    }

                    bool exactSpan  = got.Start == want.Start && got.End == want.End;
                    bool sameValues = ValuesMatch(want.Values, got.Values);
                    bool glueOnly   = !exactSpan && sameValues && DiffersOnlyByGlue(c.Input, want, got, glue);

                    if (exactSpan)
                    {
                        stats.SpanMatches++;
                        result.Overall.SpanMatches++;
                    }

                    if (exactSpan && sameValues)
                    {
                        stats.ValueMatches++;
                        result.Overall.ValueMatches++;
                    }

                    if ((exactSpan || glueOnly) && sameValues)
                    {
                        stats.ReadingMatches++;
                        result.Overall.ReadingMatches++;
                    }

                    if (glueOnly)
                    {
                        // The same reading, bounded one function word differently. Recorded so it can be read,
                        // not counted as a difference — see DiffersOnlyByGlue.
                        result.GlueOnly++;
                        if (result.Failures.Count < maxFailuresRecorded)
                        {
                            result.Failures.Add($"GLUE   \"{c.Input}\"\n         want {want.Describe()}\n         got  {got.Describe()}");
                        }
                        continue;
                    }

                    if (!exactSpan)
                    {
                        perfect = false;
                        if (result.Failures.Count < maxFailuresRecorded)
                        {
                            result.Failures.Add($"SPAN   \"{c.Input}\"\n         want {want.Describe()}\n         got  {got.Describe()}");
                        }
                    }
                    else if (!sameValues)
                    {
                        perfect = false;
                        if (result.Failures.Count < maxFailuresRecorded)
                        {
                            result.Failures.Add($"VALUE  \"{c.Input}\"\n         want {want.Describe()}\n         got  {got.Describe()}");
                        }
                    }
                }

                if (perfect) result.PerfectCases++;
            }

            return result;
        }

        private static bool Overlaps(Hit a, Hit b) => a.Start <= b.End && b.Start <= a.End;

        private static readonly Dictionary<string, Lexicon> _glue = new Dictionary<string, Lexicon>();

        private static Lexicon GlueOf(string language)
        {
            lock (_glue)
            {
                if (!_glue.TryGetValue(language, out var lexicon))
                {
                    lexicon = language switch
                    {
                        "English"       => Lexicons.For(Language.English, useUsEnglishForEnglish: true),
                        "EnglishOthers" => Lexicons.For(Language.English, useUsEnglishForEnglish: false),
                        _               => Lexicons.For(Enum.Parse<Language>(language, ignoreCase: true)),
                    };

                    _glue[language] = lexicon;
                }

                return lexicon;
            }
        }

        /// <summary>
        /// Whether two spans over the same input differ only by glue, and so say the same thing.
        ///
        /// The reference implementation is not consistent about where a function word belongs: it reports
        /// "am Wochenende" with its preposition and "am Freitag" without, "de las 5 a las 6" with its article
        /// and "5 de la tarde" without, "in 2014 through 2018" with its "in" and "in two days from today"
        /// without. A caller reads the resolution, not the offsets, so a boundary that falls either side of
        /// an article, a preposition or a comma is not a capability difference and is not scored as one.
        ///
        /// The bar is deliberately narrow. This is only reached when the two readings already match field for
        /// field, and every word in the disagreement has to be one the language's own lexicon classes as glue.
        /// A content word — a number, a unit, a part of the day — still counts as a miss even when the
        /// resolution happens to survive it.
        /// </summary>
        private static bool DiffersOnlyByGlue(string input, Hit want, Hit got, Lexicon lexicon)
        {
            int outerStart = Math.Min(want.Start, got.Start);
            int innerStart = Math.Max(want.Start, got.Start);
            int innerEnd   = Math.Min(want.End,   got.End);
            int outerEnd   = Math.Max(want.End,   got.End);

            if (innerStart > innerEnd) return false;   // the two do not overlap at all

            return IsAllGlue(input.AsSpan(outerStart, innerStart - outerStart), lexicon)
                && IsAllGlue(input.AsSpan(innerEnd + 1, outerEnd - innerEnd), lexicon);
        }

        private static bool IsAllGlue(ReadOnlySpan<char> text, Lexicon lexicon)
        {
            int i = 0;

            while (i < text.Length)
            {
                if (!char.IsLetterOrDigit(text[i])) { i++; continue; }   // spaces and punctuation carry nothing

                int start = i;
                while (i < text.Length && (char.IsLetterOrDigit(text[i]) || text[i] == '\'')) i++;

                if (!lexicon.TryGetWord(text[start..i], out var term)) return false;
                if (!IsGlue(term)) return false;
            }

            return true;
        }

        private static bool IsGlue(TermInfo term)
            => term.Is(TermKind.Filler) || term.Is(TermKind.Article)     || term.Is(TermKind.Connector)
            || term.Is(TermKind.RangeStart) || term.Is(TermKind.ClockPrefix) || term.Is(TermKind.InPrefix)
            || term.Is(TermKind.OrdinalSuffix);

        /// <summary>The first expected reading has to be produced with the same timex, type and bounds.</summary>
        private static bool ValuesMatch(List<Dictionary<string, string>> want, List<Dictionary<string, string>> got)
        {
            if (want.Count == 0) return got.Count == 0;
            if (got.Count == 0)  return false;

            foreach (var w in want)
            {
                if (!got.Any(g => Same(w, g))) return false;
            }

            return true;
        }

        private static bool Same(Dictionary<string, string> want, Dictionary<string, string> got)
        {
            foreach (var key in new[] { "timex", "type", "value", "start", "end", "Mod" })
            {
                want.TryGetValue(key, out var w);
                got.TryGetValue(key, out var g);

                if (!string.Equals(w, g, StringComparison.Ordinal)) return false;
            }

            return true;
        }

        public static string Format(string title, ParityResult r)
        {
            var sb = new StringBuilder();

            sb.AppendLine($"=== {title} ===");
            sb.AppendLine($"  cases {r.Cases}, expected entities {r.Overall.Expected}, produced {r.Overall.Produced}");
            sb.AppendLine($"  same reading    : {r.Overall.ReadingMatches,5} / {r.Overall.Expected,-5} = {r.Overall.ReadingRate,7:P1}   ({r.GlueOnly} of them bounded differently)");
            sb.AppendLine($"  span+type match : {r.Overall.SpanMatches,5} / {r.Overall.Expected,-5} = {r.Overall.SpanRate,7:P1}");
            sb.AppendLine($"  full resolution : {r.Overall.ValueMatches,5} / {r.Overall.Expected,-5} = {r.Overall.ValueRate,7:P1}");
            sb.AppendLine();

            foreach (var kv in r.ByType.OrderByDescending(k => k.Value.Expected))
            {
                sb.AppendLine($"    {kv.Key,-28} reading {kv.Value.ReadingRate,7:P1}  span {kv.Value.SpanRate,7:P1}  value {kv.Value.ValueRate,7:P1}   (n={kv.Value.Expected})");
            }

            return sb.ToString();
        }

        public static void WriteReport(string path, string title, ParityResult r, bool includeFailures = true)
        {
            var sb = new StringBuilder();
            sb.Append(Format(title, r));

            if (includeFailures)
            {
                sb.AppendLine();
                sb.AppendLine("--- differences ---");
                foreach (var f in r.Failures) sb.AppendLine(f);
            }

            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.WriteAllText(path, sb.ToString());
        }
    }
}
