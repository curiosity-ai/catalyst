using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace Catalyst.Tests.DateTimeRecognition
{
    public sealed class ParityStats
    {
        public int SpanMatches     { get; set; }
        public int ValueMatches    { get; set; }
        public int Expected        { get; set; }
        public int Produced        { get; set; }

        public double SpanRate  => Expected == 0 ? 1 : (double)SpanMatches  / Expected;
        public double ValueRate => Expected == 0 ? 1 : (double)ValueMatches / Expected;
    }

    public sealed class ParityResult
    {
        public ParityStats                      Overall  { get; } = new ParityStats();
        public Dictionary<string, ParityStats>  ByType   { get; } = new Dictionary<string, ParityStats>();
        public List<string>                     Failures { get; } = new List<string>();
        public int                              Cases    { get; set; }
        public int                              PerfectCases { get; set; }

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

                    bool exactSpan = got.Start == want.Start && got.End == want.End;

                    if (exactSpan)
                    {
                        stats.SpanMatches++;
                        result.Overall.SpanMatches++;
                    }
                    else
                    {
                        perfect = false;
                        if (result.Failures.Count < maxFailuresRecorded)
                        {
                            result.Failures.Add($"SPAN   \"{c.Input}\"\n         want {want.Describe()}\n         got  {got.Describe()}");
                        }
                    }

                    if (exactSpan && ValuesMatch(want.Values, got.Values))
                    {
                        stats.ValueMatches++;
                        result.Overall.ValueMatches++;
                    }
                    else if (exactSpan)
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
            sb.AppendLine($"  span+type match : {r.Overall.SpanMatches,5} / {r.Overall.Expected,-5} = {r.Overall.SpanRate,7:P1}");
            sb.AppendLine($"  full resolution : {r.Overall.ValueMatches,5} / {r.Overall.Expected,-5} = {r.Overall.ValueRate,7:P1}");
            sb.AppendLine();

            foreach (var kv in r.ByType.OrderByDescending(k => k.Value.Expected))
            {
                sb.AppendLine($"    {kv.Key,-28} span {kv.Value.SpanRate,7:P1}  value {kv.Value.ValueRate,7:P1}   (n={kv.Value.Expected})");
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
