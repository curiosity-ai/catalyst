using System;
using System.Collections.Generic;
using System.Diagnostics;
using Catalyst.DateTimeRecognition;
using Mosaik.Core;
using Xunit;

namespace Catalyst.Tests.DateTimeRecognition
{
    /// <summary>
    /// The reason for replacing the regular-expression engine was throughput on ordinary text, where almost
    /// nothing matches and every pattern still has to be tried. This measures both engines on the same corpus.
    /// </summary>
    public class ThroughputTests
    {
        private static string BuildCorpus()
        {
            var lines = new[]
            {
                "The quick brown fox jumps over the lazy dog and nothing here is a date at all.",
                "Please review the attached document and let me know what you think of the approach.",
                "Let's meet next friday at 3pm to go over the numbers one more time.",
                "The contract was signed on 2019-08-01 and runs for three years from that date.",
                "Shipping usually takes 2 weeks, but around christmas it can be a good deal longer.",
                "He said the release is planned for the first quarter of 2021, give or take.",
            };

            var sb = new System.Text.StringBuilder();

            for (int i = 0; i < 400; i++) sb.AppendLine(lines[i % lines.Length]);

            return sb.ToString();
        }

        [Fact]
        public void CatalystIsFasterThanTheEngineItReplaces()
        {
            var corpus    = BuildCorpus();
            var reference = new DateTime(2016, 11, 7);
            var results   = new List<DateTimeEntity>();
            var model     = DateTimeModel.For(Language.English);

            void Catalyst() { results.Clear(); model.Parse(corpus.AsSpan(), reference, results); }
            void Microsoft() { Engines.RunMicrosoft("English", corpus, reference); }

            // Warm up both, so neither pays for JIT or for building its patterns
            Catalyst();
            Microsoft();

            var catalystMs  = Time(Catalyst,  10);
            var microsoftMs = Time(Microsoft, 10);

            var line = $"corpus {corpus.Length:N0} chars, {results.Count} entities — Catalyst {catalystMs:N1} ms, Microsoft.Recognizers.Text {microsoftMs:N1} ms ({microsoftMs / catalystMs:N1}x)";
            Console.WriteLine(line);

            var path = System.IO.Path.Combine(AppContext.BaseDirectory, "parity", "throughput.txt");
            System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(path));
            System.IO.File.WriteAllText(path, line + Environment.NewLine);

            Assert.NotEmpty(results);
            Assert.True(catalystMs < microsoftMs, $"expected the new engine to be the faster one; Catalyst took {catalystMs:N1} ms and Microsoft.Recognizers.Text {microsoftMs:N1} ms");
        }

        private static double Time(Action action, int iterations)
        {
            var sw = Stopwatch.StartNew();

            for (int i = 0; i < iterations; i++) action();

            return sw.Elapsed.TotalMilliseconds / iterations;
        }
    }
}
