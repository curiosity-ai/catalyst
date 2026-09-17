using System;
using System.Collections.Generic;
using Catalyst.DateTimeRecognition;
using Mosaik.Core;
using Xunit;

namespace Catalyst.Tests.DateTimeRecognition
{
    /// <summary>
    /// The scan itself is meant to allocate nothing: the lexeme and node buffers are rented, the lexicon is
    /// probed by span, and only a hit allocates its resolution. These tests measure that rather than assume it.
    /// </summary>
    public class AllocationTests
    {
        private const string Prose =
            "The quick brown fox jumps over the lazy dog, and then it does so again, because that is what " +
            "the sentence is for. Nothing in here names a moment in time, so the scanner should walk it and " +
            "come back with nothing at all to show for the walk.";

        private static long Measure(Action action, int iterations)
        {
            // Warm up, so JIT and lexicon construction are not what gets measured
            for (int i = 0; i < 50; i++) action();

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            long before = GC.GetAllocatedBytesForCurrentThread();

            for (int i = 0; i < iterations; i++) action();

            return (GC.GetAllocatedBytesForCurrentThread() - before) / iterations;
        }

        [Fact]
        public void ScanningProseThatHoldsNoDateAllocatesOnlyTheEmptyResultList()
        {
            var model   = DateTimeModel.For(Language.English);
            var results = new List<DateTimeEntity>();

            long perCall = Measure(() =>
            {
                results.Clear();
                model.Parse(Prose.AsSpan(), new DateTime(2016, 11, 7), results);
            }, iterations: 2000);

            Assert.Empty(results);
            Assert.True(perCall == 0, $"expected an allocation-free scan, measured {perCall} bytes per call");
        }

        [Fact]
        public void ScanningAllocatesNothingBeyondWhatAHitHasToReturn()
        {
            var model   = DateTimeModel.For(Language.English);
            var results = new List<DateTimeEntity>();
            var text    = "Let's meet next friday at 3pm.";

            long perCall = Measure(() =>
            {
                results.Clear();
                model.Parse(text.AsSpan(), new DateTime(2016, 11, 7), results);
            }, iterations: 2000);

            Assert.Single(results);

            // A hit has to produce its text, its resolution objects and their strings; the scan around it must not
            // add anything of that order again.
            Assert.True(perCall < 1024, $"expected a hit to cost under 1 KB, measured {perCall} bytes per call");
        }

        [Fact]
        public void TheLexiconIsProbedWithoutMaterialisingWords()
        {
            var lexicon = EnglishLexicon.Get(dayMonthOrder: false);

            long perCall = Measure(() =>
            {
                lexicon.TryGetWord("september".AsSpan(), out _);
                lexicon.TryGetWord("notaword".AsSpan(), out _);
            }, iterations: 20000);

            Assert.Equal(0, perCall);
        }
    }
}
