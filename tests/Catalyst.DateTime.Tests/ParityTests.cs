using System;
using System.IO;
using Xunit;

namespace Catalyst.Tests.DateTimeRecognition
{
    /// <summary>
    /// Scores the Catalyst date/time engine against the Microsoft.Recognizers.Text specification suite, and
    /// against the Microsoft implementation itself, so a loss of capability shows up as a number.
    ///
    /// Three rates per language, because a span disagreement and a wrong answer are not the same thing.
    /// "Same reading" is the one that says whether the engine understood the text: the resolution matches
    /// field for field, and the span is either identical or differs only by glue — an article, a preposition,
    /// a comma — which the reference implementation itself is not consistent about. The two strict rates are
    /// reported beside it so nothing is hidden by the tolerance.
    ///
    /// The floors below are what the engine reaches today. They exist to catch a regression, not to describe
    /// a target: raise one whenever the engine beats it, and never lower one to make a change pass.
    /// </summary>
    public class ParityTests
    {
        private static string ReportPath(string name) => Path.Combine(AppContext.BaseDirectory, "parity", name + ".txt");

        [Theory]
        //                       reading  span  value
        [InlineData("English",       0.93, 0.95, 0.92)]
        [InlineData("EnglishOthers", 0.93, 0.95, 0.92)]
        [InlineData("Italian",       0.80, 0.81, 0.77)]
        [InlineData("Spanish",       0.78, 0.74, 0.71)]
        [InlineData("French",        0.75, 0.75, 0.71)]
        [InlineData("Dutch",         0.74, 0.72, 0.67)]
        [InlineData("German",        0.68, 0.74, 0.66)]
        [InlineData("Portuguese",    0.68, 0.68, 0.66)]
        public void CatalystKeepsItsParityWithMicrosoftRecognizersText(string language, double minimumReadingRate, double minimumSpanRate, double minimumValueRate)
        {
            var catalyst  = ParityReport.Run(language, Engines.RunCatalyst);
            var microsoft = ParityReport.Run(language, Engines.RunMicrosoft);

            ParityReport.WriteReport(ReportPath("catalyst-" + language),  $"Catalyst / {language}",  catalyst);
            ParityReport.WriteReport(ReportPath("microsoft-" + language), $"Microsoft / {language}", microsoft, includeFailures: false);

            Console.WriteLine(ParityReport.Format($"Catalyst / {language}",  catalyst));
            Console.WriteLine(ParityReport.Format($"Microsoft / {language}", microsoft));

            Assert.True(catalyst.Overall.Expected > 0, "the spec suite did not load");

            // The suite is Microsoft's own regression set, so its score is the ceiling this is measured against
            Assert.True(microsoft.Overall.SpanRate > 0.98, $"the reference implementation scored {microsoft.Overall.SpanRate:P1}, so the comparison is not measuring what it should");

            Assert.True(catalyst.Overall.ReadingRate >= minimumReadingRate, $"reading parity for {language} fell to {catalyst.Overall.ReadingRate:P1}, below the {minimumReadingRate:P0} floor (see {ReportPath("catalyst-" + language)})");
            Assert.True(catalyst.Overall.SpanRate    >= minimumSpanRate,    $"span parity for {language} fell to {catalyst.Overall.SpanRate:P1}, below the {minimumSpanRate:P0} floor (see {ReportPath("catalyst-" + language)})");
            Assert.True(catalyst.Overall.ValueRate   >= minimumValueRate,   $"resolution parity for {language} fell to {catalyst.Overall.ValueRate:P1}, below the {minimumValueRate:P0} floor (see {ReportPath("catalyst-" + language)})");
        }
    }
}
