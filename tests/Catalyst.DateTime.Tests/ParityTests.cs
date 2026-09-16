using System;
using System.IO;
using Xunit;

namespace Catalyst.Tests.DateTimeRecognition
{
    /// <summary>
    /// Scores the Catalyst date/time engine against the Microsoft.Recognizers.Text specification suite, and
    /// against the Microsoft implementation itself, so a loss of capability shows up as a number.
    ///
    /// The floors below are what the engine reaches today. They exist to catch a regression, not to describe
    /// a target: raise one whenever the engine beats it, and never lower one to make a change pass.
    /// </summary>
    public class ParityTests
    {
        private static string ReportPath(string name) => Path.Combine(AppContext.BaseDirectory, "parity", name + ".txt");

        [Theory]
        [InlineData("English",       0.92, 0.87)]
        [InlineData("EnglishOthers", 0.95, 0.80)]
        [InlineData("French",        0.59, 0.56)]
        [InlineData("Italian",       0.60, 0.55)]
        [InlineData("Dutch",         0.48, 0.43)]
        [InlineData("German",        0.42, 0.36)]
        [InlineData("Portuguese",    0.39, 0.36)]
        [InlineData("Spanish",       0.42, 0.38)]
        public void CatalystKeepsItsParityWithMicrosoftRecognizersText(string language, double minimumSpanRate, double minimumValueRate)
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

            Assert.True(catalyst.Overall.SpanRate  >= minimumSpanRate,  $"span parity for {language} fell to {catalyst.Overall.SpanRate:P1}, below the {minimumSpanRate:P0} floor (see {ReportPath("catalyst-" + language)})");
            Assert.True(catalyst.Overall.ValueRate >= minimumValueRate, $"resolution parity for {language} fell to {catalyst.Overall.ValueRate:P1}, below the {minimumValueRate:P0} floor (see {ReportPath("catalyst-" + language)})");
        }
    }
}
