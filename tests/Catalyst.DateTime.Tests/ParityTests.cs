using System;
using System.IO;
using Xunit;

namespace Catalyst.Tests.DateTimeRecognition
{
    /// <summary>
    /// Scores the Catalyst date/time engine against the Microsoft.Recognizers.Text specification suite, and
    /// against the Microsoft implementation itself, so a regression in capability is visible as a number.
    /// </summary>
    public class ParityTests
    {
        private static string ReportPath(string name) => Path.Combine(AppContext.BaseDirectory, "parity", name + ".txt");

        [Theory]
        [InlineData("English")]
        [InlineData("EnglishOthers")]
        [InlineData("German")]
        [InlineData("French")]
        [InlineData("Spanish")]
        [InlineData("Portuguese")]
        [InlineData("Italian")]
        [InlineData("Dutch")]
        public void CatalystIsScoredAgainstTheSpecSuite(string language)
        {
            var catalyst  = ParityReport.Run(language, Engines.RunCatalyst);
            var microsoft = ParityReport.Run(language, Engines.RunMicrosoft);

            ParityReport.WriteReport(ReportPath("catalyst-" + language),  $"Catalyst / {language}",  catalyst);
            ParityReport.WriteReport(ReportPath("microsoft-" + language), $"Microsoft / {language}", microsoft, includeFailures: false);

            Console.WriteLine(ParityReport.Format($"Catalyst / {language}",  catalyst));
            Console.WriteLine(ParityReport.Format($"Microsoft / {language}", microsoft));

            Assert.True(catalyst.Overall.Expected > 0, "the spec suite did not load");
        }
    }
}
