using System;
using System.Linq;
using Catalyst.DateTimeRecognition;
using Mosaik.Core;
using Xunit;

namespace Catalyst.Tests.DateTimeRecognition
{
    /// <summary>
    /// Behaviour the engine has to keep regardless of how the parity score moves: the shapes Mosaik reads,
    /// the languages it answers for, and the promise that ordinary prose produces nothing.
    /// </summary>
    public class DateTimeModelTests
    {
        private static readonly DateTime Reference = new DateTime(2016, 11, 7);

        private static DateTimeEntity One(string text, Language language = Language.English)
        {
            var results = DateTimeModel.For(language).Parse(text, Reference);
            Assert.True(results.Count > 0, $"nothing recognised in \"{text}\"");
            return results[0];
        }

        [Theory]
        [InlineData("I'll go back 2019-08-01",      "datetimeV2.date",      "2019-08-01", "2019-08-01")]
        [InlineData("I'll go back tomorrow",        "datetimeV2.date",      "2016-11-08", "2016-11-08")]
        [InlineData("I'll go back january 18, 2019","datetimeV2.date",      "2019-01-18", "2019-01-18")]
        [InlineData("I'll be back 7:56:30 pm",      "datetimeV2.time",      "T19:56:30",  "19:56:30")]
        [InlineData("I'll leave for 3 hours",       "datetimeV2.duration",  "PT3H",       "10800")]
        public void ResolvesTheCommonShapes(string input, string type, string timex, string value)
        {
            var hit = One(input);

            Assert.Equal(type,  hit.TypeName);
            Assert.Equal(timex, hit.Values[0].Timex);
            Assert.Equal(value, hit.Values[0].Value);
        }

        [Fact]
        public void ResolvesAPeriodWithBothEnds()
        {
            var hit = One("I'll be out next week");

            Assert.Equal("datetimeV2.daterange", hit.TypeName);
            Assert.Equal("2016-11-14", hit.Values[0].Start);
            Assert.Equal("2016-11-21", hit.Values[0].End);
        }

        [Fact]
        public void ReportsTheSpanOfTheExpressionOnly()
        {
            var hit = One("I'll go back january 18, 2019 to see them");

            Assert.Equal("january 18, 2019", hit.Text);
            Assert.Equal(13, hit.Start);
            Assert.Equal(28, hit.End);
        }

        [Fact]
        public void ProducesTheMetadataDictionaryMosaikReads()
        {
            var metadata = One("I'll go back tomorrow").Values[0].ToDictionary();

            Assert.Equal("2016-11-08", metadata["timex"]);
            Assert.Equal("date",       metadata["type"]);
            Assert.Equal("2016-11-08", metadata["value"]);
        }

        [Theory]
        [InlineData(Language.English,    "I'll go back tomorrow")]
        [InlineData(Language.German,     "Ich komme morgen zurück")]
        [InlineData(Language.French,     "Je reviendrai demain")]
        [InlineData(Language.Spanish,    "Volveré mañana")]
        [InlineData(Language.Portuguese, "Eu volto amanhã")]
        [InlineData(Language.Italian,    "Tornerò domani")]
        [InlineData(Language.Dutch,      "Ik kom morgen terug")]
        public void EveryDeclaredLanguageResolvesItsOwnWordForTomorrow(Language language, string input)
        {
            var hit = One(input, language);

            Assert.Equal("datetimeV2.date", hit.TypeName);
            Assert.Equal("2016-11-08",      hit.Values[0].Value);
        }

        [Fact]
        public void SupportedLanguagesAllBuild()
        {
            foreach (var language in new[] { Language.English, Language.German, Language.French, Language.Spanish, Language.Portuguese, Language.Italian, Language.Dutch })
            {
                Assert.True(Lexicons.IsSupported(language));
                Assert.NotNull(Lexicons.For(language));
            }
        }

        [Theory]
        [InlineData("The quick brown fox jumps over the lazy dog.")]
        [InlineData("Please review the attached document and let me know.")]
        [InlineData("")]
        public void OrdinaryProseProducesNothing(string input)
        {
            Assert.Empty(DateTimeModel.For(Language.English).Parse(input, Reference));
        }

        [Fact]
        public void ScanningIsRepeatableAcrossCalls()
        {
            var model = DateTimeModel.For(Language.English);
            var text  = "Let's meet next friday at 3pm, or on 2019-08-01 instead.";

            var first  = model.Parse(text, Reference).Select(e => e.Text + "|" + e.Values[0].Timex).ToArray();
            var second = model.Parse(text, Reference).Select(e => e.Text + "|" + e.Values[0].Timex).ToArray();

            Assert.Equal(first, second);
            Assert.NotEmpty(first);
        }
    }
}
