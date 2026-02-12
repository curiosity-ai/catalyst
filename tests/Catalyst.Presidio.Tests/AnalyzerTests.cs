using Catalyst.Presidio;
using Catalyst.Models;
using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using System.Linq;
using System;

namespace Catalyst.Presidio.Tests
{
    public class AnalyzerTests
    {
        public AnalyzerTests()
        {
            Catalyst.Models.English.Register();
        }

        [Fact]
        public async Task TestEmail()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddEmail();
            var text = "Contact me at user@example.com for more info.";
            var results = analyzer.Analyze(text);

            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "EMAIL_ADDRESS" && text.Substring(r.Start, r.Length) == "user@example.com");
        }

        [Fact]
        public async Task TestUrl()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddUrl();
            var text = "Visit https://www.example.com now.";
            var results = analyzer.Analyze(text);

            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "URL" && text.Substring(r.Start, r.Length) == "https://www.example.com");
        }

        [Fact]
        public async Task TestPhone()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddPhone();
            var text = "Call 555-123-4567 today.";
            var results = analyzer.Analyze(text);

            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "PHONE_NUMBER" && text.Substring(r.Start, r.Length) == "555-123-4567");
        }

        [Fact]
        public async Task TestCreditCard()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddCreditCard();
            var text = "My card is 1234 5678 1234 5678.";
            var results = analyzer.Analyze(text);

            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "CREDIT_CARD" && text.Substring(r.Start, r.Length) == "1234 5678 1234 5678");
        }

        [Fact]
        public async Task TestSsn()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddUsSsn();
            var text = "SSN is 123-45-6789.";
            var results = analyzer.Analyze(text);

            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "US_SSN" && text.Substring(r.Start, r.Length) == "123-45-6789");
        }

        [Fact]
        public async Task TestPassport()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddUsPassport();
            var text = "Passport # 123456789.";
            var results = analyzer.Analyze(text);

            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "US_PASSPORT" && text.Substring(r.Start, r.Length) == "123456789");
        }

        [Fact]
        public async Task TestAll()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddAllRecognizers();
            var text = "Email: test@test.com, SSN: 123-45-6789.";
            var results = analyzer.Analyze(text);

            foreach(var r in results)
            {
                Console.WriteLine($"Found: {r.EntityType} at {r.Start}-{r.End}");
            }

            Assert.Equal(2, results.Count);
            Assert.Contains(results, r => r.EntityType == "EMAIL_ADDRESS");
            Assert.Contains(results, r => r.EntityType == "US_SSN");
        }
    }
}
