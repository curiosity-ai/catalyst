using Catalyst.Presidio;
using Catalyst.Models;
using System.Threading.Tasks;
using Xunit;
using System.Linq;

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
            var analyzer = new PresidioAnalyzer();
            await analyzer.InitializeAsync();
            var text = "Contact me at user@example.com for more info.";
            var results = analyzer.Analyze(text);

            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "EMAIL_ADDRESS" && text.Substring(r.Start, r.Length) == "user@example.com");
        }

        [Fact]
        public async Task TestUrl()
        {
            var analyzer = new PresidioAnalyzer();
            await analyzer.InitializeAsync();
            var text = "Visit https://www.example.com now.";
            var results = analyzer.Analyze(text);

            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "URL" && text.Substring(r.Start, r.Length) == "https://www.example.com");
        }

        [Fact]
        public async Task TestPhone()
        {
            var analyzer = new PresidioAnalyzer();
            await analyzer.InitializeAsync();
            var text = "Call 555-123-4567 today.";
            var results = analyzer.Analyze(text);

            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "PHONE_NUMBER" && text.Substring(r.Start, r.Length) == "555-123-4567");
        }

        [Fact]
        public async Task TestCreditCard()
        {
            var analyzer = new PresidioAnalyzer();
            await analyzer.InitializeAsync();
            var text = "My card is 1234 5678 1234 5678.";
            var results = analyzer.Analyze(text);

            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "CREDIT_CARD" && text.Substring(r.Start, r.Length) == "1234 5678 1234 5678");
        }
    }
}
