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
        public async Task TestIban()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddIban();
            var text = "IBAN: GB12345678901234.";
            var results = analyzer.Analyze(text);

            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "IBAN_CODE" && text.Substring(r.Start, r.Length) == "GB12345678901234");
        }

        [Fact]
        public async Task TestItin()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddUsItin();
            var text = "ITIN: 987-65-4321.";
            var results = analyzer.Analyze(text);

            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "US_ITIN" && text.Substring(r.Start, r.Length) == "987-65-4321");
        }

        [Fact]
        public async Task TestCrypto()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddCrypto();
            var text = "Send BTC to 1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2.";
            var results = analyzer.Analyze(text);

            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "CRYPTO" && text.Substring(r.Start, r.Length) == "1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2");
        }

        [Fact]
        public async Task TestUsBank()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddUsBankNumber();
            var text = "Routing: 123456789";
            var results = analyzer.Analyze(text);
            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "US_BANK_NUMBER" && text.Substring(r.Start, r.Length) == "123456789");
        }

        [Fact]
        public async Task TestUkNhs()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddUkNhs();
            var text = "NHS: 123 456 7890";
            var results = analyzer.Analyze(text);
            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "UK_NHS" && text.Substring(r.Start, r.Length) == "123 456 7890");
        }

        [Fact]
        public async Task TestEsNif()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddEsNif();
            var text = "NIF: 12345678Z";
            var results = analyzer.Analyze(text);
            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "ES_NIF" && text.Substring(r.Start, r.Length) == "12345678Z");
        }

        [Fact]
        public async Task TestItFiscal()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddItFiscalCode();
            var text = "CF: RSSMRA80A01H501U";
            var results = analyzer.Analyze(text);
            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "IT_FISCAL_CODE" && text.Substring(r.Start, r.Length) == "RSSMRA80A01H501U");
        }

        [Fact]
        public async Task TestSgNric()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddSgNric();
            var text = "NRIC: S1234567D";
            var results = analyzer.Analyze(text);
            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "SG_NRIC_FIN" && text.Substring(r.Start, r.Length) == "S1234567D");
        }

        [Fact]
        public async Task TestAuAbn()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddAuAbn();
            var text = "ABN: 51 824 753 556";
            var results = analyzer.Analyze(text);
            Assert.NotEmpty(results);
            Assert.Contains(results, r => r.EntityType == "AU_ABN" && text.Substring(r.Start, r.Length) == "51 824 753 556");
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
