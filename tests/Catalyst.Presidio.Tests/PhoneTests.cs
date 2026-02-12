using Catalyst.Presidio;
using Catalyst.Models;
using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using System.Linq;

namespace Catalyst.Presidio.Tests
{
    public class PhoneTests
    {
        public PhoneTests()
        {
            Catalyst.Models.English.Register();
        }

        [Fact]
        public async Task TestPhoneUS()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddPhoneUS();
            var text = "Call 555-123-4567 or (555) 123-4567.";
            var results = analyzer.Analyze(text);

            Assert.Contains(results, r => r.EntityType == "PHONE_US" && text.Substring(r.Start, r.Length) == "555-123-4567");
            Assert.Contains(results, r => r.EntityType == "PHONE_US" && text.Substring(r.Start, r.Length) == "(555) 123-4567");
        }

        [Fact]
        public async Task TestPhoneUK()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddPhoneUK();
            var text = "Mobile: 07700 900077, London: 020 7946 0123, Intl: +44 1632 960960";
            var results = analyzer.Analyze(text);

            Assert.Contains(results, r => r.EntityType == "PHONE_UK" && text.Substring(r.Start, r.Length) == "07700 900077");
            Assert.Contains(results, r => r.EntityType == "PHONE_UK" && text.Substring(r.Start, r.Length) == "020 7946 0123");
            Assert.Contains(results, r => r.EntityType == "PHONE_UK" && text.Substring(r.Start, r.Length) == "+44 1632 960960");
        }

        [Fact]
        public async Task TestPhoneDE()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddPhoneDE();
            var text = "Berlin: 030 123456, Intl: +49 30 123456";
            var results = analyzer.Analyze(text);

            Assert.Contains(results, r => r.EntityType == "PHONE_DE" && text.Substring(r.Start, r.Length) == "030 123456");
            Assert.Contains(results, r => r.EntityType == "PHONE_DE" && text.Substring(r.Start, r.Length) == "+49 30 123456");
        }

        [Fact]
        public async Task TestPhoneFR()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddPhoneFR();
            var text = "Paris: 01 23 45 67 89, Intl: +33 1 23 45 67 89";
            var results = analyzer.Analyze(text);

            Assert.Contains(results, r => r.EntityType == "PHONE_FR" && text.Substring(r.Start, r.Length) == "01 23 45 67 89");
            Assert.Contains(results, r => r.EntityType == "PHONE_FR" && text.Substring(r.Start, r.Length) == "+33 1 23 45 67 89");
        }

        [Fact]
        public async Task TestPhoneBR()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddPhoneBR();
            var text = "SP: (11) 91234-5678, Intl: +55 11 91234-5678";
            var results = analyzer.Analyze(text);

            Assert.Contains(results, r => r.EntityType == "PHONE_BR" && text.Substring(r.Start, r.Length) == "(11) 91234-5678");
            Assert.Contains(results, r => r.EntityType == "PHONE_BR" && text.Substring(r.Start, r.Length) == "+55 11 91234-5678");
        }

        [Fact]
        public async Task TestPhoneAU()
        {
            var analyzer = PresidioAnalyzer.For(Language.English).AddPhoneAU();
            var text = "Sydney: 02 1234 5678, Intl: +61 2 1234 5678";
            var results = analyzer.Analyze(text);

            Assert.Contains(results, r => r.EntityType == "PHONE_AU" && text.Substring(r.Start, r.Length) == "02 1234 5678");
            Assert.Contains(results, r => r.EntityType == "PHONE_AU" && text.Substring(r.Start, r.Length) == "+61 2 1234 5678");
        }
    }
}
