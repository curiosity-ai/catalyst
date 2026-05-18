using Catalyst.Presidio;
using System.Collections.Generic;
using Xunit;

namespace Catalyst.Presidio.Tests
{
    public class AnonymizerTests
    {
        [Fact]
        public void TestMask()
        {
            var anonymizer = new PresidioAnonymizer();
            var text = "My email is user@example.com.";
            var results = new List<RecognizerResult>
            {
                new RecognizerResult(12, 27, "EMAIL_ADDRESS", 1f)
            };

            var anonymized = anonymizer.Anonymize(text, results, "mask", "*");
            Assert.Equal("My email is ****************.", anonymized);
        }

        [Fact]
        public void TestReplace()
        {
            var anonymizer = new PresidioAnonymizer();
            var text = "My email is user@example.com.";
            var results = new List<RecognizerResult>
            {
                new RecognizerResult(12, 27, "EMAIL_ADDRESS", 1f)
            };

            var anonymized = anonymizer.Anonymize(text, results, "replace", replacement: "<EMAIL>");
            Assert.Equal("My email is <EMAIL>.", anonymized);
        }

        [Fact]
        public void TestRedact()
        {
            var anonymizer = new PresidioAnonymizer();
            var text = "My email is user@example.com.";
            var results = new List<RecognizerResult>
            {
                new RecognizerResult(12, 27, "EMAIL_ADDRESS", 1f)
            };

            var anonymized = anonymizer.Anonymize(text, results, "redact");
            Assert.Equal("My email is .", anonymized);
        }

        [Fact]
        public void TestMultiple()
        {
            var anonymizer = new PresidioAnonymizer();
            var text = "Email: user@example.com, Phone: 555-1234.";
            // "user@example.com" -> 16 chars. Start: 7. End: 22.
            // "555-1234" -> 8 chars. Start: 32. End: 39.
            var results = new List<RecognizerResult>
            {
                new RecognizerResult(7, 22, "EMAIL_ADDRESS", 1f),
                new RecognizerResult(32, 39, "PHONE_NUMBER", 1f)
            };

            var anonymized = anonymizer.Anonymize(text, results, "mask", "*");
            Assert.Equal("Email: ****************, Phone: ********.", anonymized);
        }
    }
}
