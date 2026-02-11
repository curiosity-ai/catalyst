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
                new RecognizerResult { Start = 12, End = 27, EntityType = "EMAIL_ADDRESS" } // "user@example.com"
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
                new RecognizerResult { Start = 12, End = 27, EntityType = "EMAIL_ADDRESS" }
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
                new RecognizerResult { Start = 12, End = 27, EntityType = "EMAIL_ADDRESS" }
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
                new RecognizerResult { Start = 7, End = 22, EntityType = "EMAIL_ADDRESS" },
                new RecognizerResult { Start = 32, End = 39, EntityType = "PHONE_NUMBER" }
            };

            var anonymized = anonymizer.Anonymize(text, results, "mask", "*");
            Assert.Equal("Email: ****************, Phone: ********.", anonymized);
        }
    }
}
