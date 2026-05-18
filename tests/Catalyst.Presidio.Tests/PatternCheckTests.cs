using Catalyst;
using Catalyst.Models;
using Mosaik.Core;
using System;
using System.Linq;
using Xunit;

namespace Catalyst.Presidio.Tests
{
    public class PatternCheckTests
    {
        [Fact]
        public void CheckTokens()
        {
            Catalyst.Models.English.Register();
            var tokenizer = new FastTokenizer(Language.English);

            var texts = new[]
            {
                "SSN: 123-45-6789",
                "Driver: A1234567",
                "Passport: 123456789",
                "IBAN: GB12345678901234"
            };

            foreach (var text in texts)
            {
                var doc = new Document(text, Language.English);
                tokenizer.Process(doc);
                Console.WriteLine($"Text: {text}");
                foreach (var token in doc.Spans.SelectMany(s => s.Tokens))
                {
                    Console.WriteLine($"Token: '{token.Value}' Shape: '{token.ValueAsSpan.Shape()}'");
                }
                Console.WriteLine();
            }
        }
    }
}
