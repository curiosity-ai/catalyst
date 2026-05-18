using Catalyst;
using Catalyst.Models;
using Mosaik.Core;
using System;
using System.Linq;
using Xunit;

namespace Catalyst.Presidio.Tests
{
    public class PatternInvestigationTests
    {
        [Fact]
        public void Investigate()
        {
            Catalyst.Models.English.Register();
            var tokenizer = new FastTokenizer(Language.English);

            var samples = new[]
            {
                "US_BANK: 123456789",
                "UK_NHS_1: 123 456 7890",
                "UK_NHS_2: 1234567890",
                "ES_NIF_1: 12345678Z",
                "ES_NIF_2: X1234567Z",
                "IT_FISCAL: RSSMRA80A01H501U",
                "SG_NRIC: S1234567D",
                "AU_ABN_1: 51 824 753 556",
                "AU_ABN_2: 51824753556"
            };

            foreach (var text in samples)
            {
                var doc = new Document(text, Language.English);
                tokenizer.Process(doc);
                Console.WriteLine($"Text: {text}");
                foreach (var token in doc.Spans.SelectMany(s => s.Tokens))
                {
                    Console.WriteLine($"  Token: '{token.Value}' Shape: '{token.ValueAsSpan.Shape()}' Len: {token.Length}");
                }
                Console.WriteLine();
            }
        }
    }
}
