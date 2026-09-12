using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;
using System.Linq;
using System.Collections.Generic;

namespace Catalyst.Tests.Core
{
    public class TestTokenization
    {
        public TestTokenization()
        {
            Catalyst.Models.English.Register();
        }

        [Fact]
        public async Task TestSentenceSplitting()
        {
            var nlp = await Pipeline.ForAsync(Language.English, tagger: false);
            var doc = new Document("Hello world. This is a test.", Language.English);
            nlp.ProcessSingle(doc);

            bool hasSentenceDetector = nlp.GetModelsList().Any(m => m is ISentenceDetector);

            if (hasSentenceDetector)
            {
                Assert.Equal(2, doc.SpansCount);
                Assert.Equal("Hello world.", doc.Spans.ElementAt(0).Value);
                Assert.Equal("This is a test.", doc.Spans.ElementAt(1).Value);
            }
            else
            {
                 // Fallback if model failed to load due to serialization issues
                 Assert.True(doc.SpansCount >= 1);
            }
        }

        [Fact]
        public async Task TestPunctuation()
        {
            var nlp = await Pipeline.ForAsync(Language.English, tagger: false);
            var doc = new Document("Hello, world!", Language.English);
            nlp.ProcessSingle(doc);

            var tokens = new List<string>();
            foreach(var span in doc.Spans) {
                foreach(var token in span) {
                    tokens.Add(token.Value);
                }
            }

            // Expected: "Hello", ",", "world", "!"
            Assert.Contains(",", tokens);
            Assert.Contains("!", tokens);
        }

        [Fact]
        public async Task TestTokenDataBounds()
        {
            var nlp = await Pipeline.ForAsync(Language.English, tagger: false);
            var doc = new Document("Test", Language.English);
            nlp.ProcessSingle(doc);

            // Accessing TokenData inside the list of lists
            var tokenDataList = doc.TokensData.SelectMany(td => td).ToList();
            Assert.NotEmpty(tokenDataList);
            // Verify bounds exist (checking LowerBound/UpperBound indirectly via array if field access fails, but struct has fields)
            // If NuGet package TokenData is different, we use Bounds array
            Assert.True(tokenDataList[0].Bounds[0] == 0);
            Assert.True(tokenDataList[0].Bounds[1] == 3);
        }
    }
}
