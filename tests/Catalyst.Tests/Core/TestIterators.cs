using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;
using System.Linq;

namespace Catalyst.Tests.Core
{
    public class TestIterators
    {
        public TestIterators()
        {
            Catalyst.Models.English.Register();
        }

        [Fact]
        public async Task TestTokenIteration()
        {
            var text = "Hello world";
            var doc = new Document(text, Language.English);
            var nlp = await Pipeline.ForAsync(Language.English, tagger: false);
            nlp.ProcessSingle(doc);

            int count = 0;
            foreach(var span in doc)
            {
                foreach(var token in span)
                {
                    count++;
                }
            }
            Assert.Equal(2, count);
        }

        [Fact]
        public async Task TestSpanIteration()
        {
            var text = "Hello world. This is test.";
            var doc = new Document(text, Language.English);
            var nlp = await Pipeline.ForAsync(Language.English, tagger: false);
            nlp.ProcessSingle(doc);

            bool hasSentenceDetector = nlp.GetModelsList().Any(m => m is ISentenceDetector);

            int count = 0;
            foreach(var span in doc)
            {
                count++;
            }

            if (hasSentenceDetector)
            {
                Assert.Equal(2, count);
            }
            else
            {
                Assert.True(count >= 1);
            }
        }
    }
}
