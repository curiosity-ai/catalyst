using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;
using System.Linq;

namespace Catalyst.Tests.Core
{
    public class TestTagging
    {
        public TestTagging()
        {
            Catalyst.Models.English.Register();
        }

        [Fact]
        public async Task TestPOSTagging()
        {
            var text = "The cat sat on the mat.";
            var doc = new Document(text, Language.English);
            var nlp = await Pipeline.ForAsync(Language.English); // Requires tagger
            nlp.ProcessSingle(doc);

            var tokens = doc.Spans.First().ToList();

            // Just check that POS is not NONE
            Assert.NotEqual(PartOfSpeech.NONE, tokens[0].POS);
            Assert.NotEqual(PartOfSpeech.NONE, tokens[1].POS);
        }
    }
}
