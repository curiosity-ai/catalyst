using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;
using System.Linq;

namespace Catalyst.Tests.Core
{
    public class TestSpan
    {
        public TestSpan()
        {
            Catalyst.Models.English.Register();
        }

        [Fact]
        public async Task TestSpanProperties()
        {
            var text = "Hello world";
            var doc = new Document(text, Language.English);
            var nlp = await Pipeline.ForAsync(Language.English, tagger: false);
            nlp.ProcessSingle(doc);

            var span = doc.Spans.First();
            Assert.Equal(text, span.Value);
            Assert.Equal(0, span.Begin);
            Assert.Equal(text.Length - 1, span.End);
        }
    }
}
