using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Core
{
    public class TestDocument
    {
        public TestDocument()
        {
            Catalyst.Models.English.Register();
        }

        [Fact]
        public async Task TestDocumentCreation()
        {
            var text = "Hello world";
            var doc = new Document(text, Language.English);

            Assert.Equal(text, doc.Value);
            Assert.Equal(text.Length, doc.Length);
            Assert.Equal(Language.English, doc.Language);
        }

        [Fact]
        public async Task TestDocumentProcess()
        {
            var text = "Hello world. This is a test.";
            var doc = new Document(text, Language.English);
            var nlp = await Pipeline.ForAsync(Language.English, tagger: false);
            nlp.ProcessSingle(doc);

            Assert.True(doc.TokensCount > 0);
            Assert.True(doc.SpansCount > 0);
        }
    }
}
