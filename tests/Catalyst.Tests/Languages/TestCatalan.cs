using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestCatalan
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Catalan.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Catalan, tagger: false);
            var doc = new Document("Test string", Language.Catalan);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
