using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestSlovak
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Slovak.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Slovak, tagger: false);
            var doc = new Document("Test string", Language.Slovak);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
