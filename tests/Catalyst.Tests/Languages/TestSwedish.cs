using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestSwedish
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Swedish.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Swedish, tagger: false);
            var doc = new Document("Test string", Language.Swedish);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
