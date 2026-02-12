using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestGaelic_Scottish_Gaelic
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Gaelic_Scottish_Gaelic.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Gaelic_Scottish_Gaelic, tagger: false);
            var doc = new Document("Test string", Language.Gaelic_Scottish_Gaelic);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
