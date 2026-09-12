using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestJapanese
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Japanese.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Japanese, tagger: false);
            var doc = new Document("Test string", Language.Japanese);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
