using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestDutch
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Dutch.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Dutch, tagger: false);
            var doc = new Document("Test string", Language.Dutch);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
