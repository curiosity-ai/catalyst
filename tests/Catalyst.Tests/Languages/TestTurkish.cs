using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestTurkish
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Turkish.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Turkish, tagger: false);
            var doc = new Document("Test string", Language.Turkish);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
