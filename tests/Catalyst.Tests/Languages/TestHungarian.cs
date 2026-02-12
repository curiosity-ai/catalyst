using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestHungarian
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Hungarian.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Hungarian, tagger: false);
            var doc = new Document("Test string", Language.Hungarian);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
