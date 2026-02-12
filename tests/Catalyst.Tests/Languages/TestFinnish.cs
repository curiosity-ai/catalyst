using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestFinnish
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Finnish.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Finnish, tagger: false);
            var doc = new Document("Test string", Language.Finnish);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
