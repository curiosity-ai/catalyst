using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestUkrainian
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Ukrainian.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Ukrainian, tagger: false);
            var doc = new Document("Test string", Language.Ukrainian);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
