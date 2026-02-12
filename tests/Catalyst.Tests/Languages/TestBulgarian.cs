using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestBulgarian
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Bulgarian.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Bulgarian);
            var doc = new Document("Test string", Language.Bulgarian);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
