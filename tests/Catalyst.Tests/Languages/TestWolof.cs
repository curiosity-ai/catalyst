using Mosaik.Core;
using System.Threading.Tasks;
using Xunit;
using Catalyst.Models;

namespace Catalyst.Tests.Languages
{
    public class TestWolof
    {
        [Fact]
        public async Task BasicTest()
        {
            Catalyst.Models.Wolof.Register();
            // tagger: false due to known serialization issue with MessagePack
            var nlp = await Pipeline.ForAsync(Language.Wolof, tagger: false);
            var doc = new Document("Test string", Language.Wolof);
            nlp.ProcessSingle(doc);
            Assert.True(doc.TokensCount > 0);
        }
    }
}
